#!/usr/bin/env python3
"""
yolo_node: YOLOv5n-based target suspicion detector node for TurtleBot4.

- Subscribes to camera RGB (and optional depth, camera_info)
- Runs a YOLOv5n custom model on CPU
- Filters detections by target_classes and confidence
- For the best suspicious detection, estimates a target_point
  in either map or base_link frame
- Calls `/box_query` (tb4_target_selector/BoxQuery) service on cv2_node
- Logs the returned sorted box candidates.

This implements the "yolo_node" side of the service-based architecture.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
import tf_transformations

from cv_bridge import CvBridge, CvBridgeError

from tb4_target_selector.srv import BoxQuery


class CameraIntrinsics:
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


class YoloNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_node")

        # Parameters ---------------------------------------------------------
        self.declare_parameter("model_path", "/path/to/yolov5n_custom.pt")
        # TurtleBot4 OAK-D 기본 토픽 이름에 맞춘 기본값
        self.declare_parameter("rgb_topic", "/oakd/rgb/image_raw")
        self.declare_parameter("depth_topic", "/oakd/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/oakd/rgb/camera_info")
        self.declare_parameter("target_classes", ["person", "chair", "box"])
        self.declare_parameter("yolo_conf_threshold", 0.4)
        self.declare_parameter("service_name", "/box_query")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("assumed_distance_m", 1.5)
        self.declare_parameter("trigger_cooldown_sec", 1.0)

        model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.rgb_topic = (
            self.get_parameter("rgb_topic").get_parameter_value().string_value
        )
        self.depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        self.camera_info_topic = (
            self.get_parameter("camera_info_topic")
            .get_parameter_value()
            .string_value
        )
        self.target_classes: List[str] = list(
            self.get_parameter("target_classes")
            .get_parameter_value()
            .string_array_value
        )
        self.yolo_conf_threshold: float = (
            self.get_parameter("yolo_conf_threshold")
            .get_parameter_value()
            .double_value
        )
        self.service_name: str = (
            self.get_parameter("service_name").get_parameter_value().string_value
        )
        self.map_frame: str = (
            self.get_parameter("map_frame").get_parameter_value().string_value
        )
        self.base_frame: str = (
            self.get_parameter("base_frame").get_parameter_value().string_value
        )
        self.camera_frame: str = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self.assumed_distance_m: float = (
            self.get_parameter("assumed_distance_m")
            .get_parameter_value()
            .double_value
        )
        self.trigger_cooldown_sec: float = (
            self.get_parameter("trigger_cooldown_sec")
            .get_parameter_value()
            .double_value
        )

        # Load YOLOv5 model (CPU)
        self.get_logger().info(f"Loading YOLOv5 model from: {model_path}")
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_path,
            source="github",
        )
        self.model.eval()

        # Segmentation 모델(yolov5n-seg.pt 등)을 사용하면 AutoShape 미지원으로 인해
        # self.model(rgb) 호출 시 4D 텐서를 기대하다가 3D로 들어와서
        # "ValueError: not enough values to unpack (expected 4, got 3)" 가 발생한다.
        # 이 yolo_node 는 "바운딩 박스 탐지"만 필요하므로, 세그멘테이션 모델은 지원하지 않고
        # 초기화 단계에서 명시적으로 막아준다.
        model_path_lower = model_path.lower()
        if "seg" in model_path_lower:
            err_msg = (
                "Loaded YOLOv5 segmentation model (path contains 'seg'). "
                "This yolo_node currently supports ONLY detection models that output "
                "bounding boxes (e.g. yolov5n.pt or a custom detection .pt), "
                "not segmentation models (yolov5n-seg.pt). "
                "Please switch to a detection model to avoid "
                "ValueError: expected 4D tensor in SegmentationModel.forward()."
            )
            self.get_logger().error(err_msg)
            raise RuntimeError(err_msg)

        # TF / camera / depth state ------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()

        self.intrinsics: Optional[CameraIntrinsics] = None
        self.latest_depth: Optional[np.ndarray] = None

        # Service client -----------------------------------------------------
        self.box_query_client = self.create_client(BoxQuery, self.service_name)

        # Debug image publisher (/selector/debug, bounding box overlay)
        self.debug_pub = self.create_publisher(
            Image,
            "/selector/debug",
            10,
        )

        # Time-based trigger debouncing
        self.last_trigger_time: Optional[float] = None

        # Subscribers --------------------------------------------------------
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.get_logger().info(
            "YoloNode initialized:\n"
            f"  model_path={model_path}\n"
            f"  rgb_topic={self.rgb_topic}\n"
            f"  depth_topic={self.depth_topic}\n"
            f"  camera_info_topic={self.camera_info_topic}\n"
            f"  target_classes={self.target_classes}\n"
            f"  yolo_conf_threshold={self.yolo_conf_threshold}\n"
            f"  map_frame={self.map_frame}, base_frame={self.base_frame}, "
            f"camera_frame={self.camera_frame}\n"
            f"  service_name={self.service_name}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if len(msg.k) >= 9:
            fx = msg.k[0]
            fy = msg.k[4]
            cx = msg.k[2]
            cy = msg.k[5]
            self.intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    def depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except CvBridgeError:
            try:
                depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
                depth = depth_raw.astype(np.float32) / 1000.0
            except CvBridgeError as e:
                self.get_logger().error(f"Depth cv_bridge error: {e}")
                return

        self.latest_depth = depth

    def rgb_callback(self, msg: Image) -> None:
        # Debounce
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (
            self.last_trigger_time is not None
            and now_sec - self.last_trigger_time < self.trigger_cooldown_sec
        ):
            return

        if not self.box_query_client.service_is_ready():
            # Try once to wait for service
            if not self.box_query_client.wait_for_service(timeout_sec=0.1):
                return

        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB cv_bridge error: {e}")
            return

        # YOLOv5 inference
        results = self.model(rgb)
        boxes = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]
        names = results.names

        if boxes is None or len(boxes) == 0:
            return

        # Select best suspicious detection
        best_det = None
        best_conf = 0.0
        for det in boxes:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            conf = float(conf)
            cls_id = int(cls_id)
            class_name = names.get(cls_id, str(cls_id))

            if class_name not in self.target_classes:
                continue
            if conf < self.yolo_conf_threshold:
                continue
            if conf > best_conf:
                best_conf = conf
                best_det = (x1, y1, x2, y2, conf, class_name)

        if best_det is None:
            return

        x1, y1, x2, y2, conf, class_name = best_det
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)

        # 디버그용: 선택된 바운딩 박스를 그린 이미지를 /selector/debug 로 publish
        self._publish_debug_image(
            rgb=rgb,
            header=msg.header,
            bbox=(x1, y1, x2, y2),
            class_name=class_name,
            conf=conf,
        )

        header_frame_id = self.map_frame
        target_point = Point()

        # Try map-frame target using depth + TF
        success = self._estimate_target_point_map(
            u=u,
            v=v,
            stamp=msg.header.stamp,
            out_point=target_point,
        )
        if not success:
            # Fallback: base_link frame, fixed distance in front
            header_frame_id = self.base_frame
            target_point.x = self.assumed_distance_m
            target_point.y = 0.0
            target_point.z = 0.0

        # Prepare service request
        req = BoxQuery.Request()
        req.header.stamp = msg.header.stamp
        req.header.frame_id = header_frame_id
        req.target_point = target_point
        req.target_class = class_name
        req.confidence = float(conf)

        future = self.box_query_client.call_async(req)
        future.add_done_callback(self._box_query_response_cb)

        self.last_trigger_time = now_sec

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _publish_debug_image(
        self,
        rgb,
        header,
        bbox,
        class_name: str,
        conf: float,
    ) -> None:
        """Publish debug image with YOLO bounding box overlay to /selector/debug."""
        if self.debug_pub is None:
            return

        x1, y1, x2, y2 = bbox
        img = rgb.copy()

        # 바운딩 박스 그리기
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

        # 라벨 텍스트 (클래스 + confidence)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            img,
            label,
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"Failed to convert debug image: {e}")
            return

        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

    def _estimate_target_point_map(
        self,
        u: float,
        v: float,
        stamp,
        out_point: Point,
    ) -> bool:
        """Estimate a 3D point in map frame from pixel (u, v) using depth + TF."""
        if self.latest_depth is None or self.intrinsics is None:
            return False

        depth = self.latest_depth
        h, w = depth.shape[:2]
        u_i = int(round(u))
        v_i = int(round(v))
        if not (0 <= u_i < w and 0 <= v_i < h):
            return False

        z = float(depth[v_i, u_i])
        if not np.isfinite(z) or z <= 0.0:
            patch = depth[
                max(0, v_i - 2) : min(h, v_i + 3),
                max(0, u_i - 2) : min(w, u_i + 3),
            ]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]
            if valid.size == 0:
                return False
            z = float(np.median(valid))

        intr = self.intrinsics
        X_cam = (u - intr.cx) * z / intr.fx
        Y_cam = (v - intr.cy) * z / intr.fy
        Z_cam = z

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                stamp,
                timeout=Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF lookup failed (map<-{self.camera_frame}) in YOLO node: {e}"
            )
            return False

        T = self._transform_to_matrix(transform)
        p_cam = np.array([X_cam, Y_cam, Z_cam, 1.0], dtype=np.float64)
        p_map = T @ p_cam

        out_point.x = float(p_map[0])
        out_point.y = float(p_map[1])
        out_point.z = float(p_map[2])
        return True

    def _transform_to_matrix(self, transform) -> np.ndarray:
        t = transform.transform.translation
        q = transform.transform.rotation
        T = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z
        return T

    def _box_query_response_cb(self, future) -> None:
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"BoxQuery call failed: {e}")
            return

        num = len(resp.candidates)
        self.get_logger().info(f"Received {num} box candidates from cv2_node.")
        if num > 0:
            best = resp.candidates[0]
            self.get_logger().info(
                "  Best candidate: "
                f"score={best.score:.3f}, "
                f"pos=({best.pose.position.x:.2f}, {best.pose.position.y:.2f}), "
                f"size=({best.width:.2f} x {best.height:.2f})"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YoloNode (KeyboardInterrupt).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()