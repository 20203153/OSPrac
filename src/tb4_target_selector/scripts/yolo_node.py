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

import cv2
import torch

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from cv_bridge import CvBridge, CvBridgeError

from tb4_target_selector.srv import BoxQuery


class YoloNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_node")

        # Parameters ---------------------------------------------------------
        self.declare_parameter("model_path", "/path/to/yolov5n_custom.pt")
        # TurtleBot4 OAK-D 기본 토픽 이름에 맞춘 기본값
        #  - RGB:  /oakd/rgb/preview/image_raw (또는 /oakd/rgb/image_raw)
        #  - Depth: (옵션) /oakd/depth/image_raw
        #          빈 문자열("")이면 depth 사용 안 함 → 항상 base_link + assumed_distance 로 fallback
        self.declare_parameter("rgb_topic", "/oakd/rgb/preview/image_raw")
        self.declare_parameter("camera_info_topic", "/oakd/rgb/camera_info")
        self.declare_parameter("target_classes", ["box"])
        self.declare_parameter("yolo_conf_threshold", 0.4)
        self.declare_parameter("service_name", "/box_query")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("assumed_distance_m", 1.5)
        self.declare_parameter("trigger_cooldown_sec", 1.0)
        # YOLOv5n 평가 최소 간격(초). 기본 0.5초 → 최대 2Hz
        self.declare_parameter("yolo_eval_min_interval_sec", 0.5)

        model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.rgb_topic = (
            self.get_parameter("rgb_topic").get_parameter_value().string_value
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
        self.yolo_eval_min_interval_sec: float = (
            self.get_parameter("yolo_eval_min_interval_sec")
            .get_parameter_value()
            .double_value
        )

        # Load YOLOv5n model via torch.hub (CPU/GPU as available)
        self.get_logger().info(f"Loading YOLOv5n model from: {model_path}")
        try:
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=model_path,
                source="github",  # change to 'local' if using a local clone
            )
            self.model.eval()
        except Exception as e:
            self.get_logger().error(
                f"Failed to load YOLOv5n model from '{model_path}': {e}"
            )
            raise

        # 클래스 이름 캐시 (dict 또는 list)
        self.class_names = self.model.names

        self.bridge = CvBridge()

        # Service client -----------------------------------------------------
        self.box_query_client = self.create_client(BoxQuery, self.service_name)

        # Debug image publisher (/selector/debug, bounding box overlay)
        self.debug_pub = self.create_publisher(
            Image,
            "/selector/debug",
            10,
        )

        # Time-based trigger debouncing (for BoxQuery)
        self.last_trigger_time: Optional[float] = None
        # YOLOv5n 평가 주기 제한용 타임스탬프
        self.last_yolo_eval_time: Optional[float] = None

        # Subscribers --------------------------------------------------------
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10,
        )

        self.get_logger().info(
            "YoloNode initialized:\n"
            f"  model_path={model_path}\n"
            f"  rgb_topic={self.rgb_topic}\n"
            f"  target_classes={self.target_classes}\n"
            f"  yolo_conf_threshold={self.yolo_conf_threshold}\n"
            f"  base_frame={self.base_frame}\n"
            f"  service_name={self.service_name}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def rgb_callback(self, msg: Image) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # YOLOv5n 평가 주기 제한 (최대 1 / yolo_eval_min_interval_sec Hz)
        if (
            self.last_yolo_eval_time is not None
            and now_sec - self.last_yolo_eval_time < self.yolo_eval_min_interval_sec
        ):
            return

        # BoxQuery 서비스 유무와 관계없이 YOLO 추론 및 /selector/debug publish 는 항상 수행
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB cv_bridge error: {e}")
            return

        # YOLOv5 inference (torch.hub, numpy BGR image 가능)
        # 평가 간격 제한을 통과한 시점에 타임스탬프 갱신
        self.last_yolo_eval_time = now_sec
        results = self.model(rgb)
        # YOLOv5: results.xyxy[0] -> tensor [x1, y1, x2, y2, conf, cls]
        boxes = results.xyxy[0]
        names = self.class_names

        if boxes is None or len(boxes) == 0:
            # detection 이 없어도 디버그용 이미지는 항상 publish
            self._publish_debug_image(
                rgb=rgb,
                header=msg.header,
                bbox=None,
                class_name="",
                conf=0.0,
            )
            return

        # 디버깅용: YOLO raw detection 개수 로그
        self.get_logger().info(f"YOLO raw detections: {len(boxes)}")

        # Select best suspicious detection
        best_det = None
        best_conf = 0.0
        for det in boxes:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            conf = float(conf)
            cls_id = int(cls_id)
            if isinstance(names, dict):
                class_name = names.get(cls_id, str(cls_id))
            else:
                # names 가 list/tuple 인 경우
                if 0 <= cls_id < len(names):
                    class_name = names[cls_id]
                else:
                    class_name = str(cls_id)

            if class_name not in self.target_classes:
                continue
            if conf < self.yolo_conf_threshold:
                continue
            if conf > best_conf:
                best_conf = conf
                best_det = (x1, y1, x2, y2, conf, class_name)

        if best_det is None:
            # 필터링 후 남은 박스가 없어도 현재 프레임을 디버그 토픽으로 publish
            self._publish_debug_image(
                rgb=rgb,
                header=msg.header,
                bbox=None,
                class_name="",
                conf=0.0,
            )
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

        header_frame_id = self.base_frame
        target_point = Point()
        # Depth/TF를 사용하지 않고 항상 base_link 기준 고정 거리로 타겟 포인트 생성
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

        # BoxQuery 서비스가 준비된 경우에만 호출
        if self.box_query_client.service_is_ready():
            future = self.box_query_client.call_async(req)
            future.add_done_callback(self._box_query_response_cb)
        else:
            self.get_logger().warn(
                "BoxQuery service not available; skipping /box_query call. "
                "YOLO and /selector/debug are still running."
            )

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
        """Publish debug image (raw or with YOLO bounding box) to /selector/debug."""
        if self.debug_pub is None:
            return

        img = rgb.copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

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
        # bbox 가 None 이면, 아무 박스도 그리지 않은 raw 이미지 그대로 publish

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"Failed to convert debug image: {e}")
            return

        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


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