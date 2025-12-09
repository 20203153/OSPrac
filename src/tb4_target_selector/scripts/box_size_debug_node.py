#!/usr/bin/env python3
"""
box_size_debug_node: YOLOv5 + 크기 필터 디버그 노드 (TurtleBot4, ROS2 Jazzy)

기능 요약
---------
- RGB 카메라 이미지를 구독하여 YOLOv5 추론 수행
- LiDAR(/scan) 기반 거리 추정값을 이용해 bbox → 실제 크기 추정
- new_func.detect_func 의
    - detect_box  (width & length 기반, 바닥 footprint)
    - detect_box_by_height (height 기반)
  을 선택적으로 사용해 우체국 박스 4호인지 판별
- 카메라 + YOLO bounding box overlay 이미지를 /detector/debug 로 최대 30Hz 로 publish

사용 시나리오
-------------
- width_mode=True  → detect_box (footprint) 기준 테스트
- width_mode=False → detect_box_by_height (height-only) 기준 테스트
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

from new_func.detect_func import (
    detect_box,
    detect_box_by_height,
    estimate_object_size_from_bbox,
)
from new_func.yolov5_singleton import (
    load_yolov5_model,
    run_yolov5_inference,
)


class BoxSizeDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("box_size_debug_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("model_path", "yolov5n.pt")
        self.declare_parameter("rgb_topic", "/oakd/rgb/preview/image_raw")
        self.declare_parameter("camera_info_topic", "/oakd/rgb/preview/camera_info")
        # 최대 30Hz 정도 → 주기 ~0.033초
        self.declare_parameter("yolo_eval_min_interval_sec", 1.0 / 30.0)
        # width_mode=True  → width & length 기반 detect_box 사용
        # width_mode=False → height-only 기반 detect_box_by_height 사용
        self.declare_parameter("width_mode", True)
        # 박스까지의 거리 [m] (테스트 편의를 위해 파라미터로 지정, 기본 1.0m)
        self.declare_parameter("distance_to_box_m", 1.0)
        # YOLO 입력 해상도
        self.declare_parameter("img_size", 640)
        # YOLO confidence threshold
        self.declare_parameter("conf_threshold", 0.4)
        # device: "cpu", "cuda", "mps" 등
        self.declare_parameter("device", "cpu")

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
        self.yolo_eval_min_interval_sec = (
            self.get_parameter("yolo_eval_min_interval_sec")
            .get_parameter_value()
            .double_value
        )
        self.width_mode: bool = (
            self.get_parameter("width_mode").get_parameter_value().bool_value
        )
        self.distance_to_box_m: float = (
            self.get_parameter("distance_to_box_m")
            .get_parameter_value()
            .double_value
        )
        self.img_size: int = (
            self.get_parameter("img_size").get_parameter_value().integer_value
        )
        self.conf_threshold: float = (
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )
        self.device: str = (
            self.get_parameter("device").get_parameter_value().string_value
        )

        # ------------------------------------------------------------------
        # Load YOLOv5 model (Singleton)
        # ------------------------------------------------------------------
        self.get_logger().info(
            f"[BoxSizeDebugNode] Loading YOLOv5 model: {model_path} (device={self.device})"
        )
        try:
            _ = load_yolov5_model(
                model_path=model_path,
                device=self.device,
                use_half=False,
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLOv5 model: {e}")
            raise

        # ------------------------------------------------------------------
        # State (latest messages)
        # ------------------------------------------------------------------
        self.bridge = CvBridge()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_header: Optional[Image] = None
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None

        self.last_yolo_eval_time: Optional[float] = None

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.debug_pub = self.create_publisher(
            Image,
            "/detector/debug",
            10,
        )

        # ------------------------------------------------------------------
        # Subscriptions
        # ------------------------------------------------------------------
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        # ------------------------------------------------------------------
        # Timer: YOLO eval 주기 제한 (최대 30Hz)
        # ------------------------------------------------------------------
        self.timer = self.create_timer(
            self.yolo_eval_min_interval_sec,
            self.timer_callback,
        )

        self.get_logger().info(
            "BoxSizeDebugNode initialized.\n"
            f"  model_path={model_path}\n"
            f"  rgb_topic={self.rgb_topic}\n"
            f"  camera_info_topic={self.camera_info_topic}\n"
            f"  width_mode={self.width_mode}\n"
            f"  distance_to_box_m={self.distance_to_box_m}\n"
            f"  yolo_eval_min_interval_sec={self.yolo_eval_min_interval_sec}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def rgb_callback(self, msg: Image) -> None:
        """가장 최근 RGB 프레임을 저장만 해 두고, YOLO 는 timer 에서 수행."""
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB cv_bridge error: {e}")
            return

        self.latest_rgb = rgb
        self.latest_rgb_header = msg.header

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """CameraInfo 에서 fx, fy 를 추출해 저장."""
        try:
            k = msg.k
            self.fx = float(k[0])  # fx
            self.fy = float(k[4])  # fy
        except Exception as e:
            self.get_logger().warn(f"Failed to parse CameraInfo: {e}")

    # ------------------------------------------------------------------
    # Timer: YOLO + detect_box / detect_box_by_height
    # ------------------------------------------------------------------

    def timer_callback(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # 1) 필수 데이터 확인
        if self.latest_rgb is None or self.latest_rgb_header is None:
            self.get_logger().debug("No RGB frame yet; skip YOLO.")
            return
        if self.fx is None or self.fy is None:
            self.get_logger().debug("No CameraInfo yet (fx, fy); skip YOLO.")
            return

        # LiDAR 대신 파라미터 distance_to_box_m 를 그대로 사용
        distance_m = self.distance_to_box_m

        # 2) YOLO 추론 (최신 프레임 사용)
        bgr = self.latest_rgb.copy()

        results = run_yolov5_inference(
            bgr_image=bgr,
            img_size=self.img_size,
            conf_threshold=self.conf_threshold,
        )
        boxes = results.xyxy[0]  # [N, 6]: x1, y1, x2, y2, conf, cls

        if boxes is None or len(boxes) == 0:
            # detection 이 없어도 원본 이미지를 /detector/debug 로 publish
            self._publish_debug_image(
                bgr=bgr,
                overlay_boxes=None,
            )
            return

        # 3) bbox + 크기 추정 + detect_box / detect_box_by_height
        img_h, img_w = bgr.shape[:2]
        overlay = bgr.copy()

        for det in boxes:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            conf = float(conf)
            cls_id = int(cls_id)

            est_w, est_h = estimate_object_size_from_bbox(
                bbox=det,
                distance_m=distance_m,
                fx=self.fx,
                fy=self.fy,
            )

            if self.width_mode:
                # footprint (width/length) 기준
                is_target = detect_box(
                    bbox=det,
                    distance=distance_m,
                    fx=self.fx,
                    fy=self.fy,
                )
            else:
                # height-only 기준
                is_target = detect_box_by_height(
                    bbox=det,
                    distance=distance_m,
                    fx=self.fx,
                    fy=self.fy,
                )

            # 로그 출력
            self.get_logger().info(
                f"[bbox] cls={cls_id} conf={conf:.3f} "
                f"pix=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                f"size_est=({est_w:.3f}m, {est_h:.3f}m) "
                f"distance={distance_m:.3f}m "
                f"mode={'width&length' if self.width_mode else 'height-only'} "
                f"is_target={is_target}"
            )

            # bbox 그리기
            x1_i, y1_i = int(x1), int(y1)
            x2_i, y2_i = int(x2), int(y2)
            x1_i = max(0, min(img_w - 1, x1_i))
            y1_i = max(0, min(img_h - 1, y1_i))
            x2_i = max(0, min(img_w - 1, x2_i))
            y2_i = max(0, min(img_h - 1, y2_i))

            color = (0, 255, 0) if is_target else (255, 0, 0)
            cv2.rectangle(overlay, (x1_i, y1_i), (x2_i, y2_i), color, 2)

            label = (
                f"{'W/L' if self.width_mode else 'H'} "
                f"{conf:.2f} "
                f"{est_w:.2f}x{est_h:.2f}m"
            )
            cv2.putText(
                overlay,
                label,
                (x1_i, max(0, y1_i - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

        # 4) /detector/debug 로 publish
        self._publish_debug_image(
            bgr=overlay,
            overlay_boxes=True,
        )

        self.last_yolo_eval_time = now_sec

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _publish_debug_image(
        self,
        bgr: np.ndarray,
        overlay_boxes: Optional[bool],
    ) -> None:
        """BGR 이미지를 /detector/debug 로 publish."""
        if self.debug_pub is None:
            return

        try:
            msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"Failed to convert debug image: {e}")
            return

        # 최신 RGB 헤더가 있으면 timestamp/frame 을 재사용
        if self.latest_rgb_header is not None:
            msg.header = self.latest_rgb_header

        self.debug_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoxSizeDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down BoxSizeDebugNode (KeyboardInterrupt).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()