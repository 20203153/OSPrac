#!/usr/bin/env python3
"""
Simple LiDAR + YOLOv5 logger node for TurtleBot4.

- Human manually drives the robot.
- Every 10 seconds:
  * Run YOLOv5 on the latest RGB image (using singleton loader).
  * Compute the bounding-box width/height in pixels of the best detection.
  * Estimate distance to the closest object in front using LiDAR
    within ±1 degree (total 2 degrees).

This node does NOT publish /cmd_vel. It only subscribes to:
- RGB image topic
- LiDAR scan topic
"""

from __future__ import annotations

from typing import Optional, Any

import math

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError

from new_func.yolov5_singleton import (
    load_yolov5_model,
    run_yolov5_inference,
    get_yolov5_model,
)


class SimpleLidarYoloNode(Node):
    """Minimal node that combines YOLOv5 (singleton) with LiDAR."""

    def __init__(self) -> None:
        super().__init__("simple_lidar_yolo_node")

        # Parameters
        self.declare_parameter("model_path", "./box.pt")
        self.declare_parameter("rgb_topic", "/oakd/rgb/preview/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("img_size", 640)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("yolo_conf_threshold", 0.4)
        self.declare_parameter("target_class", "box")

        model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.rgb_topic: str = (
            self.get_parameter("rgb_topic").get_parameter_value().string_value
        )
        self.scan_topic: str = (
            self.get_parameter("scan_topic").get_parameter_value().string_value
        )
        self.img_size: int = (
            self.get_parameter("img_size").get_parameter_value().integer_value
        )
        self.device: str = (
            self.get_parameter("device").get_parameter_value().string_value
        )
        self.yolo_conf_threshold: float = (
            self.get_parameter("yolo_conf_threshold")
            .get_parameter_value()
            .double_value
        )
        self.target_class: str = (
            self.get_parameter("target_class").get_parameter_value().string_value
        )

        # Load YOLOv5 model (Singleton)
        self.get_logger().info(
            f"[SimpleLidarYoloNode] Loading YOLOv5 model: {model_path} (device={self.device})"
        )
        try:
            _ = load_yolov5_model(
                model_path=model_path,
                device=self.device,
                use_half=False,
            )
            model = get_yolov5_model()
            self.class_names = getattr(model, "names", {})
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLOv5 model: {e}")
            raise

        # State
        self.bridge = CvBridge()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_scan: Optional[LaserScan] = None
        self.last_eval_time: Optional[float] = None

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )

        # Main timer: check every 0.5s, run YOLO+LiDAR every 10s
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info(
            "SimpleLidarYoloNode initialized.\n"
            f"  model_path={model_path}\n"
            f"  rgb_topic={self.rgb_topic}\n"
            f"  scan_topic={self.scan_topic}\n"
            f"  img_size={self.img_size}\n"
            f"  device={self.device}\n"
            f"  yolo_conf_threshold={self.yolo_conf_threshold}\n"
            f"  target_class={self.target_class}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def rgb_callback(self, msg: Image) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB cv_bridge error: {e}")
            return

        self.latest_rgb = rgb

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def timer_callback(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # Run only every 10 seconds
        if self.last_eval_time is not None and (now_sec - self.last_eval_time) < 10.0:
            return

        if self.latest_rgb is None:
            self.get_logger().debug("No RGB frame yet; skip.")
            return
        if self.latest_scan is None:
            self.get_logger().debug("No LiDAR scan yet; skip.")
            return

        self.last_eval_time = now_sec

        # LiDAR: closest object within ±1 degree in front
        lidar_dist = self._get_front_min_distance(self.latest_scan, sector_half_deg=1.0)
        if lidar_dist is None or lidar_dist <= 0.0:
            self.get_logger().warn("LiDAR: no valid distance in front ±1 degree.")
            return

        # YOLO inference (singleton)
        bgr = self.latest_rgb.copy()
        results = run_yolov5_inference(
            bgr_image=bgr,
            img_size=self.img_size,
            conf_threshold=self.yolo_conf_threshold,
        )
        boxes = results.xyxy[0]  # [N, 6]: x1, y1, x2, y2, conf, cls

        if boxes is None or len(boxes) == 0:
            self.get_logger().info("YOLO: no detections.")
            return

        # Pick best detection for target_class
        best_det = None
        best_conf = 0.0
        class_names = getattr(self, "class_names", {})

        for det in boxes:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            conf = float(conf)
            cls_id = int(cls_id)

            if isinstance(class_names, dict):
                class_name = class_names.get(cls_id, str(cls_id))
            else:
                try:
                    class_name = class_names[cls_id]
                except Exception:
                    class_name = str(cls_id)

            if class_name != self.target_class:
                continue
            if conf < self.yolo_conf_threshold:
                continue
            if conf > best_conf:
                best_conf = conf
                best_det = (det, class_name, cls_id, conf)

        if best_det is None:
            self.get_logger().info(
                f"YOLO: no '{self.target_class}' detections above threshold."
            )
            return

        det, class_name, cls_id, conf = best_det
        x1, y1, x2, y2, _, _ = det.tolist()
        pixel_width = max(0.0, float(x2 - x1))
        pixel_height = max(0.0, float(y2 - y1))

        # Log result: pixel size + LiDAR distance
        self.get_logger().info(
            f"[YOLO+LiDAR] class={class_name}(id={cls_id}) conf={conf:.3f} "
            f"bbox_px=({pixel_width:.1f}, {pixel_height:.1f}) "
            f"lidar_distance={lidar_dist:.3f} m"
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _get_front_min_distance(
        self,
        scan: LaserScan,
        sector_half_deg: float = 1.0,
    ) -> Optional[float]:
        """
        Return the minimum valid range within ±sector_half_deg around 0 rad.
        """
        ranges = np.array(scan.ranges, dtype=np.float32)

        # Angle array
        angles = np.arange(
            scan.angle_min,
            scan.angle_min + scan.angle_increment * len(ranges),
            scan.angle_increment,
            dtype=np.float32,
        )

        if angles.shape[0] > ranges.shape[0]:
            angles = angles[: ranges.shape[0]]

        valid_mask = np.isfinite(ranges) & (ranges > 0.0)
        if not np.any(valid_mask):
            return None

        sector_half_rad = math.radians(sector_half_deg)
        front_mask = valid_mask & (np.abs(angles) <= sector_half_rad)
        front_ranges = ranges[front_mask]

        if front_ranges.size == 0:
            return None

        return float(np.min(front_ranges))


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node = SimpleLidarYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down SimpleLidarYoloNode (KeyboardInterrupt).")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()