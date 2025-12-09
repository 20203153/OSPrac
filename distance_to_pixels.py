#!/usr/bin/env python3
"""Convert physical box sizes from distance.json to pixel sizes using CameraInfo fx, fy."""

from __future__ import annotations

from typing import Any, Dict, List

import json
import os

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo


class DistanceToPixelsNode(Node):
    """
    Read fx, fy from a CameraInfo topic, then
    convert physical box sizes (meters) from a JSON file to pixel sizes.
    """

    def __init__(self) -> None:
        super().__init__("distance_to_pixels_node")

        # Parameters
        self.declare_parameter("camera_info_topic", "/oakd/rgb/preview/camera_info")
        self.declare_parameter("distance_json_path", "distance.json")
        self.declare_parameter("output_json_path", "distance_pixels.json")

        self.camera_info_topic: str = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.distance_json_path: str = (
            self.get_parameter("distance_json_path").get_parameter_value().string_value
        )
        self.output_json_path: str = (
            self.get_parameter("output_json_path").get_parameter_value().string_value
        )

        self.fx: float | None = None
        self.fy: float | None = None
        self._processed: bool = False

        # Subscribe to CameraInfo
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.get_logger().info(
            "DistanceToPixelsNode initialized. Waiting for CameraInfo...\n"
            f"  camera_info_topic={self.camera_info_topic}\n"
            f"  distance_json_path={self.distance_json_path}\n"
            f"  output_json_path={self.output_json_path}"
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Store fx, fy from first CameraInfo and then process JSON once."""
        if self._processed:
            return

        try:
            k = msg.k
            self.fx = float(k[0])  # fx
            self.fy = float(k[4])  # fy
        except Exception as e:
            self.get_logger().warn(f"Failed to parse CameraInfo: {e}")
            return

        if self.fx is None or self.fy is None:
            self.get_logger().warn("CameraInfo fx/fy are not available.")
            return

        self.get_logger().info(f"Received CameraInfo: fx={self.fx:.3f}, fy={self.fy:.3f}")
        self._process_distance_json()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _process_distance_json(self) -> None:
        if self._processed:
            return

        self._processed = True

        if not os.path.exists(self.distance_json_path):
            self.get_logger().error(
                f"distance_json_path does not exist: {self.distance_json_path}"
            )
            self._shutdown()
            return

        try:
            with open(self.distance_json_path, "r", encoding="utf-8") as f:
                data: Dict[str, List[Dict[str, Any]]] = json.load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load JSON: {e}")
            self._shutdown()
            return

        fx = float(self.fx)
        fy = float(self.fy)

        def convert_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
            """Apply inverse of estimate_object_size_from_bbox to one record."""
            distance_m = float(entry.get("distance", 0.0))
            width_m = float(entry.get("x", 0.0))
            height_m = float(entry.get("y", 0.0))

            if distance_m <= 0.0:
                pixel_w = 0.0
                pixel_h = 0.0
            else:
                # Inverse of:
                #   width_m  = (pixel_width  * distance_m) / fx
                #   height_m = (pixel_height * distance_m) / fy
                pixel_w = (width_m * fx) / distance_m
                pixel_h = (height_m * fy) / distance_m

            return {
                "distance": distance_m,
                "x_m": width_m,
                "y_m": height_m,
                "x_px": pixel_w,
                "y_px": pixel_h,
            }

        output: Dict[str, List[Dict[str, Any]]] = {}
        for key, entries in data.items():
            if not isinstance(entries, list):
                continue
            converted_list: List[Dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                converted_list.append(convert_entry(entry))
            output[key] = converted_list

        try:
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.get_logger().error(f"Failed to write output JSON: {e}")
        else:
            self.get_logger().info(
                f"Wrote pixel-converted data to {self.output_json_path}"
            )

        self._shutdown()

    def _shutdown(self) -> None:
        """Stop the node and shutdown rclpy."""
        self.get_logger().info("Shutting down DistanceToPixelsNode.")
        try:
            self.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node = DistanceToPixelsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Exiting.")
        node._shutdown()


if __name__ == "__main__":
    main()