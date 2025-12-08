#!/usr/bin/env python3
"""
cv2_node: ROI-based box (rectangle) extraction and scoring node for TurtleBot4.

- Exposes a service `/box_query` (tb4_target_selector/BoxQuery)
- Uses `/map` (nav_msgs/OccupancyGrid) and TF to interpret a target point
- Around the target point, extracts an OccupancyGrid ROI
- Uses OpenCV to detect rectangular obstacle/structure candidates
- Scores candidates and returns them sorted by score (descending) in the service response.

This matches the "cv2_node" specification in the design.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
import tf_transformations

from tb4_target_selector.msg import BoxCandidate
from tb4_target_selector.srv import BoxQuery


@dataclass
class MapBox:
    x: float
    y: float
    width: float
    height: float
    angle: float


class Cv2BoxSelectorNode(Node):
    def __init__(self) -> None:
        super().__init__("cv2_box_selector_node")

        # Parameters ---------------------------------------------------------
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("roi_size_m", 2.0)
        self.declare_parameter("occ_threshold", 50)
        self.declare_parameter("min_area_pix", 20)
        self.declare_parameter("max_candidates", 10)
        self.declare_parameter("service_name", "/box_query")

        # Optional expected box size (for size-based scoring), 0.0 disables
        self.declare_parameter("expected_box_width_m", 0.0)
        self.declare_parameter("expected_box_height_m", 0.0)

        self.map_topic: str = (
            self.get_parameter("map_topic").get_parameter_value().string_value
        )
        self.map_frame: str = (
            self.get_parameter("map_frame").get_parameter_value().string_value
        )
        self.base_frame: str = (
            self.get_parameter("base_frame").get_parameter_value().string_value
        )
        self.roi_size_m: float = (
            self.get_parameter("roi_size_m").get_parameter_value().double_value
        )
        self.occ_threshold: int = (
            self.get_parameter("occ_threshold").get_parameter_value().integer_value
        )
        self.min_area_pix: int = (
            self.get_parameter("min_area_pix").get_parameter_value().integer_value
        )
        self.max_candidates: int = (
            self.get_parameter("max_candidates").get_parameter_value().integer_value
        )
        self.service_name: str = (
            self.get_parameter("service_name").get_parameter_value().string_value
        )

        self.expected_box_width_m: float = (
            self.get_parameter("expected_box_width_m")
            .get_parameter_value()
            .double_value
        )
        self.expected_box_height_m: float = (
            self.get_parameter("expected_box_height_m")
            .get_parameter_value()
            .double_value
        )

        # TF -----------------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Map cache ----------------------------------------------------------
        self.map_lock = threading.Lock()
        self.map_array: Optional[np.ndarray] = None  # shape: (H, W)
        self.map_info: Optional[MapMetaData] = None
        self.map_header: Optional[Header] = None

        # Subscribers --------------------------------------------------------
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10,
        )

        # Service server -----------------------------------------------------
        self.srv = self.create_service(
            BoxQuery,
            self.service_name,
            self.handle_box_query,
        )

        self.get_logger().info(
            "Cv2BoxSelectorNode initialized:\n"
            f"  map_topic={self.map_topic}\n"
            f"  map_frame={self.map_frame}\n"
            f"  base_frame={self.base_frame}\n"
            f"  roi_size_m={self.roi_size_m}\n"
            f"  occ_threshold={self.occ_threshold}\n"
            f"  min_area_pix={self.min_area_pix}\n"
            f"  max_candidates={self.max_candidates}\n"
            f"  expected_box_width_m={self.expected_box_width_m}\n"
            f"  expected_box_height_m={self.expected_box_height_m}\n"
            f"  service_name={self.service_name}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid) -> None:
        with self.map_lock:
            data = np.array(msg.data, dtype=np.int16)
            if data.size != msg.info.width * msg.info.height:
                self.get_logger().warn(
                    f"OccupancyGrid data size mismatch: {data.size} vs "
                    f"{msg.info.width * msg.info.height}"
                )
                return
            self.map_array = data.reshape((msg.info.height, msg.info.width))
            self.map_info = msg.info
            self.map_header = msg.header

    def handle_box_query(
        self,
        request: BoxQuery.Request,
        response: BoxQuery.Response,
    ) -> BoxQuery.Response:
        """
        Service callback:
        - Transform request.target_point into map frame (if needed)
        - Extract ROI around that point from /map
        - Find rectangular occupied regions via OpenCV
        - Score and sort candidates
        - Fill response.candidates (sorted by descending score)
        """
        # Snapshot map
        with self.map_lock:
            if self.map_array is None or self.map_info is None:
                self.get_logger().warn("No map data available for BoxQuery.")
                response.header = Header()
                response.candidates = []
                return response
            map_array = self.map_array.copy()
            map_info = self.map_info

        # Determine target point in map frame
        target_map = self._transform_point_to_map(
            point=request.target_point,
            from_frame=request.header.frame_id,
            stamp=request.header.stamp,
        )
        if target_map is None:
            self.get_logger().warn(
                f"Failed to transform target point to map frame "
                f"(from_frame='{request.header.frame_id}')"
            )
            response.header = Header()
            response.candidates = []
            return response

        x_obj, y_obj = target_map.x, target_map.y

        boxes = self._extract_boxes_from_map_roi(
            map_array=map_array,
            map_info=map_info,
            x_obj=x_obj,
            y_obj=y_obj,
        )

        if not boxes:
            self.get_logger().debug("No box candidates found in ROI.")
            response.header = Header()
            response.candidates = []
            return response

        candidates: List[BoxCandidate] = []
        exp_size: Optional[Tuple[float, float]] = None
        if self.expected_box_width_m > 0.0 and self.expected_box_height_m > 0.0:
            exp_size = (self.expected_box_width_m, self.expected_box_height_m)

        for box in boxes:
            score = self._compute_score(
                X_box=box.x,
                Y_box=box.y,
                W_box=box.width,
                H_box=box.height,
                x_obj=x_obj,
                y_obj=y_obj,
                target_class=request.target_class,
                yolo_conf=request.confidence,
                expected_size=exp_size,
            )
            cand = BoxCandidate()
            cand.related_class = request.target_class
            cand.score = float(score)
            cand.pose = Pose()
            cand.pose.position.x = float(box.x)
            cand.pose.position.y = float(box.y)
            cand.pose.position.z = 0.0
            cand.pose.orientation.x = 0.0
            cand.pose.orientation.y = 0.0
            cand.pose.orientation.z = 0.0
            cand.pose.orientation.w = 1.0
            cand.width = float(box.width)
            cand.height = float(box.height)
            candidates.append(cand)

        # Sort by score (descending)
        candidates.sort(key=lambda c: c.score, reverse=True)
        if self.max_candidates > 0 and len(candidates) > self.max_candidates:
            candidates = candidates[: self.max_candidates]

        # Fill response
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame
        response.header = header
        response.candidates = candidates

        self.get_logger().info(
            f"BoxQuery: target_class={request.target_class}, "
            f"confidence={request.confidence:.3f}, "
            f"num_candidates={len(candidates)}"
        )
        if candidates:
            best = candidates[0]
            self.get_logger().info(
                f"  Best candidate: score={best.score:.3f}, "
                f"pos=({best.pose.position.x:.2f}, {best.pose.position.y:.2f}), "
                f"size=({best.width:.2f} x {best.height:.2f})"
            )

        return response

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _transform_point_to_map(
        self,
        point: Point,
        from_frame: str,
        stamp,
    ) -> Optional[Point]:
        """
        Transform a point expressed in from_frame into map_frame.
        If from_frame is already map_frame or empty, returns the point as-is.
        """
        if from_frame == "" or from_frame == self.map_frame:
            # Already in map frame (or unspecified -> assume map)
            p = Point()
            p.x = float(point.x)
            p.y = float(point.y)
            p.z = float(point.z)
            return p

        # Use TF to transform from from_frame -> map_frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                from_frame,
                stamp if stamp.sec != 0 or stamp.nanosec != 0 else Time().to_msg(),
                timeout=Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed (map<-{from_frame}): {e}")
            return None

        T = self._transform_to_matrix(transform)
        p_vec = np.array([point.x, point.y, point.z, 1.0], dtype=np.float64)
        p_map = T @ p_vec

        p = Point()
        p.x = float(p_map[0])
        p.y = float(p_map[1])
        p.z = float(p_map[2])
        return p

    def _transform_to_matrix(self, transform) -> np.ndarray:
        """Convert geometry_msgs/TransformStamped to 4x4 homogeneous matrix."""
        t = transform.transform.translation
        q = transform.transform.rotation
        T = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z
        return T

    def _extract_boxes_from_map_roi(
        self,
        map_array: np.ndarray,
        map_info: MapMetaData,
        x_obj: float,
        y_obj: float,
    ) -> List[MapBox]:
        """Extract obstacle boxes from occupancy map ROI around (x_obj, y_obj)."""
        resolution = map_info.resolution
        width = map_info.width
        height = map_info.height
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y

        # World -> pixel indices
        i_obj = int((x_obj - origin_x) / resolution)
        j_obj = int((y_obj - origin_y) / resolution)

        half = int(self.roi_size_m / (2.0 * resolution))

        i_min = max(0, i_obj - half)
        i_max = min(width - 1, i_obj + half)
        j_min = max(0, j_obj - half)
        j_max = min(height - 1, j_obj + half)

        if i_min >= i_max or j_min >= j_max:
            return []

        # map_array is indexed as [row (y=j), col (x=i)]
        roi = map_array[j_min : j_max + 1, i_min : i_max + 1]

        # Occupancy to binary mask
        binary = np.zeros_like(roi, dtype=np.uint8)
        binary[roi > self.occ_threshold] = 255

        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # OpenCV 2/3 vs 4 호환을 위해 반환값 개수에 따라 분기
        contours_info = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours_info) == 2:
            contours, _ = contours_info
        else:
            _, contours, _ = contours_info

        boxes: List[MapBox] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_pix:
                continue
            rect = cv2.minAreaRect(c)  # (center(x,y), (w,h), angle)
            (cx_pix, cy_pix), (w_pix, h_pix), angle = rect

            # Convert ROI coords back to full map pixel indices
            cx_full = cx_pix + i_min
            cy_full = cy_pix + j_min

            x_map = origin_x + cx_full * resolution
            y_map = origin_y + cy_full * resolution
            w_m = w_pix * resolution
            h_m = h_pix * resolution

            boxes.append(
                MapBox(
                    x=float(x_map),
                    y=float(y_map),
                    width=float(w_m),
                    height=float(h_m),
                    angle=float(angle),
                )
            )

        return boxes

    def _compute_score(
        self,
        X_box: float,
        Y_box: float,
        W_box: float,
        H_box: float,
        x_obj: float,
        y_obj: float,
        target_class: str,
        yolo_conf: float,
        expected_size: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Compute a heuristic score for a box candidate."""
        dist = math.sqrt((X_box - x_obj) ** 2 + (Y_box - y_obj) ** 2)
        dist_score = 1.0 / (1.0 + dist)  # 0..1, closer is better

        # Size-based score: favor sizes close to expected_size, if provided
        size_score = 1.0
        if expected_size is not None:
            exp_w, exp_h = expected_size
            size_error = abs(W_box - exp_w) + abs(H_box - exp_h)
            size_score = 1.0 / (1.0 + size_error)

        # YOLO confidence as a direct factor
        conf_score = max(0.0, min(1.0, float(yolo_conf)))

        # Simple weighted sum (can be adjusted per project needs)
        score = 0.5 * dist_score + 0.3 * size_score + 0.2 * conf_score
        return float(score)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Cv2BoxSelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Cv2BoxSelectorNode (KeyboardInterrupt).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()