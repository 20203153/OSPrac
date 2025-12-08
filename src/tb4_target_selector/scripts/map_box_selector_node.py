#!/usr/bin/env python3
"""
ROS2 Jazzy node for Turtlebot4 that:
- subscribes to YOLO detection results (/yolo/detections)
- estimates an approximate target position in the map frame
- extracts an OccupancyGrid ROI around that position
- uses OpenCV to find rectangular obstacle/structure candidates
- scores them and publishes candidate list and best target box.

This implements the 'map_box_selector_node' described in the design.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
import tf_transformations

# Custom message types (need to be defined in your ROS2 package):
# tb4_target_selector/msg/Detection.msg
# tb4_target_selector/msg/YoloDetections.msg
# tb4_target_selector/msg/BoxCandidate.msg
# tb4_target_selector/msg/BoxCandidateArray.msg
# tb4_target_selector/msg/BestTargetBox.msg
from tb4_target_selector.msg import (
    Detection,
    YoloDetections,
    BoxCandidate,
    BoxCandidateArray,
    BestTargetBox,
)
from tb4_target_selector.srv import BoxQuery


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class MapBox:
    x: float
    y: float
    width: float
    height: float
    angle: float


class MapBoxSelectorNode(Node):
    def __init__(self) -> None:
        super().__init__("map_box_selector_node")

        # Parameters ---------------------------------------------------------
        self.declare_parameter("target_classes", ["person", "chair", "target_obj"])
        self.declare_parameter("yolo_conf_threshold", 0.4)
        self.declare_parameter("roi_size_m", 2.0)
        self.declare_parameter("occ_threshold", 50)
        self.declare_parameter("min_area_pix", 20)
        self.declare_parameter("trigger_cooldown_sec", 1.0)
        self.declare_parameter("assumed_distance_m", 1.5)
        # Physical box size (Korea Post box #4) and tolerance [m]
        self.declare_parameter("box_width_m", 0.41)
        self.declare_parameter("box_height_m", 0.31)
        self.declare_parameter("box_tolerance_m", 0.10)
        self.declare_parameter("enable_size_filter", True)

        self.declare_parameter("yolo_topic", "/yolo/detections")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("map_frame", "map")

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
        self.roi_size_m: float = (
            self.get_parameter("roi_size_m").get_parameter_value().double_value
        )
        self.occ_threshold: int = (
            self.get_parameter("occ_threshold").get_parameter_value().integer_value
        )
        self.min_area_pix: int = (
            self.get_parameter("min_area_pix").get_parameter_value().integer_value
        )
        self.trigger_cooldown_sec: float = (
            self.get_parameter("trigger_cooldown_sec")
            .get_parameter_value()
            .double_value
        )
        self.assumed_distance_m: float = (
            self.get_parameter("assumed_distance_m")
            .get_parameter_value()
            .double_value
        )
        self.box_width_m: float = (
            self.get_parameter("box_width_m").get_parameter_value().double_value
        )
        self.box_height_m: float = (
            self.get_parameter("box_height_m").get_parameter_value().double_value
        )
        self.box_tolerance_m: float = (
            self.get_parameter("box_tolerance_m").get_parameter_value().double_value
        )
        self.enable_size_filter: bool = (
            self.get_parameter("enable_size_filter").get_parameter_value().bool_value
        )

        self.yolo_topic: str = (
            self.get_parameter("yolo_topic").get_parameter_value().string_value
        )
        self.map_topic: str = (
            self.get_parameter("map_topic").get_parameter_value().string_value
        )
        self.depth_topic: str = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        self.camera_info_topic: str = (
            self.get_parameter("camera_info_topic")
            .get_parameter_value()
            .string_value
        )
        self.base_frame: str = (
            self.get_parameter("base_frame").get_parameter_value().string_value
        )
        self.camera_frame: str = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self.map_frame: str = (
            self.get_parameter("map_frame").get_parameter_value().string_value
        )

        # TF -----------------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State caches -------------------------------------------------------
        self.camera_intrinsics: Optional[CameraIntrinsics] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_stamp: Optional[Time] = None
        self.latest_depth_frame_id: Optional[str] = None

        self.map_lock = threading.Lock()
        self.map_array: Optional[np.ndarray] = None  # 2D (H, W)
        self.map_info: Optional[MapMetaData] = None
        self.map_header: Optional[Header] = None

        self.last_trigger_time: Optional[float] = None

        # Accumulated box candidates over the exploration session
        self.saved_candidates: List[BoxCandidate] = []

        # Publishers ---------------------------------------------------------
        self.candidates_pub = self.create_publisher(
            BoxCandidateArray,
            "/target_box_candidates",
            10,
        )
        self.best_pub = self.create_publisher(
            BestTargetBox,
            "/best_target_box",
            10,
        )

        # Subscribers --------------------------------------------------------
        self.yolo_sub = self.create_subscription(
            YoloDetections,
            self.yolo_topic,
            self.yolo_callback,
            10,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
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

        # BoxQuery service: return stored candidates around a queried point
        self.box_query_srv = self.create_service(
            BoxQuery,
            "/box_query",
            self.handle_box_query,
        )

        self.get_logger().info(
            "MapBoxSelectorNode initialized:\n"
            f"  yolo_topic={self.yolo_topic}\n"
            f"  map_topic={self.map_topic}\n"
            f"  depth_topic={self.depth_topic}\n"
            f"  camera_info_topic={self.camera_info_topic}\n"
            f"  target_classes={self.target_classes}\n"
            f"  yolo_conf_threshold={self.yolo_conf_threshold}\n"
            f"  roi_size_m={self.roi_size_m}\n"
            f"  occ_threshold={self.occ_threshold}\n"
            f"  min_area_pix={self.min_area_pix}\n"
            f"  trigger_cooldown_sec={self.trigger_cooldown_sec}\n"
            f"  assumed_distance_m={self.assumed_distance_m}\n"
            f"  box_width_m={self.box_width_m}, box_height_m={self.box_height_m}, box_tolerance_m={self.box_tolerance_m}\n"
            f"  enable_size_filter={self.enable_size_filter}\n"
            f"  frames: map={self.map_frame}, base={self.base_frame}, camera={self.camera_frame}"
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
            self.camera_intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    def depth_callback(self, msg: Image) -> None:
        try:
            depth = self._image_to_depth(msg)
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")
            return

        self.latest_depth = depth
        self.latest_depth_stamp = msg.header.stamp
        self.latest_depth_frame_id = msg.header.frame_id

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

    def yolo_callback(self, msg: YoloDetections) -> None:
        # Debounce / cooldown
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_trigger_time is not None:
            if now_sec - self.last_trigger_time < self.trigger_cooldown_sec:
                return

        with self.map_lock:
            if self.map_array is None or self.map_info is None:
                return
            map_array = self.map_array.copy()
            map_info = self.map_info
            map_header = self.map_header

        # Select best detection matching target classes and confidence
        best_det: Optional[Detection] = None
        best_conf: float = 0.0
        for det in msg.detections:
            if det.class_name not in self.target_classes:
                continue
            if det.confidence < self.yolo_conf_threshold:
                continue
            if det.confidence > best_conf:
                best_conf = det.confidence
                best_det = det

        if best_det is None:
            return

        # Try to estimate target world position
        target_xy = self._estimate_target_world_xy(best_det, msg.header)
        if target_xy is None:
            # Fallback using camera direction and assumed distance
            target_xy = self._fallback_target_world_xy(msg.header)
            if target_xy is None:
                self.get_logger().warn("Failed to estimate target world position.")
                return

        x_obj, y_obj = target_xy

        # ROI는 항상 "탐지된 물체 추정 위치(x_obj, y_obj)"를 중심으로 자른다.
        # 이렇게 해야, 로봇-박스 거리가 1m 수준일 때도 박스가 ROI 경계에 걸려 잘리는 상황을 줄일 수 있다.
        robot_pose = self._lookup_robot_pose_xy_yaw()

        boxes = self._extract_boxes_from_map_roi(
            map_array=map_array,
            map_info=map_info,
            x_obj=x_obj,
            y_obj=y_obj,
        )

        if not boxes:
            return

        scored_candidates: List[Tuple[float, MapBox]] = []
        for box in boxes:
            score = self._score_box(box, x_obj, y_obj, robot_pose)
            scored_candidates.append((score, box))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        candidates_msg = BoxCandidateArray()
        header = Header()
        if map_header is not None:
            header.stamp = map_header.stamp
            header.frame_id = map_header.frame_id or self.map_frame
        else:
            header.stamp = msg.header.stamp
            header.frame_id = self.map_frame
        candidates_msg.header = header

        candidates: List[BoxCandidate] = []
        for score, box in scored_candidates:
            candidate = BoxCandidate()
            candidate.related_class = best_det.class_name
            candidate.score = float(score)
            candidate.pose = Pose()
            candidate.pose.position.x = float(box.x)
            candidate.pose.position.y = float(box.y)
            candidate.pose.position.z = 0.0
            candidate.pose.orientation.x = 0.0
            candidate.pose.orientation.y = 0.0
            candidate.pose.orientation.z = 0.0
            candidate.pose.orientation.w = 1.0
            candidate.width = float(box.width)
            candidate.height = float(box.height)
            candidates.append(candidate)

        # Limit number of candidates (e.g., top 5)
        max_candidates = 5
        candidates_msg.candidates = candidates[:max_candidates]

        # 저장용 누적 리스트 업데이트 (탐사 세션 전체에서 사용)
        self._update_saved_candidates(candidates_msg.candidates)

        best_msg = BestTargetBox()
        best_msg.header = candidates_msg.header
        if scored_candidates:
            best_score, best_box = scored_candidates[0]
            best_candidate = BoxCandidate()
            best_candidate.related_class = best_det.class_name
            best_candidate.score = float(best_score)
            best_candidate.pose = Pose()
            best_candidate.pose.position.x = float(best_box.x)
            best_candidate.pose.position.y = float(best_box.y)
            best_candidate.pose.position.z = 0.0
            best_candidate.pose.orientation.x = 0.0
            best_candidate.pose.orientation.y = 0.0
            best_candidate.pose.orientation.z = 0.0
            best_candidate.pose.orientation.w = 1.0
            best_candidate.width = float(best_box.width)
            best_candidate.height = float(best_box.height)
            best_msg.best_candidate = best_candidate
            best_msg.valid = True
        else:
            best_msg.valid = False

        self.candidates_pub.publish(candidates_msg)
        self.best_pub.publish(best_msg)

        self.last_trigger_time = now_sec

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _image_to_depth(self, msg: Image) -> np.ndarray:
        """Convert a depth Image to float32 meters."""
        # Lazy import to avoid cv_bridge dependency in type hints
        from cv_bridge import CvBridge, CvBridgeError

        bridge = CvBridge()
        try:
            depth = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except CvBridgeError:
            depth_raw = bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            depth = depth_raw.astype(np.float32) / 1000.0
        return depth

    def _estimate_target_world_xy(
        self,
        det: Detection,
        header: Header,
    ) -> Optional[Tuple[float, float]]:
        """Estimate target (x, y) in map frame using depth and camera intrinsics."""
        if self.latest_depth is None or self.camera_intrinsics is None:
            return None

        depth = self.latest_depth
        h, w = depth.shape[:2]
        u = int(det.x_center)
        v = int(det.y_center)
        if not (0 <= u < w and 0 <= v < h):
            return None

        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0.0:
            # Try small neighborhood average
            patch = depth[max(0, v - 2) : min(h, v + 3), max(0, u - 2) : min(w, u + 3)]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]
            if valid.size == 0:
                return None
            z = float(np.median(valid))

        intr = self.camera_intrinsics
        X_cam = (u - intr.cx) * z / intr.fx
        Y_cam = (v - intr.cy) * z / intr.fy
        Z_cam = z

        point_cam = np.array([X_cam, Y_cam, Z_cam, 1.0], dtype=np.float64)

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                header.stamp,
                timeout=Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed (map<-camera) in estimate: {e}")
            return None

        T = self._transform_to_matrix(transform)
        point_map = T @ point_cam
        x_map = float(point_map[0])
        y_map = float(point_map[1])
        return x_map, y_map

    def _fallback_target_world_xy(self, header: Header) -> Optional[Tuple[float, float]]:
        """Fallback when depth is unavailable: use camera direction and assumed distance."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                header.stamp,
                timeout=Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed (map<-camera) in fallback: {e}")
            return None

        T = self._transform_to_matrix(transform)
        # Camera origin in map
        origin = T @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        # Camera forward (x-axis) direction in map
        forward = T @ np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        dir_vec = forward[:3]
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-6:
            return None
        dir_unit = dir_vec / norm

        target = origin[:3] + dir_unit * self.assumed_distance_m
        return float(target[0]), float(target[1])

    def _lookup_robot_pose_xy_yaw(self) -> Optional[Tuple[float, float, float]]:
        """Lookup robot pose (base_frame) in map frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        q = transform.transform.rotation
        yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )[2]
        return float(x), float(y), float(yaw)

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

        i_obj = int((x_obj - origin_x) / resolution)
        j_obj = int((y_obj - origin_y) / resolution)

        half = int(self.roi_size_m / (2.0 * resolution))

        i_min = max(0, i_obj - half)
        i_max = min(width - 1, i_obj + half)
        j_min = max(0, j_obj - half)
        j_max = min(height - 1, j_obj + half)

        if i_min >= i_max or j_min >= j_max:
            return []

        # Note: map_array is indexed as [row (y=j), col (x=i)]
        roi = map_array[j_min : j_max + 1, i_min : i_max + 1]

        # Occupancy to binary mask
        binary = np.zeros_like(roi, dtype=np.uint8)
        binary[roi > self.occ_threshold] = 255

        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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

            # 물리 크기 기반 필터 (우체국 박스 4호: 0.41m x 0.31m, ± box_tolerance_m)
            if self.enable_size_filter:
                tol = self.box_tolerance_m
                tw = self.box_width_m
                th = self.box_height_m

                # 정방향 (width ≈ tw, height ≈ th)
                width_ok = (tw - tol) <= w_m <= (tw + tol)
                height_ok = (th - tol) <= h_m <= (th + tol)
                # 90도 회전된 경우 (width ≈ th, height ≈ tw)
                width_ok_swapped = (tw - tol) <= h_m <= (tw + tol)
                height_ok_swapped = (th - tol) <= w_m <= (th + tol)

                if not ((width_ok and height_ok) or (width_ok_swapped and height_ok_swapped)):
                    continue

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

    def _update_saved_candidates(self, new_candidates: List[BoxCandidate]) -> None:
        """Accumulate box candidates over the exploration session."""
        if not new_candidates:
            return
        self.saved_candidates.extend(new_candidates)

    def _score_box(
        self,
        box: MapBox,
        x_obj: float,
        y_obj: float,
        robot_pose: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        """Compute a heuristic score for a box candidate."""
        dx = box.x - x_obj
        dy = box.y - y_obj
        dist = math.hypot(dx, dy)

        # Distance-based score (0..1), closer to x_obj,y_obj is better.
        dist_score = 1.0 / (1.0 + dist)

        # Size-based score: penalize too large or too small boxes.
        area = box.width * box.height
        # Assume we prefer boxes around 1.0 m^2 (tunable).
        preferred_area = 1.0
        size_score = math.exp(-abs(area - preferred_area))

        angle_score = 1.0
        if robot_pose is not None:
            rx, ry, ryaw = robot_pose
            vx = box.x - rx
            vy = box.y - ry
            if vx != 0.0 or vy != 0.0:
                box_yaw = math.atan2(vy, vx)
                diff = abs(self._normalize_angle(box_yaw - ryaw))
                # Prefer boxes roughly in front of the robot (<= 90deg).
                angle_score = max(0.0, math.cos(diff))

        score = 0.6 * dist_score + 0.3 * size_score + 0.1 * angle_score
        return float(score)

    def handle_box_query(self, request: BoxQuery.Request, response: BoxQuery.Response):
        """Service callback: return stored box candidates around a queried point.

        - Request.header.frame_id: frame of target_point (e.g. "map", "base_link")
        - Request.target_point: query center
        - Request.target_class: optional class filter ("" = no filter)
        - Request.confidence: minimum score threshold
        """
        target_xy = self._point_to_map_xy(request.header, request.target_point)
        if target_xy is None:
            self.get_logger().warn("BoxQuery: failed to transform target_point to map frame.")
            response.header = request.header
            response.candidates = []
            return response

        tx, ty = target_xy
        half = self.roi_size_m / 2.0

        filtered: List[BoxCandidate] = []
        for cand in self.saved_candidates:
            # Class filter (if provided)
            if request.target_class and cand.related_class != request.target_class:
                continue
            # Score threshold
            if cand.score < request.confidence:
                continue

            dx = cand.pose.position.x - tx
            dy = cand.pose.position.y - ty
            # 정사각 반경 1m (roi_size_m=2.0 기준) 안에 있는 후보만 사용
            if abs(dx) <= half and abs(dy) <= half:
                filtered.append(cand)

        # 스코어 내림차순 정렬
        filtered.sort(key=lambda c: c.score, reverse=True)

        response.header.stamp = self.get_clock().now().to_msg()
        response.header.frame_id = self.map_frame
        response.candidates = filtered
        return response

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _point_to_map_xy(
        self,
        header: Header,
        point: Point,
    ) -> Optional[Tuple[float, float]]:
        """Transform a Point in an arbitrary frame to (x, y) in map frame."""
        # Already in map frame
        if not header.frame_id or header.frame_id == self.map_frame:
            return float(point.x), float(point.y)

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                header.frame_id,
                header.stamp,
                timeout=Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed in _point_to_map_xy: {e}")
            return None

        T = self._transform_to_matrix(transform)
        p = np.array([point.x, point.y, 0.0, 1.0], dtype=np.float64)
        p_map = T @ p
        return float(p_map[0]), float(p_map[1])


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = MapBoxSelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down MapBoxSelectorNode (KeyboardInterrupt).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()