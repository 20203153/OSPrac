#!/usr/bin/env python3
"""
lidar_forward: LiDAR + YOLOv5 기반 전진 측정 스크립트 (ROS2 + TurtleBot4)

최신 컨셉
---------
- 전진 제어:
  - LiDAR 피드백을 사용하지 않고, "기계적으로 step_m 만큼 전진"한다고 가정.
  - /cmd_vel 로 일정 속도(forward_speed)로 일정 시간 동안 전진 → 약 step_m 만큼 이동했다고 본다.
- LiDAR:
  - 제어에는 사용하지 않고, **기록/로그용**으로만 사용.
  - 각 스텝에서 정면 방향 LiDAR 거리를 로그에 남겨 참고용으로만 본다.
- 거리 축:
  - JSON 의 distance 는 "개념상 거리 축"을 사용:
    - base_distance_m 에서 시작해서 step_m 씩 줄어드는 값 (예: 2.0, 1.9, 1.8, ...)
    - 실제 LiDAR 오차에 영향 받지 않는다.
- 크기 축:
  - JSON 의 x, y 는 YOLO bbox 의 width/height [픽셀 단위].

JSON 출력 형식
--------------
{
    "{casename}": [
        {
            "distance": float,  # 개념상 거리 (예: 2.0, 1.9, 1.8 ...) [m], 소숫점 둘째 자리
            "x": float,         # YOLO bbox width [px]
            "y": float          # YOLO bbox height [px]
        },
        ...
    ]
}

실행 예시
---------
python lidar_forward.py \
  --ros-args \
    -p casename:=test_case_01 \
    -p model_path:=./yolov5n.pt \
    -p rgb_topic:=/oakd/rgb/preview/image_raw \
    -p camera_info_topic:=/oakd/rgb/preview/camera_info \
    -p scan_topic:=/scan \
    -p cmd_vel_topic:=/cmd_vel \
    -p step_m:=0.10 \
    -p forward_speed:=0.10 \
    -p yolo_conf_threshold:=0.4 \
    -p target_class:=box \
    -p device:=cpu

주의
----
- 실제 TurtleBot4 환경에서 /cmd_vel 제어를 수행하므로, 사용 전에
  안전한 환경에서 테스트해야 한다.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import json
import math
import warnings

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

from new_func.yolov5_singleton import (
    load_yolov5_model,
    run_yolov5_inference,
    get_yolov5_model,
)

# torch.cuda.amp.autocast 관련 FutureWarning 무시
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\\.cuda\\.amp\\.autocast.*",
)


class LidarForwardNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_forward_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("casename", "default_case")
        self.declare_parameter("model_path", "yolov5n.pt")
        self.declare_parameter("rgb_topic", "/oakd/rgb/preview/image_raw")
        self.declare_parameter("camera_info_topic", "/oakd/rgb/preview/camera_info")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # 개념상 전진 스텝 (약 10cm)
        self.declare_parameter("step_m", 0.10)

        # 전진 속도 [m/s] (예: 0.1 m/s → 1초 전진 = 0.1m 이동 가정)
        self.declare_parameter("forward_speed", 0.10)

        # 개념상 시작 거리 [m] (예: 2.0m 에서 시작)
        self.declare_parameter("base_distance_m", 2.0)

        # YOLO 관련 파라미터
        self.declare_parameter("yolo_conf_threshold", 0.4)
        self.declare_parameter("target_class", "box")
        self.declare_parameter("img_size", 640)
        self.declare_parameter("device", "cpu")

        # YOLO 평가 최소 간격 (초)
        self.declare_parameter("yolo_eval_min_interval_sec", 1.0)

        # 파라미터 값 읽기
        self.casename: str = (
            self.get_parameter("casename").get_parameter_value().string_value
        )
        model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.rgb_topic: str = (
            self.get_parameter("rgb_topic").get_parameter_value().string_value
        )
        self.camera_info_topic: str = (
            self.get_parameter("camera_info_topic")
            .get_parameter_value()
            .string_value
        )
        self.scan_topic: str = (
            self.get_parameter("scan_topic").get_parameter_value().string_value
        )
        self.cmd_vel_topic: str = (
            self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        )
        self.step_m: float = (
            self.get_parameter("step_m").get_parameter_value().double_value
        )
        self.forward_speed: float = (
            self.get_parameter("forward_speed").get_parameter_value().double_value
        )
        self.base_distance_m: float = (
            self.get_parameter("base_distance_m").get_parameter_value().double_value
        )
        self.yolo_conf_threshold: float = (
            self.get_parameter("yolo_conf_threshold")
            .get_parameter_value()
            .double_value
        )
        self.target_class: str = (
            self.get_parameter("target_class").get_parameter_value().string_value
        )
        self.img_size: int = (
            self.get_parameter("img_size").get_parameter_value().integer_value
        )
        self.device: str = (
            self.get_parameter("device").get_parameter_value().string_value
        )
        self.yolo_eval_min_interval_sec: float = (
            self.get_parameter("yolo_eval_min_interval_sec")
            .get_parameter_value()
            .double_value
        )

        # ------------------------------------------------------------------
        # Load YOLOv5 model (Singleton)
        # ------------------------------------------------------------------
        self.get_logger().info(
            f"[LidarForwardNode] Loading YOLOv5 model: {model_path} (device={self.device})"
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

        # ------------------------------------------------------------------
        # State (latest messages)
        # ------------------------------------------------------------------
        self.bridge = CvBridge()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_header: Optional[Image] = None
        self.latest_scan: Optional[LaserScan] = None
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None

        # sampling state
        self.last_yolo_eval_time: Optional[float] = None
        # 샘플링 간격: 10초마다 한 번씩 샘플 기록 (로봇이 10cm 전진했다고 가정)
        self.last_sample_time: Optional[float] = None
        self.done: bool = False

        # LiDAR 기록용
        self.current_distance_m: Optional[float] = None
        self.last_lidar_log_time: Optional[float] = None

        # 기록용 데이터
        # [{"distance": float, "x": float, "y": float}, ...]
        #  - distance: LiDAR 로 측정한 실제 전방 거리 [m]
        #  - x, y: YOLO bbox width/height [px]
        self.records: List[Dict[str, float]] = []

        # ------------------------------------------------------------------
        # Publishers / Subscribers
        # ------------------------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10,
        )

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
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )

        # 메인 타이머: 상태 머신 + YOLO + 이동 제어
        self.timer = self.create_timer(
            0.1,  # 10Hz 정도로 상태 업데이트
            self.timer_callback,
        )

        self.get_logger().info(
            "LidarForwardNode initialized.\n"
            f"  casename={self.casename}\n"
            f"  model_path={model_path}\n"
            f"  rgb_topic={self.rgb_topic}\n"
            f"  camera_info_topic={self.camera_info_topic}\n"
            f"  scan_topic={self.scan_topic}\n"
            f"  cmd_vel_topic={self.cmd_vel_topic}\n"
            f"  step_m={self.step_m}\n"
            f"  forward_speed={self.forward_speed}\n"
            f"  base_distance_m={self.base_distance_m}\n"
            f"  yolo_conf_threshold={self.yolo_conf_threshold}\n"
            f"  target_class={self.target_class}\n"
            f"  yolo_eval_min_interval_sec={self.yolo_eval_min_interval_sec}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def rgb_callback(self, msg: Image) -> None:
        """마지막 RGB 프레임 저장."""
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB cv_bridge error: {e}")
            return

        self.latest_rgb = rgb
        self.latest_rgb_header = msg.header

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """CameraInfo 에서 fx, fy 추출."""
        try:
            k = msg.k
            self.fx = float(k[0])  # fx
            self.fy = float(k[4])  # fy
        except Exception as e:
            self.get_logger().warn(f"Failed to parse CameraInfo: {e}")

    def scan_callback(self, msg: LaserScan) -> None:
        """마지막 LiDAR 스캔 저장."""
        self.latest_scan = msg

    # ------------------------------------------------------------------
    # Timer: 상태 머신 + YOLO + 이동 제어
    # ------------------------------------------------------------------

    def timer_callback(self) -> None:
        if self.done:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # 필수 데이터 확보 여부 확인
        if self.latest_rgb is None or self.latest_rgb_header is None:
            self.get_logger().debug("No RGB frame yet; skip.")
            return
        if self.fx is None or self.fy is None:
            self.get_logger().debug("No CameraInfo yet (fx, fy); skip.")
            return
        if self.latest_scan is None:
            self.get_logger().debug("No LiDAR scan yet; skip.")
            return

        # LiDAR: 정면 거리 추정
        lidar_dist = self._estimate_forward_distance_from_scan(self.latest_scan)
        if lidar_dist is None or lidar_dist <= 0.0:
            self.get_logger().warn("Invalid LiDAR distance; skip this cycle.")
            return

        self.current_distance_m = float(lidar_dist)

        # LiDAR 전방 거리 로그 (1초에 한 번만 출력)
        if (
            self.last_lidar_log_time is None
            or now_sec - self.last_lidar_log_time > 1.0
        ):
            self.get_logger().info(
                f"LiDAR forward distance: {self.current_distance_m:.3f} m"
            )
            self.last_lidar_log_time = now_sec

        # 샘플 간격(10초) 제어: 최근 샘플 이후 10초가 지나지 않았다면 대기
        if (
            self.last_sample_time is not None
            and now_sec - self.last_sample_time < 10.0
        ):
            return

        # YOLO 호출 주기 제한 (옵션)
        if (
            self.last_yolo_eval_time is not None
            and now_sec - self.last_yolo_eval_time < self.yolo_eval_min_interval_sec
        ):
            return

        # YOLO 추론
        bgr = self.latest_rgb.copy()
        results = run_yolov5_inference(
            bgr_image=bgr,
            img_size=self.img_size,
            conf_threshold=self.yolo_conf_threshold,
        )
        boxes = results.xyxy[0]  # [N, 6]: x1, y1, x2, y2, conf, cls

        self.last_yolo_eval_time = now_sec

        if boxes is None or len(boxes) == 0:
            # 더 이상 목표 Object 가 검출되지 않으면 종료
            self.get_logger().info("YOLO: no detections. Finishing run.")
            self._finish_and_print()
            return

        # target_class 에 해당하는 detection 중 가장 conf 높은 것 선택
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
            # 목표 class 가 없으면 종료
            self.get_logger().info(
                f"YOLO: no '{self.target_class}' detections. Finishing run."
            )
            self._finish_and_print()
            return

        det, class_name, cls_id, conf = best_det

        # YOLO bbox 의 픽셀 폭/높이 계산 (x, y 방향)
        x1, y1, x2, y2, _, _ = det.tolist()
        pixel_width = max(0.0, float(x2 - x1))
        pixel_height = max(0.0, float(y2 - y1))

        # 기록 추가:
        #  - distance: LiDAR 로 측정한 실제 전방 거리 [m]
        #  - x, y: YOLO bbox width/height [px]
        distance_val = (
            float(self.current_distance_m)
            if self.current_distance_m is not None
            else 0.0
        )
        record = {
            "distance": distance_val,
            "x": float(pixel_width),
            "y": float(pixel_height),
        }
        self.records.append(record)

        self.get_logger().info(
            f"[sample] class={class_name}(id={cls_id}) conf={conf:.3f} "
            f"lidar_dist={distance_val:.3f}m "
            f"bbox_pix=({pixel_width:.1f}px, {pixel_height:.1f}px) "
            f"step_index={len(self.records)}"
        )

        # 다음 샘플 타이밍 기록 (10초 후에 다음 샘플을 측정한다고 가정)
        self.last_sample_time = now_sec
        self.get_logger().info(
            "Next sample will be taken after 10 seconds "
            "(assume robot has moved additional 0.10 m forward)."
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _estimate_forward_distance_from_scan(
        self, scan: LaserScan
    ) -> Optional[float]:
        """
        LiDAR (/scan) 에서 **정면 방향** 물체까지의 거리를 추정 (기록/로그용).

        - angle_min, angle_increment 정보를 사용해 "정면 ± sector_half_deg" 범위만 사용
        - 해당 sector 내 유효 range 들의 중앙값(median)을 사용해 튀는 값을 완화
        """
        ranges = np.array(scan.ranges, dtype=np.float32)

        # 각도 배열 계산
        angles = np.arange(
            scan.angle_min,
            scan.angle_min + scan.angle_increment * len(ranges),
            scan.angle_increment,
            dtype=np.float32,
        )

        # 기본 sector: 정면(0 rad) 기준 ±10도
        sector_half_deg = 10.0
        sector_half_rad = math.radians(sector_half_deg)

        valid_mask = np.isfinite(ranges) & (ranges > 0.0)
        front_mask = valid_mask & (np.abs(angles) <= sector_half_rad)
        front_ranges = ranges[front_mask]

        if front_ranges.size > 0:
            return float(np.median(front_ranges))

        all_valid = ranges[valid_mask]
        if all_valid.size == 0:
            return None

        return float(np.median(all_valid))

    def _forward(self) -> None:
        """
        로봇을 전진시키는 cmd_vel publish 함수였으나,
        현재 버전에서는 **forward 움직임을 완전히 배제**하기 위해
        정지 Twist(0.0 m/s)만 publish 한다.
        """
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def _stop(self) -> None:
        """로봇 정지."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def _finish_and_print(self) -> None:
        """최종 JSON 결과를 pretty print 하고 종료."""
        self._stop()
        self.done = True

        result = {self.casename: self.records}
        pretty = json.dumps(result, indent=4, ensure_ascii=False)
        # stdout 으로 출력
        print(pretty)

        # 노드 종료
        self.get_logger().info("LidarForwardNode finished. Shutting down.")
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarForwardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down LidarForwardNode (KeyboardInterrupt).")
    finally:
        node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
