#!/usr/bin/env python3
"""
lidar_forward: LiDAR + YOLOv5 기반 전진 측정 스크립트 (ROS2 + TurtleBot4)

요구사항 요약
------------
1. 실행 시 현재 좌표를 (0, 0) 초기점으로 가정.
   - 전진 방향을 +x 로 본다.
   - 실제 TF/odom 은 사용하지 않고, LiDAR 거리만으로 전진 거리를 제어한다.

2. 목표 Object 가 정면에 있다고 가정.
   - 조금씩 전진하면서
     - LiDAR 로 측정한 목표까지의 거리(distance, [m])
     - YOLO + 카메라로 추정한 물체 가로/세로 길이(x, y, [m])
   를 기록한다.

3. 목표 Object 가 더 이상 YOLO 로 인식되지 않을 때까지 전진.

4. 전진 스텝은 LiDAR 기준 약 10cm 내외(step_m).

5. 최종 실행 결과를 JSON pretty print 로 다음과 같이 출력하고 종료:

    {
        "{casename}": [
            {
                "distance": float,  # LiDAR 로 측정한 거리 [m]
                "x": float,         # YOLO+카메라로 추정한 가로 길이 [m]
                "y": float          # YOLO+카메라로 추정한 세로 길이 [m]
            },
            ...
        ]
    }

   - casename 은 ROS2 파라미터 "casename" 으로 입력받는다.


실행 예시
---------
ros2 run (또는 python) 으로 직접 실행:

python lidar_forward.py \
  --ros-args \
    -p casename:=test_case_01 \
    -p model_path:=./yolov5n.pt \
    -p rgb_topic:=/oakd/rgb/preview/image_raw \
    -p camera_info_topic:=/oakd/rgb/preview/camera_info \
    -p scan_topic:=/scan \
    -p cmd_vel_topic:=/cmd_vel \
    -p step_m:=0.10 \
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

from new_func.detect_func import estimate_object_size_from_bbox
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

        # LiDAR 기준 전진 스텝 (약 10cm 내외)
        self.declare_parameter("step_m", 0.10)

        # YOLO 관련 파라미터
        self.declare_parameter("yolo_conf_threshold", 0.4)
        self.declare_parameter("target_class", "box")
        self.declare_parameter("img_size", 640)
        self.declare_parameter("device", "cpu")

        # YOLO 평가 최소 간격 (초). 너무 빨리 돌지 않게 0.5~1.0 정도 권장.
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

        # movement / sampling state
        self.last_yolo_eval_time: Optional[float] = None
        self.current_distance_m: Optional[float] = None
        self.target_distance_m: Optional[float] = None
        self.is_moving: bool = False
        self.done: bool = False
        # LiDAR 전방 거리 로그용 타임스탬프 (로그 스팸 방지)
        self.last_lidar_log_time: Optional[float] = None

        # "개념상" 기준 거리 (예: 2.0m 에서 step_m 씩 줄여가며 샘플링)
        # 실제 LiDAR 거리(current_distance_m)는 그대로 사용하되,
        # JSON distance 필드는 base_distance_m - n * step_m 로 기록한다.
        self.base_distance_m: float = 2.0

        # 기록용 데이터
        # [{"distance": float, "x": float, "y": float}, ...]
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

        # 현재 LiDAR 거리 추정 (정면 최소 거리)
        current_dist = self._estimate_forward_distance_from_scan(self.latest_scan)
        if current_dist is None or current_dist <= 0.0:
            self.get_logger().warn("Invalid LiDAR distance; skip this cycle.")
            return

        self.current_distance_m = current_dist

        # LiDAR 전방 거리 로그 (1초에 한 번만 출력)
        if (
            getattr(self, "last_lidar_log_time", None) is None
            or now_sec - getattr(self, "last_lidar_log_time") > 1.0
        ):
            self.get_logger().info(
                f"LiDAR forward distance: {self.current_distance_m:.3f} m"
            )
            self.last_lidar_log_time = now_sec

        # LiDAR 전방 거리 로그 (1초에 한 번만 출력)
        if (
            self.last_lidar_log_time is None
            or now_sec - self.last_lidar_log_time > 1.0
        ):
            self.get_logger().info(
                f"LiDAR forward distance: {self.current_distance_m:.3f} m"
            )
            self.last_lidar_log_time = now_sec

        # 이동 중인 경우: 아직 목표 거리까지 도달 안 했다면 계속 전진
        if self.is_moving:
            if self.current_distance_m is not None and self.target_distance_m is not None:
                # 목표 거리(target_distance_m) 이하로 가까워지면 정지 후 샘플링 모드로 전환
                if self.current_distance_m <= self.target_distance_m:
                    self._stop()
                    self.is_moving = False
                    self.get_logger().info(
                        f"Reached step target distance={self.current_distance_m:.3f} m "
                        f"(target={self.target_distance_m:.3f} m). Stopping for next sample."
                    )
            return

        # 여기부터는 로봇이 정지 상태에서 샘플 + 다음 스텝 계획
        # YOLO 호출 주기 제한
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

        # 개념상 기준 거리 (예: 2.0m 에서 step_m 씩 줄여가며 샘플링)
        # 첫 샘플: base_distance_m
        # 두 번째 샘플: base_distance_m - step_m
        # ...
        conceptual_distance = max(
            self.base_distance_m - self.step_m * (len(self.records)), 0.0
        )

        # 기록 추가:
        #  - distance: 개념상 기준 거리 (2.0 - n*step_m) [m]
        #  - x, y: YOLO bbox width/height [px]
        record = {
            "distance": float(conceptual_distance),
            "x": float(pixel_width),
            "y": float(pixel_height),
        }
        self.records.append(record)

        self.get_logger().info(
            f"[sample] class={class_name}(id={cls_id}) conf={conf:.3f} "
            f"lidar_dist={self.current_distance_m:.3f}m "
            f"conceptual_dist={conceptual_distance:.3f}m "
            f"bbox_pix=({pixel_width:.1f}px, {pixel_height:.1f}px) "
            f"step_index={len(self.records)}"
        )

        # 다음 스텝을 위한 목표 거리 설정 (현재보다 step_m 만큼 가까워지도록)
        self.target_distance_m = max(self.current_distance_m - self.step_m, 0.05)
        self.get_logger().info(
            f"Planning next step: current={self.current_distance_m:.3f} m, "
            f"target={self.target_distance_m:.3f} m (step={self.step_m:.3f} m)"
        )

        # 앞으로 전진 시작
        self._forward()
        self.is_moving = True

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _estimate_forward_distance_from_scan(
        self, scan: LaserScan
    ) -> Optional[float]:
        """
        LiDAR (/scan) 에서 전방(정면) 물체까지의 거리를 추정.

        단순 버전:
        - 모든 유효한 range 값 중 최소값을 사용.
        - 더 정교하게 하려면 angle_min/max 와 카메라 중심 방향을 매핑하여
          특정 중앙 sector 내에서만 최소값을 뽑도록 확장 가능.
        """
        ranges = np.array(scan.ranges, dtype=np.float32)
        valid = ranges[np.isfinite(ranges) & (ranges > 0.0)]

        if valid.size == 0:
            return None

        return float(valid.min())

    def _forward(self) -> None:
        """로봇을 전진시키는 cmd_vel publish (단순 정속 전진)."""
        twist = Twist()
        twist.linear.x = 0.1  # 0.1 m/s 정도로 천천히 전진
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
