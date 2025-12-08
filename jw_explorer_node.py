#!/usr/bin/env python3
import math
import time
from typing import Optional, Tuple, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from rclpy.exceptions import ParameterAlreadyDeclaredException # 예외 처리를 위해 추가

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import MarkerArray

import tf2_ros

# frontier_core.py 를 같은 패키지(turtlebot4_explorer) 안에 두고 사용
from .frontier_core import FrontierSearch, Point
# [NEW] 박스 디텍터 모듈 임포트
from .box_detector import BoxDetector


class TaskResult:
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3


class FrontierCostmapAdapter:
    """
    frontier_core.FrontierSearch 에 넘기기 위한 Costmap 래퍼.
    """

    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0
        self.data = np.zeros(0, dtype=np.uint8)

    def update_from_msg(self, msg: OccupancyGrid) -> None:
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        # origin orientation -> yaw
        q = msg.info.origin.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.origin_yaw = math.atan2(siny_cosp, cosy_cosp)

        # nav2 costmap은 uint8(0..255)
        arr = np.array(msg.data, dtype=np.uint8)
        self.data = arr.reshape(-1)

    # --- FrontierSearch용 인터페이스 ---

    def getSizeInCellsX(self) -> int:
        return self.width

    def getSizeInCellsY(self) -> int:
        return self.height

    def getResolution(self) -> float:
        return self.resolution

    def getCharMap(self):
        return self.data

    def indexToCells(self, index: int) -> Tuple[int, int]:
        mx = index % self.width
        my = index // self.width
        return mx, my

    def mapToWorld(self, mx: int, my: int) -> Tuple[float, float]:
        # grid 좌표계 (origin 기준, 회전 전)
        gx = (mx + 0.5) * self.resolution
        gy = (my + 0.5) * self.resolution

        # origin yaw 적용
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        dx = cos_yaw * gx - sin_yaw * gy
        dy = sin_yaw * gx + cos_yaw * gy

        x = self.origin_x + dx
        y = self.origin_y + dy
        return x, y

    def worldToMap(self, wx: float, wy: float) -> Tuple[bool, int, int]:
        # origin 기준으로 평행이동
        dx = wx - self.origin_x
        dy = wy - self.origin_y

        # world -> grid (역회전)
        cos_yaw = math.cos(-self.origin_yaw)
        sin_yaw = math.sin(-self.origin_yaw)
        gx = cos_yaw * dx - sin_yaw * dy
        gy = sin_yaw * dx + cos_yaw * dy

        mx = int(gx / self.resolution)
        my = int(gy / self.resolution)

        if mx < 0 or my < 0 or mx >= self.width or my >= self.height:
            return False, 0, 0

        return True, mx, my

    def getIndex(self, mx: int, my: int) -> int:
        return my * self.width + mx


class ExplorerNode(Node):
    """
    개선된 프론티어 기반 탐색 노드.
    """

    def __init__(self) -> None:
        super().__init__('explorer_node')

        # -----------------------
        # Parameters
        # -----------------------
        self.declare_parameter('total_time_limit', 600.0)
        self.declare_parameter('goal_timeout', 40.0)
        self.declare_parameter('costmap_threshold', 220)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        
        # 탐색 관련 파라미터
        self.declare_parameter('spin_speed', 0.5) # rad/s
        self.declare_parameter('goal_retry_limit', 3) 
        self.declare_parameter('blacklist_radius', 0.5) 

        # FrontierSearch 튜닝용 파라미터
        self.declare_parameter('frontier_min_size', 0.5)
        self.declare_parameter('frontier_potential_scale', 1e-3)
        self.declare_parameter('frontier_gain_scale', 1.0)
        
        # [NEW] 박스 탐지 파라미터
        self.declare_parameter('box_min_area', 0.07) # m^2
        self.declare_parameter('box_max_area', 0.25) # m^2

        # [FIX] use_sim_time 중복 선언 방지
        # 런치 파일에서 이미 선언된 경우 예외 처리 또는 확인 후 선언
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        # 값 로드
        self.total_time_limit = self.get_parameter('total_time_limit').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.costmap_threshold = self.get_parameter('costmap_threshold').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.costmap_topic = self.get_parameter('costmap_topic').value
        self.spin_speed = self.get_parameter('spin_speed').value
        self.goal_retry_limit = self.get_parameter('goal_retry_limit').value
        self.blacklist_radius = self.get_parameter('blacklist_radius').value

        self.frontier_min_size = self.get_parameter('frontier_min_size').value
        self.frontier_potential_scale = self.get_parameter('frontier_potential_scale').value
        self.frontier_gain_scale = self.get_parameter('frontier_gain_scale').value
        
        box_min = self.get_parameter('box_min_area').value
        box_max = self.get_parameter('box_max_area').value

        # 시간 관리
        self.start_time = self.get_clock().now()

        # 위치 저장소
        self.initial_pose: Optional[PoseStamped] = None  # 시작 위치 저장용

        # Costmap 저장소
        self.costmap_msg: Optional[OccupancyGrid] = None
        self.costmap_array: Optional[np.ndarray] = None

        # SLAM 맵 저장소 (/map)
        self.map_msg: Optional[OccupancyGrid] = None

        # SLAM 맵 구독
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # FrontierSearch용 costmap 래퍼
        self.frontier_costmap = FrontierCostmapAdapter()
        self.frontier_search = FrontierSearch(
            costmap=self.frontier_costmap,
            potential_scale=self.frontier_potential_scale,
            gain_scale=self.frontier_gain_scale,
            min_frontier_size=self.frontier_min_size,
            logger=self.get_logger().get_child('frontier'),
        )

        # [NEW] Box Detector 및 Publisher 초기화
        self.box_detector = BoxDetector(min_area=box_min, max_area=box_max)
        self.box_marker_pub = self.create_publisher(MarkerArray, '/detected_box_markers', 10)

        # TF & Nav2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Velocity Publisher (for spinning)
        qos = QoSProfile(depth=10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        # Action state
        self._goal_handle = None
        self._result_future = None
        
        # Blacklist: 실패했거나 접근 불가능한 지점들
        self.blacklist: List[Tuple[float, float]] = []

        # Costmap 구독
        self.create_subscription(
            OccupancyGrid,
            self.costmap_topic,
            self.costmap_callback,
            10
        )

        self.get_logger().info('Full-Cycle ExplorerNode initialized.')
        # 파라미터 값 확인용 로그
        try:
            use_sim = self.get_parameter("use_sim_time").value
            self.get_logger().info(f'Params: Map={self.map_frame}, Base={self.base_frame}, SimTime={use_sim}')
        except Exception:
            self.get_logger().warn('Could not read use_sim_time parameter.')

    # ------------------------------------------------------------------
    # Costmap handling
    # ------------------------------------------------------------------
    def costmap_callback(self, msg: OccupancyGrid) -> None:
        self.costmap_msg = msg
        try:
            data = np.array(msg.data, dtype=np.int16)
            self.costmap_array = data.reshape((msg.info.height, msg.info.width))
        except Exception as exc:
            self.get_logger().error(f'Failed to reshape costmap: {exc}')
            self.costmap_array = None

        try:
            self.frontier_costmap.update_from_msg(msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to update frontier costmap: {exc}')


    def map_callback(self, msg: OccupancyGrid) -> None:
        # SLAM에서 오는 /map OccupancyGrid 저장
        self.map_msg = msg


    # ------------------------------------------------------------------
    # Robot Control & TF
    # ------------------------------------------------------------------
    def lookup_robot_pose(self) -> Optional[PoseStamped]:
        try:
            # 1. 변환 가능 여부 확인
            if not self.tf_buffer.can_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(), timeout=Duration(seconds=1.0)
            ):
                self.get_logger().warn(
                    f'[TF] Cannot transform "{self.map_frame}" to "{self.base_frame}". '
                    'Waiting for TF tree...'
                )
                return None
            
            # 2. 변환 시도
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        
        except tf2_ros.LookupException as e:
            self.get_logger().error(f'[TF] LookupException: {e}')
            return None
        except tf2_ros.ConnectivityException as e:
            self.get_logger().error(f'[TF] ConnectivityException: {e}')
            return None
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().error(f'[TF] ExtrapolationException: {e} (Check "use_sim_time")')
            return None
        except Exception as e:
            self.get_logger().error(f'[TF] Unknown Error: {e}')
            return None

        # 3. 변환 성공 시 Pose 생성
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = transform.header.stamp
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.orientation = transform.transform.rotation
        return pose

    def spin_robot(self, duration_sec: float) -> None:
        """제자리에서 지정된 시간 동안 회전"""
        self.get_logger().info(f'[ACTION] Spinning for {duration_sec:.1f}s...')
        twist = Twist()
        twist.angular.z = self.spin_speed
        
        start = self.get_clock().now()
        while rclpy.ok():
            now = self.get_clock().now()
            if (now - start).nanoseconds / 1e9 > duration_sec:
                break
            self.cmd_vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # 정지
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        time.sleep(1.0) # 안정화 대기

    # ------------------------------------------------------------------
    # Nav2 Action
    # ------------------------------------------------------------------
    def _nav_goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('[Nav2] Goal rejected.')
            self._goal_handle = None
            self._result_future = None
            return
        self._goal_handle = goal_handle
        self._result_future = goal_handle.get_result_async()

    def goToPose(self, pose: PoseStamped) -> None:
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('navigate_to_pose server not available.')

        future = self.nav_to_pose_client.send_goal_async(goal_msg)
        future.add_done_callback(self._nav_goal_response_callback)

    def isTaskComplete(self) -> bool:
        return self._result_future is not None and self._result_future.done()

    def getResult(self) -> int:
        if self._result_future is None or not self._result_future.done():
            return TaskResult.UNKNOWN
        result = self._result_future.result()
        status = result.status
        self._goal_handle = None
        self._result_future = None
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.FAILED

    def cancelTask(self) -> None:
        if self._goal_handle:
            try:
                self._goal_handle.cancel_goal_async()
            except:
                pass

    # ------------------------------------------------------------------
    # Frontier Logic
    # ------------------------------------------------------------------
    def is_blacklisted(self, x: float, y: float) -> bool:
        for bx, by in self.blacklist:
            dist = math.hypot(x - bx, y - by)
            if dist < self.blacklist_radius:
                return True
        return False

    def is_safe_goal(self, x: float, y: float) -> bool:
        if self.costmap_array is None:
            return False
        
        ok, mx, my = self.frontier_costmap.worldToMap(x, y)
        if not ok: return False
        
        h, w = self.costmap_array.shape
        if mx < 0 or mx >= w or my < 0 or my >= h:
            return False

        cell_cost = int(self.costmap_array[my, mx])
        
        # Free(0) ~ CostmapThreshold 미만이면 OK
        if 0 <= cell_cost < self.costmap_threshold:
            return True
        
        return False

    def sample_frontier_goal(self) -> Optional[Tuple[float, float]]:
        if self.costmap_array is None:
            return None

        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            return None
        
        rx = robot_pose.pose.position.x
        ry = robot_pose.pose.position.y
        robot_point = Point(x=rx, y=ry)

        frontiers = self.frontier_search.search_from(robot_point)
        if not frontiers:
            return None

        # cost 오름차순 정렬된 frontiers
        for f in frontiers:
            if self.is_blacklisted(f.centroid.x, f.centroid.y):
                continue
            
            # 후보: middle, centroid, initial
            candidates = [f.middle, f.centroid, f.initial]
            
            # 로봇과 너무 가까우면 스킵 (이미 도달)
            dist_to_robot = math.hypot(f.middle.x - rx, f.middle.y - ry)
            if dist_to_robot < 0.3:
                continue

            for cand in candidates:
                cx, cy = cand.x, cand.y
                if self.is_safe_goal(cx, cy):
                    self.get_logger().info(
                        f'[FRONTIER] Found goal at ({cx:.2f}, {cy:.2f}) '
                        f'dist={dist_to_robot:.2f}, size={f.size}'
                    )
                    return (cx, cy)

        self.get_logger().info('[FRONTIER] Frontiers exist but are unsafe or blacklisted.')
        return None

    # ------------------------------------------------------------------
    # Main Logic
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.get_logger().info('[START] Exploration Node Started.')
        
        # 1. 초기화 및 준비 대기
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=60.0):
            self.get_logger().error('Nav2 Action Server not ready.')
            return

        while rclpy.ok() and self.costmap_array is None:
            self.get_logger().info('Waiting for Costmap...')
            rclpy.spin_once(self, timeout_sec=1.0)

        # 2. 시작 위치 저장 (Return Home 용)
        self.get_logger().info('[INIT] Waiting for initial robot pose...')
        while rclpy.ok():
            pose = self.lookup_robot_pose()
            if pose:
                self.initial_pose = pose
                self.get_logger().info(
                    f'[INIT] Initial Pose Captured: ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})'
                )
                break
            rclpy.spin_once(self, timeout_sec=0.5)

        # 3. [필수] 시작 시 무조건 360도 회전
        spin_duration = (2.0 * math.pi) / self.spin_speed
        self.get_logger().info('[INIT] Performing mandatory initial 360 spin...')
        self.spin_robot(spin_duration + 0.5)

        no_frontier_retry_count = 0

        # 4. 메인 탐색 루프
        while rclpy.ok():
            now = self.get_clock().now()
            if (now - self.start_time).nanoseconds / 1e9 > self.total_time_limit:
                self.get_logger().info('[DONE] Total time limit reached.')
                break

            goal_xy = self.sample_frontier_goal()

            if goal_xy is None:
                no_frontier_retry_count += 1
                self.get_logger().warn(f'[RETRY] No valid frontier found. Count: {no_frontier_retry_count}/{self.goal_retry_limit}')
                
                if no_frontier_retry_count > self.goal_retry_limit:
                    self.get_logger().info('[DONE] Exploration Completed (No frontiers left).')
                    break 
                
                self.spin_robot(4.0)
                continue
            
            no_frontier_retry_count = 0
            gx, gy = goal_xy

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = self.map_frame
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = gx
            goal_pose.pose.position.y = gy
            goal_pose.pose.orientation.w = 1.0 

            self.goToPose(goal_pose)
            
            nav_start = self.get_clock().now()
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)

                if self.isTaskComplete():
                    result = self.getResult()
                    if result == TaskResult.SUCCEEDED:
                        self.get_logger().info('[NAV] Reached frontier.')
                        self.spin_robot(3.0)
                    else:
                        self.get_logger().warn('[NAV] Failed to reach frontier.')
                        self.blacklist.append((gx, gy))
                        self.spin_robot(2.0)
                    break
                
                if (self.get_clock().now() - nav_start).nanoseconds / 1e9 > self.goal_timeout:
                    self.get_logger().warn('[NAV] Goal timeout. Canceling.')
                    self.cancelTask()
                    self.blacklist.append((gx, gy))
                    break

        # 5. [종료 후] 시작 위치로 복귀 (Return to Home)
        if self.initial_pose:
            self.get_logger().info('------------------------------------------------')
            self.get_logger().info('[RETURN] Exploration finished. Returning to START POSE...')
            self.get_logger().info('------------------------------------------------')
            
            self.goToPose(self.initial_pose)
            
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.isTaskComplete():
                    res = self.getResult()
                    if res == TaskResult.SUCCEEDED:
                        self.get_logger().info('[RETURN] Successfully returned to start pose.')
                    else:
                        self.get_logger().warn('[RETURN] Failed to return to start pose.')
                    break
        else:
            self.get_logger().warn('[RETURN] Initial pose not set. Cannot return.')

        # -------------------------------------------------------------
        # [MODIFIED] 6. 박스 탐지 및 지속적 시각화 (Idle Loop)
        # -------------------------------------------------------------
        self.get_logger().info('------------------------------------------------')
        self.get_logger().info('[BOX] Starting Continuous Box Detection & Publishing...')
        self.get_logger().info('[DONE] Exploration Node is now in IDLE mode.')
        
        # 마지막으로 맵 업데이트를 위해 한 번 스핀
        rclpy.spin_once(self, timeout_sec=0.5)

        try:
            while rclpy.ok():
                # 1. SLAM 맵이 있으면 박스 탐지 및 마커 생성
                if self.map_msg is not None:
                    # 현재 시간으로 마커 생성
                    markers = self.box_detector.detect_boxes(self.map_msg, self.get_clock().now())
                    
                    # 2. 마커 퍼블리시 (지속적으로 쏨)
                    self.box_marker_pub.publish(markers)
                else:
                    self.get_logger().warn('[BOX] Waiting for map data...')

                # 3. 노드 생존 유지 (1초 간격으로 처리)
                rclpy.spin_once(self, timeout_sec=1.0)
                
        except KeyboardInterrupt:
            self.get_logger().info('[EXIT] Interrupted by user.')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorerNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()