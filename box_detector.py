import math
import numpy as np
import cv2
from typing import List, Tuple, Optional

from rclpy.time import Time
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

class BoxDetector:
    def __init__(self, min_area: float = 0.02, max_area: float = 0.5):
        """
        min_area: 박스로 인식할 최소 면적 (m^2)
        max_area: 박스로 인식할 최대 면적 (m^2)
        """
        self.min_area = min_area
        self.max_area = max_area

    def detect_boxes(self, costmap_msg: OccupancyGrid, current_time: Time) -> MarkerArray:
        marker_array = MarkerArray()
        
        # 이전 마커들을 지우기 위해 DELETEALL 마커를 하나 추가할 수도 있지만,
        # 여기서는 ID를 0부터 덮어쓰는 방식으로 진행합니다.
        
        if costmap_msg is None:
            return marker_array

        width = costmap_msg.info.width
        height = costmap_msg.info.height
        resolution = costmap_msg.info.resolution
        origin_x = costmap_msg.info.origin.position.x
        origin_y = costmap_msg.info.origin.position.y
        
        # Quaternion to Yaw
        q = costmap_msg.info.origin.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        origin_yaw = math.atan2(siny_cosp, cosy_cosp)

        # 1. 맵 데이터를 이미지로 변환 (slam_toolbox: -1, 0, 100 가정)
        data = np.array(costmap_msg.data, dtype=np.int16).reshape((height, width))

        # 이진화:
        #   - 100 (occupied)만 흰색(255)
        #   - 0 (free), -1 (unknown)은 전부 0 (배경)으로 처리
        binary_img = np.where(data == 100, 255, 0).astype(np.uint8)

        # 2. 윤곽선(Contours) 검출
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_id = 0
        for cnt in contours:
            # 3. 면적 계산 (픽셀 -> 미터)
            pixel_area = cv2.contourArea(cnt)
            real_area = pixel_area * (resolution * resolution)

            # 면적 필터링
            if self.min_area <= real_area <= self.max_area:
                # 4. 중심점(Centroid) 계산
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]

                # 5. Grid -> World 좌표 변환
                cos_yaw = math.cos(origin_yaw)
                sin_yaw = math.sin(origin_yaw)

                wx_local = cX * resolution
                wy_local = cY * resolution

                final_x = origin_x + (wx_local * cos_yaw - wy_local * sin_yaw)
                final_y = origin_y + (wx_local * sin_yaw + wy_local * cos_yaw)

                # 6. 마커 생성
                marker = Marker()
                marker.header.frame_id = costmap_msg.header.frame_id
                marker.header.stamp = current_time.to_msg()
                marker.ns = "detected_boxes"
                marker.id = box_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                marker.pose.position.x = final_x
                marker.pose.position.y = final_y
                marker.pose.position.z = 0.2  # 바닥에서 좀 더 위로

                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.4
                marker.scale.y = 0.4
                marker.scale.z = 0.4

                marker.color.r = 0.6
                marker.color.g = 1.0
                marker.color.b = 0.2
                marker.color.a = 0.9  # 약간 투명도

                marker.lifetime = Duration().to_msg()  # 0 = 무한 유지

                marker_array.markers.append(marker)
                box_id += 1

        return marker_array
