#!/usr/bin/env python3
"""
Launch file for tb4_target_selector:

- yolo_node: YOLOv5n 기반 타겟 의심 탐지 + BoxQuery 서비스 클라이언트
- cv2_node: /map ROI + OpenCV 사각형 추정 + BoxQuery 서비스 서버
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path = LaunchConfiguration("model_path")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    declare_model_path = DeclareLaunchArgument(
        "model_path",
        default_value="/path/to/yolov5n_custom.pt",
        description="Path to YOLOv5n custom weights (.pt)",
    )

    declare_rgb_topic = DeclareLaunchArgument(
        "rgb_topic",
        default_value="/camera/image_raw",
        description="RGB image topic for YOLO node",
    )

    declare_depth_topic = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/depth/image_raw",
        description="Depth image topic for YOLO node",
    )

    declare_camera_info_topic = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/camera_info",
        description="CameraInfo topic for YOLO node",
    )

    yolo_node = Node(
        package="tb4_target_selector",
        executable="yolo_node.py",
        name="yolo_node",
        output="screen",
        parameters=[
            {
                "model_path": model_path,
                "rgb_topic": rgb_topic,
                "depth_topic": depth_topic,
                "camera_info_topic": camera_info_topic,
            }
        ],
    )

    cv2_node = Node(
        package="tb4_target_selector",
        executable="cv2_node.py",
        name="cv2_node",
        output="screen",
        parameters=[{}],
    )

    return LaunchDescription(
        [
            declare_model_path,
            declare_rgb_topic,
            declare_depth_topic,
            declare_camera_info_topic,
            cv2_node,
            yolo_node,
        ]
    )