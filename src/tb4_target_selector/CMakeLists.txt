cmake_minimum_required(VERSION 3.8)
project(tb4_target_selector)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/YoloDetections.msg"
  "msg/BoxCandidate.msg"
  "msg/BoxCandidateArray.msg"
  "msg/BestTargetBox.msg"
  "srv/BoxQuery.srv"
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

install(PROGRAMS
  scripts/map_box_selector_node.py
  scripts/cv2_node.py
  scripts/yolo_node.py
  scripts/box_size_debug_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# box_size_debug_node 가 사용하는 공용 Python 모듈(detect_func, yolov5_singleton)도 함께 설치
install(FILES
  scripts/detect_func.py
  scripts/yolov5_singleton.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  msg
  srv
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf2_ros
)

ament_package()