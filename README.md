# tb4_target_selector (ROS2 Jazzy, TurtleBot4)

TurtleBot4 + ROS2 Jazzy 환경에서 다음 파이프라인을 구현하는 패키지/코드 모음이다.

1. YOLOv5n 커스텀 모델로 카메라 이미지에서 **타겟 물체 의심**을 탐지 (`/yolo/detections`)
2. 의심 탐지가 발생하면, 해당 물체의 **대략적인 월드 좌표(map frame)** 를 추정
3. 그 주변의 SLAM 맵(`/map`, `nav_msgs/OccupancyGrid`)을 **ROI로 잘라 OpenCV**로 사각형 장애물/구조물을 추정
4. 각 사각형 후보를 스코어링하고,
   - 우선순위 리스트를 `/target_box_candidates` 로 publish
   - 최고 우선순위 박스를 `/best_target_box` 로 publish

---

## 1. 패키지 구조 개요

워크스페이스 루트 기준:

```text
OSPrac/
  box_detector.py
  box_detector.md
  map_box_selector_node.py            # 설계/실험용 단독 스크립트 (원본)
  tb4_target_selector/
    package.xml
    CMakeLists.txt
    msg/
      Detection.msg
      YoloDetections.msg
      BoxCandidate.msg
      BoxCandidateArray.msg
      BestTargetBox.msg
    scripts/
      map_box_selector_node.py        # ROS2 패키지용 실행 스크립트
  progress.md
  README.md
```

실제 ROS2 패키지는 `tb4_target_selector/` 디렉터리를 기준으로 colcon build 한다.

---

## 2. 커스텀 메시지 정의

### 2.1 YOLO Detection

[`Detection.msg`](tb4_target_selector/msg/Detection.msg:1):

```text
int32 class_id
string class_name
float32 confidence
float32 x_center
float32 y_center
float32 width
float32 height
```

- 한 개의 YOLO detection 을 표현.
- `x_center, y_center, width, height` 는 이미지 픽셀 좌표계 기준.

[`YoloDetections.msg`](tb4_target_selector/msg/YoloDetections.msg:1):

```text
std_msgs/Header header
Detection[] detections
```

- `/yolo/detections` 토픽 타입.
- `header.frame_id` 는 카메라 프레임(`camera_link` 등)을 사용.

### 2.2 맵 박스 후보

[`BoxCandidate.msg`](tb4_target_selector/msg/BoxCandidate.msg:1):

```text
string related_class           # ex. "person", "chair"
float32 score                  # [0, 1]
geometry_msgs/Pose pose        # center pose in map frame (z ~= 0)
float32 width                  # [m]
float32 height                 # [m]
```

[`BoxCandidateArray.msg`](tb4_target_selector/msg/BoxCandidateArray.msg:1):

```text
std_msgs/Header header
BoxCandidate[] candidates
```

[`BestTargetBox.msg`](tb4_target_selector/msg/BestTargetBox.msg:1):

```text
std_msgs/Header header
BoxCandidate best_candidate
bool valid
```

---

## 3. 핵심 노드: map_box_selector_node

ROS2 노드 구현은 [`tb4_target_selector/scripts/map_box_selector_node.py`](tb4_target_selector/scripts/map_box_selector_node.py:1)에 위치한다.

### 3.1 구독 토픽

- `/yolo/detections` (`tb4_target_selector/YoloDetections`)
- `/map` (`nav_msgs/OccupancyGrid`)
- `/camera/depth/image_raw` (`sensor_msgs/Image`, 32FC1 또는 16UC1)
- `/camera/camera_info` (`sensor_msgs/CameraInfo`)
- TF:
  - `map → odom → base_link → camera_link` 체인

### 3.2 발행 토픽

- `/target_box_candidates` (`tb4_target_selector/BoxCandidateArray`)
- `/best_target_box` (`tb4_target_selector/BestTargetBox`)

### 3.3 주요 파라미터

[`MapBoxSelectorNode.__init__()`](tb4_target_selector/scripts/map_box_selector_node.py:59)에서 rclpy 파라미터로 선언:

- `target_classes` (string[], 기본 `["person", "chair", "target_obj"]`)
- `yolo_conf_threshold` (float, 기본 0.4)
- `roi_size_m` (float, 기본 2.0)
- `occ_threshold` (int, 기본 50)
- `min_area_pix` (int, 기본 20)
- `trigger_cooldown_sec` (float, 기본 1.0)
- `assumed_distance_m` (float, 기본 1.5)
- 프레임/토픽:
  - `yolo_topic` (기본 `/yolo/detections`)
  - `map_topic` (기본 `/map`)
  - `depth_topic` (기본 `/camera/depth/image_raw`)
  - `camera_info_topic` (기본 `/camera/camera_info`)
  - `map_frame` (기본 `map`)
  - `base_frame` (기본 `base_link`)
  - `camera_frame` (기본 `camera_link`)

---

## 4. ROS2 패키지 빌드 방법

### 4.1 워크스페이스 구성 예시

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 이 리포지토리를 src/ 아래에 클론했다고 가정
# ~/ros2_ws/src/OSPrac/ 에 현재 파일들이 존재
```

colcon build 시, `OSPrac/tb4_target_selector` 가 하나의 ROS2 패키지로 인식된다.

```bash
cd ~/ros2_ws
colcon build --packages-select tb4_target_selector
source install/setup.bash
```

### 4.2 노드 실행

```bash
# 맵, slam_toolbox, Nav2 등이 이미 올라와 있다고 가정
ros2 run tb4_target_selector map_box_selector_node.py
```

필요하다면 파라미터를 함께 지정:

```bash
ros2 run tb4_target_selector map_box_selector_node.py \
  --ros-args \
    -p yolo_topic:=/yolo/detections \
    -p map_topic:=/map \
    -p depth_topic:=/camera/depth/image_raw \
    -p camera_info_topic:=/camera/camera_info \
    -p target_classes:="['person','chair','target_obj']" \
    -p roi_size_m:=2.0 \
    -p yolo_conf_threshold:=0.4
```

---

## 5. YOLO 노드 연동 (개념)

- 별도의 YOLO 노드를 구현/사용해 `/yolo/detections` 를 발행해야 한다.
- 메시지 타입은 [`YoloDetections.msg`](tb4_target_selector/msg/YoloDetections.msg:1)를 따른다.
- 각 detection:
  - `class_name` 이 `target_classes` 화이트리스트에 포함
  - `confidence` 가 `yolo_conf_threshold` 이상
- 위 조건을 만족하는 detection 중에서 가장 confidence 높은 것 1개가 `map_box_selector_node`의 트리거가 된다.

---

## 6. 참고: box_detector (YOLOv5n + 크기 필터)

이 리포지토리에는 우체국 박스 4호 실제 크기를 이용한 별도 노드 설계도 포함되어 있다.

- 문서: [`box_detector.md`](box_detector.md:1)
- 구현: [`box_detector.py`](box_detector.py:1)

이 노드는 카메라/Depth/YOLOv5n 을 사용해 **실제 물리적 크기가 목표와 일치하는 박스만 필터링**하는 역할을 한다.
`tb4_target_selector` 패키지의 map 기반 후보 박스 선택 로직과 조합해 사용할 수 있다.

---

## 7. jw_explorer_node / frontier_core 와의 연동

워크스페이스 루트에는 프론티어 기반 탐색 노드와 코어 로직이 별도 Python 스크립트로 존재한다.

- 탐색 노드: [`jw_explorer_node.py`](jw_explorer_node.py:1)
- 프론티어 코어: [`frontier_core.py`](frontier_core.py:1)

### 7.1 좌표계 / 토픽 호환성

`ExplorerNode` 클래스는 다음과 같은 기본 파라미터를 사용한다:

- `map_frame` = `"map"`
- `base_frame` = `"base_link"`
- SLAM 맵: `/map` (`nav_msgs/OccupancyGrid`)
- 글로벌 코스트맵: `/global_costmap/costmap` (`nav_msgs/OccupancyGrid`)

`tb4_target_selector` 의 두 핵심 노드(`yolo_node`, `cv2_node`)도 동일한 좌표계/맵을 전제로 한다.

- `cv2_node`:
  - `map_topic` 기본값: `/map`
  - `map_frame` 기본값: `map`
  - `base_frame` 기본값: `base_link`
- `yolo_node`:
  - `map_frame` 기본값: `map`
  - `base_frame` 기본값: `base_link`
  - `camera_frame` 기본값: `camera_link` (TF 트리에서 카메라 프레임 이름만 일치시키면 됨)

따라서 `jw_explorer_node.py` 를 사용해 프론티어 탐색을 수행하면서,
동시에 `tb4_target_selector` 의 `yolo_node` + `cv2_node` 를 띄워도
맵/좌표계 측면에서 충돌 없이 함께 동작할 수 있다.

### 7.2 실행 예시 (탐색 + 타겟 셀렉터 병행)

1. Nav2 + SLAM (`slam_toolbox`) + TF 트리(map, odom, base_link, camera_link) 가 이미 동작 중이라고 가정한다.
2. 별도 터미널에서 프론티어 탐색 노드를 Python으로 실행한다 (ROS 패키지로 옮기지 않은 순수 스크립트 케이스):

```bash
cd ~/ros2_ws/src/OSPrac
python3 jw_explorer_node.py
```

3. 또 다른 터미널에서 `tb4_target_selector` 패키지를 빌드/실행한다:

```bash
cd ~/ros2_ws
colcon build --packages-select tb4_target_selector
source install/setup.bash

# YOLO + CV2 노드 통합 런치
ros2 launch tb4_target_selector target_selector.launch.py \
  model_path:=/absolute/path/to/your_yolov5n_custom.pt \
  rgb_topic:=/camera/image_raw \
  depth_topic:=/camera/depth/image_raw \
  camera_info_topic:=/camera/camera_info
```

이 구성에서:

- `jw_explorer_node` 는 `/map` + `/global_costmap/costmap` 을 이용해 프론티어를 찾아 Nav2 로 이동
- `tb4_target_selector` 는 동일한 `/map` 과 TF 트리를 사용하면서
  카메라 이미지 기반 YOLO 탐지 + BoxQuery 서비스(`/box_query`) 를 통해
  특정 타겟 주변의 맵 상 사각형(박스) 후보를 추출해 줄 수 있다.

향후 통합 방향 예시:

- `ExplorerNode` 내부에서
  - `BoxQuery` 서비스를 직접 호출해 특정 위치 주변의 후보 박스를 조회하거나,
  - `/detected_box_markers` 대신 `BoxCandidate[]` 를 이용한 고정 포인트/접근 전략을 구성하는 식으로
- 프론티어 탐색과 타겟 박스 선택 로직을 더 강하게 결합할 수 있다.