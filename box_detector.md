# box_detector (YOLOv5n + Size Filter ROS2 Node)

ROS2 Jazzy 환경에서 **YOLOv5n 커스텀 모델(`box` 레이블)**과 **대한민국 우체국 박스 4호 실제 크기(410 × 310 × 280 mm)**, 그리고 **길이 오차 허용값(기본 5cm)**를 함께 사용하여 **진짜 4호 상자만 탐색**하는 노드에 대한 문서이다.

구현 파일: `box_detector.py`  
핵심 클래스: `BoxDetectorWithSizeNode`


---

## 1. 노드 개요

### 1.1 기능 요약

- 입력:
  - RGB 이미지: `sensor_msgs/msg/Image`
  - Depth 이미지: `sensor_msgs/msg/Image`
- 내부 처리:
  1. YOLOv5n 커스텀 모델로 `box` 라벨에 해당하는 Bounding Box 탐지
  2. 각 Bounding Box 영역의 Depth 값을 샘플링해 3D 포인트 구하기
  3. 3D 포인트들의 최소/최대 좌표로부터 실제 3D 크기(dX, dY, dZ) 추정
  4. 목표 박스 크기 (0.41, 0.31, 0.28)m 와 비교 후, 축 순서를 무시하고 오차 허용범위(기본 0.05m) 이내인지 검사
- 출력:
  - `/box_detector/box_pose` (`geometry_msgs/PoseStamped`)
    - `pose.position`: 카메라 좌표계에서 박스 중심 (X, Y, Z)
    - `pose.orientation`: `w=1.0` (기본 단위 quaternion, 방향은 아직 미사용)
    - `header.frame_id`: RGB 이미지 frame


### 1.2 전제 조건

- ROS2 Jazzy
- ROS 패키지(apt 설치):
  - `rclpy`
  - `sensor_msgs`
  - `geometry_msgs`
  - `cv_bridge`
- Python 패키지(pip 설치):
  - `torch`
  - `numpy`
  - `opencv-python`
- YOLOv5n 커스텀 모델
  - `box` 라벨로 이미 학습된 `.pt` weight 파일 (예: `box_yolov5n_best.pt`)
- Depth / RGB 동기
  - Depth와 RGB가 정렬(aligned)되어 있거나, 크게 어긋나지 않는 전제
- 카메라 내파라미터
  - fx, fy, cx, cy 값 알고 있어야 함


---

## 2. ROS 인터페이스

### 2.1 파라미터

모두 ROS2 파라미터로 설정 가능하다.

| 파라미터 이름          | 타입   | 기본값                               | 설명 |
|------------------------|--------|--------------------------------------|------|
| `model_path`           | string | `/path/to/box_yolov5n.pt`           | 학습 완료된 YOLOv5n `box` 모델 `.pt` 파일 경로 (local path) |
| `rgb_topic`            | string | `/camera/color/image_raw`           | RGB 이미지 토픽 이름 |
| `depth_topic`          | string | `/camera/aligned_depth_to_color/image_raw` | Depth 이미지 토픽 이름 |
| `conf_threshold`       | double | `0.4`                               | YOLO confidence threshold |
| `length_tolerance_m`   | double | `0.05`                              | 길이/너비/높이 허용 오차(미터 단위, 기본 5cm) |
| `fx`                   | double | `615.0`                             | 카메라 focal length x |
| `fy`                   | double | `615.0`                             | 카메라 focal length y |
| `cx`                   | double | `320.0`                             | 카메라 principal point x |
| `cy`                   | double | `240.0`                             | 카메라 principal point y |


### 2.2 구독 토픽

- `rgb_topic` (기본: `/camera/color/image_raw`)
  - 타입: `sensor_msgs/msg/Image`
  - 용도: YOLOv5n 추론 입력용 RGB 이미지

- `depth_topic` (기본: `/camera/aligned_depth_to_color/image_raw`)
  - 타입: `sensor_msgs/msg/Image`
  - 용도: RGB와 정렬된 Depth 이미지 (단위 m 또는 mm)


### 2.3 발행 토픽

- `/box_detector/box_pose`
  - 타입: `geometry_msgs/msg/PoseStamped`
  - frame:
    - `header.frame_id` = RGB frame_id (예: `camera_color_optical_frame`)
  - 의미:
    - position.x, y, z: 카메라 좌표계에서의 박스 중심 3D 좌표
    - orientation: `(x=0, y=0, z=0, w=1)` (기본 단위 quaternion)


---

## 3. 알고리즘 상세

### 3.1 YOLOv5n 기반 초기 탐지

1. RGB 이미지 수신 → OpenCV BGR로 변환
2. YOLOv5n 모델 호출:

   ```python
   results = self.model(rgb)
   boxes = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]
   names = results.names
   ```

3. 각 detection에 대해:
   - 클래스 이름 조회:

     ```python
     class_name = names.get(cls_id, str(cls_id))
     ```

   - 필터링 조건:
     - `class_name == "box"`
     - `conf >= conf_threshold`

4. 조건을 만족하는 bbox만 **크기 계산 단계로 전달**.


### 3.2 Depth + 카메라 내파라미터를 이용한 3D 크기 추정

#### 3.2.1 Depth 캐시 구조

- Depth 콜백에서 가장 최신 depth 이미지를 저장:

  ```python
  self.latest_depth = depth  # 2D np.ndarray (float32, meters)
  self.latest_depth_stamp = msg.header.stamp
  self.latest_depth_frame_id = msg.header.frame_id
  ```

- Depth 인코딩 처리:
  - 우선 `32FC1` 시도 (단위: meter)
  - 실패 시 `16UC1` → float32 로 변환 후 `/1000.0` (mm → m)


#### 3.2.2 bbox 내부 샘플링 & 3D 포인트 계산

함수: `estimate_3d_size_and_center(bbox, depth, intrinsics, sample_grid=5)`

1. bbox = (x1, y1, x2, y2) 픽셀 좌표를 depth 이미지 범위 내로 클램핑
2. `sample_grid` (기본 5) 만큼 x, y 방향 등분:

   ```python
   xs = np.linspace(x1, x2, num=sample_grid, dtype=np.int32)
   ys = np.linspace(y1, y2, num=sample_grid, dtype=np.int32)
   ```

3. 각 (u, v) 에 대해:
   - depth 값 z = depth[v, u] (단위 m)
   - z <= 0 또는 NaN/inf 인 경우 스킵
   - 카메라 좌표계로 변환:

     \[
     X = (u - c_x) \cdot \frac{z}{f_x}, \quad
     Y = (v - c_y) \cdot \frac{z}{f_y}, \quad
     Z = z
     \]

4. 유효한 3D 포인트가 3개 미만이면 추정 불가 → `None` 반환


#### 3.2.3 3D Bounding Box 와 중심 계산

유효 3D 포인트 집합 `pts` 에 대해:

```python
min_xyz = pts.min(axis=0)  # (min_x, min_y, min_z)
max_xyz = pts.max(axis=0)  # (max_x, max_y, max_z)

dims = max_xyz - min_xyz   # (dx, dy, dz)
center = (min_xyz + max_xyz) / 2.0  # (cx, cy, cz)
```

- `dims` = 카메라 좌표계 기준 3D 크기
- `center` = 카메라 좌표계 기준 박스 중심


### 3.3 목표 박스 크기와의 비교 (길이 오차 적용)

#### 3.3.1 목표 크기 상수

- 우체국 박스 4호 실제 크기 (단위 m):

  ```python
  TARGET_DIMENSIONS_M = (0.41, 0.31, 0.28)
  ```

- 길이 허용 오차 기본값 (단위 m):

  ```python
  DEFAULT_LENGTH_TOLERANCE_M = 0.05  # 5cm
  ```

  이는 ROS2 파라미터 `length_tolerance_m` 로 변경 가능.

#### 3.3.2 축 순서 무시 후 비교

함수: `is_size_match(measured_dims, target_dims, tolerance_m)`

1. measured_dims = (dx, dy, dz)  
   target_dims = (L, W, H)

2. 둘 다 정렬:

   ```python
   m_sorted = sorted(measured_dims)
   t_sorted = sorted(target_dims)
   ```

3. 각 축 차이를 비교:

   ```python
   for m, t in zip(m_sorted, t_sorted):
       if abs(m - t) > tolerance_m:
           return False
   return True
   ```

- 이렇게 하면 **회전/축 배치에 상관없이** 단순히 세 축 길이가 target과 비슷한지 확인 가능
- 예: 측면을 보고 있든, 약간 기울어져 있든, 3D bounding box의 세 축 길이 조합이 (0.41, 0.31, 0.28)m 와 근접하면 통과


### 3.4 최종 박스 후보 채택 & Pose 출력

- 위 조건까지 모두 만족(`is_size_match(...) == True`)하면:

  ```python
  pose_msg = PoseStamped()
  pose_msg.header.stamp = msg.header.stamp
  pose_msg.header.frame_id = msg.header.frame_id  # RGB frame

  pose_msg.pose.position.x = center_cam[0]
  pose_msg.pose.position.y = center_cam[1]
  pose_msg.pose.position.z = center_cam[2]

  pose_msg.pose.orientation.x = 0.0
  pose_msg.pose.orientation.y = 0.0
  pose_msg.pose.orientation.z = 0.0
  pose_msg.pose.orientation.w = 1.0

  self.pose_pub.publish(pose_msg)
  ```

- 로그:

  - 성공 (크기 매칭된 박스):

    ```text
    Detected box (size matched): class='box', conf=..., dims(m)=(dx, dy, dz), center_cam(m)=(cx, cy, cz)
    ```

  - 실패 (크기 불일치):

    ```text
    Rejected box (size mismatch): class='box', conf=..., dims(m)=(dx, dy, dz)
    ```


---

## 4. 실행 방법

### 4.1 단독 실행 (python 직접 실행)

ROS2 Jazzy 워크스페이스를 source 한 후:

```bash
# ROS2 환경 로드
source /opt/ros/jazzy/setup.bash
# 또는 사용 중인 ros2_ws/install/setup.bash

cd /Users/jioo0224/Desktop/KMU/OSPrac

python box_detector.py \
  --ros-args \
    -p model_path:=/absolute/path/to/your_box_yolov5n.pt \
    -p rgb_topic:=/camera/color/image_raw \
    -p depth_topic:=/camera/aligned_depth_to_color/image_raw \
    -p fx:=615.0 \
    -p fy:=615.0 \
    -p cx:=320.0 \
    -p cy:=240.0 \
    -p conf_threshold:=0.4 \
    -p length_tolerance_m:=0.05
```

- `model_path` 에는 **이미 학습된 YOLOv5n `box` 모델의 로컬 `.pt` 경로**를 지정
- 카메라 intrinsic 값은 실제 장비에 맞게 수정해야 한다.


### 4.2 ROS 패키지에 통합 (예시 개념)

1. Python 패키지 디렉터리에 `box_detector.py` 를 넣고 `setup.py` 의 `entry_points`에 등록:

   ```python
   entry_points={
       "console_scripts": [
           "box_detector = your_pkg_name.box_detector:main",
       ],
   }
   ```

2. 빌드 후:

   ```bash
   colcon build --packages-select your_pkg_name
   source install/setup.bash
   ```

3. 실행:

   ```bash
   ros2 run your_pkg_name box_detector \
     --ros-args \
       -p model_path:=/absolute/path/to/your_box_yolov5n.pt \
       -p length_tolerance_m:=0.05
   ```


---

## 5. SLAM/탐색과의 연계 아이디어

1. `/box_detector/box_pose` 를 구독하는 별도 노드에서:
   - camera frame → map frame TF 변환
   - SLAM 맵 상에 4호 박스 위치를 마킹 (예: Marker, Grid, Costmap layer 등)

2. Nav2와 연계:
   - `/box_detector/box_pose` 기반으로 “박스 근처로 접근 목표 포즈” 생성
   - 박스 주변에서 추가 촬영/검증을 수행해 데이터 수집 품질 향상

3. 탐색 전략:
   - 초기에는 Nav2 waypoint/Frontier 기반으로 맵 전체를 순회
   - 순회 중에 `/box_detector/box_pose` 가 여러 번 쌓이는 위치를 “후보 위치”로 간주
   - 후보 위치 근처를 재방문해 더 다양한 각도/거리에서 촬영


---

## 6. 파라미터 튜닝 팁

- `conf_threshold`:
  - 값이 낮으면 검출 수는 많아지지만, 오탐도 증가
  - 0.4~0.6 사이에서 조정하며 실험 권장

- `length_tolerance_m`:
  - 기본값 0.05 (5cm)
  - 더 엄격하게: 0.03 (3cm) → 진짜 4호 상자만 남기고 싶을 때
  - 더 완화: 0.10 (10cm) → 비슷한 크기의 상자까지 모두 포함하고 싶을 때

- `sample_grid` (코드 내부 상수):
  - 5x5 → 속도/정확도 균형
  - depth 노이즈가 심하면 샘플 수를 늘리고, median/평균 등 전처리를 추가하는 것도 고려 가능


---

이 문서(`box_detector.md`)는 `box_detector.py` 노드의 **사용법과 내부 알고리즘(YOLO + Depth 기반 크기 필터링)**을 정리한 것으로,  
향후 Nav2/SLAM 연동, 멀티 박스 탐색 전략 설계 시 참고 문서로 사용할 수 있다.