다음 내용은 그대로 슬라이드 목차/해설로 옮길 수 있도록, 코드 역할·실험 절차·문제점·수치 결과 중심으로 정리했다.

---

## 1. [`new_func/detect_func.py`](new_func/detect_func.py)의 역할

### 1-1. 목적

- YOLOv5가 출력한 바운딩 박스(x1, y1, x2, y2)와 **물체까지의 거리(distance, LiDAR/깊이센서)**, 그리고 **카메라 내파라미터(fx, fy)** 를 이용해서  
  → **실제 물리 크기(미터)** 를 추정하고,  
  → 그 크기가 **우체국 박스 4호**인지 여부를 필터링하는 유틸리티 함수 모음.

### 1-2. 핵심 상수

- 우체국 박스 4호 실제 크기 (단위: m)
  - `TARGET_BOX_WIDTH_M = 0.41`, `TARGET_BOX_LENGTH_M = 0.31`, `TARGET_BOX_HEIGHT_M = 0.28`  
    → [`detect_func`](new_func/detect_func.py:8)의 물리적 기준값.
- 허용 오차
  - `DEFAULT_TOLERANCE_M = 0.10` (10cm) [`detect_func`](new_func/detect_func.py:14)
  - 실제 박스 크기 주변 ±10cm 안에 들어오면 “같은 물체”로 간주.

### 1-3. 크기 추정 수식

핵심 함수는 [`detect_func.estimate_object_size_from_bbox()`](new_func/detect_func.py:40).

- 입력
  - `bbox`: YOLOv5 xyxy 형식 `[x1, y1, x2, y2, (conf), (cls)]`
  - `distance_m`: 카메라–물체 거리 (m, LiDAR/깊이센서에서 측정)
  - `fx`, `fy`: 카메라 focal length (pixel; CameraInfo에서 가져옴)
- 내부 수식 (pinhole 카메라 모델 가정):

\[
\text{pixel\_width} = x_2 - x_1,\quad
\text{pixel\_height} = y_2 - y_1
\]

\[
\text{width\_m} = \frac{\text{pixel\_width} \cdot \text{distance\_m}}{fx}
\]

\[
\text{height\_m} = \frac{\text{pixel\_height} \cdot \text{distance\_m}}{fy}
\]

- 출력: `(width_m, height_m)` : 이미지 상 bbox가 의미하는 **실제 물체의 가로/세로 길이(m)**

### 1-4. 크기 기반 타겟 판별 함수

1. [`detect_func.detect_box()`](new_func/detect_func.py:75)

   - 전체 footprint (가로·세로) 기준으로 판별.
   - 절차
     1. `estimate_object_size_from_bbox()`로 `(est_w, est_h)` 추정
     2. `(est_w, est_h)`와 타겟 실제 크기 `(TARGET_BOX_WIDTH_M, TARGET_BOX_HEIGHT_M)`를  
        각각 (작은 축, 큰 축) 순서로 정렬 후 비교
     3. 두 축 모두에서 `|추정길이 - 실제길이| ≤ tolerance_m` 이면 `True` (타겟으로 간주)

   - 의도: 박스가 눕거나 세워져도, “긴 변/짧은 변” 기준으로 보면 실제 크기와 맞으면 필터 통과.

2. [`detect_func.detect_box_by_height()`](new_func/detect_func.py:108)

   - **세로 픽셀 길이(높이)** 하나만 보고 판별하는 보조 함수.
   - 박스를 측면에서 보거나, 일부만 보이는 경우에도 세로 길이가 실제 높이(0.28m)와 맞으면 True.
   - 높이만 쓰기 때문에 상황에 따라 footprint보다 튜닝/해석이 단순.

---

## 1.1. 디버깅용: [`test_yolov5_detect_box.py`](test_yolov5_detect_box.py)의 역할

### 목적

- [`detect_func`](new_func/detect_func.py)의 크기 추정/필터 로직이 잘 동작하는지 **오프라인 단일 이미지/테스트케이스로 검증**하는 스크립트.
- 실제 ROS/로봇 없이, 이미지 + 메타데이터(거리)를 기반으로 YOLO + size filter를 반복 실험.

### 주요 기능 정리

1. YOLOv5 싱글톤 로딩
   - [`yolov5_singleton.load_yolov5_model()`](new_func/yolov5_singleton.py:1)를 사용해 모델 1회 로딩 후 재사용.

2. 테스트 이미지 전처리
   - [`crop_center_square_and_resize_to_250()`](test_yolov5_detect_box.py:26)
     - 원본 이미지에서 중앙 정사각형을 잘라 250×250으로 리사이즈.
     - 실제 OAK-D `/oakd/rgb/preview/image_raw` 250×250 프리뷰와 비슷한 조건을 재현.

3. 거리 메타데이터 사용
   - `testcase/test_*.png` 와 동일 이름의 `test_*.json` 파일에서 `distance` 값을 읽어와  
     → 각 이미지에 대해 **실제 카메라–박스 거리(m)** 를 설정.

4. 크기 추정 및 필터 동작 검증
   - 각 detection에 대해:
     - [`estimate_object_size_from_bbox()`](new_func/detect_func.py:40)로 물리 크기 추정.
     - [`detect_func.detect_box()`](new_func/detect_func.py:75) 결과 (footprint 기준)
     - [`detect_func.detect_box_by_height()`](new_func/detect_func.py:108) 결과 (높이 기준)
   - 터미널에
     - 클래스, 신뢰도, bbox 값, 추정 크기(m), 필터 통과 여부를 로그 출력.
   - 시각화
     - footprint 기준 결과를 색깔(타겟: 초록, 비타겟: 파랑)으로 표시해 `result/*_footprint.png` 저장.
     - height 기준 결과를 (타겟: 주황, 비타겟: 회색)으로 표시해 `result/*_height.png` 저장.

### 발표 포인트

- 이 스크립트는 **실제 로봇 없이**도:
  - 카메라 화각, 리사이즈 방식, 거리 메타정보를 고정하고
  - 크기 추정 수식과 허용오차가 적절한지 **눈으로 직접 검증**하기 위한 도구.
- 이 과정을 통해 `tolerance_m`, `fx/fy` 등의 값을 조정하고,  
  → 이후 ROS 노드/실로봇 실험으로 넘어가는 “사전 디버깅 단계”로 사용.

---

## 1.2. LiDAR 적용: [`simple_lidar_yolo.py`](simple_lidar_yolo.py)의 역할과 문제점

### 1.2-a. 노드 목적

- ROS2 환경에서 동작하는 [`SimpleLidarYoloNode`](simple_lidar_yolo.py:39):
  - 사람이 **직접 로봇을 조종**하면서,
  - 일정 주기(10초마다)마다
    1. 최신 RGB 프레임에 YOLOv5 추론 실행
    2. 타겟 클래스(예: `"box"`)에 대한 **최고 신뢰도 bounding box 픽셀 크기** 계산
    3. LiDAR `/scan` 토픽에서 전방 ±1° 내 최소 거리값을 가져와,  
       → 해당 시점 박스까지의 거리로 가정
    4. 이 두 값을 함께 로그로 남김.

- 즉,
  - 입력: RGB 이미지 + LiDAR 스캔
  - 출력: 로그 한 줄
    - `bbox_px=(pixel_width, pixel_height)`, `lidar_distance=... m`
  - 이 로그들을 모아, **“픽셀 크기 ↔ 실제 거리” 관계를 데이터로 수집**하는 용도.

### 1.2-b. 내부 동작 요약

1. 파라미터
   - `model_path`, `rgb_topic`, `scan_topic`, `img_size`, `device`, `yolo_conf_threshold`, `target_class` 등 [`SimpleLidarYoloNode.__init__`](simple_lidar_yolo.py:42).
2. YOLOv5 싱글톤 로딩
   - [`load_yolov5_model()`](new_func/yolov5_singleton.py:1) 사용, 클래스 이름 캐시.
3. 콜백
   - `rgb_callback`: RGB 이미지 수신 후 `self.latest_rgb`에 보관.
   - `scan_callback`: LiDAR 스캔 수신 후 `self.latest_scan`에 보관.
4. 타이머
   - 0.5초마다 `timer_callback()` 실행.
   - 단, `last_eval_time` 기준으로 **10초에 한 번만** YOLO+LiDAR 계산.

5. LiDAR 거리 계산
   - [`_get_front_min_distance()`](simple_lidar_yolo.py:231)
     - 전체 `scan.ranges`에서 유효한 값만 골라,
     - 각 샘플의 각도 배열을 만든 뒤
     - `|angle| ≤ 1°` 범위의 샘플만 남김
     - 그 중 최소값을 전방 거리로 사용.

6. YOLO + 타겟 박스 선택
   - `results.xyxy[0]`에서
     - `class_name == target_class`
     - `conf ≥ yolo_conf_threshold`
     - 위 조건 만족하는 것 중 **가장 conf가 큰 detection** 1개 선택.

7. 로그 출력
   - 선택된 detection의 `(x1,y1,x2,y2)`로부터
     - `pixel_width`, `pixel_height` 계산
   - 최종 로그:
     - `"[YOLO+LiDAR] class=... conf=... bbox_px=(w_px, h_px) lidar_distance=... m"` [`simple_lidar_yolo.py`](simple_lidar_yolo.py:221)

---

### 1.2.1. 실제 로봇(TurtleBot4)이 전진하지 않은 문제

- 설계 상 [`SimpleLidarYoloNode`](simple_lidar_yolo.py:39)는 **/cmd_vel을 publish하지 않는 “로깅 전용 노드”**다.
  - 실제 전진 제어는 별도의 노드(예: [`lidar_forward.py`](lidar_forward.py:1) 계열)에서 수행.
- 실험 과정에서 나타난 문제:
  1. LiDAR 전방 거리 계산이 **매우 보수적**(±1°만 사용)이고,
  2. 노이즈·반사·센서 데드존 등으로 인해 유효한 값이 부족하거나,
  3. 전방에 항상 벽/기타 구조물이 있어 “충돌 위험”으로 판단된 경우
- 그 결과:
  - 안전 조건(예: “전방 거리가 일정 m 이상이어야 전진”)을 만족하지 못해  
    → 실제 전진 명령이 거의 나가지 않거나, 즉시 멈추는 현상 발생.
  - 이 경험 때문에,
    - “LiDAR 기반 자동 전진 실험”은 신뢰도가 낮다고 판단,
    - 오히려 사용자가 로봇을 수동 조종하고, 센서 값만 로깅하는 전략으로 전환.

발표용 요약 포인트:
- “LiDAR만으로 안전 조건을 걸어 자동 전진시키려 했으나, 전방 거리 측정의 불안정·과도한 보수성 때문에 로봇이 잘 움직이지 않는 문제가 있었다. 이후 **수동 주행 + 로그 수집 방식**으로 전략을 바꾸었다.”

---

### 1.2.2. LiDAR로 측정한 object와의 거리가 이상했던 문제

실험 중 관찰된 대표적인 문제 원인:

1. **시야/각도 불일치**
   - 카메라가 보고 있는 박스가 반드시 LiDAR 전방 ±1° 안에 들어오지 않는다.
   - 이 경우 LiDAR는 **박스가 아닌 뒷배경(벽, 다른 물체)** 거리를 반환할 수 있음.
   - 결과적으로 “YOLO는 박스를 보고 있는데, LiDAR는 더 멀리 있는 벽까지의 거리”를 기록하는 케이스가 빈번.

2. **센서 노이즈 및 Inf 값**
   - `scan.ranges`에는 `inf`, 0, 이상 값이 많이 포함됨.
   - `_get_front_min_distance()`는 이 중 **유효한 것들 중 최소값**만 사용하므로,
     - 국소적인 노이즈에 민감,
     - 바닥/근거리 장애물 하나만 있어도 거리값이 급격히 줄어드는 문제.

3. **센서 위치 차이(카메라 vs LiDAR)**
   - 카메라와 LiDAR는 로봇 상에서 물리적으로 다른 위치·높이에 장착.
   - 실제 박스 중심까지의 거리는 두 센서 기준 좌표계에서 다르게 측정될 수 있는데,
   - 본 실험에서는 이를 정교하게 보정하지 않고 “LiDAR 전방 최소값 ≈ 박스까지의 거리”로 단순 가정.

이러한 이유로:

- 같은 픽셀 높이의 박스라도 **LiDAR 거리 값이 들쑥날쑥**하게 측정.
- 이후 “픽셀 크기 ↔ 거리” 회귀를 했을 때, 데이터가 완전히 깨지진 않았지만 **상당한 분산**이 존재.

---

### 1.3. LiDAR–픽셀 관계의 1차 선형보간 적합도: 0.95

- [`simple_lidar_yolo.py`](simple_lidar_yolo.py:221) 로그로부터
  - x축: LiDAR 거리 (m)
  - y축: bbox 픽셀 크기 (예: `pixel_height`)
  - 또는 그 역(x: 픽셀, y: 거리) 형태로 데이터를 수집.
- 이 데이터에 대해 **1차 함수(선형)** 형태로 보간/회귀를 수행했을 때,
  - 결정계수(R²) 기준 약 **0.95 수준의 적합도**를 얻음.
- 해석:
  - 전반적인 트렌드(“멀수록 작게, 가까울수록 크게”)는 꽤 잘 맞는다.
  - 다만 앞에서 언급한 LiDAR 거리 이상치·노이즈 때문에 **완벽한 직선 관계는 아니며**, 일부 샘플이 크게 벗어나는 구간 존재.
- 발표 시 포인트:
  - “수동 주행을 통해 수집한 LiDAR–픽셀 데이터는 **일차 함수로 어느 정도 잘 설명(R²≈0.95)**되었고, 이를 기반으로 이후 거리 추정 모델을 시도했다.”

---

## 2. simple_lidar_yolo 기반 수동 주행 결과 재이용 절차

여기서는 “simple_lidar_yolo로 모은 로그 데이터를 어떻게 다시 활용했는지”의 흐름을 슬라이드용으로 정리한다.

### 2-1. [`new_func/detect_func.py`](new_func/detect_func.py)의 역할 재강조

- [`detect_func`](new_func/detect_func.py)의 핵심은:

\[
\text{width\_m} = \frac{\text{pixel\_width} \cdot \text{distance\_m}}{fx}
\]

\[
\text{height\_m} = \frac{\text{pixel\_height} \cdot \text{distance\_m}}{fy}
\]

- 즉,
  - **거리(distance_m)가 주어졌을 때**,  
    YOLO bbox 픽셀 크기와 카메라 내파라미터를 이용해 실제 크기(m)를 계산.
  - 이를 사용해 “이 bbox가 실제 우체국 박스 4호 크기와 얼마나 비슷한가?”를 판별.

→ **정방향(Forward) 기능**:  
“거리 → (픽셀 크기, 카메라 Intrinsic) → 실제 크기 → 박스 여부 판별”

### 2-2. 이 함수를 “역으로” 사용하려는 시도

실험 목표:

- LiDAR 없이도, 즉 **카메라 + YOLO만으로 거리 추정을 하고 싶다.**
- 이를 위해 이미 수집해 둔 “LiDAR 거리 ↔ 픽셀 크기” 데이터를 이용하여,  
  `detect_func`의 수식을 **역방향으로 사용하는 절차**를 구성.

개념적 역산:

- 정방향 수식:  
  \[
  \text{height\_m} = \frac{\text{pixel\_height} \cdot \text{distance\_m}}{fy}
  \]
- 목표: 거리 추정 (distance_m)을 구하고 싶다면,

\[
\text{distance\_m} = \frac{\text{height\_m} \cdot fy}{\text{pixel\_height}}
\]

- 여기서 `height_m`는 **우체국 박스 4호 실제 높이**(0.28m)로 고정할 수 있다.
- LiDAR로 거리(distance_m)를 알고 있는 상태에서 여러 샘플을 모은 뒤,
  - “실제 distance_m”과
  - “픽셀 높이(pixel_height)”의 관계를 **거꾸로 회귀**하여,
  - 나중에는 LiDAR 없이도
    - YOLO bbox의 세로 픽셀 길이만 보고
    - distance_m을 추정하는 모델을 만들려는 시도.

정리하면:

1. simple_lidar_yolo 로그에서 `(pixel_height, lidar_distance)` 샘플 수집
2. 이들로 1차 회귀/보간 모델 `distance_est = f(pixel_height)` 학습
3. 이후 LiDAR가 없을 때는,
   - YOLO bbox의 `pixel_height`만 가지고
   - `distance_est = f(pixel_height)`로 대략적인 거리 추정
4. 그 결과를 다시 [`detect_func.detect_box()`](new_func/detect_func.py:75) 등에 넣어  
   “크기/거리 동시에 만족하는 타겟 박스 필터”를 구현하려는 흐름.

### 2-3. 역방향 모델의 선형 적합도: 약 0.88, 사용성 한계

- 위와 같이 역방향(픽셀→거리) 관계를 1차 함수로 피팅했을 때,
  - 결정계수 R²가 **약 0.88** 수준으로 나타남.
- 이는:
  - 전반적인 경향은 맞지만
  - 로봇 자율주행에서 신뢰할 만큼 **정확도가 충분히 높지 않다**는 의미.
- 원인(해석):
  - 앞서 언급한 LiDAR 거리 이상치, 센서 정렬 문제, 노이즈로 인해
    - “distance → pixel” 방향은 R²≈0.95로 꽤 양호했지만,
    - 이를 다시 “pixel → distance”로 뒤집으면 **오차가 더 커짐**.
  - 특히 실제 응용에서는 “몇십 cm의 오차”도 경로 계획/충돌 회피에 치명적일 수 있음.

발표용 결론:

- “수동 주행 + LiDAR–픽셀 로그로부터, 카메라 단독 거리 추정 모델을 시도했지만,
  역방향 선형 적합도는 **R²≈0.88 수준**에 머물렀고,  
  자율주행 의사결정에 쓰기에는 **신뢰도가 부족**하다고 판단하여,  
  **생산 환경에는 적용하지 않는 것으로 결정**했다.”

---

## 3. [`box_size_helper.py`](box_size_helper.py)를 이용한 YOLO bbox → 실제 크기 추정

### 3-1. 목적

- [`box_size_helper.BoxSizeHelper`](box_size_helper.py:49)는
  - YOLO 모델 로딩/추론과는 완전히 분리된
  - **순수한 “픽셀 크기 → 실제 크기” 변환 전용 헬퍼 클래스**.
- 역할:
  1. ROS2 [`CameraInfo`](sensor_msgs/msg/CameraInfo.msg:1) 메시지에서 fx, fy를 1번만 읽어 캐싱.
  2. YOLO가 반환한 bbox 리스트와 거리(m)를 입력으로 받아,
     - 픽셀 단위 크기 (x_px, y_px)
     - 미터 단위 크기 (x_m, y_m)
     를 계산해줌.
  3. 필요 시 여러 bbox 중 “세로 픽셀 크기가 가장 큰 것”만 뽑아 변환.

### 3-2. 주요 메서드

1. 카메라 내부 파라미터 설정
   - [`BoxSizeHelper.set_camera_info()`](box_size_helper.py:74)
     - fx, fy를 직접 숫자로 설정.
   - [`BoxSizeHelper.set_camera_info_from_msg()`](box_size_helper.py:79)
     - `CameraInfo.K[0]`, `CameraInfo.K[4]`에서 fx, fy 자동 추출.

2. 픽셀 크기 계산
   - [`BoxSizeHelper.get_pixel_size()`](box_size_helper.py:109)
     - bbox `[x1, y1, x2, y2, ...]` → `(pixel_width, pixel_height)`.

3. 실제 크기(m) 추정
   - [`BoxSizeHelper.estimate_size_from_bbox()`](box_size_helper.py:121)
     - 수식은 [`detect_func.estimate_object_size_from_bbox()`](new_func/detect_func.py:40)와 동일:
       - `(width_m, height_m)` = `(pixel_width * distance_m / fx, pixel_height * distance_m / fy)`.

4. 편의 변환 메서드
   - [`BoxSizeHelper.convert_bbox()`](box_size_helper.py:168)
     - 하나의 bbox에 대해:
       ```json
       {
         "distance": distance_m,
         "x_px": pixel_width,
         "y_px": pixel_height,
         "x_m": width_m,
         "y_m": height_m
       }
       ```
       형태의 dict 반환.
   - [`BoxSizeHelper.convert_boxes()`](box_size_helper.py:206)
     - 여러 bbox 중에서 “픽셀 높이(y_px)가 가장 큰 것” 하나만 골라 위와 같은 dict로 변환.
     - 실제 응용에서 “카메라에 가장 크게 보이는(가장 가까운) 물체”를 고르는 용도로 사용 가능.

### 3-3. 발표용 메시지

- [`new_func/detect_func.py`](new_func/detect_func.py)와 [`box_size_helper.py`](box_size_helper.py)는 같은 수학적 기반 위에 있다.
  - 둘 다 **pinhole 카메라 모델**을 사용해
    - “픽셀 크기 ↔ 실제 물리 크기(미터)”를 연결.
- 차이점:
  - `detect_func`는 **특정 타겟 박스(우체국 박스 4호) 필터링**에 초점을 둔 함수 모음.
  - `BoxSizeHelper`는
    - 카메라 정보 캐싱,
    - bbox 여러 개 중 선택,
    - 픽셀/미터 변환을 묶어서 제공하는 **일반화된 유틸리티 클래스**.
- 이를 통해:
  - YOLOv5, YOLOv8 등 어떤 모델이든,
  - ROS2 노드 어디에서든,
  - 동일한 방식으로 **바운딩 박스 픽셀 크기를 실제 미터 단위 객체 크기**로 추정할 수 있는 기반을 제공하게 된다.

---

위 내용을 기반으로, 발표 자료에서는 다음과 같이 흐름을 구성할 수 있다:

1. `detect_func`로 **크기 기반 타겟 필터 개념** 소개
2. `test_yolov5_detect_box.py`로 **오프라인 검증/튜닝 절차** 설명
3. `simple_lidar_yolo.py`로 실제 **LiDAR + YOLO 동시 로깅** 및
   - 로봇 전진 실패, 거리 이상치 문제,
   - 그럼에도 불구하고 얻어진 **R²≈0.95** 수준의 1차 선형 관계 소개
4. 그 데이터를 역으로 활용해 **카메라 단독 거리 추정(R²≈0.88)**을 시도했지만
   - 실사용에는 부족하다고 판단한 결론
5. 마지막으로 `box_size_helper.py`를 통해
   - 향후 어떤 YOLO/카메라 조합에도 재사용 가능한
   - **“bbox → 실제 크기(m)” 변환 인프라**를 정리했다는 점을 강조.

이 구조로 정리하면, PowerPoint의 AI 기능이 슬라이드(배경/문단/도식)를 자동 구성할 때도 “문제 정의 → 실험 → 분석 → 한계 → 일반화 유틸” 흐름이 잘 드러나도록 만들 수 있다.