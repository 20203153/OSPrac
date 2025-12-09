# YOLOv5 기반 박스 크기 필터 및 통합 테스트 사용법

이 문서는 이 저장소에서 제공하는 박스 크기 필터 유틸과 통합 테스트 스크립트의 사용법을 정리한다.

관련 파일:
- [`new_func/detect_func.py`](new_func/detect_func.py:1) — 크기 추정 및 박스 판별 함수
- [`new_func/yolov5_singleton.py`](new_func/yolov5_singleton.py:1) — YOLOv5 Singleton 로더/러너
- [`test_yolov5_detect_box.py`](test_yolov5_detect_box.py:1) — 단일 이미지/테스트케이스용 통합 테스트
- `testcase/test_*.png`, `testcase/test_*.json` — 테스트 입력 이미지 및 거리 메타데이터
- `result/*.png` — 테스트 결과 시각화 이미지 출력 위치

## 1. detect_func 모듈

파일: [`detect_func.py`](new_func/detect_func.py:1)

### 1.1 상수

- 목표 박스(우체국 박스 4호) 실제 크기 [미터]:
  - 가로(width): [`TARGET_BOX_WIDTH_M`](new_func/detect_func.py:9) = 0.41
  - 세로(length): [`TARGET_BOX_LENGTH_M`](new_func/detect_func.py:10) = 0.31
  - 높이(height): [`TARGET_BOX_HEIGHT_M`](new_func/detect_func.py:11) = 0.28
- 허용 오차: [`DEFAULT_TOLERANCE_M`](new_func/detect_func.py:13) = 0.10 (±10cm)

### 1.2 크기 추정 함수

- [`estimate_object_size_from_bbox()`](new_func/detect_func.py:38)

```python
from new_func.detect_func import estimate_object_size_from_bbox

width_m, height_m = estimate_object_size_from_bbox(
    bbox,            # YOLOv5 xyxy: [x1, y1, x2, y2, (conf), (cls)]
    distance_m,      # 카메라–물체 거리 [m]
    fx, fy,          # CameraInfo.K[0], CameraInfo.K[4]
)
```

- 반환값:
  - `width_m`: 이미지 x 방향(가로) 실제 길이 추정
  - `height_m`: 이미지 y 방향(세로) 실제 길이 추정

### 1.3 바닥 크기 기반 판별 함수

- [`detect_box()`](new_func/detect_func.py:73)

```python
from new_func.detect_func import detect_box

is_target = detect_box(
    bbox=bbox,
    distance=distance_m,
    fx=fx,
    fy=fy,
    # 기본 target_size_m=(0.41, 0.31), tolerance_m=0.10 사용
)
```

- 역할:
  - YOLOv5 바운딩박스가 **바닥 footprint (0.41 × 0.31 m)** 와 ±10cm 이내인지 판별.
  - 박스가 정면/윗면 등으로 찍혀서 가로×세로가 잘 보이는 경우에 적합.

### 1.4 높이 기반 판별 함수

- [`detect_box_by_height()`](new_func/detect_func.py:105)

```python
from new_func.detect_func import detect_box_by_height

is_target_h = detect_box_by_height(
    bbox=bbox,
    distance=distance_m,
    fx=fx,
    fy=fy,
    # 기본 target_height_m=0.28, tolerance_m=0.10
)
```

- 역할:
  - YOLO 바운딩박스의 **세로 길이(이미지 y 방향)**만 사용해 박스 높이(0.28m)와 ±10cm 이내인지 판별.
  - 박스를 **측면(세로×높이)** 으로 봐서 바닥이 잘 안 보이는 경우에도 높이만으로 True 가능.

ROS2 노드에서 두 함수를 함께 쓰는 예:

```python
from new_func.detect_func import detect_box, detect_box_by_height

is_target_foot = detect_box(bbox, distance=distance_m, fx=fx, fy=fy)
is_target_h    = detect_box_by_height(bbox, distance=distance_m, fx=fx, fy=fy)

is_target = is_target_foot or is_target_h
```

## 2. YOLOv5 Singleton 모듈

파일: [`yolov5_singleton.py`](new_func/yolov5_singleton.py:1)

### 2.1 모델 로드 (Singleton)

- [`load_yolov5_model()`](new_func/yolov5_singleton.py:13)

```python
from new_func.yolov5_singleton import load_yolov5_model

model = load_yolov5_model(
    model_path="./yolov5n.pt",
    device="cuda",   # 또는 "cpu", "mps"
    use_half=False,
)
```

- 최초 1회만 weight 를 로드하고, 이후 호출은 같은 인스턴스를 재사용한다.

### 2.2 추론 함수

- [`run_yolov5_inference()`](new_func/yolov5_singleton.py:63)

```python
from new_func.yolov5_singleton import run_yolov5_inference

results = run_yolov5_inference(
    bgr_image=bgr_image,   # OpenCV BGR
    img_size=640,
    conf_threshold=0.4,
)

boxes = results.xyxy[0]  # [N, 6]: x1, y1, x2, y2, conf, cls
```

## 3. 통합 테스트 스크립트: test_yolov5_detect_box.py

파일: [`test_yolov5_detect_box.py`](test_yolov5_detect_box.py:1)

### 3.1 입력 데이터 구조

- 이미지: `testcase/test_*.png`
- 메타데이터(JSON): `testcase/test_*.json`
  - 예시(`distance` 필드 사용):

```json
{
  "distance": 1.2
}
```

각 테스트케이스에 대해:
- RGB 이미지를 로드하고, 중앙 정사각형을 잘라 `250×250` 으로 리사이즈.
- 동일 basename 의 JSON 이 존재하면, `distance` 값을 읽어 해당 케이스의 거리[m]로 사용.

### 3.2 실행 방법

```bash
python test_yolov5_detect_box.py \
  --model_path ./yolov5n.pt \
  --testcase_path ./testcase \
  --fx 300.0 --fy 300.0 \
  --img_size 640 \
  --conf_threshold 0.4 \
  --distance_m 1.0 \
  --device cpu
```

- `--fx`, `--fy` 는 실제 환경에서는 `CameraInfo.K[0]`, `CameraInfo.K[4]` 를 넣는 것이 가장 정확하다.
- `--distance_m` 는 JSON 에 `distance` 가 없을 때 사용되는 기본 거리.
- `--device` 는 `cpu`, `cuda`, `mps` 를 사용할 수 있다. 각각 CPU 연산, CUDA(Nvidia GPU) 연산, MPS(Apple 가속기) 연산을 의미한다.

### 3.3 테스트 로직

각 `test_*.png` 에 대해:

1. YOLOv5 추론으로 bbox 리스트(`results.xyxy[0]`)를 얻는다.
2. 각 bbox 에 대해 [`estimate_object_size_from_bbox()`](new_func/detect_func.py:38) 로 실제 크기를 추정한다.
3. 두 가지 테스트를 수행한다.

#### Test 1: 바닥 footprint 기준 [`detect_box()`](new_func/detect_func.py:73)

- 바닥 크기(0.41×0.31m) 기준으로 타겟인지 여부를 계산.
- 결과는 `result/<basename>_footprint.png` 로 저장된다.
- 색상:
  - 초록: [`detect_box()`](new_func/detect_func.py:73) == True
  - 파랑: False

#### Test 2: 높이 기준 [`detect_box_by_height()`](new_func/detect_func.py:105)

- bbox 세로 길이만 사용해 높이(0.28m) ±10cm 범위인지 검사.
- 결과는 `result/<basename>_height.png` 로 저장된다.
- 색상:
  - 주황: [`detect_box_by_height()`](new_func/detect_func.py:105) == True
  - 회색: False

### 3.4 ROS2 노드 통합 시 참조

실제 TurtleBot4 ROS2 노드에서는:

1. 카메라 토픽으로부터 RGB 이미지와 CameraInfo (fx, fy)를 구독한다.
2. YOLOv5 추론 결과에서 bbox 를 얻는다.
3. depth/TF 등을 이용해 박스까지의 거리 `distance_m` 을 계산한다.
4. [`detect_box()`](new_func/detect_func.py:73) 와 [`detect_box_by_height()`](new_func/detect_func.py:105) 를 함께 사용해
   - 바닥 footprint 와
   - 측면 높이
   두 기준 중 하나라도 만족하면 우체국 박스 4호로 인식하도록 구성할 수 있다.

테스트 스크립트는 이 전체 파이프라인을 오프라인 이미지/JSON 세트에 대해 재현해 보는 용도로 사용하면 된다.