"""Utility functions for YOLOv5 bounding-box based size checking."""

from typing import Sequence, Tuple, Union

Number = Union[int, float]
BBoxLike = Sequence[Number]

# 물리적인 목표 박스 크기 (우체국 박스 4호, 단위: 미터)
TARGET_BOX_WIDTH_M: float = 0.41
TARGET_BOX_HEIGHT_M: float = 0.31

# 허용 오차 (단위: 미터) - 기본 0.10m = 10cm
DEFAULT_TOLERANCE_M: float = 0.10

# 카메라 내파라미터 (예: OAK-D, /oakd/rgb/preview/image_raw 250x250 기준 예시 값)
# 실제 값은 각 장비의 CameraInfo(/oakd/rgb/preview/camera_info 등)에서 fx, fy 를 읽어
# estimate_object_size_from_bbox() 호출 시 인자로 넘겨주는 것이 가장 정확하다.
# 여기서는 OAK-D 720p (~870px)에서 250x250 프리뷰로 리사이즈된 경우를 가정해
# 대략 축소 비율을 반영한 예시값(~300px)을 사용한다.
FX: float = 300.0
FY: float = 300.0


def _parse_bbox_xyxy(bbox: BBoxLike) -> Tuple[float, float, float, float]:
    """
    YOLOv5 xyxy 형식의 bounding box 로부터 (x1, y1, x2, y2)를 float 로 추출한다.

    YOLOv5 PyTorch 결과는 보통 [x1, y1, x2, y2, conf, cls] 형태의 1차원 Tensor 이거나,
    동일한 순서의 list/tuple 이라고 가정한다.
    """
    if len(bbox) < 4:
        raise ValueError("bbox must have at least 4 elements: [x1, y1, x2, y2, ...]")

    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return float(x1), float(y1), float(x2), float(y2)


def estimate_object_size_from_bbox(
    bbox: BBoxLike,
    distance_m: float,
    fx: float,
    fy: float,
) -> Tuple[float, float]:
    """
    주어진 거리(distance_m)에서 bbox 픽셀 크기를 이용해
    물체의 대략적인 실제 가로/세로 길이(미터)를 추정한다.

    단순한 pinhole 카메라 모델을 사용:
        width_m  = (bbox_pixel_width  * distance_m) / fx
        height_m = (bbox_pixel_height * distance_m) / fy

    Args:
        bbox: YOLOv5 xyxy 형식 bbox (길이 >= 4 인 시퀀스)
        distance_m: 카메라와 물체 사이의 거리 (미터)
        fx, fy: 카메라 focal length (픽셀 단위, 보통 CameraInfo.K[0], CameraInfo.K[4])

    Returns:
        (width_m, height_m): 물체의 가로/세로 길이(미터)
    """
    x1, y1, x2, y2 = _parse_bbox_xyxy(bbox)
    pixel_width = max(0.0, float(x2 - x1))
    pixel_height = max(0.0, float(y2 - y1))

    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive.")

    width_m = (pixel_width * distance_m) / fx
    height_m = (pixel_height * distance_m) / fy

    return width_m, height_m


def detect_box(
    bbox: BBoxLike,
    distance: float,
    fx: float = FX,
    fy: float = FY,
    target_size_m: Tuple[float, float] = (TARGET_BOX_WIDTH_M, TARGET_BOX_HEIGHT_M),
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> bool:
    """
    YOLOv5 bounding box 와 물체까지의 거리(distance)를 이용해
    "우체국 박스 4호" 와 유사한 크기인지 여부를 판별한다.

    - bbox 는 YOLOv5 xyxy 형식 [x1, y1, x2, y2, (conf), (cls)] 를 가정
    - distance 는 카메라-물체 거리(미터)
    - fx, fy 는 CameraInfo 에서 읽어온 focal length (픽셀 단위, CameraInfo topic -> K[0], K[4])
    - target_size_m 은 (width, height) [m]
    - tolerance_m 는 각 축별 허용 오차 [m] (기본 0.10m = 10cm)

    크기 비교 시, 물체가 눕거나 세워져 있을 수 있으므로
    (width, height) 순서를 무시하고 작은 축/큰 축으로 정렬해 비교한다.
    """
    est_w, est_h = estimate_object_size_from_bbox(bbox, distance_m=distance, fx=fx, fy=fy)

    # 축 순서를 무시하고 비교 (작은 쪽 / 큰 쪽)
    est_dims = sorted((est_w, est_h))
    tgt_dims = sorted(target_size_m)

    diff0 = abs(est_dims[0] - tgt_dims[0])
    diff1 = abs(est_dims[1] - tgt_dims[1])

    return (diff0 <= tolerance_m) and (diff1 <= tolerance_m)