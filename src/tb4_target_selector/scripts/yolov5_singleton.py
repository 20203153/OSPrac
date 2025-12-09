"""Singleton-style helper for loading and running a YOLOv5 model."""

from typing import Optional, Any

import numpy as np
import cv2
import torch

_YOLO_MODEL: Optional[Any] = None
_YOLO_DEVICE: Optional[torch.device] = None


def load_yolov5_model(
    model_path: str,
    device: Optional[str] = None,
    use_half: bool = False,
) -> Any:
    """
    YOLOv5 모델을 Singleton 패턴으로 로드한다.

    - 최초 1회만 실제 weight 를 로드하고,
      이후부터는 이미 생성된 모델 인스턴스를 그대로 재사용한다.

    Args:
        model_path: YOLOv5 .pt weight 파일 경로
        device: "cpu", "cuda", "cuda:0" 등 (None 이면 자동 선택)
        use_half: True 이고 CUDA 사용 시 half precision 으로 변환

    Returns:
        로드된 YOLOv5 모델 객체
    """
    global _YOLO_MODEL, _YOLO_DEVICE

    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _YOLO_DEVICE = torch.device(device)

    # 기존 프로젝트에서 사용하던 방식과 동일하게,
    # torch.hub 를 통해 Ultralytics YOLOv5 모델을 로드한다고 가정한다.
    # (이미 로컬에 ultralytics/yolov5가 설치되어 있어야 함)
    _YOLO_MODEL = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=model_path,
        source="github",
    )

    _YOLO_MODEL.to(_YOLO_DEVICE)

    if use_half and _YOLO_DEVICE.type != "cpu":
        _YOLO_MODEL.half()

    _YOLO_MODEL.eval()

    return _YOLO_MODEL


def get_yolov5_model() -> Any:
    """
    이미 로드된 YOLOv5 모델을 반환한다.

    아직 load_yolov5_model() 이 호출되지 않았다면 RuntimeError 를 발생시킨다.
    """
    if _YOLO_MODEL is None:
        raise RuntimeError(
            "YOLOv5 model is not loaded yet. Call load_yolov5_model() first."
        )
    return _YOLO_MODEL


def run_yolov5_inference(
    bgr_image: np.ndarray,
    img_size: int = 640,
    conf_threshold: Optional[float] = None,
) -> Any:
    """
    Singleton 으로 로드된 YOLOv5 모델을 이용해 추론을 수행한다.

    Args:
        bgr_image: OpenCV BGR 이미지 (H, W, 3)
        img_size: YOLOv5 입력 해상도 (기본 640)
        conf_threshold: 필요하면 이 값으로 모델 conf 를 설정 (None 이면 건드리지 않음)

    Returns:
        YOLOv5 results 객체 (results.xyxy[0] 등으로 bbox 접근 가능)
    """
    model = get_yolov5_model()

    if conf_threshold is not None:
        # Ultralytics YOLOv5 는 model.conf 로 confidence threshold 를 설정한다.
        model.conf = float(conf_threshold)

    # YOLOv5 는 내부에서 RGB 를 기준으로 동작하므로 BGR→RGB 변환
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 모델은 Numpy 배열도 직접 받을 수 있다.
    results = model(rgb_image, size=img_size)

    return results