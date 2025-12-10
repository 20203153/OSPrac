#!/usr/bin/env python3
"""
BoxSizeHelper: YOLO 바운딩박스 픽셀 크기를 실제 미터 단위로 변환하는 순수 헬퍼 클래스.

- 이 클래스는 **YOLO 모델을 로딩하거나 추론하지 않는다.**
- 이미 다른 Node / 코드에서 얻은 바운딩박스(bbox) 데이터와,
  처음 한 번 캐싱한 CameraInfo (fx, fy)를 사용해서:
    1) 픽셀 단위 bbox 크기 계산
    2) 거리(distance, m) + fx, fy 를 이용해 미터 단위 실제 크기 계산
  만 담당한다.

사용 예시 (다른 Node 안에서):

    from box_size_helper import BoxSizeHelper
    from sensor_msgs.msg import CameraInfo

    helper = BoxSizeHelper()

    # CameraInfo 콜백
    def camera_info_callback(msg: CameraInfo):
        helper.set_camera_info_from_msg(msg)

    # RGB + YOLO 추론이 이미 끝난 상황에서:
    def handle_detection(det, distance_m: float):
        # det: [x1, y1, x2, y2, (conf), (cls)] 형식
        pix_w, pix_h = helper.get_pixel_size(det)
        width_m, height_m = helper.estimate_size_from_bbox(det, distance_m)

        conv = helper.convert_bbox(det, distance_m)
        # conv = {
        #   "distance": distance_m,
        #   "x_px": pix_w,
        #   "y_px": pix_h,
        #   "x_m": width_m,
        #   "y_m": height_m,
        # }
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

from sensor_msgs.msg import CameraInfo

Number = Union[int, float]
BBoxLike = Sequence[Number]


class BoxSizeHelper:
    """
    - CameraInfo 에서 fx, fy 를 한 번 캐싱해 두고,
    - YOLO 스타일 bbox([x1, y1, x2, y2, ...])와 거리(m)를 넘기면
      픽셀 단위/미터 단위 크기를 계산하는 헬퍼 클래스.
    """

    def __init__(self, camera_info: Optional[CameraInfo] = None) -> None:
        """
        Args:
            camera_info: 초기화 시점에 바로 사용할 CameraInfo 메시지(옵션).
                         주어지면 fx, fy 를 자동으로 채운다.
        """
        # Camera intrinsics
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None

        if camera_info is not None:
            # 초기화 시 바로 CameraInfo 를 이용해 fx, fy 를 세팅
            self.set_camera_info_from_msg(camera_info)

    # ------------------------------------------------------------------
    # CameraInfo / Intrinsics 설정
    # ------------------------------------------------------------------

    def set_camera_info(self, fx: float, fy: float) -> None:
        """카메라 내부 파라미터 fx, fy 를 직접 설정한다."""
        self.fx = float(fx)
        self.fy = float(fy)

    def set_camera_info_from_msg(self, msg: CameraInfo) -> None:
        """
        ROS2 sensor_msgs/CameraInfo 메시지에서 fx, fy 를 추출해 저장한다.

        보통 CameraInfo.K[0] = fx, CameraInfo.K[4] = fy.
        """
        try:
            k = msg.k
            self.fx = float(k[0])
            self.fy = float(k[4])
        except Exception as e:  # pragma: no cover - 단순 유틸
            raise ValueError(f"Failed to parse CameraInfo: {e}") from e

    # ------------------------------------------------------------------
    # BBox 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_bbox_xyxy(bbox: BBoxLike) -> Tuple[float, float, float, float]:
        """
        YOLOv5 xyxy 형식의 bounding box 로부터 (x1, y1, x2, y2)를 float 로 추출한다.

        bbox: [x1, y1, x2, y2, (conf), (cls)] 형식을 가정.
        """
        if len(bbox) < 4:
            raise ValueError("bbox must have at least 4 elements: [x1, y1, x2, y2, ...]")

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        return float(x1), float(y1), float(x2), float(y2)

    def get_pixel_size(self, bbox: BBoxLike) -> Tuple[float, float]:
        """
        바운딩박스의 픽셀 단위 (width, height)를 계산한다.

        Returns:
            (pixel_width, pixel_height)
        """
        x1, y1, x2, y2 = self._parse_bbox_xyxy(bbox)
        pixel_width = max(0.0, float(x2 - x1))
        pixel_height = max(0.0, float(y2 - y1))
        return pixel_width, pixel_height

    def estimate_size_from_bbox(
        self,
        bbox: BBoxLike,
        distance_m: float,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        주어진 거리(distance_m)에서 bbox 픽셀 크기를 이용해
        물체의 대략적인 실제 가로/세로 길이(미터)를 추정한다.

        new_func.detect_func.estimate_object_size_from_bbox() 와 동일한 수식 사용:

            width_m  = (bbox_pixel_width  * distance_m) / fx
            height_m = (bbox_pixel_height * distance_m) / fy

        Args:
            bbox: YOLOv5 xyxy 형식 bbox (길이 >= 4 인 시퀀스)
            distance_m: 카메라와 물체 사이의 거리 (미터)
            fx, fy: 카메라 focal length (픽셀 단위).
                    None 이면, 이 클래스에 저장된 self.fx, self.fy 를 사용.

        Returns:
            (width_m, height_m): 물체의 가로/세로 길이(미터)
        """
        if fx is None or fy is None:
            if self.fx is None or self.fy is None:
                raise ValueError(
                    "Camera intrinsics fx/fy are not set. "
                    "Call set_camera_info() or set_camera_info_from_msg() first."
                )
            fx_use = float(self.fx)
            fy_use = float(self.fy)
        else:
            fx_use = float(fx)
            fy_use = float(fy)

        if distance_m <= 0.0:
            raise ValueError("distance_m must be positive.")

        pixel_width, pixel_height = self.get_pixel_size(bbox)

        width_m = (pixel_width * distance_m) / fx_use
        height_m = (pixel_height * distance_m) / fy_use

        return width_m, height_m

    def convert_bbox(
        self,
        bbox: BBoxLike,
        distance_m: float,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
    ) -> dict:
        """
        하나의 bbox 에 대해
        - 픽셀 단위 크기 (x_px, y_px)
        - 미터 단위 크기 (x_m, y_m)
        를 모두 계산해 dict 로 반환하는 편의 메서드.

        Returns:
            {
                "distance": distance_m,
                "x_px": pixel_width,
                "y_px": pixel_height,
                "x_m": width_m,
                "y_m": height_m,
            }
        """
        pixel_width, pixel_height = self.get_pixel_size(bbox)
        width_m, height_m = self.estimate_size_from_bbox(
            bbox=bbox,
            distance_m=distance_m,
            fx=fx,
            fy=fy,
        )

        return {
            "distance": float(distance_m),
            "x_px": float(pixel_width),
            "y_px": float(pixel_height),
            "x_m": float(width_m),
            "y_m": float(height_m),
        }

    def convert_boxes(
        self,
        boxes: Any,
        distance_m: float,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
    ) -> Optional[dict]:
        """
        여러 bbox 가 들어왔을 때,
        - 픽셀 기준으로 "세로 길이(y_px)가 가장 큰" 하나만 선택해 변환 결과를 반환한다.

        Args:
            boxes: YOLO 결과의 bbox 리스트/배열 (예: results.xyxy[0])
            distance_m: 카메라-물체 거리 [m]
            fx, fy: 카메라 focal length (옵션, None 이면 self.fx/self.fy 사용)

        Returns:
            가장 픽셀 높이가 큰 bbox 에 대해 convert_bbox() 결과를 반환하거나,
            boxes 가 비어 있으면 None 반환.
        """
        if boxes is None or len(boxes) == 0:
            return None

        best_det = None
        best_height_px = -1.0

        for det in boxes:
            _, h_px = self.get_pixel_size(det)
            if h_px > best_height_px:
                best_height_px = h_px
                best_det = det

        if best_det is None:
            return None

        return self.convert_bbox(
            bbox=best_det,
            distance_m=distance_m,
            fx=fx,
            fy=fy,
        )