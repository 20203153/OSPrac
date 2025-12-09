#!/usr/bin/env python3
"""Integration test for YOLOv5 singleton + detect_box size filter."""

import argparse
import os
import glob
import json

import cv2

from new_func.detect_func import detect_box, detect_box_by_height, estimate_object_size_from_bbox
from new_func.yolov5_singleton import load_yolov5_model, run_yolov5_inference


import warnings


# torch.cuda.amp.autocast 관련 FutureWarning 무시
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast.*",
)


def crop_center_square_and_resize_to_250(bgr):
    """
    원본 이미지에서 중앙 정사각형 영역을 잘라낸 뒤 250x250으로 리사이즈한다.

    - OAK-D /oakd/rgb/preview/image_raw 250x250 프리뷰와 비슷한 전처리를
      오프라인 테스트 이미지에도 적용하기 위한 유틸.
    """
    h, w = bgr.shape[:2]
    side = min(h, w)

    # 중앙 정사각형 좌표 계산
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    x2 = x1 + side
    y2 = y1 + side

    crop = bgr[y1:y2, x1:x2].copy()
    preview = cv2.resize(crop, (250, 250), interpolation=cv2.INTER_AREA)
    return preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv5 on a single image and test detect_box() filter.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./box.pt",
        help="Path to YOLOv5 .pt weight file.",
    )
    parser.add_argument(
        "--testcase_path",
        type=str,
        default="./testcase",
        help="Directory path containing test*.png images for batch test.",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=300.0,
        help="Camera focal length fx in pixels (default ~300 for 250x250 preview).",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=300.0,
        help="Camera focal length fy in pixels (default ~300 for 250x250 preview).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="YOLOv5 inference size (default: 640).",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.4,
        help="YOLOv5 confidence threshold (default: 0.4).",
    )
    parser.add_argument(
        "--distance_m",
        type=float,
        default=1.0,
        help="Target distance in meters (default: 1.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="YOLOv5 inference device (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Load YOLOv5 model (singleton)
    _ = load_yolov5_model(
        model_path=args.model_path,
        device=args.device,
        use_half=False,
    )

    # 2) Collect testcase images: test*.png under testcase_path
    pattern = os.path.join(args.testcase_path, "test_*.png")
    image_paths = sorted(glob.glob(pattern))

    if not image_paths:
        raise RuntimeError(f"No testcase images found matching {pattern}")

    for img_path in image_paths:
        print(f"\n=== Testcase: {img_path} ===")

        # 2-1) Load image
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Failed to read image: {img_path}")
            continue

        # 2-2) 중앙 정사각형을 잘라 250x250 프리뷰로 리사이즈
        preview_250 = crop_center_square_and_resize_to_250(bgr)

        # 3) Run inference on 250x250 preview image
        results = run_yolov5_inference(
            bgr_image=preview_250,
            img_size=args.img_size,
            conf_threshold=args.conf_threshold,
        )

        boxes = results.xyxy[0]  # tensor [N, 6]: x1, y1, x2, y2, conf, cls

        if boxes is None or len(boxes) == 0:
            print("No detections.")
            continue

        print(f"Detections: {len(boxes)}")

        # 이 테스트케이스에 대한 distance_m:
        # 기본값은 CLI args.distance_m, 있으면 같은 이름의 JSON(test_*.json)에서 distance 필드로 override
        distance_m = args.distance_m
        json_base = os.path.splitext(os.path.basename(img_path))[0] + ".json"
        json_path = os.path.join(os.path.dirname(img_path), json_base)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if isinstance(meta, dict) and "distance" in meta:
                    distance_m = float(meta["distance"])
                    print(f"Using distance from {json_path}: {distance_m} m")
            except Exception as e:
                print(f"Failed to load distance from {json_path}: {e}. Using default {distance_m} m")

        # ------------------------------------------------------------------
        # Test 1: footprint 기준 detect_box 결과 시각화
        # ------------------------------------------------------------------
        print("  [Test 1] detect_box (footprint width/height match)")
        preview_vis_foot = preview_250.copy()

        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2, conf, cls_id = bbox.tolist()

            est_w, est_h = estimate_object_size_from_bbox(
                bbox=bbox,
                distance_m=distance_m,
                fx=args.fx,
                fy=args.fy,
            )

            is_target_foot = detect_box(
                bbox=bbox,
                distance=distance_m,
                fx=args.fx,
                fy=args.fy,
            )

            print(
                f"[1:{i}] cls={int(cls_id)} conf={conf:.3f} "
                f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                f"size_est=({est_w:.3f}m, {est_h:.3f}m) "
                f"is_target_footprint={is_target_foot}"
            )

            # preview_250 (250x250) 위에 bbox 그리기
            x1_i, y1_i = int(x1), int(y1)
            x2_i, y2_i = int(x2), int(y2)

            # footprint 기준 타겟: 초록, 나머지: 파랑
            color = (0, 255, 0) if is_target_foot else (255, 0, 0)
            cv2.rectangle(preview_vis_foot, (x1_i, y1_i), (x2_i, y2_i), color, 2)

        # ------------------------------------------------------------------
        # Test 2: 높이 기준 detect_box_by_height 결과 시각화
        # ------------------------------------------------------------------
        print("  [Test 2] detect_box_by_height (height-only match)")
        preview_vis_height = preview_250.copy()

        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2, conf, cls_id = bbox.tolist()

            est_w, est_h = estimate_object_size_from_bbox(
                bbox=bbox,
                distance_m=distance_m,
                fx=args.fx,
                fy=args.fy,
            )

            is_target_height = detect_box_by_height(
                bbox=bbox,
                distance=distance_m,
                fx=args.fx,
                fy=args.fy,
            )

            print(
                f"[2:{i}] cls={int(cls_id)} conf={conf:.3f} "
                f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                f"size_est=({est_w:.3f}m, {est_h:.3f}m) "
                f"is_target_height={is_target_height}"
            )

            x1_i, y1_i = int(x1), int(y1)
            x2_i, y2_i = int(x2), int(y2)

            # 높이 기준 타겟: 주황, 나머지: 회색
            color = (0, 165, 255) if is_target_height else (128, 128, 128)
            cv2.rectangle(preview_vis_height, (x1_i, y1_i), (x2_i, y2_i), color, 2)

        # 결과 이미지 저장 (케이스당 2개: footprint / height)
        os.makedirs("result", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path_foot = os.path.join("result", f"{base_name}_footprint.png")
        out_path_height = os.path.join("result", f"{base_name}_height.png")

        cv2.imwrite(out_path_foot, preview_vis_foot)
        cv2.imwrite(out_path_height, preview_vis_height)
        print(f"Saved footprint test preview to: {out_path_foot}")
        print(f"Saved height test preview to: {out_path_height}")


if __name__ == "__main__":
    main()