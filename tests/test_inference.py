"""src/inference.py 단위 테스트.

ultralytics YOLO는 conftest.mock_yolo 픽스처로 항상 Mock 처리한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.inference import load_model, process_results, run_inference


def test_process_results_schema(fake_yolo_result_with_detections) -> None:
    """process_results() 출력에 필수 키가 모두 존재한다."""
    out = process_results(
        fake_yolo_result_with_detections,
        filename="sample.jpg",
        model_name="yolov8n.pt",
    )
    required = {"filename", "detections", "inference_time_ms"}
    assert required.issubset(out.keys()), f"누락된 키: {required - out.keys()}"
    assert out["filename"] == "sample.jpg"
    assert isinstance(out["detections"], list)
    assert len(out["detections"]) == 2


def test_empty_detections(fake_yolo_result_empty) -> None:
    """탐지가 없으면 detections == [], detection_count == 0."""
    out = process_results(
        fake_yolo_result_empty, filename="empty.jpg", model_name="yolov8n.pt"
    )
    assert out["detections"] == []
    assert out["detection_count"] == 0


def test_bbox_integer_coords(fake_yolo_result_with_detections) -> None:
    """모든 detection의 bbox x1/y1/x2/y2는 int 타입이어야 한다."""
    out = process_results(
        fake_yolo_result_with_detections,
        filename="sample.jpg",
        model_name="yolov8n.pt",
    )
    assert out["detections"], "탐지 결과가 비어 있어 검증 불가"
    for det in out["detections"]:
        bbox = det["bbox"]
        for key in ("x1", "y1", "x2", "y2"):
            assert isinstance(bbox[key], int), (
                f"{key}={bbox[key]!r} (type={type(bbox[key]).__name__}) is not int"
            )


def test_inference_time_positive(
    tmp_path: Path,
    sample_bgr_image: np.ndarray,
    mock_yolo,
) -> None:
    """run_inference() 결과의 inference_time_ms > 0 이어야 한다."""
    import cv2
    img_path = tmp_path / "img.jpg"
    cv2.imwrite(str(img_path), sample_bgr_image)

    model = load_model("yolov8n.pt")
    out = run_inference(model, str(img_path))

    assert out["inference_time_ms"] > 0, (
        f"inference_time_ms={out['inference_time_ms']} not > 0"
    )
    mock_yolo.predict.assert_called_once()
