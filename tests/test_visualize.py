"""src/visualize.py 단위 테스트."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.visualize import (
    draw_detections,
    get_class_color,
    process_all,
    save_annotated,
)


def test_draw_detections_shape(
    sample_bgr_image: np.ndarray, sample_detections: list
) -> None:
    """draw_detections() 반환 이미지는 원본과 동일한 shape여야 한다."""
    annotated = draw_detections(sample_bgr_image, sample_detections)
    assert annotated.shape == sample_bgr_image.shape
    assert annotated.dtype == sample_bgr_image.dtype


def test_draw_no_detections(sample_bgr_image: np.ndarray) -> None:
    """탐지 리스트가 비어 있으면 원본과 픽셀 단위로 동일해야 한다."""
    annotated = draw_detections(sample_bgr_image, [])
    assert annotated.shape == sample_bgr_image.shape
    assert np.array_equal(annotated, sample_bgr_image)


def test_annotated_output_created(
    tmp_path: Path,
    sample_bgr_image: np.ndarray,
    sample_detections: list,
) -> None:
    """save_annotated() 호출 후 결과 파일이 디스크에 생성되어야 한다."""
    src_path = tmp_path / "original.jpg"
    out_path = tmp_path / "results" / "annotated.jpg"
    cv2.imwrite(str(src_path), sample_bgr_image)

    save_annotated(str(src_path), sample_detections, str(out_path))

    assert out_path.exists(), f"Annotated output not found at {out_path}"
    loaded = cv2.imread(str(out_path))
    assert loaded is not None and loaded.size > 0


def test_color_palette_uniqueness() -> None:
    """인접한 10개 class_id의 색상이 서로 모두 다르다."""
    colors = [get_class_color(i) for i in range(10)]
    assert len(set(colors)) == 10, (
        f"중복 색상 발견: {colors}"
    )


def test_process_all_smoke(
    tmp_path: Path,
    sample_bgr_image: np.ndarray,
    sample_detections: list,
) -> None:
    """process_all()이 prediction JSON을 어노테이션 이미지와 차트로 변환한다."""
    input_dir = tmp_path / "input"
    pred_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"
    input_dir.mkdir()
    pred_dir.mkdir()

    img_path = input_dir / "img.jpg"
    cv2.imwrite(str(img_path), sample_bgr_image)

    entry = {
        "filename": "img.jpg",
        "detections": sample_detections,
        "detection_count": len(sample_detections),
        "inference_time_ms": 1.2,
    }
    (pred_dir / "img.json").write_text(json.dumps(entry))

    n = process_all(str(pred_dir), str(input_dir), str(out_dir))
    assert n == 1
    assert (out_dir / "img_annotated.jpg").exists()
    assert (out_dir / "detection_summary.png").exists()
