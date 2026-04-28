"""src/preprocess.py 단위 테스트."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocess import (
    SUPPORTED_EXTS,
    load_input,
    preprocess_image,
    save_manifest,
)


def test_preprocess_image_output_size(sample_bgr_image: np.ndarray) -> None:
    """preprocess_image() 출력 shape == (640, 640, 3)."""
    out = preprocess_image(sample_bgr_image)
    assert out.shape == (640, 640, 3)


def test_preprocess_image_dtype(sample_bgr_image: np.ndarray) -> None:
    """preprocess_image() 출력 dtype == np.uint8."""
    out = preprocess_image(sample_bgr_image)
    assert out.dtype == np.uint8


def test_letterbox_no_distortion(sample_bgr_image: np.ndarray) -> None:
    """letterbox 패딩 적용 후 원본 종횡비가 유지되어야 한다.

    480×640 → 640×640 변환 시: scale = min(640/480, 640/640) = 1.0,
    new_w = 640, new_h = 480. 위/아래로 80픽셀씩 회색(128) 패딩이 들어간다.
    """
    out = preprocess_image(sample_bgr_image)
    h, w = sample_bgr_image.shape[:2]
    scale = min(640 / h, 640 / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    pad_y = (640 - new_h) // 2
    pad_x = (640 - new_w) // 2

    if pad_y > 0:
        top_strip = out[:pad_y, :, :]
        assert np.all(top_strip == 128), "위쪽 패딩이 회색(128)이 아닙니다"
    if pad_x > 0:
        left_strip = out[:, :pad_x, :]
        assert np.all(left_strip == 128), "왼쪽 패딩이 회색(128)이 아닙니다"


def test_load_unsupported_format(tmp_path: Path, caplog) -> None:
    """지원하지 않는 형식만 있으면 ValueError(빈 결과), 경고 로그 발생."""
    bad = tmp_path / "note.txt"
    bad.write_text("not an image")

    import logging
    caplog.set_level(logging.WARNING, logger="src.preprocess")

    with pytest.raises(ValueError):
        load_input(str(tmp_path))

    assert any("Unsupported" in rec.message for rec in caplog.records), (
        "지원하지 않는 형식에 대한 경고 로그가 없습니다"
    )


def test_empty_directory_raises(tmp_path: Path) -> None:
    """완전히 빈 디렉토리는 ValueError를 발생시켜야 한다."""
    with pytest.raises(ValueError):
        load_input(str(tmp_path))


def test_manifest_keys(tmp_path: Path) -> None:
    """save_manifest()로 기록한 entry가 필수 키를 모두 포함해야 한다."""
    entries = [
        {
            "filename": "a.jpg",
            "original_path": "/in/a.jpg",
            "preprocessed_path": "/out/a.jpg",
            "width": 640,
            "height": 640,
            "source_type": "image",
        }
    ]
    save_manifest(entries, str(tmp_path))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert isinstance(manifest, list)
    assert len(manifest) == 1
    required = {"filename", "preprocessed_path", "width", "height", "source_type"}
    assert required.issubset(manifest[0].keys())


def test_load_input_reads_real_image(tmp_path: Path, sample_bgr_image: np.ndarray) -> None:
    """디렉토리에서 .jpg 파일을 정상적으로 로드한다 (smoke)."""
    img_path = tmp_path / "img.jpg"
    cv2.imwrite(str(img_path), sample_bgr_image)
    items = load_input(str(tmp_path))
    assert len(items) == 1
    name, arr = items[0]
    assert name == "img.jpg"
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3
