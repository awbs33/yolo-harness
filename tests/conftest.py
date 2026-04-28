"""공통 pytest 픽스처.

ultralytics YOLO는 pytest-mock으로 격리하여 실제 모델을 다운로드하지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 프로젝트 루트를 sys.path에 추가하여 src 모듈을 임포트할 수 있게 한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """480×640×3 uint8 BGR 합성 이미지."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[dict]:
    """person + car 2개 탐지 결과 리스트."""
    return [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.95,
            "bbox": {"x1": 50, "y1": 30, "x2": 200, "y2": 400},
        },
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.82,
            "bbox": {"x1": 300, "y1": 100, "x2": 580, "y2": 350},
        },
    ]


class _FakeBoxes:
    """ultralytics Boxes 객체를 흉내내는 더미.

    .xyxy / .conf / .cls 는 numpy 배열 그대로 반환한다 (이미 cpu().numpy() 처리된 것처럼).
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.xyxy)


class _FakeResult:
    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        names: dict,
        orig_shape=(480, 640),
    ):
        self.boxes = _FakeBoxes(xyxy, conf, cls)
        self.names = names
        self.orig_shape = orig_shape


@pytest.fixture
def fake_yolo_result_with_detections():
    """탐지 2개(person, car)가 포함된 가짜 ultralytics 결과 리스트."""
    xyxy = np.array(
        [[50.0, 30.0, 200.0, 400.0], [300.0, 100.0, 580.0, 350.0]],
        dtype=np.float32,
    )
    conf = np.array([0.95, 0.82], dtype=np.float32)
    cls = np.array([0, 2], dtype=np.float32)
    names = {0: "person", 2: "car"}
    return [_FakeResult(xyxy, conf, cls, names)]


@pytest.fixture
def fake_yolo_result_empty():
    """탐지 결과가 비어 있는 가짜 ultralytics 결과 리스트."""
    xyxy = np.zeros((0, 4), dtype=np.float32)
    conf = np.zeros((0,), dtype=np.float32)
    cls = np.zeros((0,), dtype=np.float32)
    names = {0: "person"}
    return [_FakeResult(xyxy, conf, cls, names)]


@pytest.fixture
def mock_yolo(mocker, fake_yolo_result_with_detections):
    """ultralytics YOLO 클래스를 Mock 처리한다.

    실제 모델을 다운로드하지 않고, predict()는 fake_yolo_result_with_detections를
    반환한다. src.inference.YOLO 심볼을 패치하여 load_model()이 사용한다.
    """
    mock_model = mocker.MagicMock()
    mock_model.names = {0: "person", 2: "car"}
    mock_model.model_name = "yolov8n.pt"
    mock_model.predict.return_value = fake_yolo_result_with_detections

    yolo_cls = mocker.patch("src.inference.YOLO", autospec=False)
    yolo_cls.return_value = mock_model
    return mock_model
