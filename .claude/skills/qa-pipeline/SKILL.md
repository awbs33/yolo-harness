---
name: qa-pipeline
description: "YOLOv8 파이프라인의 pytest 테스트 스위트를 작성하고 실행한다. 전처리/추론/시각화 단위 테스트, CLI 통합 테스트, 엣지 케이스 검증, Mock 기반 ultralytics 격리 포함. 'QA', '테스트 작성', 'pytest', '파이프라인 검증', '단위 테스트', '통합 테스트', 'test suite' 요청 시 반드시 이 스킬을 사용할 것."
---

# QA 파이프라인 스킬

## 목표
`tests/` 디렉토리에 pytest 테스트를 작성하고 실행하여 파이프라인 정확성을 검증한다.

## 테스트 구조

```
tests/
├── conftest.py         - 공통 픽스처
├── test_preprocess.py  - 전처리 단위 테스트
├── test_inference.py   - 추론 단위 테스트
├── test_visualize.py   - 시각화 단위 테스트
└── test_integration.py - CLI 통합 테스트
```

## conftest.py 핵심 픽스처

```python
import pytest
import numpy as np
import json
from pathlib import Path

@pytest.fixture
def sample_bgr_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def sample_detections():
    return [
        {"class_id": 0, "class_name": "person", "confidence": 0.95,
         "bbox": {"x1": 50, "y1": 30, "x2": 200, "y2": 400}},
        {"class_id": 2, "class_name": "car", "confidence": 0.82,
         "bbox": {"x1": 300, "y1": 100, "x2": 580, "y2": 350}}
    ]

@pytest.fixture
def mock_yolo(mocker):
    # ultralytics YOLO를 Mock하여 실제 모델 다운로드 없이 테스트
    mock = mocker.MagicMock()
    mock.names = {0: "person", 2: "car"}
    # predict 결과를 합성 탐지 결과로 대체
    ...
    return mock
```

## test_preprocess.py 커버리지

| 테스트 함수 | 검증 항목 |
|------------|----------|
| `test_preprocess_image_output_size` | 출력 shape == (640, 640, 3) |
| `test_preprocess_image_dtype` | dtype == np.uint8 |
| `test_letterbox_no_distortion` | 패딩 적용 후 비율 유지 확인 |
| `test_load_unsupported_format` | 경고 로그 발생, 빈 리스트 반환 |
| `test_empty_directory_raises` | ValueError 발생 |
| `test_manifest_keys` | manifest 항목에 필수 키 존재 |

## test_inference.py 커버리지

| 테스트 함수 | 검증 항목 |
|------------|----------|
| `test_process_results_schema` | 필수 키(filename, detections, inference_time_ms) 존재 |
| `test_empty_detections` | detections == [], detection_count == 0 |
| `test_bbox_integer_coords` | bbox x1/y1/x2/y2 모두 int 타입 |
| `test_inference_time_positive` | inference_time_ms > 0 |

## test_visualize.py 커버리지

| 테스트 함수 | 검증 항목 |
|------------|----------|
| `test_draw_detections_shape` | 원본과 동일한 shape 반환 |
| `test_draw_no_detections` | 빈 리스트 → 원본 이미지와 동일 |
| `test_annotated_output_created` | results/ 파일 생성 확인 |
| `test_color_palette_uniqueness` | 인접 10개 class_id의 색상이 모두 다름 |

## test_integration.py

```python
import subprocess
import sys

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--input" in result.stdout

def test_full_pipeline(tmp_path, sample_image_dir):
    result = subprocess.run(
        [sys.executable, "main.py",
         "--input", str(sample_image_dir),
         "--output", str(tmp_path / "results")],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert (tmp_path / "results").exists()
```

## 실행 방법
```bash
pytest tests/ -v --tb=short
```

## QA 리포트 형식 (_workspace/04_qa_report.md)
```markdown
# QA Report — {timestamp}

## 결과 요약
- 전체: {N}개 | 통과: {P}개 | 실패: {F}개

## 실패 목록
| 테스트 | 오류 | 수정 제안 |
|--------|------|----------|
| ... | ... | ... |

## 통과 목록
- test_preprocess_image_output_size ✓
- ...
```
