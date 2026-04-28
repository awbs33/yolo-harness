---
name: yolo-inference
description: "ultralytics YOLOv8을 사용한 객체 탐지 추론 모듈(src/inference.py)과 requirements.txt를 구현한다. YOLO 모델 로드, 이미지 배치 추론, 신뢰도/IOU 필터링, 탐지 결과 JSON 저장, 추론 통계 요약 포함. '추론 구현', 'inference 모듈', 'YOLOv8 추론', '객체 탐지 실행', 'detection', 'requirements.txt' 요청 시 반드시 이 스킬을 사용할 것."
---

# YOLOv8 추론 스킬

## 목표
`src/inference.py`를 구현하여 ultralytics YOLOv8로 객체 탐지 추론을 수행하고, `requirements.txt`를 생성한다.

## 구현 명세: src/inference.py

### 의존성
```python
from ultralytics import YOLO
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
```

### 함수 명세

**`load_model(model_path: str = "yolov8n.pt") -> YOLO`**
- `YOLO(model_path)` 호출 (없으면 ultralytics 자동 다운로드)
- 로드 시간 측정 및 로그 출력
- 실패 시 `RuntimeError("Model load failed. Install ultralytics: pip install ultralytics")`

**`run_inference(model: YOLO, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict[str, Any]`**
- `model.predict(source=image_path, conf=conf_threshold, iou=iou_threshold, verbose=False)`
- `time.perf_counter()`로 추론 시간 측정
- `process_results()`로 결과 변환 후 반환

**`process_results(results, filename: str, model_name: str) -> Dict[str, Any]`**
- ultralytics Results 객체 파싱
- 반환 스키마:
  ```json
  {
    "filename": "image.jpg",
    "model": "yolov8n.pt",
    "image_width": 640,
    "image_height": 640,
    "detections": [
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 400}
      }
    ],
    "detection_count": 1,
    "inference_time_ms": 12.5
  }
  ```
- bbox 좌표: `int(box.xyxy[0][i])` 로 정수 변환
- class_name: `model.names[int(class_id)]`

**`run_batch_inference(model: YOLO, manifest_path: str, output_dir: str, **kwargs) -> str`**
- `manifest.json`을 읽어 모든 이미지에 순차 추론
- 결과를 `{output_dir}/{stem}.json`으로 저장
- 완료 후 `summary.json` 생성:
  ```json
  {"total_images": 10, "total_detections": 42, "avg_inference_ms": 15.2,
   "class_counts": {"person": 20, "car": 22}}
  ```
- `output_dir` 경로 반환

## requirements.txt 내용
```
ultralytics>=8.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
numpy>=1.24.0
pytest>=7.4.0
pytest-mock>=3.11.0
```

## 장치 처리
ultralytics가 CUDA 자동 감지. 별도 device 설정 불필요. CPU 폴백도 자동.

## 출력 경로
- 개별 결과: `_workspace/02_predictions/{stem}.json`
- 요약 통계: `_workspace/02_predictions/summary.json`
