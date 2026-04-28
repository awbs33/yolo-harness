---
name: preprocess-input
description: "YOLOv8 파이프라인의 이미지/영상 입력 전처리 모듈(src/preprocess.py)과 main.py 진입점을 구현한다. './images' 경로 파일 로드, OpenCV BGR→RGB 변환, 640×640 letterbox 리사이즈, 영상 프레임 추출, manifest.json 생성 포함. '전처리 구현', '입력 처리', 'preprocess 모듈', 'main.py 작성', '파이프라인 진입점' 요청 시 이 스킬을 사용할 것."
---

# 입력 전처리 스킬

## 목표
`src/preprocess.py`와 `main.py`를 구현하여 YOLOv8 입력 규격에 맞는 전처리 파이프라인과 CLI 진입점을 구성한다.

## 구현 명세: src/preprocess.py

### 의존성
```python
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
```

### 지원 형식
- 이미지: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- 영상: `.mp4`, `.avi`, `.mov`

### 함수 명세

**`load_input(input_path: str) -> List[Tuple[str, np.ndarray]]`**
- 경로의 모든 지원 파일을 로드하여 `(filename, bgr_image)` 리스트 반환
- 영상은 `preprocess_video()`로 처리
- 지원하지 않는 형식: `logging.warning()` 후 건너뜀
- 빈 디렉토리: `ValueError("No supported files found in {input_path}")` 발생

**`preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray`**
- BGR → RGB 변환 (`cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`)
- letterbox 리사이즈: 종횡비 유지, 빈 공간은 회색(128, 128, 128) 패딩
- dtype: uint8 유지 (ultralytics가 내부 정규화 처리)
- letterbox를 쓰는 이유: 단순 resize는 종횡비를 왜곡하여 탐지 정확도를 떨어뜨린다

**`preprocess_video(video_path: str, frame_interval: int = 1) -> List[Tuple[str, np.ndarray]]`**
- `cv2.VideoCapture`로 프레임 추출
- `frame_interval` 프레임마다 1장 (기본: 전 프레임)
- 파일명 컨벤션: `{video_stem}_frame{idx:06d}.jpg`

**`save_manifest(items: List[dict], output_dir: str) -> None`**
- `manifest.json` 저장, 항목 형식:
  ```json
  {
    "filename": "img.jpg",
    "original_path": "./images/img.jpg",
    "preprocessed_path": "_workspace/01_preprocessed/img.jpg",
    "width": 640,
    "height": 640,
    "source_type": "image"
  }
  ```

### letterbox 구현 패턴
```python
h, w = image.shape[:2]
scale = min(target_size[0] / h, target_size[1] / w)
new_h, new_w = int(h * scale), int(w * scale)
resized = cv2.resize(image, (new_w, new_h))
canvas = np.full((target_size[0], target_size[1], 3), 128, dtype=np.uint8)
y_off = (target_size[0] - new_h) // 2
x_off = (target_size[1] - new_w) // 2
canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
```

## 구현 명세: main.py

```python
import argparse
from src.preprocess import load_input, preprocess_image, save_manifest
from src.inference import load_model, run_batch_inference
from src.visualize import process_all

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Pipeline")
    parser.add_argument("--input",  default="./images",   help="Input directory")
    parser.add_argument("--output", default="./results",  help="Output directory")
    parser.add_argument("--model",  default="yolov8n.pt", help="YOLOv8 model weights")
    parser.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou",    type=float, default=0.45, help="IOU threshold")
    args = parser.parse_args()
    # 파이프라인 실행 (preprocess → inference → visualize)
    ...

if __name__ == "__main__":
    main()
```

## 출력 경로
- 전처리 이미지: `_workspace/01_preprocessed/{filename}`
- 매니페스트: `_workspace/01_preprocessed/manifest.json`
