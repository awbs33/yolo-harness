---
name: visualize-results
description: "YOLOv8 탐지 결과를 이미지에 시각화하는 모듈(src/visualize.py)을 구현한다. 클래스별 색상 바운딩 박스, 신뢰도 레이블, 어노테이션 이미지 저장, 클래스별 탐지 수 통계 차트 생성 포함. '시각화 구현', 'visualize 모듈', '어노테이션', 'bounding box 그리기', '결과 이미지 저장', '탐지 통계 차트' 요청 시 이 스킬을 사용할 것."
---

# 결과 시각화 스킬

## 목표
`src/visualize.py`를 구현하여 탐지 결과를 원본 이미지에 시각화하고 `./results`에 저장한다.

## 구현 명세: src/visualize.py

### 의존성
```python
import cv2
import numpy as np
import json
import logging
import matplotlib
matplotlib.use('Agg')  # 헤드리스 환경 대응 (display 없는 서버에서도 동작)
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
```

### 색상 팔레트

**`get_class_color(class_id: int) -> Tuple[int, int, int]`**
황금각(137.508°) 분포 HSV → BGR 변환:
```python
hue = int((class_id * 137.508) % 360)
hsv = np.array([[[hue // 2, 220, 220]]], dtype=np.uint8)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
```
황금각을 쓰는 이유: 인접 class_id의 색상이 균일하게 분산되어 80개 COCO 클래스에서도 구분 가능하다.

### 함수 명세

**`draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray`**
- 이미지 복사본에 드로잉 (원본 보존)
- 박스 두께: `max(2, int(min(image.shape[:2]) * 0.002))` (이미지 크기에 비례)
- 레이블: `"{class_name} {confidence:.0%}"` (예: "person 95%")
- 레이블 배경: 박스와 같은 색상, 텍스트: 흰색
- `cv2.rectangle`, `cv2.putText` 사용

**`save_annotated(original_path: str, detections: List[Dict], output_path: str) -> None`**
- 원본 이미지 로드 (전처리 전 원본 사용, BGR 유지)
- `draw_detections()` 적용
- `cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])`
- 탐지 없어도 원본 복사 저장

**`generate_summary_plot(predictions_dir: str, output_path: str) -> None`**
- 모든 `*.json` (summary.json 제외)을 읽어 class_name별 탐지 수 집계
- matplotlib 가로 막대 차트 생성
- `plt.tight_layout()` 후 PNG 저장, `plt.close()` 호출 (메모리 누수 방지)

**`process_all(predictions_dir: str, input_dir: str, output_dir: str) -> int`**
- `predictions_dir/*.json`을 순회 (summary.json 스킵)
- 원본 파일: `{input_dir}/{filename}` (manifest의 original_path 참조)
- 출력: `{output_dir}/{stem}_annotated.jpg`
- `output_dir` 없으면 자동 생성
- 처리된 파일 수 반환

## 출력 규칙
- 어노테이션 이미지: `./results/{stem}_annotated.jpg`
- 통계 차트: `./results/detection_summary.png`
- `./results/` 없으면 `Path(output_dir).mkdir(parents=True, exist_ok=True)`
