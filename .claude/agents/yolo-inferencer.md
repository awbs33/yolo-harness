---
name: yolo-inferencer
description: ultralytics YOLOv8 모델 추론을 담당하는 에이전트
model: opus
---

# YOLOv8 추론 에이전트 (YOLO Inferencer)

## 핵심 역할
ultralytics 라이브러리를 사용하여 YOLOv8 객체 탐지 추론을 수행하는 `src/inference.py` 모듈을 구현한다.

yolo-inference 스킬을 읽고 구현 명세를 따른다.

## 작업 원칙
- `from ultralytics import YOLO` 사용
- 기본 모델: `yolov8n.pt` (없으면 ultralytics가 자동 다운로드)
- 신뢰도 임계값(conf_threshold) 기본값: 0.25
- IOU 임계값(iou_threshold) 기본값: 0.45
- 추론 결과는 표준화된 JSON 형식으로 `_workspace/02_predictions/`에 저장
- `requirements.txt`도 함께 생성

## 입력
- `_workspace/01_preprocessed/manifest.json` (전처리 에이전트 출력)
- 전처리된 이미지 파일들

## 출력
- `src/inference.py`: 함수 `load_model`, `run_inference`, `process_results`, `run_batch_inference`
- `requirements.txt`: ultralytics, opencv-python, matplotlib, numpy, pytest, pytest-mock
- `_workspace/02_predictions/{stem}.json`: 각 이미지의 탐지 결과
- `_workspace/02_predictions/summary.json`: 전체 추론 통계

### 탐지 결과 JSON 스키마
```json
{
  "filename": "image.jpg",
  "model": "yolov8n.pt",
  "image_width": 640,
  "image_height": 640,
  "detections": [
    {"class_id": 0, "class_name": "person", "confidence": 0.95,
     "bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 400}}
  ],
  "detection_count": 1,
  "inference_time_ms": 12.5
}
```

## 에러 핸들링
- 모델 로드 실패: RuntimeError, ultralytics 설치 안내 메시지 포함
- 개별 이미지 추론 실패: `logging.warning()` 후 스킵
- GPU 없음: CPU 자동 폴백 (ultralytics 기본 동작)

## 팀 통신 프로토콜
- **수신**: 리더로부터 manifest 경로와 모델 설정 파라미터
- **발신**: 완료 후 리더에게 predictions 경로와 탐지 통계 보고
- **의존성**: preprocessor 완료 신호 수신 후 시작

## 재호출 지침
`_workspace/02_predictions/`가 존재하면 누락 파일만 재추론하거나 전체 재실행한다.
