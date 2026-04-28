"""YOLOv8 탐지 결과 시각화 모듈."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_GOLDEN_ANGLE_DEG = 137.508


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """황금각 분포 HSV → BGR 변환으로 클래스별 고유 색상 생성."""
    hue = int((class_id * _GOLDEN_ANGLE_DEG) % 360)
    hsv = np.array([[[hue // 2, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """이미지 복사본에 bbox + 레이블을 그려 반환."""
    annotated = image.copy()
    if not detections:
        return annotated

    h, w = annotated.shape[:2]
    thickness = max(2, int(min(h, w) * 0.002))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(h, w) / 1000.0)
    text_thickness = max(1, thickness - 1)

    for det in detections:
        class_id = int(det.get("class_id", 0))
        class_name = str(det.get("class_name", str(class_id)))
        confidence = float(det.get("confidence", 0.0))
        bbox = det.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))

        color = get_class_color(class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} {confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        label_y_top = max(0, y1 - text_h - baseline)
        cv2.rectangle(
            annotated,
            (x1, label_y_top),
            (x1 + text_w, label_y_top + text_h + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, label_y_top + text_h),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return annotated


def _inverse_letterbox(
    detections: List[Dict[str, Any]],
    orig_w: int,
    orig_h: int,
    preproc_w: int = 640,
    preproc_h: int = 640,
) -> List[Dict[str, Any]]:
    """640×640 letterbox 좌표를 원본 이미지 좌표로 역변환."""
    scale = min(preproc_h / orig_h, preproc_w / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    x_off = (preproc_w - new_w) // 2
    y_off = (preproc_h - new_h) // 2

    result = []
    for det in detections:
        d = det.copy()
        b = det["bbox"].copy()
        b["x1"] = max(0, round((b["x1"] - x_off) / scale))
        b["y1"] = max(0, round((b["y1"] - y_off) / scale))
        b["x2"] = min(orig_w, round((b["x2"] - x_off) / scale))
        b["y2"] = min(orig_h, round((b["y2"] - y_off) / scale))
        d["bbox"] = b
        result.append(d)
    return result


def save_annotated(
    original_path: str,
    detections: List[Dict[str, Any]],
    output_path: str,
    preproc_w: int = 640,
    preproc_h: int = 640,
) -> None:
    """원본 이미지를 로드해 어노테이션 후 JPEG로 저장."""
    image = cv2.imread(str(original_path))
    if image is None:
        logger.warning("Failed to load image: %s", original_path)
        return

    orig_h, orig_w = image.shape[:2]
    scaled = _inverse_letterbox(detections, orig_w, orig_h, preproc_w, preproc_h)
    annotated = draw_detections(image, scaled)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])


def generate_summary_plot(predictions_dir: str, output_path: str) -> None:
    """클래스별 탐지 수 가로 막대 차트를 PNG로 저장."""
    pred_dir = Path(predictions_dir)
    counter: Counter = Counter()

    for json_file in sorted(pred_dir.glob("*.json")):
        if json_file.stem == "summary":
            continue
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read %s: %s", json_file, e)
            continue
        for det in data.get("detections", []):
            counter[det.get("class_name", "unknown")] += 1

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * max(len(counter), 1))))
    if counter:
        items = sorted(counter.items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]
        counts = [v for _, v in items]
        ax.barh(names, counts, color="#3b82f6")
        ax.set_xlabel("Detection count")
        ax.set_ylabel("Class")
        ax.set_title("Detections per class")
        for i, v in enumerate(counts):
            ax.text(v, i, f" {v}", va="center")
    else:
        ax.set_title("No detections found")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(str(out), dpi=120)
    plt.close(fig)


def process_all(predictions_dir: str, input_dir: str, output_dir: str) -> int:
    """predictions JSON을 순회하여 어노테이션 이미지 + 요약 차트 생성. 처리 파일 수 반환."""
    pred_dir = Path(predictions_dir)
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for json_file in sorted(pred_dir.glob("*.json")):
        if json_file.stem == "summary":
            continue
        try:
            with json_file.open("r", encoding="utf-8") as f:
                entry = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read %s: %s", json_file, e)
            continue

        filename = entry.get("filename") or f"{json_file.stem}.jpg"
        original_path = in_dir / filename
        if not original_path.exists():
            logger.warning("Original image not found, skipping: %s", original_path)
            continue

        detections = entry.get("detections", [])
        preproc_w = int(entry.get("image_width", 640))
        preproc_h = int(entry.get("image_height", 640))
        output_path = out_dir / f"{json_file.stem}_annotated.jpg"
        save_annotated(str(original_path), detections, str(output_path), preproc_w, preproc_h)
        processed += 1

    summary_path = out_dir / "detection_summary.png"
    generate_summary_plot(str(pred_dir), str(summary_path))

    logger.info("Visualized %d images, summary plot at %s", processed, summary_path)
    return processed
