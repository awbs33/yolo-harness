"""YOLOv8 객체 탐지 추론 모듈."""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics가 설치되지 않았습니다. 실행: pip install ultralytics"
    )

logger = logging.getLogger(__name__)


def load_model(model_path: str = "yolov8n.pt") -> "YOLO":
    """YOLO 모델 로드. 실패 시 RuntimeError."""
    start = time.perf_counter()
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Model load failed: {e}. Install ultralytics: pip install ultralytics"
        ) from e
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info("Loaded model %s in %.1f ms", model_path, elapsed_ms)
    return model


def process_results(results, filename: str, model_name: str) -> Dict[str, Any]:
    """ultralytics Results를 표준 JSON 스키마로 변환."""
    if not results:
        return {
            "filename": filename,
            "model": model_name,
            "image_width": 0,
            "image_height": 0,
            "detections": [],
            "detection_count": 0,
            "inference_time_ms": 0.0,
        }

    result = results[0]
    names = getattr(result, "names", None) or {}

    image_height, image_width = 0, 0
    if getattr(result, "orig_shape", None) is not None:
        image_height, image_width = int(result.orig_shape[0]), int(result.orig_shape[1])
    elif getattr(result, "orig_img", None) is not None:
        h, w = result.orig_img.shape[:2]
        image_height, image_width = int(h), int(w)

    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy
        confs = boxes.conf
        cls_ids = boxes.cls
        try:
            xyxy = xyxy.cpu().numpy()
            confs = confs.cpu().numpy()
            cls_ids = cls_ids.cpu().numpy()
        except AttributeError:
            pass

        for i in range(len(boxes)):
            class_id = int(cls_ids[i])
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            x1, y1, x2, y2 = xyxy[i]
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confs[i]),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
            })

    return {
        "filename": filename,
        "model": model_name,
        "image_width": image_width,
        "image_height": image_height,
        "detections": detections,
        "detection_count": len(detections),
        "inference_time_ms": 0.0,
    }


def run_inference(
    model: "YOLO",
    image_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """단일 이미지 추론 + 시간 측정."""
    image_path_obj = Path(image_path)
    model_name = getattr(model, "model_name", None) or getattr(
        model, "ckpt_path", None
    ) or "yolov8n.pt"
    if isinstance(model_name, Path):
        model_name = model_name.name
    elif isinstance(model_name, str):
        model_name = Path(model_name).name

    start = time.perf_counter()
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    inference_time_ms = (time.perf_counter() - start) * 1000.0

    output = process_results(results, image_path_obj.name, model_name)
    output["inference_time_ms"] = round(inference_time_ms, 2)
    return output


def run_batch_inference(
    model: "YOLO",
    manifest_path: str,
    output_dir: str,
    conf: float = 0.25,
    iou: float = 0.45,
) -> str:
    """manifest.json을 읽어 배치 추론 + summary.json 생성."""
    manifest_file = Path(manifest_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if isinstance(manifest, dict) and "images" in manifest:
        entries = manifest["images"]
    elif isinstance(manifest, list):
        entries = manifest
    else:
        entries = []

    total_detections = 0
    inference_times = []
    class_counter: Counter = Counter()
    processed = 0

    for entry in entries:
        if isinstance(entry, dict):
            img_path = (
                entry.get("preprocessed_path")
                or entry.get("output_path")
                or entry.get("path")
                or entry.get("image_path")
                or entry.get("filename")
            )
        else:
            img_path = entry

        if not img_path:
            logger.warning("Skipping entry without path: %s", entry)
            continue

        img_path_obj = Path(img_path)
        if not img_path_obj.is_absolute():
            candidate = manifest_file.parent / img_path_obj
            if candidate.exists():
                img_path_obj = candidate

        if not img_path_obj.exists():
            logger.warning("Image not found, skipping: %s", img_path_obj)
            continue

        try:
            result = run_inference(
                model,
                str(img_path_obj),
                conf_threshold=conf,
                iou_threshold=iou,
            )
        except Exception as e:
            logger.warning("Inference failed for %s: %s", img_path_obj, e)
            continue

        stem = img_path_obj.stem
        out_file = output_path / f"{stem}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        total_detections += result["detection_count"]
        inference_times.append(result["inference_time_ms"])
        for det in result["detections"]:
            class_counter[det["class_name"]] += 1
        processed += 1

    avg_ms = (sum(inference_times) / len(inference_times)) if inference_times else 0.0
    summary = {
        "total_images": processed,
        "total_detections": total_detections,
        "avg_inference_ms": round(avg_ms, 2),
        "class_counts": dict(class_counter),
    }
    summary_file = output_path / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "Batch inference done: %d images, %d detections, avg %.2f ms",
        processed,
        total_detections,
        avg_ms,
    )
    return str(output_path)
