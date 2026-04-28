"""YOLOv8 객체 탐지 파이프라인 진입점.

파이프라인 순서: preprocess → inference → visualize
"""

import argparse
import logging
from pathlib import Path

import cv2

from src.preprocess import load_input, preprocess_image, save_manifest


PREPROCESSED_DIR = Path("_workspace/01_preprocessed")


def run_preprocess(input_dir: str) -> Path:
    """입력 디렉토리의 파일을 전처리하고 manifest.json을 생성한다."""
    items = load_input(input_dir)

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for filename, bgr_image in items:
        processed = preprocess_image(bgr_image)
        out_path = PREPROCESSED_DIR / filename
        # ultralytics 호환을 위해 BGR로 다시 변환하여 저장한다.
        bgr_out = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr_out)

        source_type = "video_frame" if "_frame" in Path(filename).stem else "image"
        manifest_entries.append({
            "filename": filename,
            "original_path": str(Path(input_dir) / filename) if source_type == "image" else input_dir,
            "preprocessed_path": str(out_path),
            "width": processed.shape[1],
            "height": processed.shape[0],
            "source_type": source_type,
        })

    save_manifest(manifest_entries, str(PREPROCESSED_DIR))
    logging.info("Preprocessed %d items → %s", len(manifest_entries), PREPROCESSED_DIR)
    return PREPROCESSED_DIR / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Pipeline")
    parser.add_argument("--input",  default="./images",   help="Input directory")
    parser.add_argument("--output", default="./results",  help="Output directory")
    parser.add_argument("--model",  default="yolov8n.pt", help="YOLOv8 model weights")
    parser.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou",    type=float, default=0.45, help="IOU threshold")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 1) 전처리
    manifest_path = run_preprocess(args.input)

    # 2) 추론
    from src.inference import load_model, run_batch_inference
    model = load_model(args.model)
    predictions_dir = run_batch_inference(
        model,
        str(manifest_path),
        "_workspace/02_predictions",
        conf=args.conf,
        iou=args.iou,
    )

    # 3) 시각화
    from src.visualize import process_all
    process_all(str(predictions_dir), args.input, args.output)

    logging.info("Pipeline complete. Results saved to %s", args.output)


if __name__ == "__main__":
    main()
