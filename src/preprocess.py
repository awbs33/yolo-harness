"""YOLOv8 입력 전처리 모듈.

이미지/영상 파일 로드, BGR→RGB 변환, 640×640 letterbox 리사이즈,
영상 프레임 추출, manifest.json 생성을 담당한다.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov"}
SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS


def load_input(input_path: str) -> List[Tuple[str, np.ndarray]]:
    """입력 경로의 모든 지원 파일을 로드하여 (filename, bgr_image) 리스트를 반환한다.

    영상 파일은 preprocess_video()로 처리하여 프레임 단위로 확장한다.
    지원하지 않는 형식은 경고 로그 후 건너뛴다.
    """
    path = Path(input_path)
    if not path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    items: List[Tuple[str, np.ndarray]] = []

    if path.is_file():
        candidates = [path]
    else:
        candidates = sorted(p for p in path.iterdir() if p.is_file())

    for fp in candidates:
        ext = fp.suffix.lower()
        if ext in IMAGE_EXTS:
            image = cv2.imread(str(fp))
            if image is None:
                logger.warning("Failed to read image: %s", fp)
                continue
            items.append((fp.name, image))
        elif ext in VIDEO_EXTS:
            frames = preprocess_video(str(fp))
            items.extend(frames)
        else:
            logger.warning("Unsupported file format, skipping: %s", fp)

    if not items:
        raise ValueError(f"No supported files found in {input_path}")

    return items


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """BGR → RGB 변환 후 letterbox 방식으로 target_size에 맞춰 리사이즈한다.

    종횡비를 유지하기 위해 빈 공간은 회색(128, 128, 128)으로 패딩한다.
    dtype은 uint8을 유지한다 (ultralytics가 내부 정규화 처리).
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(rgb, (new_w, new_h))
    canvas = np.full((target_size[0], target_size[1], 3), 128, dtype=np.uint8)
    y_off = (target_size[0] - new_h) // 2
    x_off = (target_size[1] - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def preprocess_video(
    video_path: str,
    frame_interval: int = 1,
) -> List[Tuple[str, np.ndarray]]:
    """cv2.VideoCapture로 영상에서 프레임을 추출한다.

    frame_interval 프레임마다 1장씩 추출 (기본: 모든 프레임).
    파일명 컨벤션: {video_stem}_frame{idx:06d}.jpg
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Failed to open video: %s", video_path)
        return []

    stem = Path(video_path).stem
    frames: List[Tuple[str, np.ndarray]] = []
    idx = 0
    extracted = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                fname = f"{stem}_frame{extracted:06d}.jpg"
                frames.append((fname, frame))
                extracted += 1
            idx += 1
    finally:
        cap.release()

    return frames


def save_manifest(items: List[dict], output_dir: str) -> None:
    """전처리 결과 메타데이터를 manifest.json으로 저장한다."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
