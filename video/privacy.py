"""Fast, local face detection and anonymization for the live display."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
YUNET_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"


@lru_cache(maxsize=1)
def _yunet():
    """OpenCV Zoo YuNet is compact and works in OpenCV builds without Haar support."""
    if not YUNET_MODEL.exists():
        raise FileNotFoundError(
            f"Face model missing: {YUNET_MODEL}. Download the documented OpenCV Zoo YuNet model."
        )
    return cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), 0.75, 0.3, 5000)


def detect_faces(
    frame: np.ndarray, scale_factor: float = 1.12, min_neighbors: int = 5, min_size: int = 28
) -> list[list[int]]:
    """Return face boxes as integer ``[x1, y1, x2, y2]`` in a BGR frame."""
    if frame is None or frame.size == 0:
        raise ValueError("detect_faces requires a non-empty frame")
    height, width = frame.shape[:2]
    detector = _yunet()
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    output: list[list[int]] = []
    for face in faces:
        x, y, w, h = (int(v) for v in face[:4])
        if w >= min_size and h >= min_size:
            output.append([max(0, x), max(0, y), min(width, x + w), min(height, y + h)])
    return output


def blur_faces(frame: np.ndarray, faces: list[list[int]], kernel_ratio: float = 0.45) -> np.ndarray:
    """Copy ``frame`` and strongly blur each face ROI; never modifies the input."""
    if frame is None or frame.size == 0:
        raise ValueError("blur_faces requires a non-empty frame")
    out = frame.copy()
    height, width = out.shape[:2]
    for x1, y1, x2, y2 in faces:
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        size = max(3, int(min(roi.shape[:2]) * kernel_ratio) | 1)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (size, size), 0)
    return out
