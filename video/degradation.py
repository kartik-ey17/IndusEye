"""Deliberate frame degradations for robustness evaluation.

These functions make a reasonably clean frame worse so we can later
compare detector behavior with and without `process_frame`.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

DegradeFn = Callable[[np.ndarray], np.ndarray]


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("degradation requires a non-empty frame")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def simulate_low_light(frame: np.ndarray, gain: float = 0.28, bias: int = -18) -> np.ndarray:
    """Darken the frame to mimic poorly lit CCTV."""
    img = _as_bgr(frame).astype(np.float32)
    img = img * float(gain) + float(bias)
    return np.clip(img, 0, 255).astype(np.uint8)


def simulate_noise(frame: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    """Additive Gaussian sensor noise."""
    img = _as_bgr(frame)
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def simulate_motion_blur(frame: np.ndarray, kernel_size: int = 15, angle_deg: float = 12.0) -> np.ndarray:
    """1D kernel motion blur (camera shake / subject motion)."""
    img = _as_bgr(frame)
    k = max(3, int(kernel_size) | 1)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    matrix = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def simulate_blur(frame: np.ndarray, ksize: int = 11) -> np.ndarray:
    """Defocus-style Gaussian blur."""
    img = _as_bgr(frame)
    k = max(3, int(ksize) | 1)
    return cv2.GaussianBlur(img, (k, k), 0)


def simulate_compression(frame: np.ndarray, quality: int = 12) -> np.ndarray:
    """JPEG quantization artifacts from aggressive compression."""
    img = _as_bgr(frame)
    quality = int(np.clip(quality, 1, 95))
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def simulate_occlusion(frame: np.ndarray, coverage: float = 0.18) -> np.ndarray:
    """Opaque rectangle covering part of the frame (object / dirt / overlay)."""
    img = _as_bgr(frame).copy()
    h, w = img.shape[:2]
    coverage = float(np.clip(coverage, 0.05, 0.6))
    box_w = max(8, int(w * coverage))
    box_h = max(8, int(h * coverage * 0.9))
    x = int(w * 0.55)
    y = int(h * 0.15)
    x2 = min(w, x + box_w)
    y2 = min(h, y + box_h)
    img[y:y2, x:x2] = (20, 20, 20)
    return img


DEGRADATIONS: dict[str, DegradeFn] = {
    "low_light": simulate_low_light,
    "noise": simulate_noise,
    "motion_blur": simulate_motion_blur,
    "blur": simulate_blur,
    "compression": simulate_compression,
    "occlusion": simulate_occlusion,
}


def apply_degradation(frame: np.ndarray, name: str) -> np.ndarray:
    """Apply a named degradation, or return the frame unchanged for 'none'."""
    if name in (None, "", "none"):
        return frame
    if name not in DEGRADATIONS:
        available = ", ".join(sorted(DEGRADATIONS))
        raise ValueError(f"Unknown degradation '{name}'. Choose from: none, {available}")
    return DEGRADATIONS[name](frame)
