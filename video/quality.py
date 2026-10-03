"""Lightweight frame-quality statistics (no ML).

Scores are cheap OpenCV/NumPy features used to decide preprocessing
and to label evaluation conditions.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Thresholds tuned for typical webcam / 720p–1080p BGR frames.
LOW_LIGHT_BRIGHTNESS = 55.0
BLURRY_LAPLACIAN_VAR = 80.0
NOISY_RESIDUAL_STD = 8.5


def _as_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def analyze_quality(frame: np.ndarray) -> dict[str, Any]:
    """Return brightness, blur, and noise estimates plus human-readable flags.

    - brightness: mean gray intensity in [0, 255]
    - blur_score: Laplacian variance (lower => more blur)
    - noise_score: std of high-frequency residual after a light Gaussian
    - flags: subset of {"low_light", "blurry", "noisy"}
    """
    if frame is None or frame.size == 0:
        raise ValueError("analyze_quality requires a non-empty frame")

    gray = _as_gray(frame)
    brightness = float(np.mean(gray))

    # Variance of Laplacian is a standard no-reference blur metric.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # Residual after mild blur approximates sensor/compression grain.
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - smoothed.astype(np.float32)
    noise_score = float(np.std(residual))

    flags: list[str] = []
    if brightness < LOW_LIGHT_BRIGHTNESS:
        flags.append("low_light")
    if blur_score < BLURRY_LAPLACIAN_VAR:
        flags.append("blurry")
    if noise_score > NOISY_RESIDUAL_STD:
        flags.append("noisy")

    return {
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2),
        "noise_score": round(noise_score, 2),
        "flags": flags,
    }
