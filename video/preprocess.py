"""Cheap surveillance-frame preprocessing for near-real-time inference.

Target interface:
    process_frame(frame, mode="auto") -> processed_frame

The pipeline is intentionally small: resize, optional low-light/CLAHE,
light denoise, optional unsharp. It is not a restoration research stack.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from video.quality import analyze_quality

DEFAULT_MAX_WIDTH = 1280
_CLAHE_CACHE: dict[tuple[float, tuple[int, int]], cv2.CLAHE] = {}
_GAMMA_LUTS: dict[float, np.ndarray] = {}


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("process_frame requires a non-empty frame")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def resize_frame(frame: np.ndarray, max_width: int = DEFAULT_MAX_WIDTH) -> np.ndarray:
    """Downscale wide frames so later ML stays near real time. Upscaling is skipped."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, int(round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    """Lift dark pixels with a LUT gamma curve. gamma > 1 brightens."""
    if abs(gamma - 1.0) < 1e-3:
        return frame
    key = round(float(gamma), 3)
    table = _GAMMA_LUTS.get(key)
    if table is None:
        inv = 1.0 / max(gamma, 1e-6)
        table = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)]).astype("uint8")
        _GAMMA_LUTS[key] = table
    return cv2.LUT(frame, table)


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Local contrast on L in LAB — cheap and effective for CCTV low light."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    cache_key = (round(float(clip_limit), 2), (8, 8))
    clahe = _CLAHE_CACHE.get(cache_key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        _CLAHE_CACHE[cache_key] = clahe
    l_eq = clahe.apply(l_channel)
    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(frame: np.ndarray) -> np.ndarray:
    """Mild spatial denoise. Gaussian is much faster than NL-means for live video."""
    return cv2.GaussianBlur(frame, (3, 3), 0.8)


def sharpen(frame: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """Unsharp mask. Keep amount modest so grain is not amplified."""
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


def _gamma_for_brightness(brightness: float) -> float:
    if brightness < 25:
        return 1.85
    if brightness < 40:
        return 1.55
    if brightness < 55:
        return 1.3
    return 1.0


def process_frame(
    frame: np.ndarray,
    mode: str = "auto",
    max_width: int = DEFAULT_MAX_WIDTH,
    quality: Optional[dict] = None,
) -> np.ndarray:
    """Preprocess one BGR frame for a downstream detector.

    Modes:
        auto       — inspect quality and apply only needed ops
        low_light  — gamma + CLAHE
        denoise    — mild Gaussian
        sharpen    — unsharp mask (blur reduction)
        enhance    — CLAHE + light denoise + mild sharpen
        none       — resize only
    """
    out = _ensure_bgr(frame)
    out = resize_frame(out, max_width=max_width)

    mode = (mode or "auto").lower()
    if mode == "none":
        return out

    flags: set[str] = set()
    brightness = 128.0
    if mode == "auto":
        quality = quality or analyze_quality(out)
        flags = set(quality.get("flags", []))
        brightness = float(quality.get("brightness", 128.0))
    elif quality:
        flags = set(quality.get("flags", []))
        brightness = float(quality.get("brightness", 128.0))

    want_low_light = mode in ("low_light", "enhance") or (mode == "auto" and "low_light" in flags)
    want_denoise = mode in ("denoise", "enhance") or (mode == "auto" and "noisy" in flags)
    want_sharpen = mode in ("sharpen", "enhance") or (
        mode == "auto" and "blurry" in flags and "noisy" not in flags
    )

    if want_low_light:
        out = apply_gamma(out, _gamma_for_brightness(brightness if mode == "auto" else 35.0))

    # Denoise before CLAHE/sharpen so local contrast does not amplify grain.
    if want_denoise:
        out = denoise(out)

    if want_low_light:
        out = apply_clahe(out, clip_limit=2.2 if "low_light" in flags or mode == "low_light" else 1.8)

    if want_sharpen:
        out = sharpen(out, amount=0.45 if mode == "auto" else 0.6)

    return out


def warmup() -> None:
    """Prime CLAHE and LUTs so the first live frame is not a stall."""
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    apply_gamma(dummy, 1.55)
    apply_clahe(dummy, clip_limit=1.8)
    apply_clahe(dummy, clip_limit=2.2)


warmup()
