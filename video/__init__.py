"""Video robustness layer for SentinelAI.

This package is the first stage of the pipeline:
messy surveillance frame -> quality analysis + cheap preprocessing -> cleaner frame for ML.
"""

from video.quality import analyze_quality
from video.preprocess import process_frame
from video.degradation import (
    simulate_low_light,
    simulate_noise,
    simulate_motion_blur,
    simulate_blur,
    simulate_compression,
    simulate_occlusion,
    apply_degradation,
)

__all__ = [
    "analyze_quality",
    "process_frame",
    "simulate_low_light",
    "simulate_noise",
    "simulate_motion_blur",
    "simulate_blur",
    "simulate_compression",
    "simulate_occlusion",
    "apply_degradation",
]
