"""Synthetic-frame latency check for the preprocessing path."""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np

from video.preprocess import process_frame
from video.quality import analyze_quality


def synthetic_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, size=(height, width, 3), dtype=np.uint8)
    cv2.rectangle(base, (200, 150), (500, 480), (30, 80, 200), -1)
    cv2.circle(base, (900, 360), 90, (20, 180, 40), -1)
    return base


def measure_latency(iterations: int = 40, warmup: int = 5) -> dict:
    frame = synthetic_frame()
    for _ in range(warmup):
        process_frame(frame, mode="auto")

    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        processed = process_frame(frame, mode="auto")
        times_ms.append((time.perf_counter() - start) * 1000.0)

    quality = analyze_quality(processed)
    return {
        "iterations": iterations,
        "mean_ms": round(statistics.mean(times_ms), 2),
        "p95_ms": round(statistics.quantiles(times_ms, n=20)[18], 2)
        if len(times_ms) >= 20
        else round(max(times_ms), 2),
        "max_ms": round(max(times_ms), 2),
        "output_shape": list(processed.shape),
        "quality": quality,
    }


if __name__ == "__main__":
    result = measure_latency()
    print("Preprocess latency (1280x720 synthetic frame):")
    print(f"  mean {result['mean_ms']} ms | p95 {result['p95_ms']} ms | max {result['max_ms']} ms")
    print(f"  output {result['output_shape']} | quality {result['quality']}")
