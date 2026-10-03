"""Headless check that quality, degradation, and preprocess stay consistent and fast."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video.degradation import DEGRADATIONS, apply_degradation
from video.preprocess import process_frame
from video.quality import analyze_quality


def make_scene() -> np.ndarray:
    h, w = 720, 1280
    img = np.full((h, w, 3), 165, dtype=np.uint8)
    cv2.rectangle(img, (120, 100), (480, 560), (45, 85, 210), -1)
    cv2.circle(img, (900, 300), 110, (25, 170, 70), -1)
    for i in range(12):
        x = 80 + i * 90
        cv2.line(img, (x, 40), (x + 40, 680), (15, 15, 15), 3)
    return img


def main() -> int:
    source = make_scene()
    out_dir = ROOT / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== SentinelAI pipeline verification ===")
    base_q = analyze_quality(source)
    print(f"clean quality: {base_q}")

    times_ms: list[float] = []
    process_frame(source, mode="auto")  # warmup OpenCV kernels / CLAHE
    for name in ["none", *sorted(DEGRADATIONS)]:
        degraded = apply_degradation(source, name)
        t0 = time.perf_counter()
        processed = process_frame(degraded, mode="auto")
        elapsed = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed)
        dq = analyze_quality(degraded)
        pq = analyze_quality(processed)
        assert processed.ndim == 3 and processed.shape[2] == 3
        assert processed.dtype == np.uint8
        pair = np.hstack(
            (
                cv2.resize(degraded, (640, 360)),
                cv2.resize(processed, (640, 360)),
            )
        )
        path = out_dir / f"verify_{name}.jpg"
        cv2.imwrite(str(path), pair)
        print(
            f"{name:14}  {elapsed:6.2f} ms  "
            f"in_flags={dq['flags']}  out_bright={pq['brightness']}"
        )

    mean_ms = sum(times_ms) / len(times_ms)
    print(f"mean preprocess latency: {mean_ms:.2f} ms")
    if mean_ms > 80:
        print("WARNING: preprocessing is slower than the ~real-time budget on this machine.")
    else:
        print("latency within a near-real-time budget for 720p on this machine.")

    # Low-light should actually get brighter after auto mode.
    dark = apply_degradation(source, "low_light")
    recovered = process_frame(dark, mode="auto")
    if analyze_quality(recovered)["brightness"] <= analyze_quality(dark)["brightness"]:
        print("WARNING: low-light correction did not raise mean brightness.")
        return 1

    print("verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
