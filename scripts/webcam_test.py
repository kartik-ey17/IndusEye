"""Live webcam test: original vs preprocessed side by side.

Keys:
  q / ESC  quit
  d        cycle simulated degradation on the incoming frame
  p        toggle preprocessing
  s        save a snapshot pair under data/samples/
"""

from __future__ import annotations

import argparse
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

DEGRADE_CYCLE = ["none", *sorted(DEGRADATIONS.keys())]


def _label(image: np.ndarray, lines: list[str], origin=(10, 28)) -> None:
    y = origin[1]
    for line in lines:
        cv2.putText(
            image,
            line,
            (origin[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (origin[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 24


def compose_pair(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    h = min(original.shape[0], processed.shape[0])
    w = min(original.shape[1], processed.shape[1])
    left = cv2.resize(original, (w, h))
    right = cv2.resize(processed, (w, h))
    return np.hstack((left, right))


def save_snapshot(original: np.ndarray, processed: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"webcam_{stamp}.jpg"
    cv2.imwrite(str(path), compose_pair(original, processed))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentinelAI webcam preprocessing test")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--mode", default="auto", help="process_frame mode")
    parser.add_argument("--degrade", default="none", help="starting degradation name")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="process N frames without a GUI (for CI / machines without a display)",
    )
    parser.add_argument("--frames", type=int, default=30, help="frame count in headless mode")
    parser.add_argument("--width", type=int, default=1280, help="capture width hint")
    parser.add_argument("--height", type=int, default=720, help="capture height hint")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main() -> int:
    args = parse_args()
    cap = open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        print("Could not open webcam. Check the camera index with --camera.")
        print("Synthetic path: python scripts/verify_pipeline.py")
        return 1

    degrade_name = args.degrade if args.degrade in DEGRADE_CYCLE else "none"
    preprocess_on = True
    fps = 0.0
    last = time.perf_counter()
    processed_count = 0
    latencies_ms: list[float] = []

    print("Webcam open. Keys: q quit | d cycle degradation | p toggle preprocess | s save")
    print(f"Starting degrade={degrade_name} mode={args.mode} headless={args.headless}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read a frame from the webcam.")
                return 1

            degraded = apply_degradation(frame, degrade_name)
            quality = analyze_quality(degraded)

            t0 = time.perf_counter()
            processed = process_frame(degraded, mode=args.mode) if preprocess_on else degraded
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)
            processed_count += 1

            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            if args.headless:
                if processed_count >= args.frames:
                    mean_ms = sum(latencies_ms) / len(latencies_ms)
                    print(f"Processed {processed_count} frames")
                    print(f"Mean preprocess latency: {mean_ms:.2f} ms")
                    print(f"Approx FPS: {fps:.1f}")
                    print(f"Last quality: {quality}")
                    out = save_snapshot(degraded, processed, ROOT / "data" / "samples")
                    print(f"Saved {out}")
                    return 0
                continue

            canvas = compose_pair(degraded, processed)
            flags = ",".join(quality["flags"]) or "ok"
            _label(
                canvas,
                [
                    f"FPS {fps:.1f}  preprocess {latency_ms:.1f} ms  [{'ON' if preprocess_on else 'OFF'}]",
                    f"degrade={degrade_name}  flags={flags}",
                    f"bright={quality['brightness']} blur={quality['blur_score']} noise={quality['noise_score']}",
                    "left=input  right=processed   q=quit d=degrade p=toggle s=save",
                ],
            )
            cv2.imshow("SentinelAI webcam test", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                idx = (DEGRADE_CYCLE.index(degrade_name) + 1) % len(DEGRADE_CYCLE)
                degrade_name = DEGRADE_CYCLE[idx]
                print(f"degradation -> {degrade_name}")
            if key == ord("p"):
                preprocess_on = not preprocess_on
            if key == ord("s"):
                path = save_snapshot(degraded, processed, ROOT / "data" / "samples")
                print(f"saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if latencies_ms:
        print(f"Mean preprocess latency: {sum(latencies_ms) / len(latencies_ms):.2f} ms over {len(latencies_ms)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
