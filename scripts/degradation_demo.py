"""Show each degradation next to the preprocessed recovery on a still or webcam frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video.degradation import DEGRADATIONS, apply_degradation
from video.preprocess import process_frame
from video.quality import analyze_quality


def synthetic_scene(width: int = 960, height: int = 540) -> np.ndarray:
    img = np.full((height, width, 3), 170, dtype=np.uint8)
    cv2.rectangle(img, (80, 70), (320, 420), (40, 90, 210), -1)
    cv2.rectangle(img, (380, 180), (620, 460), (30, 160, 90), -1)
    cv2.circle(img, (780, 220), 80, (20, 40, 200), -1)
    cv2.putText(img, "SentinelAI", (360, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    return img


def tile_comparisons(source: np.ndarray) -> np.ndarray:
    cells = []
    names = ["none", *sorted(DEGRADATIONS)]
    for name in names:
        degraded = apply_degradation(source, name)
        processed = process_frame(degraded, mode="auto")
        h = 240
        scale = h / degraded.shape[0]
        w = int(degraded.shape[1] * scale)
        left = cv2.resize(degraded, (w, h))
        right = cv2.resize(processed, (w, h))
        pair = np.hstack((left, right))
        q = analyze_quality(degraded)
        flags = ",".join(q["flags"]) or "ok"
        cv2.putText(pair, f"{name} | {flags}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(pair, f"{name} | {flags}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cells.append(pair)

    width = max(c.shape[1] for c in cells)
    padded = []
    for cell in cells:
        if cell.shape[1] < width:
            pad = np.zeros((cell.shape[0], width - cell.shape[1], 3), dtype=np.uint8)
            cell = np.hstack((cell, pad))
        padded.append(cell)
    return np.vstack(padded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentinelAI degradation demo")
    parser.add_argument("--image", type=str, default="", help="optional still image path")
    parser.add_argument("--save", type=str, default="", help="write mosaic to this path")
    parser.add_argument("--show", action="store_true", help="open an OpenCV window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image:
        source = cv2.imread(args.image)
        if source is None:
            print(f"Could not read image: {args.image}")
            return 1
    else:
        source = synthetic_scene()

    mosaic = tile_comparisons(source)
    out_path = Path(args.save) if args.save else ROOT / "data" / "samples" / "degradation_mosaic.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mosaic)
    print(f"Wrote {out_path}  shape={mosaic.shape}  (left=degraded, right=processed)")

    if args.show:
        cv2.imshow("SentinelAI degradation demo", mosaic)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
