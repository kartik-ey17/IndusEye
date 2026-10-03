"""Annotate an image, video, or webcam with SentinelAI detections.

Examples:
  python scripts/test_video.py --source 0
  python scripts/test_video.py --source path/to/clip.mp4
  python scripts/test_video.py --source path/to/image.jpg --preprocess
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incidents.pipeline import IncidentProcessor, process_frame as process_sentinel_frame
from ml.inference import Detection, model_info

COLORS = {
    "person": (80, 180, 80),
    "knife": (0, 165, 255),
    "gun": (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentinelAI detector video/image test")
    parser.add_argument("--source", default="0", help="webcam index, image path, or video path")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--preprocess", action="store_true", help="run Stage 1 process_frame first")
    parser.add_argument("--camera-id", default="CAM-01", help="identifier stored with any incident")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = all / until q)")
    parser.add_argument("--save", default="", help="optional output video/image path")
    return parser.parse_args()


def draw(frame, detections: list[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = COLORS.get(det.class_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def open_source(source: str) -> tuple[str, object]:
    path = Path(source)
    if path.exists() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image {path}")
        return "image", image
    if path.exists():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video {path}")
        return "video", cap
    try:
        index = int(source)
    except ValueError as exc:
        raise FileNotFoundError(f"Unknown source {source}") from exc
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {index}")
    return "webcam", cap


def main() -> int:
    args = parse_args()
    info = model_info(args.weights)
    print(json.dumps(info, indent=2))

    kind, handle = open_source(args.source)
    writer = None
    processed = 0
    last = time.perf_counter()
    fps = 0.0
    processor = IncidentProcessor()

    def handle_frame(frame):
        nonlocal processed, last, fps, writer
        t0 = time.perf_counter()
        result = process_sentinel_frame(
            frame,
            {
                "camera_id": args.camera_id,
                "preprocess": args.preprocess,
                "detection_confidence": args.conf,
                "weights": args.weights,
            },
            processor=processor,
        )
        frame = result["display_frame"]
        dets = result["detections"]
        infer_ms = (time.perf_counter() - t0) * 1000.0
        incident = result["incident"]
        now = time.perf_counter()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
        cv2.putText(
            frame,
            f"FPS {fps:.1f}  total {infer_ms:.1f}ms  n={len(dets)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            3,
        )
        cv2.putText(
            frame,
            f"FPS {fps:.1f}  total {infer_ms:.1f}ms  n={len(dets)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            1,
        )
        processed += 1
        if args.save:
            out_path = Path(args.save)
            if kind == "image":
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), frame)
            else:
                if writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                writer.write(frame)
        if incident and incident["status"] == "created":
            print("INCIDENT", json.dumps(incident))
        return frame, dets

    try:
        if kind == "image":
            frame, dets = handle_frame(handle)
            print(json.dumps(dets, indent=2))
            if not args.headless:
                cv2.imshow("SentinelAI detector", frame)
                cv2.waitKey(0)
            return 0

        while True:
            ok, frame = handle.read()
            if not ok or frame is None:
                break
            frame, dets = handle_frame(frame)
            if not args.headless:
                cv2.imshow("SentinelAI detector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if args.frames and processed >= args.frames:
                break
    finally:
        if kind != "image":
            handle.release()
        if writer is not None:
            writer.release()
        # opencv-python-headless intentionally has no HighGUI backend.
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"processed_frames={processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
