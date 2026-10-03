"""Evaluate a trained detector on the official val/test split. Numbers come from Ultralytics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import default_device, load_config, resolve
from ml.inference import default_weights_path, model_info, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SentinelAI detector")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--latency-images", type=int, default=64)
    return parser.parse_args()


def _extract_metrics(results) -> dict:
    box = results.box
    speed = results.speed or {}
    inference_ms = float(speed.get("inference", 0.0) or 0.0)
    fps = (1000.0 / inference_ms) if inference_ms > 0 else None
    per_class = {}
    names = results.names if isinstance(results.names, dict) else {i: n for i, n in enumerate(results.names)}
    maps = getattr(box, "maps", None)
    if maps is not None:
        for idx, value in enumerate(maps):
            per_class[str(names.get(idx, idx))] = {"mAP50-95": float(value)}
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "per_class_mAP50-95": per_class,
        "ultralytics_speed_ms": {k: float(v) for k, v in speed.items()},
        "ultralytics_inference_fps": fps,
    }


def _latency(weights: Path, image_dir: Path, n: int, imgsz: int) -> dict:
    files = sorted(image_dir.glob("*.jpg"))[: max(n, 8)]
    if not files:
        return {"error": f"no jpg files in {image_dir}"}
    frames = []
    for path in files:
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)
    if not frames:
        return {"error": "failed to read sample images"}

    predict(frames[0], weights=weights, imgsz=imgsz)  # warmup
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        predict(frame, weights=weights, imgsz=imgsz)
        times.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(sum(times) / len(times))
    return {
        "n_images": len(times),
        "mean_latency_ms": round(mean_ms, 3),
        "p95_latency_ms": round(float(np.percentile(times, 95)), 3),
        "max_latency_ms": round(float(max(times)), 3),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else None,
        "includes_python_overhead": True,
    }


def evaluate(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    cfg = load_config()
    tuned_weights = resolve(cfg["weights_out"])
    weights = Path(args.weights) if args.weights else tuned_weights
    if not weights.exists():
        raise FileNotFoundError(
            "Fine-tuned weights are required for evaluation. Train first so models/best.pt exists; "
            "COCO fallback metrics would not measure the SentinelAI 3-class detector."
        )
    data = args.data or str(resolve(cfg["dataset_yaml"]))
    imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]
    device = args.device if args.device is not None else default_device(cfg.get("device", "0"))

    model = YOLO(str(weights) if Path(str(weights)).exists() else str(weights))
    results = model.val(
        data=data,
        split=args.split,
        imgsz=imgsz,
        device=device,
        plots=True,
        project=str(ROOT / "runs" / "detect"),
        name=f"eval_{args.split}",
        exist_ok=True,
        verbose=True,
    )
    metrics = _extract_metrics(results)
    split_images = resolve(cfg["dataset_dir"]) / "images" / args.split
    latency = _latency(weights if Path(str(weights)).exists() else Path(str(results.save_dir)), split_images, args.latency_images, imgsz)

    confusion = None
    conf_path = Path(results.save_dir) / "confusion_matrix.png"
    if conf_path.exists():
        confusion = str(conf_path)

    payload = {
        "weights": str(weights),
        "model_info": model_info(weights),
        "split": args.split,
        "data": data,
        "imgsz": imgsz,
        "metrics": metrics,
        "latency": latency,
        "confusion_matrix_png": confusion,
        "ultralytics_save_dir": str(results.save_dir),
    }

    out_json = ROOT / "evaluation" / "model_metrics.json"
    out_csv = ROOT / "evaluation" / "model_metrics.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "precision", "recall", "mAP50", "mAP50-95", "mean_latency_ms", "fps"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": args.split,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "mAP50": metrics["mAP50"],
                "mAP50-95": metrics["mAP50-95"],
                "mean_latency_ms": latency.get("mean_latency_ms"),
                "fps": latency.get("fps"),
            }
        )
    print(json.dumps({k: payload[k] for k in ("weights", "split", "metrics", "latency")}, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    return payload


def main() -> int:
    evaluate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
