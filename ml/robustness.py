"""Robustness experiment: clean vs degraded vs degraded+Stage-1 preprocess.

Uses the official YOLO val split so labels stay aligned. Photometric/blur/noise
degradations do not change box geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import default_device, load_config, resolve
from ml.inference import default_weights_path
from video.degradation import simulate_blur, simulate_low_light, simulate_noise
from video.preprocess import process_frame

CONDITIONS = {
    "clean": None,
    "low_light": simulate_low_light,
    "blur": simulate_blur,
    "noise": simulate_noise,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector robustness under surveillance degradations")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="optional cap on images (0 = all)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    return parser.parse_args()


def _copy_condition(src_images: Path, src_labels: Path, dest_root: Path, name: str, fn, preprocess: bool, limit: int) -> int:
    img_out = dest_root / name / "images"
    lbl_out = dest_root / name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    files = sorted(src_images.glob("*.jpg"))
    if limit:
        files = files[:limit]
    count = 0
    for img_path in tqdm(files, desc=name):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        if fn is not None:
            frame = fn(frame)
        if preprocess:
            frame = process_frame(frame, mode="auto")
        cv2.imwrite(str(img_out / img_path.name), frame)
        label = src_labels / (img_path.stem + ".txt")
        if label.exists():
            shutil.copy2(label, lbl_out / label.name)
        else:
            (lbl_out / (img_path.stem + ".txt")).write_text("", encoding="utf-8")
        count += 1
    return count


def _write_yaml(dest: Path, condition: str) -> Path:
    yaml_path = dest / f"{condition}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dest.as_posix()}",
                f"train: {condition}/images",
                f"val: {condition}/images",
                "names:",
                "  0: person",
                "  1: knife",
                "  2: gun",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def _metrics(results) -> dict:
    box = results.box
    speed = results.speed or {}
    inf_ms = float(speed.get("inference", 0.0) or 0.0)
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "latency_ms": inf_ms,
        "fps": (1000.0 / inf_ms) if inf_ms > 0 else None,
    }


def run(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    cfg = load_config()
    tuned_weights = resolve(cfg["weights_out"])
    weights = Path(args.weights) if args.weights else tuned_weights
    if not weights.exists():
        raise FileNotFoundError(
            "Fine-tuned weights are required for robustness metrics. Train first so models/best.pt exists."
        )
    data_root = resolve(cfg["dataset_dir"])
    src_images = data_root / "images" / args.split
    src_labels = data_root / "labels" / args.split
    if not src_images.exists():
        raise FileNotFoundError(f"Missing {src_images}. Run python -m ml.prepare_dataset first.")

    dest = ROOT / "data" / "robustness" / args.split
    rows = []
    jobs = []
    for condition, fn in CONDITIONS.items():
        jobs.append((condition, fn, False))
        if condition != "clean":
            jobs.append((f"{condition}+preprocess", fn, True))
    jobs.insert(1, ("clean+preprocess", None, True))

    for name, fn, prep in jobs:
        n = _copy_condition(src_images, src_labels, dest, name, fn, prep, args.limit)
        yaml_path = _write_yaml(dest, name)
        rows.append({"condition": name, "n_images": n, "yaml": str(yaml_path)})

    model = YOLO(str(weights) if Path(str(weights)).exists() else str(weights))
    imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]
    device = args.device if args.device is not None else default_device(cfg.get("device", "0"))

    table = []
    for row in rows:
        results = model.val(
            data=row["yaml"],
            split="val",
            imgsz=imgsz,
            device=device,
            plots=False,
            verbose=False,
            project=str(ROOT / "runs" / "detect"),
            name=f"robust_{row['condition']}",
            exist_ok=True,
        )
        metrics = _metrics(results)
        entry = {"condition": row["condition"], "n_images": row["n_images"], **metrics}
        table.append(entry)
        print(entry)

    payload = {
        "weights": str(weights),
        "split": args.split,
        "limit": args.limit or "all",
        "notes": (
            "Degradations are geometry-preserving (brightness, blur, noise) so the original "
            "YOLO labels remain valid. Preprocess uses video.process_frame(mode='auto') from Stage 1."
        ),
        "results": table,
    }
    out_json = ROOT / "evaluation" / "robustness_metrics.json"
    out_csv = ROOT / "evaluation" / "robustness_metrics.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "n_images", "precision", "recall", "mAP50", "mAP50-95", "latency_ms", "fps"],
        )
        writer.writeheader()
        for entry in table:
            writer.writerow(entry)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    return payload


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
