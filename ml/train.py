"""Fine-tune YOLO11n on the SentinelAI 3-class detection set."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import default_device, load_config, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentinelAI YOLO11 detector")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help="0, cpu, or empty for auto")
    parser.add_argument("--model", default=None, help="base checkpoint, default yolo11n.pt")
    parser.add_argument("--data", default=None, help="dataset yaml")
    parser.add_argument("--name", default="sentinel_yolo11n")
    return parser.parse_args()


def train(args: argparse.Namespace):
    from ultralytics import YOLO

    cfg = load_config()
    data = args.data or str(resolve(cfg["dataset_yaml"]))
    if not Path(data).exists():
        raise FileNotFoundError(f"Dataset yaml missing: {data}. Run python -m ml.prepare_dataset first.")

    model_name = args.model or cfg["model"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch = args.batch if args.batch is not None else cfg["batch"]
    imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]
    device = args.device if args.device is not None else default_device(cfg.get("device", "0"))

    print(f"Transfer learning from {model_name} -> 3 classes (person, knife, gun)")
    model = YOLO(model_name)
    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=cfg.get("workers", 2),
        patience=cfg.get("patience", 8),
        seed=cfg.get("seed", 42),
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.01),
        weight_decay=cfg.get("weight_decay", 0.0005),
        warmup_epochs=cfg.get("warmup_epochs", 3.0),
        close_mosaic=cfg.get("close_mosaic", 8),
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.7),
        hsv_v=cfg.get("hsv_v", 0.55),
        degrees=cfg.get("degrees", 8.0),
        translate=cfg.get("translate", 0.12),
        scale=cfg.get("scale", 0.55),
        shear=cfg.get("shear", 2.0),
        perspective=cfg.get("perspective", 0.0),
        flipud=cfg.get("flipud", 0.0),
        fliplr=cfg.get("fliplr", 0.5),
        mosaic=cfg.get("mosaic", 0.85),
        mixup=cfg.get("mixup", 0.1),
        copy_paste=cfg.get("copy_paste", 0.05),
        erasing=cfg.get("erasing", 0.35),
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        plots=True,
        val=True,
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    dest = resolve(cfg["weights_out"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, dest)
        last = run_dir / "weights" / "last.pt"
        if last.exists():
            shutil.copy2(last, dest.with_name("last.pt"))
        print(f"Copied {best} -> {dest}")
    else:
        print(f"WARNING: {best} was not written")
    return dest


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
