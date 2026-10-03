"""Download and convert a public weapon-detection dataset into YOLO format.

Source (Hugging Face parquet, CC BY 4.0):
  Subh775/WeaponDetection
  derivative of Roboflow Universe "weapon-detection" by yolov7test.

Original published split sizes (dataset card):
  train 5871 / validation 1491 / test 2295  (9657 images, 29 raw classes)

This script remaps a subset of those classes to SentinelAI's MVP labels:
  person, knife, gun
and writes only images that contain at least one remapped box.

Run from repo root:
  python -m ml.prepare_dataset
"""

from __future__ import annotations

import json
import random
import sys
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import CLASS_NAMES, load_config, resolve
from video.degradation import simulate_blur, simulate_compression, simulate_low_light, simulate_noise

HF_DATASET = "Subh775/WeaponDetection"
HF_PAGE = "https://huggingface.co/datasets/Subh775/WeaponDetection"
ROBOFLOW_PAGE = "https://universe.roboflow.com/yolov7test-pdxwq/weapon-detection-m7tpo"

PERSON_SRC = {"person", "Person"}
KNIFE_SRC = {"knife", "Knife", "Knife_Deploy", "Knife_Weapon"}
GUN_SRC = {
    "Pistol",
    "pistol",
    "pistols",
    "handgun",
    "Guns",
    "guns",
    "Rifle",
    "rifle",
    "Shotgun",
    "shotgun",
    "Heavy Gun",
    "Long guns",
    "Guns perspective",
    "heavyweapon",
}

SPLIT_MAP = {"train": "train", "validation": "val", "val": "val", "test": "test"}


def _class_list(dataset) -> list[str]:
    feat = dataset["train"].features["objects"]
    inner = feat.feature
    cats = None
    if hasattr(inner, "names"):
        return list(inner.names)
    try:
        cats = inner["category"]
    except Exception:
        cats = None
    if cats is not None and hasattr(cats, "names"):
        return list(cats.names)
    nested = getattr(cats, "feature", None) if cats is not None else None
    if nested is not None and hasattr(nested, "names"):
        return list(nested.names)
    # Fallback: documented order from the dataset card.
    return [
        "weapons",
        "Aggressor",
        "Blood",
        "Guns",
        "Guns perspective",
        "Hand",
        "Heavy Gun",
        "Knife",
        "Knife_Deploy",
        "Knife_Weapon",
        "Long guns",
        "Person",
        "Pistol",
        "Rifle",
        "Shotgun",
        "Stabbing",
        "Victim",
        "al",
        "guns",
        "handgun",
        "heavyweapon",
        "larga",
        "person",
        "pistol",
        "pistols",
        "rifle",
        "shotgun",
        "violence",
        "weapon",
    ]


def _map_name(raw: str) -> str | None:
    if raw in PERSON_SRC:
        return "person"
    if raw in KNIFE_SRC:
        return "knife"
    if raw in GUN_SRC:
        return "gun"
    return None


def _yolo_line(cls_id: int, x: float, y: float, w: float, h: float, iw: int, ih: int) -> str | None:
    if iw <= 0 or ih <= 0 or w <= 1 or h <= 1:
        return None
    x1 = max(0.0, min(float(x), iw))
    y1 = max(0.0, min(float(y), ih))
    x2 = max(0.0, min(float(x) + float(w), iw))
    y2 = max(0.0, min(float(y) + float(h), ih))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    xc = (x1 + x2) / 2.0 / iw
    yc = (y1 + y2) / 2.0 / ih
    nw = bw / iw
    nh = bh / ih
    if nw <= 0 or nh <= 0:
        return None
    return f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def _write_pair(out_root: Path, split: str, stem: str, bgr: np.ndarray, lines: list[str]) -> None:
    img_path = out_root / "images" / split / f"{stem}.jpg"
    lbl_path = out_root / "labels" / split / f"{stem}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), bgr)
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def prepare(cctv_copy_frac: float = 0.12, seed: int = 42, limit_per_split: int = 0) -> dict:
    from datasets import load_dataset

    cfg = load_config()
    out_root = resolve(cfg["dataset_dir"])
    stats_path = ROOT / "evaluation" / "dataset_stats.json"

    print(f"Loading {HF_DATASET} ...")
    dataset = load_dataset(HF_DATASET)
    raw_names = _class_list(dataset)
    print(f"Raw classes ({len(raw_names)}): {raw_names}")

    name_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    published_counts = {name: len(dataset[name]) for name in dataset}
    kept = {split: 0 for split in ("train", "val", "test")}
    dropped = {split: 0 for split in ("train", "val", "test")}
    box_counts: dict[str, Counter] = {s: Counter() for s in kept}
    image_class_presence: dict[str, Counter] = {s: Counter() for s in kept}
    extra_cctv = 0
    rng = random.Random(seed)
    degraders = [simulate_low_light, simulate_noise, simulate_blur, simulate_compression]

    for split_name, split_ds in dataset.items():
        yolo_split = SPLIT_MAP.get(split_name, split_name)
        if yolo_split not in kept:
            continue
        # A capped conversion is useful for a CPU-only smoke fine-tune. It is
        # deterministic and is recorded in dataset_stats.json; 0 means full set.
        rows = split_ds.select(range(min(len(split_ds), limit_per_split))) if limit_per_split else split_ds
        for index, row in enumerate(tqdm(rows, desc=f"convert {yolo_split}")):
            objects = row["objects"]
            cats = objects["category"]
            boxes = objects["bbox"]
            image = row["image"]
            bgr = _pil_to_bgr(image)
            height, width = bgr.shape[:2]
            lines: list[str] = []
            present: set[str] = set()
            local_boxes: Counter = Counter()
            for cat, box in zip(cats, boxes):
                raw = raw_names[int(cat)] if isinstance(cat, (int, np.integer)) else str(cat)
                mapped = _map_name(raw)
                if mapped is None:
                    continue
                line = _yolo_line(name_to_id[mapped], box[0], box[1], box[2], box[3], width, height)
                if line is None:
                    continue
                lines.append(line)
                present.add(mapped)
                local_boxes[mapped] += 1
            if not lines:
                dropped[yolo_split] += 1
                continue
            box_counts[yolo_split].update(local_boxes)
            stem = f"{yolo_split}_{int(row.get('image_id', index)):06d}"
            _write_pair(out_root, yolo_split, stem, bgr, lines)
            kept[yolo_split] += 1
            for name in present:
                image_class_presence[yolo_split][name] += 1

            if yolo_split == "train" and cctv_copy_frac > 0 and rng.random() < cctv_copy_frac:
                degraded = rng.choice(degraders)(bgr)
                _write_pair(out_root, yolo_split, f"{stem}_cctv", degraded, lines)
                extra_cctv += 1

    yaml_path = resolve(cfg["dataset_yaml"])
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_body = "\n".join(
        [
            f"path: {out_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    yaml_path.write_text(yaml_body, encoding="utf-8")

    stats = {
        "source": {
            "huggingface": HF_DATASET,
            "url": HF_PAGE,
            "original_roboflow": ROBOFLOW_PAGE,
            "license": "CC BY 4.0",
            "usage_notes": (
                "Attribution required (CC BY 4.0). This is a derivative parquet packaging "
                "of a Roboflow Universe weapon-detection set. SentinelAI remaps a subset of "
                "the original 29 classes to person/knife/gun and drops other labels "
                "(blood, aggressor, victim, generic 'weapon', etc.). Images without any "
                "remapped box are excluded. The detector flags visual indicators only."
            ),
            "limitations": [
                "Original class names are noisy and overlapping (Pistol vs pistol vs guns).",
                "Knife vs gun images are internet/CCTV mixed, not a controlled camera network.",
                "Person boxes exist but are not as consistently annotated as a dedicated pedestrian set.",
                "Generic 'weapon' boxes were not mapped because they mix blades and firearms.",
                "Labels do not encode legality or intent.",
            ],
            "published_split_counts": published_counts,
            "published_raw_class_count": len(raw_names),
            "published_raw_classes": raw_names,
        },
        "sentinel_classes": CLASS_NAMES,
        "remap": {
            "person": sorted(PERSON_SRC),
            "knife": sorted(KNIFE_SRC),
            "gun": sorted(GUN_SRC),
        },
        "after_remap": {
            "images_kept": kept,
            "images_dropped_no_target_class": dropped,
            "boxes_by_class": {k: dict(v) for k, v in box_counts.items()},
            "images_containing_class": {k: dict(v) for k, v in image_class_presence.items()},
            "extra_cctv_train_copies": extra_cctv,
            "total_kept_images": int(sum(kept.values())),
        },
        "split_policy": (
            "Hugging Face card splits are preserved (train/validation/test). "
            "No reshuffle. Extra CCTV-style photometric copies are train-only "
            "and reuse the original boxes (geometry-preserving degradations)."
        ),
        "conversion_limit_per_split": limit_per_split or "all",
        "dataset_yaml": str(yaml_path),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats["after_remap"], indent=2))
    print(f"Wrote {yaml_path}")
    print(f"Wrote {stats_path}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and convert WeaponDetection to YOLO")
    parser.add_argument("--limit-per-split", type=int, default=0, help="deterministic cap for a quick CPU run (0 = all)")
    parser.add_argument("--cctv-copy-frac", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit_per_split < 0:
        parser.error("--limit-per-split must be non-negative")
    prepare(args.cctv_copy_frac, args.seed, args.limit_per_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
