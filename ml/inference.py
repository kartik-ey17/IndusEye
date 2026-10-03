"""Application-facing detector API. Later stages should import from here, not Ultralytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ml.config import CLASS_NAMES, ROOT, load_config, resolve

# COCO names used only by the pretrained fallback (no dedicated gun class).
_COCO_KEEP = {"person": "person", "knife": "knife"}


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: list[float]  # xyxy pixel coordinates [x1, y1, x2, y2]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_weights_path() -> Path:
    cfg = load_config()
    tuned = resolve(cfg["weights_out"])
    if tuned.exists():
        return tuned
    return resolve(cfg.get("fallback_pretrained", "yolo11n.pt"))


@lru_cache(maxsize=4)
def _load_yolo_cached(weight_text: str):
    from ultralytics import YOLO

    path = Path(weight_text)
    if not path.exists() and path.name.endswith(".pt"):
        # Ultralytics will fetch official weights by name (e.g. yolo11n.pt).
        return YOLO(path.name), path.name
    return YOLO(str(path)), str(path)


def _load_yolo(weights: Path | str | None = None):
    path = Path(weights) if weights else default_weights_path()
    return _load_yolo_cached(str(path))


def predict(
    frame: np.ndarray,
    conf: float | None = None,
    iou: float | None = None,
    weights: str | Path | None = None,
    imgsz: int | None = None,
) -> list[Detection]:
    """Run detection on one BGR frame and return a stable list of detections."""
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("predict() requires a non-empty frame")

    cfg = load_config()
    conf = cfg["conf"] if conf is None else conf
    iou = cfg["iou"] if iou is None else iou
    imgsz = cfg["imgsz"] if imgsz is None else imgsz

    model, loaded = _load_yolo(weights)
    names = model.names
    # Fine-tuned 3-class head vs COCO-80 fallback.
    tuned = any(str(n).lower() == "gun" for n in names.values()) if isinstance(names, dict) else "gun" in names

    results = model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        max_det=cfg.get("max_det", 50),
    )
    detections: list[Detection] = []
    if not results:
        return detections

    boxes = results[0].boxes
    if boxes is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    for box, score, cls_id in zip(xyxy, scores, classes):
        raw_name = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(names[int(cls_id)])
        if tuned:
            class_name = raw_name
            if class_name not in CLASS_NAMES:
                continue
        else:
            class_name = _COCO_KEEP.get(raw_name)
            if class_name is None:
                continue
        detections.append(
            Detection(
                class_name=class_name,
                confidence=float(round(float(score), 4)),
                bbox=[float(round(float(v), 1)) for v in box.tolist()],
            )
        )
    return detections


def model_info(weights: str | Path | None = None) -> dict[str, Any]:
    model, loaded = _load_yolo(weights)
    names = model.names
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names)]
    else:
        class_names = list(names)
    return {
        "weights": loaded,
        "class_names": class_names,
        "uses_finetuned_head": "gun" in class_names,
        "task": getattr(model, "task", "detect"),
        "project_root": str(ROOT),
    }
