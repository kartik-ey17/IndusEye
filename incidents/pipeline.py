"""Privacy-first integration: preprocess -> detect -> blur display -> incident event."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.config import ROOT, load_config
from ml.inference import Detection, predict
from video.preprocess import process_frame as preprocess_frame
from video.privacy import blur_faces, detect_faces


WEAPON_CLASSES = {"knife", "gun"}


@dataclass(frozen=True)
class IncidentEvent:
    id: str
    event_type: str
    confidence: float
    timestamp: str
    camera_id: str
    latitude: float | None
    longitude: float | None
    detections: list[dict[str, Any]]
    evidence_path: str | None
    status: str
    person_bbox: list[float] | None = None
    weapon_bbox: list[float] | None = None
    weapon: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def associate_person(weapon: Detection, people: list[Detection], max_distance_ratio: float = 0.75) -> Detection | None:
    """Associate a weapon with its nearest person using box containment/distance."""
    wx, wy = _center(weapon.bbox)
    best: tuple[float, Detection] | None = None
    for person in people:
        x1, y1, x2, y2 = person.bbox
        px, py = _center(person.bbox)
        diagonal = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        # A weapon inside a person box is the strongest simple association.
        if x1 <= wx <= x2 and y1 <= wy <= y2:
            return person
        distance = ((wx - px) ** 2 + (wy - py) ** 2) ** 0.5
        if distance <= diagonal * max_distance_ratio and (best is None or distance < best[0]):
            best = (distance, person)
    return best[1] if best else None


class IncidentProcessor:
    """Stateful cooldown tracker. One continuous weapon sighting creates one evidence item."""

    def __init__(self, evidence_dir: str | Path | None = None, weapon_threshold: float | None = None, cooldown_seconds: float | None = None):
        cfg = load_config()
        incident_cfg = cfg.get("incident", {})
        self.weapon_threshold = float(weapon_threshold if weapon_threshold is not None else incident_cfg.get("weapon_confidence_threshold", 0.50))
        self.cooldown_seconds = float(cooldown_seconds if cooldown_seconds is not None else incident_cfg.get("cooldown_seconds", 8.0))
        self.evidence_dir = Path(evidence_dir) if evidence_dir else ROOT / "incidents"
        self._active: IncidentEvent | None = None
        self._last_seen = 0.0
        existing = [int(p.stem.split("-")[1]) for p in self.evidence_dir.glob("INC-*.jpg") if p.stem.split("-")[-1].isdigit()]
        self._counter = max(existing, default=0)
        self._lock = threading.Lock()

    def _next_id(self) -> str:
        self._counter += 1
        return f"INC-{self._counter:04d}"

    def _save_evidence(self, event: IncidentEvent, display_frame: np.ndarray) -> str:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.evidence_dir / f"{event.id}.jpg"
        cv2.imwrite(str(image_path), display_frame)
        metadata_path = self.evidence_dir / f"{event.id}.json"
        metadata_path.write_text(json.dumps(event.to_dict(), indent=2), encoding="utf-8")
        return str(image_path)

    def update(self, detections: list[Detection], display_frame: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any] | None:
        now = time.monotonic()
        weapons = [d for d in detections if d.class_name in WEAPON_CLASSES and d.confidence >= self.weapon_threshold]
        with self._lock:
            if not weapons:
                if self._active and now - self._last_seen >= self.cooldown_seconds:
                    self._active = None
                return None
            weapon = max(weapons, key=lambda d: d.confidence)
            people = [d for d in detections if d.class_name == "person"]
            person = associate_person(weapon, people)
            self._last_seen = now
            if self._active:
                return {**self._active.to_dict(), "status": "active"}

            timestamp = datetime.now(timezone.utc).isoformat()
            event_type = "armed_person" if person else "weapon_detected"
            event = IncidentEvent(
                id=self._next_id(), event_type=event_type, confidence=weapon.confidence,
                timestamp=timestamp, camera_id=str(metadata.get("camera_id", "CAM-01")),
                latitude=metadata.get("latitude"), longitude=metadata.get("longitude"),
                detections=[d.to_dict() for d in detections], evidence_path=None, status="created",
                person_bbox=person.bbox if person else None, weapon_bbox=weapon.bbox, weapon=weapon.class_name,
            )
            # Evidence uses the already face-blurred, annotated display—not the raw stream.
            evidence_path = self._save_evidence(event, display_frame)
            event = IncidentEvent(**{**event.to_dict(), "evidence_path": evidence_path})
            (self.evidence_dir / f"{event.id}.json").write_text(json.dumps(event.to_dict(), indent=2), encoding="utf-8")
            self._active = event
            return event.to_dict()


def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    colors = {"person": (80, 180, 80), "knife": (0, 165, 255), "gun": (0, 0, 255)}
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = colors.get(det.class_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


_DEFAULT_PROCESSOR: IncidentProcessor | None = None


def process_frame(frame: np.ndarray, metadata: dict[str, Any] | None = None, processor: IncidentProcessor | None = None) -> dict[str, Any]:
    """Return a privacy-safe display frame, stable detections, faces, event, and timings."""
    if frame is None or frame.size == 0:
        raise ValueError("process_frame requires a non-empty frame")
    metadata = metadata or {}
    started = time.perf_counter()
    pre_start = time.perf_counter()
    detector_frame = preprocess_frame(frame, mode=metadata.get("preprocess_mode", "auto")) if metadata.get("preprocess", True) else frame.copy()
    preprocess_ms = (time.perf_counter() - pre_start) * 1000.0
    infer_start = time.perf_counter()
    detections = predict(
        detector_frame, conf=metadata.get("detection_confidence"), weights=metadata.get("weights")
    )
    inference_ms = (time.perf_counter() - infer_start) * 1000.0
    face_start = time.perf_counter()
    faces = detect_faces(detector_frame)
    display = blur_faces(detector_frame, faces)
    privacy_ms = (time.perf_counter() - face_start) * 1000.0
    _draw_detections(display, detections)
    global _DEFAULT_PROCESSOR
    if processor is None:
        if _DEFAULT_PROCESSOR is None:
            _DEFAULT_PROCESSOR = IncidentProcessor()
        processor = _DEFAULT_PROCESSOR
    incident = processor.update(detections, display, metadata)
    if incident:
        label = f"INCIDENT {incident['event_type']}: {incident.get('weapon', 'weapon')}"
        cv2.rectangle(display, (0, 0), (min(display.shape[1], 440), 34), (0, 0, 180), -1)
        cv2.putText(display, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "display_frame": display,
        "detections": [d.to_dict() for d in detections],
        "faces": faces,
        "incident": incident,
        "metrics": {
            "total_latency_ms": round(elapsed, 3),
            "preprocessing_latency_ms": round(preprocess_ms, 3),
            "inference_latency_ms": round(inference_ms, 3),
            "privacy_latency_ms": round(privacy_ms, 3),
            "fps": round(1000.0 / elapsed, 2) if elapsed else None,
        },
    }
