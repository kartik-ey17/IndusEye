"""Own the long-lived CV processor, current browser location, and webcam safely."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

from backend.config import Settings
from backend.db import IncidentRepository
from incidents.pipeline import IncidentProcessor, process_frame
from ml.inference import model_info

LOG = logging.getLogger(__name__)


class SentinelRuntime:
    def __init__(self, settings: Settings, repository: IncidentRepository):
        self.settings, self.repository = settings, repository
        self.processor = IncidentProcessor(settings.evidence_dir, settings.weapon_confidence_threshold, settings.cooldown_seconds)
        self.location = {"latitude": settings.demo_latitude, "longitude": settings.demo_longitude}
        self.model_error: str | None = None
        self.model: dict[str, Any] | None = None
        self._camera: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

    def load_model(self) -> None:
        try:
            self.model = model_info(self.settings.model_path)
            self.model_error = None
        except Exception as exc:
            self.model_error = str(exc)
            LOG.exception("Detector unavailable")

    def health(self) -> dict[str, Any]:
        return {"status": "ok" if self.model_error is None else "degraded", "model": self.model, "model_error": self.model_error}

    def set_location(self, latitude: float | None, longitude: float | None) -> dict[str, float | None]:
        self.location = {"latitude": latitude, "longitude": longitude}
        return self.location.copy()

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        if frame is None or frame.size == 0:
            raise ValueError("Empty or invalid image frame")
        metadata = {
            "camera_id": self.settings.camera_id, "latitude": self.location["latitude"],
            "longitude": self.location["longitude"], "preprocess": True, "weights": self.settings.model_path,
        }
        try:
            result = process_frame(frame, metadata, processor=self.processor)
            if result["incident"] and result["incident"]["status"] == "created":
                self.repository.create(result["incident"])
            return result
        except Exception:
            LOG.exception("Frame processing failed")
            raise

    def _open_camera(self) -> cv2.VideoCapture:
        if self._camera and self._camera.isOpened():
            return self._camera
        source: int | str = int(self.settings.camera_source) if self.settings.camera_source.isdigit() else self.settings.camera_source
        self._camera = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
        if not self._camera.isOpened():
            self._camera.release()
            raise RuntimeError(f"Cannot open camera source {self.settings.camera_source}")
        return self._camera

    def mjpeg(self):
        delay = 1.0 / max(self.settings.stream_fps, 1.0)
        while True:
            try:
                with self._lock:
                    ok, frame = self._open_camera().read()
                if not ok:
                    raise RuntimeError("Camera returned an invalid frame")
                result = self.process(frame)
                ok, encoded = cv2.imencode(".jpg", result["display_frame"], [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    raise RuntimeError("Could not encode processed frame")
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            except GeneratorExit:
                return
            except Exception as exc:
                LOG.warning("MJPEG frame skipped: %s", exc)
                time.sleep(0.5)
            time.sleep(delay)

    def close(self) -> None:
        if self._camera:
            self._camera.release()
