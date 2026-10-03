"""Single place for backend demo settings; environment variables override defaults."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    camera_id: str = Field(default_factory=lambda: os.getenv("SENTINEL_CAMERA_ID", "CAM-01"))
    camera_source: str = Field(default_factory=lambda: os.getenv("SENTINEL_CAMERA_SOURCE", "0"))
    weapon_confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("SENTINEL_WEAPON_THRESHOLD", "0.50")))
    cooldown_seconds: float = Field(default_factory=lambda: float(os.getenv("SENTINEL_COOLDOWN_SECONDS", "8")))
    demo_latitude: float | None = Field(default_factory=lambda: _optional_float("SENTINEL_DEMO_LATITUDE", "12.9716"))
    demo_longitude: float | None = Field(default_factory=lambda: _optional_float("SENTINEL_DEMO_LONGITUDE", "77.5946"))
    model_path: str | None = Field(default_factory=lambda: os.getenv("SENTINEL_MODEL_PATH") or None)
    database_path: Path = ROOT / "data" / "sentinel.db"
    evidence_dir: Path = ROOT / "incidents"
    stream_fps: float = Field(default_factory=lambda: float(os.getenv("SENTINEL_STREAM_FPS", "12")))
    host: str = Field(default_factory=lambda: os.getenv("SENTINEL_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("SENTINEL_PORT", "8000")))


def _optional_float(name: str, default: str) -> float | None:
    value = os.getenv(name, default)
    return None if value.strip().lower() in {"", "none", "null"} else float(value)


settings = Settings()
