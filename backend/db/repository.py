"""Minimal SQLite persistence for incident API records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class IncidentRepository:
    def __init__(self, path: Path):
        self.path = path

    def _connection(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY, event_type TEXT NOT NULL, confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL, camera_id TEXT NOT NULL,
                    latitude REAL, longitude REAL, evidence_image TEXT,
                    detections_json TEXT NOT NULL
                )"""
            )

    def create(self, event: dict[str, Any]) -> None:
        with self._connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO incidents
                (id,event_type,confidence,timestamp,camera_id,latitude,longitude,evidence_image,detections_json)
                VALUES (:id,:event_type,:confidence,:timestamp,:camera_id,:latitude,:longitude,:evidence_image,:detections_json)""",
                {
                    "id": event["id"], "event_type": event["event_type"], "confidence": event["confidence"],
                    "timestamp": event["timestamp"], "camera_id": event["camera_id"],
                    "latitude": event.get("latitude"), "longitude": event.get("longitude"),
                    "evidence_image": event.get("evidence_path"), "detections_json": json.dumps(event["detections"]),
                },
            )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["detections"] = json.loads(data.pop("detections_json"))
        data["evidence_url"] = f"/api/incidents/{data['id']}/evidence" if data.get("evidence_image") else None
        return data

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]

    def get(self, incident_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        return self._row(row) if row else None
