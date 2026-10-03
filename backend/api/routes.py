from __future__ import annotations

import logging
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.services import SentinelRuntime

LOG = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class LocationInput(BaseModel):
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


def _runtime(request: Request) -> SentinelRuntime:
    return request.app.state.runtime


@router.get("/health")
def health(request: Request):
    return _runtime(request).health()


@router.get("/incidents")
def list_incidents(request: Request, limit: int = 100):
    return _runtime(request).repository.list(max(1, min(limit, 500)))


@router.get("/incidents/{incident_id}")
def get_incident(incident_id: str, request: Request):
    incident = _runtime(request).repository.get(incident_id)
    if not incident:
        raise HTTPException(404, "Incident not found")
    return incident


@router.get("/incidents/{incident_id}/evidence")
def evidence(incident_id: str, request: Request):
    incident = _runtime(request).repository.get(incident_id)
    if not incident or not incident.get("evidence_image"):
        raise HTTPException(404, "Evidence not found")
    path = Path(incident["evidence_image"])
    if not path.is_file():
        raise HTTPException(404, "Evidence image is unavailable")
    return FileResponse(path, media_type="image/jpeg", filename=path.name)


@router.post("/location")
def set_location(body: LocationInput, request: Request):
    return _runtime(request).set_location(body.latitude, body.longitude)


@router.post("/process-frame")
async def process_uploaded_frame(request: Request):
    """Process raw JPEG/PNG bytes; convenient for a browser capture canvas."""
    content = await request.body()
    if not content or len(content) > 15 * 1024 * 1024:
        raise HTTPException(400, "Send one JPEG/PNG image smaller than 15 MB")
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(400, "Request body is not a decodable image")
    try:
        result = _runtime(request).process(frame)
    except Exception as exc:
        raise HTTPException(503, f"Frame processing unavailable: {exc}") from exc
    ok, encoded = cv2.imencode(".jpg", result["display_frame"], [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise HTTPException(500, "Could not encode processed display frame")
    payload = {key: result[key] for key in ("detections", "faces", "incident", "metrics")}
    payload["display_frame_data_url"] = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    return payload


@router.get("/stream")
def stream(request: Request):
    """Privacy-preserving processed webcam stream (multipart MJPEG)."""
    return StreamingResponse(_runtime(request).mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")
