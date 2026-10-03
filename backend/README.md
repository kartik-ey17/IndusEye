# SentinelAI backend

The backend is a local FastAPI wrapper around the existing preprocessing, YOLO, face-blur, and incident modules. It stores incidents in `data/sentinel.db`; evidence remains under `incidents/`.

```powershell
pip install -r requirements.txt
uvicorn backend.main:app --reload
# Or use configured SENTINEL_HOST / SENTINEL_PORT:
python -m backend.run
```

Open `http://127.0.0.1:8000/docs` for interactive requests.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Model/error health state |
| `GET /api/stream` | Privacy-safe MJPEG webcam stream |
| `POST /api/process-frame` | Raw JPEG/PNG bytes to detections, faces, incident, timing JSON, and a face-blurred annotated `display_frame_data_url` |
| `GET /api/incidents` | Persisted incident records; each has `evidence_url` |
| `GET /api/incidents/{id}` | One incident |
| `GET /api/incidents/{id}/evidence` | Face-blurred evidence image |
| `POST /api/location` | JSON `{"latitude": 12.97, "longitude": 77.59}` |

Settings are centralized in `backend/config.py`; `SENTINEL_HOST`, `SENTINEL_PORT`, `SENTINEL_CAMERA_SOURCE`, `SENTINEL_CAMERA_ID`, `SENTINEL_WEAPON_THRESHOLD`, `SENTINEL_COOLDOWN_SECONDS`, `SENTINEL_DEMO_LATITUDE`, `SENTINEL_DEMO_LONGITUDE`, and `SENTINEL_MODEL_PATH` are supported environment overrides.
