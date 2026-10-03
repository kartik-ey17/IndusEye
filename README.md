# IndusEye

Real-time surveillance incident-indicator prototype. It detects visual objects, not crimes, intent, or legality. The MVP classes are `person`, `knife`, and `gun`; knife is the priority. Stage 1's video-quality and preprocessing code is retained under `video/`.

## Final demo architecture

```text
Camera / CCTV / browser webcam
        ↓
Video quality analysis + selective preprocessing
        ↓
YOLO detection → YuNet face detection → face-blurred display
        ↓
Incident threshold + person association + cooldown
        ↓
Face-blurred evidence image + SQLite metadata → FastAPI → Next.js dashboard
```

Implemented now: the local webcam/browser-to-dashboard flow. Future scope, not included in this MVP: fight detection, multi-camera orchestration, drone ingestion, edge acceleration, alert routing, and authentication.

## Dataset

The reproducible source is [Subh775/WeaponDetection](https://huggingface.co/datasets/Subh775/WeaponDetection), a CC BY 4.0 derivative of the [yolov7test Roboflow Universe weapon-detection set](https://universe.roboflow.com/yolov7test-pdxwq/weapon-detection-m7tpo). Its dataset card reports 5,871 train, 1,491 validation, and 2,295 test images (9,657 total), with 29 raw detection classes.

`ml.prepare_dataset` preserves those splits and maps only documented source labels to `person`, `knife`, and `gun`; images without a mapped box are omitted. It records actual retained image/box counts in `evaluation/dataset_stats.json`, so no post-remap statistics are guessed. Attribution is required. The labels are noisy/overlapping, combine web and CCTV imagery, and do not indicate criminal intent.

```powershell
python -m ml.prepare_dataset
python -m ml.prepare_dataset --limit-per-split 250  # CPU smoke run only
```

## Detector, training, and augmentation

YOLO11n uses official COCO weights for transfer learning, then fine-tunes a three-class head. Run `python -m ml.train --device 0` on GPU. A CPU functional check is `python -m ml.train --device cpu --epochs 1 --batch 2 --imgsz 416`. The best and last checkpoints are copied to `models/best.pt` and `models/last.pt`.

Settings live in `ml/config.yaml`: brightness/saturation variation supports illumination changes; scale/translation/mosaic handle distance and framing; random erasing models occlusion. Blur, noise, and JPEG degradation are only injected into a small train-only CCTV copy set, avoiding systematic loss of knife-edge detail.

## Privacy and incident workflow

Every frame passed through `incidents.process_frame` is face-detected locally with OpenCV YuNet and its returned `display_frame` has every detected face blurred. The raw feed is not returned or saved as routine output. The small YuNet model is stored at `models/face_detection_yunet_2023mar.onnx` from the [OpenCV Zoo YuNet model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

Only a `knife` or `gun` detection at or above `incident.weapon_confidence_threshold` (default `0.50`, set once in `ml/config.yaml`) begins an incident. A nearby/containing person box produces `armed_person`; otherwise it is `weapon_detected`. This is a visual incident indicator, not a legal finding. A stateful `IncidentProcessor` uses `incident.cooldown_seconds` (default 8 seconds): a continuous sighting creates one event and evidence image, then remains active instead of creating one alert per frame.

Evidence is an annotated, face-blurred frame at `incidents/INC-0001.jpg`, accompanied by JSON detection metadata. It is a distinct authorized-operator artifact; the continuous display is never switched to an unblurred stream.

```python
from incidents import process_frame

result = process_frame(frame, {"camera_id": "CAM-01"})
# result: display_frame, detections, faces, incident, metrics
```

## Evaluation and robustness

`python -m ml.evaluate --split test --device 0` writes measured precision, recall, mAP50, mAP50-95, confusion-matrix path, latency, and FPS. `python -m ml.robustness --split val --device 0` compares clean, low-light, blur, and noise images, plus each condition after `video.process_frame(mode="auto")`. Degradations preserve geometry, so labels remain valid. Results are written only after execution to `evaluation/model_metrics.*` and `evaluation/robustness_metrics.*`; no metrics are invented.

## Stable application API

```python
from ml.inference import predict
detections = predict(frame)  # OpenCV BGR frame
# [{"class_name": "knife", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}]
```

The model is cached after its first load. Before a fine-tune exists, COCO YOLO11n is an explicit fallback for `person` and `knife` only; it has no dedicated gun class.

## Demo

```powershell
python scripts/test_video.py --source 0
python scripts/test_video.py --source clip.mp4 --preprocess --save outputs/annotated.mp4
python scripts/test_video.py --source photo.jpg --headless --save outputs/annotated.jpg
```

The retained Stage 1 checks are `python scripts/verify_pipeline.py` and `python scripts/degradation_demo.py`.

## Local backend API

The backend is a thin FastAPI service over the existing CV pipeline—there is no copied detector or face-blur implementation. It initializes SQLite at `data/sentinel.db`, streams anonymized frames as MJPEG, and writes incident records only when the existing incident engine confirms a weapon indicator.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. The frontend-facing API is:

| Endpoint | Use |
| --- | --- |
| `GET /api/health` | Service/model state and load error, if any |
| `GET /api/stream` | Face-blurred processed webcam MJPEG feed |
| `POST /api/process-frame` | Send raw JPEG/PNG bytes; receive detections, faces, incident and latency metrics |
| `GET /api/incidents` / `GET /api/incidents/{id}` | SQLite incident records |
| `GET /api/incidents/{id}/evidence` | Face-blurred evidence image URL returned in each incident |
| `POST /api/location` | Browser location JSON, e.g. `{"latitude":12.9716,"longitude":77.5946}` |

The browser location overrides configurable demo coordinates. All backend settings are centralized in `backend/config.py`; use `SENTINEL_HOST`, `SENTINEL_PORT`, `SENTINEL_CAMERA_SOURCE`, `SENTINEL_CAMERA_ID`, `SENTINEL_WEAPON_THRESHOLD`, `SENTINEL_COOLDOWN_SECONDS`, `SENTINEL_DEMO_LATITUDE`, `SENTINEL_DEMO_LONGITUDE`, and `SENTINEL_MODEL_PATH` to override them. `python -m backend.run` uses configured host/port without reload.

With the server running, smoke-test a real image request:

```powershell
python scripts/backend_smoke_test.py --image data/samples/verify_none.jpg
```

## Operator dashboard

The Next.js dashboard in `frontend/` keeps Python as the inference engine. Browser `getUserMedia` captures camera frames, sends them to `POST /api/process-frame`, and renders only the server-returned anonymized/annotated image—never the raw camera element. It polls incidents every three seconds and requests browser location once; if permission is denied, it labels the backend-provided coordinates as demo fallback.

Start the backend first, then in a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.local.example .env.local
npm run dev
```

Open `http://127.0.0.1:3000`. By default `.env.local.example` targets `http://127.0.0.1:8000`. If FastAPI runs elsewhere, set `NEXT_PUBLIC_API_BASE_URL` before starting Next, for example:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8010
```

The dashboard handles an offline backend, unavailable/denied camera permission, absent incidents, and missing evidence without exposing an unprocessed stream.

## Final evaluation and demo checklist

Generate reproducible local runtime evidence:

```powershell
python -m evaluation.final_validation
```

This writes warmed-up CPU measurements for clean, low-light, blur, noise, and preprocessing variants to `evaluation/final_results.csv`, plus PPT assets under `evaluation/assets/`. Precision, recall, mAP50, and mAP50-95 are intentionally blank in that report: this checkout has no `models/best.pt`, converted labelled test data, or historical model-metric file. Restore those artifacts and run `python -m ml.evaluate --split test` and `python -m ml.robustness --split val` for valid supervised metrics. Never present fallback COCO metrics as a weapon-detector fine-tune.

For the judge flow: start FastAPI and the dashboard; confirm `SYSTEM: ONLINE` and `PRIVACY: ACTIVE`; grant browser camera permission; confirm the visible feed has blurred faces; then use a locally available fine-tuned `models/best.pt` and a clearly visible knife to generate a threshold-qualified incident. Confirm the resulting card shows evidence, timestamp, camera ID, and browser or demo coordinates. The fallback COCO YOLO11n may not reliably recognize real knives and has no gun class.

Presentation and defense material is in `PRESENTATION_NOTES.md` and `DEFENSE.md`.
