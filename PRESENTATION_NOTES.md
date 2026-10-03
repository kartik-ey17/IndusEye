# SentinelAI — presentation notes

## Slide 1 — Problem

Real surveillance footage is rarely benchmark-clean: low light, blur, noise, occlusion, and camera motion reduce detector reliability. Operators also need privacy safeguards before they can use live video responsibly.

## Slide 2 — Solution

SentinelAI is a privacy-preserving local video-analysis prototype. It enhances difficult frames, identifies visual incident indicators, blurs faces in the live display, and records a cooldown-controlled evidence event when a weapon indicator is confirmed.

## Slide 3 — Architecture

Use `evaluation/assets/architecture.svg`.

Camera/CCTV/Drone → video quality and preprocessing → YOLO detection → YuNet face anonymization → incident engine → evidence + SQLite metadata → FastAPI → Next.js operator dashboard.

Implemented now: local webcam/browser capture, preprocessing, detector interface, privacy blur, incident cooldown, SQLite/evidence, API, and dashboard. Future extensions: drone ingestion, multi-camera orchestration, edge deployment, and alert routing.

## Slide 4 — Robust video processing

Use `clean_frame.jpg`, `low_light.jpg`, and `processed_frame.jpg` from `evaluation/assets/`. Explain that quality flags selectively apply gamma/CLAHE, light denoise, and sharpening rather than applying costly restoration to every frame.

## Slide 5 — ML model

The intended fine-tune is YOLO11n transfer learning on the CC BY 4.0 WeaponDetection derivative, remapped to `person`, `knife`, and `gun`. Training augmentation includes illumination, scale/framing, occlusion, and a small CCTV-style degradation copy set. This checkout currently has the official COCO YOLO11n fallback, not `models/best.pt`; it supports person/knife fallback only. Do not claim gun detection or fine-tune mAP without the checkpoint and labelled test split.

## Slide 6 — Privacy layer

YuNet detects faces locally and the system Gaussian-blurs each face before the frame reaches the dashboard. The dashboard intentionally hides the raw browser-video element and shows only the returned privacy-safe image.

## Slide 7 — Incident detection

A `knife` or `gun` at the configured threshold creates `weapon_detected`; an associated nearby person creates `armed_person`. The incident engine uses a configurable cooldown, saves an annotated face-blurred evidence image, and persists metadata in SQLite. It identifies a visual indicator, not a crime or intent.

## Slide 8 — Deployment

FastAPI accepts browser JPEG frames and returns a blurred/annotated display image, detections, incident state, and timing. Next.js polls incidents and renders evidence cards. This is intentionally lightweight enough for a hackathon laptop.

## Slide 9 — Results

Use `evaluation/final_results.csv` and `evaluation/assets/runtime_latency_chart.svg`. These are steady-state one-frame CPU measurements from the local sample and include preprocessing, inference, privacy, end-to-end latency, and FPS for clean/low-light/blur/noise with and without preprocessing. Precision, recall, mAP50, and mAP50-95 are intentionally blank because this checkout lacks a labelled converted test set and fine-tuned checkpoint.

## Slide 10 — Future work

Fight detection, stronger temporal/tracking logic, multi-camera deployment, drone integration, edge acceleration, reliable alert routing, and a properly evaluated surveillance-specific fine-tune.
