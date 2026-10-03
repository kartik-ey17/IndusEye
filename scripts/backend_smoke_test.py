"""Run after starting uvicorn: verifies health, frame processing, and incident listing."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", default="data/samples/verify_none.jpg")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    print("health", requests.get(f"{base}/api/health", timeout=20).json())
    image = Path(args.image)
    response = requests.post(f"{base}/api/process-frame", content=image.read_bytes(), headers={"content-type": "image/jpeg"}, timeout=90)
    response.raise_for_status()
    frame = response.json()
    print("frame", frame)
    incidents = requests.get(f"{base}/api/incidents", timeout=20).json()
    print("incidents", incidents)
    if frame.get("incident"):
        incident_id = frame["incident"]["id"]
        fetched = requests.get(f"{base}/api/incidents/{incident_id}", timeout=20)
        fetched.raise_for_status()
        print("retrieved_incident", fetched.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
