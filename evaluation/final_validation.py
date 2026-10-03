"""Generate reproducible final-demo runtime and robustness evidence.

This is deliberately not a substitute for labelled mAP evaluation. If a
fine-tuned checkpoint plus held-out YOLO data exist, run ``ml.evaluate`` and
``ml.robustness`` as documented; this script records only measurements it can
actually make from the local demo frame.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from incidents.pipeline import IncidentProcessor, process_frame
from video.degradation import simulate_blur, simulate_low_light, simulate_noise
from video.preprocess import process_frame as preprocess_frame

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "evaluation" / "assets"


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure final SentinelAI demo pipeline")
    parser.add_argument("--image", default=str(ROOT / "data" / "samples" / "verify_none.jpg"))
    args = parser.parse_args()
    clean = cv2.imread(args.image)
    if clean is None:
        raise FileNotFoundError(f"Cannot read demo image: {args.image}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    # Load/cache detector and face model before measuring steady-state demo frames.
    process_frame(clean, {"camera_id": "WARMUP", "preprocess": True}, processor=IncidentProcessor(evidence_dir=ROOT / "evaluation" / ".validation_evidence"))
    conditions = {"clean": clean, "low_light": simulate_low_light(clean), "blur": simulate_blur(clean), "noise": simulate_noise(clean)}
    rows: list[dict[str, object]] = []
    for condition, image in conditions.items():
        for preprocessed in (False, True):
            # Dedicated processors prevent a previous frame's cooldown state from
            # affecting this measurement.
            processor = IncidentProcessor(evidence_dir=ROOT / "evaluation" / ".validation_evidence")
            result = process_frame(image, {"camera_id": "EVAL", "preprocess": preprocessed}, processor=processor)
            label = condition + ("+preprocess" if preprocessed else "")
            cv2.imwrite(str(ASSETS / f"{label}.jpg"), result["display_frame"])
            rows.append({
                "condition": label,
                "n_frames": 1,
                "detections": len(result["detections"]),
                "faces_blurred": len(result["faces"]),
                "preprocessing_latency_ms": result["metrics"]["preprocessing_latency_ms"],
                "inference_latency_ms": result["metrics"]["inference_latency_ms"],
                "privacy_latency_ms": result["metrics"]["privacy_latency_ms"],
                "end_to_end_latency_ms": result["metrics"]["total_latency_ms"],
                "fps": result["metrics"]["fps"],
                "precision": "",
                "recall": "",
                "mAP50": "",
                "mAP50_95": "",
                "metric_note": "Runtime-only measurement; no local labelled fine-tuned evaluation set/checkpoint.",
            })
    cv2.imwrite(str(ASSETS / "clean_frame.jpg"), clean)
    cv2.imwrite(str(ASSETS / "processed_frame.jpg"), preprocess_frame(conditions["low_light"], mode="auto"))
    # These are actual pipeline outputs. The available reference frame has no
    # weapon prediction, so the names do not imply a detection occurred.
    cv2.imwrite(str(ASSETS / "privacy_processed_frame.jpg"), cv2.imread(str(ASSETS / "clean+preprocess.jpg")))
    cv2.imwrite(str(ASSETS / "detection_pipeline_output.jpg"), cv2.imread(str(ASSETS / "clean.jpg")))
    with (ROOT / "evaluation" / "final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # Compact SVG bar chart generated strictly from measured end-to-end latency.
    maximum = max(float(row["end_to_end_latency_ms"]) for row in rows)
    bars = []
    for index, row in enumerate(rows):
        value = float(row["end_to_end_latency_ms"]); y = 45 + index * 38; width = 440 * value / maximum
        bars.append(f'<text x="10" y="{y + 14}" fill="#dce8eb" font-size="12">{row["condition"]}</text><rect x="185" y="{y}" width="{width:.1f}" height="20" fill="#55dfd0"/><text x="{192 + width:.1f}" y="{y + 14}" fill="#dce8eb" font-size="12">{value:.1f} ms</text>')
    height = 70 + len(rows) * 38
    (ASSETS / "runtime_latency_chart.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" viewBox="0 0 760 {height}"><rect width="100%" height="100%" fill="#081014"/><text x="10" y="25" fill="#55dfd0" font-size="18" font-family="Arial">Measured end-to-end pipeline latency (one frame/condition)</text>{"".join(bars)}</svg>', encoding="utf-8")
    print(f"Wrote {ROOT / 'evaluation' / 'final_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
