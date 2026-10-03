# SentinelAI detector (stage 2)

Lightweight YOLO11n fine-tune for visual incident *indicators*: `person`, `knife`, `gun`.

Knife detection is the critical demo class. The model does not decide whether a crime occurred.

## Dataset

Primary source:

- Hugging Face: [Subh775/WeaponDetection](https://huggingface.co/datasets/Subh775/WeaponDetection)
- Original listing: Roboflow Universe weapon-detection by yolov7test ([dataset card citation](https://universe.roboflow.com/yolov7test-pdxwq/weapon-detection-m7tpo))
- License: **CC BY 4.0** (attribution required)
- Task: object detection (COCO-style boxes in parquet)
- Published split sizes on the dataset card: **train 5,871 / validation 1,491 / test 2,295** (9,657 images, 29 raw class names)

Those 29 names are noisy (`Pistol` vs `pistol`, `Knife` vs `Knife_Weapon`, `Person` vs `person`). `ml/prepare_dataset.py` remaps a documented subset to three SentinelAI classes and **drops** images that have none of those boxes. Exact kept-image / box counts are written by the converter to `evaluation/dataset_stats.json` — do not quote estimated numbers.

Related academic set (not the training dump used here, cited because it is the standard knife/pistol detection corpus):

- [OD-WeaponDetection](https://github.com/ari-dasci/OD-WeaponDetection) (DaSCI / Pérez-Hernández et al.), CC BY-SA 4.0
- Knife detection: 2,078 images; pistol detection: 3,000 images; Pascal VOC XML

Limitations: internet + CCTV mix, overlapping weapon labels, person boxes are secondary, no legal/intent annotation.

```bash
python -m ml.prepare_dataset
```

Writes YOLO folders under `data/yolo/` and `data/yolo/sentinel.yaml`. A fraction of train images also gets a geometry-preserving CCTV copy (low light / noise / blur / JPEG) using Stage 1 degradations.

## Model

- Architecture: **YOLO11n** (Ultralytics nano, real-time)
- Init: official COCO `yolo11n.pt` (transfer learning, not from scratch)
- Head: 3 classes after remapping
- Fallback if `models/best.pt` is missing: COCO `yolo11n.pt` via `ml.inference.predict`, which keeps only `person` and `knife` (COCO has no gun class)

## Training

```bash
python -m ml.train
```

Hyperparameters live in `ml/config.yaml` (also copied conceptually here):

| Setting | Value | Why |
| --- | --- | --- |
| epochs | 25 (patience 8) | fine-tune, stop if val stalls |
| imgsz | 640 | detector default, near real-time |
| batch | 8 | fits ~6 GB laptop GPUs |
| lr0 / lrf | 0.01 / 0.01 | Ultralytics transfer default scale |
| hsv_v 0.55 | brightness | low-light CCTV |
| hsv_s 0.70 | saturation | contrast / illumination mix |
| scale 0.55 | zoom | subject distance to camera |
| translate 0.12 / degrees 8 | framing jitter | PTZ / handheld shake (mild) |
| mosaic 0.85 / mixup 0.10 | dense scenes | multiple people / objects |
| erasing 0.35 | random erase | occlusion by hands, overlays, objects |
| fliplr 0.5, flipud 0 | left-right only | cameras are not upside down |
| copy_paste 0.05 | rare instance paste | extra knife/gun context |

Blur, Gaussian noise, and JPEG compression are **not** piled on every batch (that can destroy the knife edge signal). They are injected as a small train-only CCTV copy set plus the robustness experiment at eval time.

Albumentations is installed so Ultralytics can apply its optional blur / CLAHE / compression hooks when the version supports them.

Weights: `models/best.pt` (best val), `models/last.pt`.

## Evaluation

```bash
python -m ml.evaluate --split test
python -m ml.robustness --split val
```

`evaluate` writes `evaluation/model_metrics.json` and `.csv` with precision, recall, mAP50, mAP50-95, confusion-matrix path if Ultralytics saved one, and measured latency/FPS.

`robustness` rebuilds the val images as:

1. clean
2. clean + Stage 1 preprocess
3. low light
4. low light + preprocess
5. blur
6. blur + preprocess
7. noise
8. noise + preprocess

and runs the **same** weights on each. Table: `evaluation/robustness_metrics.json` / `.csv`.

## Inference API

```python
from ml.inference import predict

dets = predict(frame)  # BGR numpy array
# [{"class_name": "knife", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}, ...]
```

Later stages should not import Ultralytics.

```bash
python scripts/test_video.py --source 0
python scripts/test_video.py --source clip.mp4 --preprocess
python scripts/test_video.py --source photo.jpg --save out.jpg --headless
```
