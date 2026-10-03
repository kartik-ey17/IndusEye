# Presentation assets

These assets are generated locally by `python -m evaluation.final_validation` from `data/samples/verify_none.jpg` and the real pipeline.

- `architecture.svg`: implemented system architecture.
- `clean_frame.jpg`, `low_light.jpg`, `blur.jpg`, `noise.jpg`: actual clean/degraded inputs processed in the final validation run.
- `*_preprocess.jpg`, `processed_frame.jpg`: Stage 1 preprocessing outputs.
- `privacy_processed_frame.jpg`, `detection_pipeline_output.jpg`: actual privacy/detection pipeline outputs. The reference scene generated no weapon detection, so these must not be presented as a knife-positive example.
- `runtime_latency_chart.svg`: generated from the measured `final_results.csv` values.
- `dashboard_screenshot.png`: captured from the running local dashboard. Headless capture cannot receive a real webcam permission grant, so it documents the privacy-safe empty-camera/error-state UI rather than pretending to be a live weapon event.

An authentic incident-evidence image and a knife-positive dashboard screenshot require a real knife-positive run using `models/best.pt` or a physical camera demonstration. They are intentionally not synthesized here.
