# V09 — YOLO11-seg Built-in Tracker + V08 Re-ID Logic

## Architecture

V09 uses two complementary layers:

1. **YOLO11s-seg + Ultralytics ByteTrack** handles frame-to-frame geometric tracking and provides a persistent `yolo_track_id`.
2. **The V08 `ReIDTracker`** remains the identity layer: OSNet body embeddings, face evidence, gallery protection, conservative identity confirmation, occlusion/lost states, recovery, duplicate arbitration, metrics, and presence accounting.

This avoids replacing the V08 identity safeguards with a second appearance tracker.

## Data flow

`frame -> YOLO11s-seg -> ByteTrack -> Detection(box, confidence, mask, yolo_track_id) -> V08 ReIDTracker -> identity/gallery/presence`

The segmentation mask is still converted to a bounding-box-local mask and used for the OSNet crop, preserving the V08 mask-aware Re-ID behavior.

## Model

The V09 default detector is `Models/yolo11s-seg.pt`.

The default built-in tracker is Ultralytics `bytetrack.yaml`. It can be changed with `REID_YOLO_TRACKER` when the deployment needs another supported Ultralytics tracker configuration.

## Important behavior

- `persist=True` keeps the YOLO tracker state across frames.
- `classes=[0]` restricts tracking to people for the COCO YOLO11 model.
- YOLO IDs are stored as `Detection.yolo_track_id` for diagnostics/integration.
- V08 track IDs remain the internal identity/lifecycle IDs so existing metrics and gallery ownership semantics stay compatible.
- Re-ID is still conditional on the V08 `reid_interval` and uncertainty/occlusion rules.
- Face/person ambiguity gates and identity-owner locking are unchanged.

## Run

Use the existing V08 entry points. The default detector now resolves to `Models/yolo11s-seg.pt` on the V09 branch.

Example:

```bash
python run_pipeline.py --source path/to/video.mp4
```

## Validation checklist

1. Confirm `Models/yolo11s-seg.pt` exists.
2. Run a short 100–300 frame video.
3. Check `yolo_tracker_ids` in detector profiling/debug output.
4. Compare V08 vs V09 using the existing metrics: ID switches, track fragments, recovery success/failure, false identity claims, processing FPS, and Re-ID latency.
5. Test crossing, short occlusion, long occlusion, entry/exit, and two people wearing visually similar clothing.
