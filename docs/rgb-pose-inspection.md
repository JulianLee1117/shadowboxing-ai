# Inspect local RGB and pose features

`python -m ml.inspect_rgb_pose` prepares numeric RGB and pose tensors for local dataset inspection. It does not train a recognizer, assign boxer identities, import action labels, or change the running coach. Models must already exist locally; the command downloads nothing.

Install the optional dependencies in an isolated environment using `ml/requirements-extract.txt`. Create a provenance file such as:

```json
{
  "purpose": "inspection-only",
  "sourceGroup": "original-recording-001",
  "mediaRights": "operator declaration; not independently verified",
  "videoSha256": "replace-with-the-input-video-sha256"
}
```

`videoSha256` is optional, but must match when supplied. A rights declaration is metadata, not permission verification. Use a local RTMPose COCO17 manifest following `ml/models.example.json` with verified artifact paths and the actual model input color order.

```sh
python -m ml.inspect_rgb_pose input.mp4 \
  --manifest local-models/manifest.json \
  --provenance provenance.json \
  --start-ms 8500 --duration-ms 1000 --max-frames 30 \
  --output inspection-new
```

The output directory must not exist. Every person above the fixed detector threshold is retained. Box order is local to a frame, not a stable identity; no boxer is selected by image-left/right, clothing color or size. A zero-box frame never invokes whole-image pose fallback. Models use CPU explicitly, and native pose scores remain unclamped, including values above one. These scores do not establish joint accuracy or identity.

`features.npz` contains numeric arrays and can be opened with `numpy.load(path, allow_pickle=False)`:

| Array                       | Meaning                                                       |
| --------------------------- | ------------------------------------------------------------- |
| `decoder_ms`                | Decoder-reported frame timestamps                             |
| `scene_rgb`                 | RGB scene pixels, N×112×112×3                                 |
| `causal_scene_delta`        | Signed current-minus-previous scene pixels; first delta zero  |
| `instance_frame_index`      | Source sample for each unassigned person instance             |
| `person_rgb`                | Person RGB crops with a bounded 15% margin                    |
| `native_xy`, `native_score` | Normalized COCO17 coordinates and raw SimCC scores            |
| `bbox_xyxy`, `crop_xyxy`    | Detector and crop bounds in full-image normalized coordinates |

`inspection.json` includes decoded frame indices, pixel hashes, raw detections, frame-local person instances, model hashes, input/source hashes and coverage. Inputs, model files and adapter helpers are fingerprinted before processing and checked again before the success manifest is published. Nonfinite coordinates/scores fail the run. A failed run may leave partial image/tensor files, but no completed success manifest; use a new directory to retry.

OpenCV timestamps are decoder-relative and are **not verified native/container PTS**. The command neither derives timestamps from frame index/FPS nor infers annotation alignment. Match annotation frame indexing to a separately verified native PTS table before downstream use. Coverage distinguishes an observed interval boundary, a frame limit, and decoder end/read failure; reaching decoder end does not certify complete decoding. `observationExtractionMs` includes pose inference and feature packaging, while extraction wall time excludes model loading. Neither is a live latency benchmark.

Bounds are 300 delivered frames, 1,000 total person instances, 32 instances per frame, 64 MiB of retained numeric payload, and 4096×2160 decoded pixels. Sequential decoding, including skipped preroll, is limited to 3,600 decoded frames and a 120-second loop deadline checked between frames. Start time is capped at 120 seconds and requested duration at 30 seconds. These are payload/work limits, not a process RSS guarantee or a hard interrupt for an individual decoder/inference call. Exceeding a limit fails rather than silently dropping people or choosing an identity.

All exports state `trainingEligible: false`. Before training, review media rights, stable person identity, frame-to-time mapping, annotation scope/background, occlusion masks, and splits grouped by original recording and subject. Unlabeled intervals are not automatically background. RGB context can complement pose, but a paired RGB+pose versus pose-only experiment must demonstrate any benefit on fixed labels and splits.

Tests run with `python -m unittest ml.tests.test_inspect_rgb_pose -v`; optional image/numerical cases skip without NumPy/OpenCV. The generic provenance and resource checks remain available in standard-library CI. Private prototypes and their output hashes remain separate from this public adapter version.
