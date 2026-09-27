# Offline benchmarking

The evaluator is runnable with Python 3.10+ and the standard library. It reads the application's version `1.0` JSON session export. It does not upload video, download a model, or require a Python environment installation. Tests validate the scoring logic; no real-world boxing accuracy has been established.

Run these commands from the repository root:

```bash
python3 -m unittest discover -s ml/tests -v
python3 -m ml.evaluate /absolute/path/session.json --output /absolute/path/report.json
```

The second command always reports capture/inference diagnostics. Event accuracy is withheld unless `annotationsComplete` is explicitly `true` in the export, or the operator deliberately asserts completeness:

```bash
python3 -m ml.evaluate /absolute/path/day1.json /absolute/path/day2.json \
  --annotations-complete --output /absolute/path/report.json
```

Annotate **every jab/cross throughout the full recording**, including missed detections, before asserting completeness. Do not annotate only the app's proposed punches. Remaining time is treated as negative/background evidence. Mark intervals that cannot be assessed as `unobservable`; use `other` for observable non-target movement. A few hand-picked examples cannot establish precision, recall, or false activations per minute. An empty but fully reviewed idle recording is valid; an unreviewed empty annotation array is not.

## Matching and report semantics

- Match exact technique label and anatomical `left`/`right` hand, with temporal intersection-over-union **at least 0.5**. The preview's mirror setting is irrelevant. A wrong label or wrong hand produces an unmatched prediction and a missed annotated event.
- Use maximum-cardinality one-to-one bipartite matching. Descending IoU orders candidate edges, but the objective is the number of valid pairs, not the summed IoU. Duplicate detections cannot both match one punch.
- The standalone Python evaluator retains its historical default recall scope of jab and cross. For the six-punch browser detector, pass `--labels jab,cross,hook,uppercut`; `npm run benchmark` now does this by default and records the protocol. An omitted reference class is outside the recall denominator; unmatched predictions still count as false positives. `unsupportedTruthCount` makes omitted classes explicit. Use `--labels jab,cross` only for a deliberately restricted historical comparison, and freeze that protocol before testing.
- Predictions fully contained in the union of `unobservable` intervals or target annotations with `hand: "unknown"` are excluded, with IDs and intervals listed. Excluded duration is subtracted from false-event exposure. A prediction crossing an exclusion boundary is still evaluated. Conflicting assessable truth/exclusion intervals cause a clear error requiring annotation adjudication.
- Precision is TP/(TP+FP), recall TP/(TP+FN), and F1 2TP/(2TP+FP+FN). An undefined ratio is JSON `null`, not zero or perfect accuracy. Per-class counts, aggregate counts, per-session results and matched IDs are retained for inspection.
- False events per minute is unmatched evaluated predictions divided by evaluated recording minutes. It is **not** false coaching cues per minute: cue scheduling and coach judgment require separate annotation.
- Inference and timing percentiles use linear interpolation between sorted neighboring ranks. Aggregate event precision pools counts rather than averaging sessions. A mixed complete/incomplete input set is labeled `partial_annotated_benchmark` and reports `accuracySessionCount`.

Synthetic/demo sessions are rejected by default, including either `source: "demo"` or `model: "synthetic"`. `--allow-synthetic` exists solely for software checks and marks the **entire report** `synthetic_software_check`, even when mixed with real sessions. Do not use this flag for a hardware or coaching validation report.

The evaluator rejects unsupported schema versions, malformed intervals, duplicate event/annotation/session IDs, invalid physical hands, decreasing frame timestamps and nonfinite required timing values. Output cannot overwrite a supplied input session. The original export remains unchanged.

## Coverage and timing

Default required landmarks are MediaPipe indices **11,12,13,14,15,16,23,24**: both shoulders, elbows, wrists and hips. A joint must have finite normalized coordinates inside the image and available confidence at least 0.5; when both visibility and presence exist, both must pass. Missing points or confidence are unassessable. Change these explicitly with `--required-joints` and `--min-confidence` when defining another criterion.

This is **landmark coverage among processed frames**, not the master plan's criterion assessment coverage. Missing/dropped frames are not automatically counted as assessable. Frame cadence, duplicate timestamps, large gaps, reported skipped frames and per-joint coverage accompany the percentage so sparse processing cannot masquerade as continuous observation.

`PoseFrame.inferenceMs` supplies actual per-frame inference timings. The evaluator recomputes p50/p95 from frames rather than trusting the exported summary. `frameAgeMs`, when present, measures browser frame callback to completed inference; it does not measure sensor capture delay. `PunchEvent.detectedAtMs`, when present, is compared with the **matched annotation's end** to compute event-finalization delay on the shared source-relative clock. Timing coverage is reported. Missing telemetry is not synthesized from punch duration. Negative finalization delay means detection preceded the annotated end; inspect the boundary convention rather than clipping the value to zero.

The report does not automatically declare any [release gate](project-plan.md#release-gates) passed. Coach-labeled criterion correctness, unconditional and eligible-event coverage, confidence bounds accounting for session/participant clustering, emitted cue errors, and retained learning improvement need separate study data. The current app's heuristic `score` is not a calibrated confidence probability and is not used for matching.

## Reproducible personal and cohort tests

Keep entire recording sessions together before trimming or augmentation. Tune on development sessions, freeze parameters and the annotation guide, then evaluate later-day sessions. For a broader product, reserve entire people and environments and report those separately from personal results. Record stance, device, room, view, light, model manifest and annotation provenance. Keep footage and private JSON exports out of Git.

The initial app only detects experimental jab/cross events. The correct first report is allowed to show poor recall or precision; the purpose is to find the error sources before approving cues. Inspect false-positive IDs, missed events, wrong-hand cases and low-coverage frames in replay. A coach should label execution criteria independently of the model's output.

## Optional RTMPose / RTMW extraction

`ml/extract.py` is an optional saved-video adapter targeting **rtmlib 0.0.16**, with pinned dependencies in `ml/requirements-extract.txt`. RTMPose-M and RTMW-L inference have run locally on the target Mac against development footage. Selected joint estimates improve over Full, but these experiments do not establish a replacement live pipeline. The adapter defaults to verified ONNX Runtime CPU sessions; `--provider coreml` explicitly opts into Core ML with CPU node fallback. The [official rtmlib API](https://github.com/Tau-J/rtmlib) supports explicit detector/pose paths and MMPose ordering with `to_openpose=False`.

Create a separate environment when running that experiment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements-extract.txt
.venv/bin/python -m ml.extract /absolute/path/clip.mp4 \
  --manifest /absolute/path/model-manifest.json \
  --output /absolute/path/rtm-poses.json
```

On a supported Mac, add `--provider coreml` and choose a fresh output path. The extractor fails when Core ML is unavailable or no Core ML node execution is observed; it does not silently relabel an entirely CPU run. It retains CPU fallback for unsupported graph nodes, disables whole-session automatic fallback, and writes ONNX Runtime profiles beside the output in `OUTPUT.profiles/`. Profiles, provider settings and session-creation timing are recorded separately from frame inference timing. Keep these artifacts private with the source recording. Existing outputs or profile directories are not overwritten.

The detector uses `RequireStaticInputShapes=1`, matching the live service. This keeps dynamic post-NMS outputs on CPU graph partitions: Core ML cannot execute a zero-length dynamic output when no person is detected. Pose retains its default graph policy. The exact per-model options are recorded in `providerOptionsByModel`; this is not a whole-session CPU retry. Zero people still produces an abstention. Older offline outputs using the previous dynamic detector partition policy remain separate evidence; do not assume numerical equivalence or overwrite them.

Provider registration alone does not prove acceleration or identify GPU/Neural Engine execution. The adapter uses provider defaults without a compute-unit override; the [ONNX Runtime Core ML documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) describes the available hardware and format options. Profiles identify executed provider partitions, not their share of compute. Profiling also adds overhead. Compare identical decoded inputs, coordinates, abstentions and timing before considering a live integration: numerically different low-confidence joints can remain even when arm estimates look similar. This option is an offline research tool, not browser Core ML support or sustained real-time validation.

Use `ml/models.example.json` as the manifest format. Download the chosen model artifacts from the recorded official URLs, inspect their terms, extract the ONNX files locally, and set each `path` relative to the manifest file. The example records a YOLOX-m detector and RTMPose-m body model. Input sizes are `[width,height]`. The script **does not download weights or silently select default models**. It records original source URLs, local paths, file SHA-256, input sizes, package versions, platform and input-video SHA-256. An optional `expectedSha256` is enforced. A newly computed local hash is provenance, not independent proof of artifact authenticity.

Set `pose.inputColorOrder` from the chosen model's exported preprocessing contract. The example uses `RGB`, matching its archive's `pipeline.json` `Normalize.to_rgb=true`. In the pinned rtmlib version, the adapter explicitly reorders the pose input; the detector continues to receive BGR. Omission retains upstream BGR behavior and is recorded, rather than silently changing an old experiment. Verify every new model's contract separately.

For RTMW, set `family` to `rtmw-wholebody`, select a verified 133-point COCO-WholeBody model and its correct input size from the official model zoo, and use its source URL. The extractor rejects a point-count mismatch. It abstains if detection yields zero or multiple people rather than silently changing identity. Mirrors and background people may therefore reduce coverage; inspect the person-count diagnostics.

Mapping is deliberately conservative:

| COCO / first 17 COCO-WholeBody index | Anatomical point | MediaPipe output index |
|---|---|---|
| 0 | Nose | 0 |
| 5, 6 | Left/right shoulder | 11, 12 |
| 7, 8 | Left/right elbow | 13, 14 |
| 9, 10 | Left/right wrist | 15, 16 |
| 11, 12 | Left/right hip | 23, 24 |
| 13, 14 | Left/right knee | 25, 26 |
| 15, 16 | Left/right ankle | 27, 28 |

All other MediaPipe slots have zero confidence. No eyes, ears, fingers, heels or foot-index landmarks are approximated. No `z` or world coordinates are fabricated. The research extractor retains its legacy compatibility `visibility` mapping plus separate unclamped native scores in `nativeKeypoints`. Native replay/live frames use the `score` field and an explicit SimCC estimator policy; never treat the compatibility field as calibrated MediaPipe visibility. Preserve native scores when adapting research output. Coordinates are divided by their respective image width and height, and anatomical sides are never mirrored. COCO-WholeBody ordering is defined by [MMPose's dataset metadata](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py).

The adapter writes `artifactType: "research-pose-series"` with the same `PoseFrame` layout, not a browser `Session`: the extractor does not recognize punches and its research artifact has a different provenance contract. The app now supports native RTMPose/RTMW through the local bridge. Do not rename it to MediaPipe `full` or submit an empty prediction list as a model accuracy benchmark. Native keypoints are also retained separately in their original COCO order. Use the replay integration below to compare downstream events while retaining the actual model identity.

Default timestamps come from OpenCV's video position in milliseconds, normalized to the first decoded frame. Non-increasing/unavailable timestamps fail clearly. Only for a verified constant-frame-rate source may `--assume-cfr` use frame index / reported FPS; this assumption is recorded. `--max-frames 30` provides a bounded smoke test and marks the output truncated. Pose timings include detector plus pose and first-call warm-up, and exclude decoding; the final frame duration is estimated and labeled as such. A meaningful performance test needs a complete representative clip and separate warm-up accounting.

For paired model comparisons, `--frames-manifest /path/frames.json` reads exactly the ordered local decoded images supplied to every model. Format: `{"frames":[{"file":"000001.jpg","ptsMs":0},{"file":"000002.jpg","ptsMs":33.367}]}`. Files must exist inside the manifest directory and timestamps must strictly increase. Their hashes and the manifest hash are retained. The required video argument remains source provenance; verify that the decoded images came from that video. Decoding, JPEG encoding, initialization and tracker history can materially change poses, so compare challengers against a fresh baseline on **the same image sequence**, not against live-capture landmarks.

## Replay and compare current counting rules

For the routine development loop, independently label complete sessions once, then run:

```bash
npm run benchmark -- data/pilot/day1-labeled.json data/pilot/day2-labeled.json \
  --output-dir data/pilot/runs/unique-experiment
```

This command preserves all inputs, refuses an existing output directory, replays the current production TypeScript detector, and evaluates saved and new detections with identical labels and matching rules. `comparison.json` reports matches, misses, unmatched predictions, and exactly which reference actions were recovered or lost. Each derived replay records input and detector SHA-256 fingerprints. An edit during a multi-session run causes failure rather than mixing detector versions. Complete annotation flags are required; synthetic sessions are rejected. A failed run may leave partial diagnostic files, but only a successful run writes `comparison.json`. Saved sessions may use different detector versions: for a controlled version-to-version comparison, supply derived sessions from one previously frozen version.

For one session or a replacement pose model:

```bash
npm run replay -- data/pilot/session-labeled.json --output data/pilot/current-rules.json
npm run replay -- data/pilot/session-labeled.json \
  --poses data/pilot/rtm-poses.json --model rtmpose-m-rgb-cpu \
  --video data/pilot/original.webm --pose-offset-ms=0 \
  --output data/pilot/rtm-events.json
python3 -m ml.evaluate data/pilot/rtm-events.json --output data/pilot/rtm-report.json
```

Replacement replay requires an explicit research model identifier, complete pose-series flag, model and timestamp provenance, and a video whose SHA matches the extraction source. The operator must still verify the labeled session belongs to that video. Offset maps pose timestamps onto the session clock (`sessionTime = poseTime + offset`); supply it explicitly for sessions with a nonzero video origin. No interpolation or missing peak reconstruction occurs. The result is a separate `detector-benchmark-session`, accepted by the evaluator only with its required provenance. Unknown offline capture skips remain null. Browser model identities cannot be reused for challenger models.

These are development comparisons. Replaying tuned recordings cannot establish held-out recognition accuracy, calibrated confidence, sustained live performance or correct form advice. Keep all footage, labels and experiment outputs under ignored `data/pilot/`.
