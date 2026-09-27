# Compare personal recognizers

`ml.replay_personal` runs the actual causal native recognizer over saved RTMPose/RTMW observations. It does not start a camera, upload data, retrain weights, change the service or overwrite recordings. Use the optional local recognizer environment described in [personal recognition](personal-recognition.md).

```sh
.venv/bin/python -m ml.replay_personal \
  data/round-one-labeled.json data/round-two-labeled.json \
  --manifest data/candidate/manifest.json \
  --baseline-manifest data/current/manifest.json \
  --labels jab,cross,hook,uppercut \
  --peak-reference data/round-one-video-reference.json \
  --peak-reference data/round-two-video-reference.json \
  --trace \
  --output-dir data/comparisons/new-experiment
```

The output directory must be new. Supply `--labels` once for a shared reviewed recall scope, or once per input in input order. Use all four labels only when the reference review covers all four families. The historical strict evaluator still counts unmatched predictions outside the recall scope; on a partial-family review, inspect per-class results and video before calling those predictions false actions. Omit both peak-reference options when independent video peaks are unavailable; the strict interval evaluation still runs. Incomplete annotations withhold recognition accuracy instead of treating unreviewed time as background. Native pose/model provenance, finite causal decisions and strictly increasing source times are checked. Every bundle must pass its runtime, feature, pose and checkpoint hashes. No model is downloaded or silently substituted.

## Three distinct results

For each supplied input, the command writes:

1. **Captured report:** evaluates the original exported events without changing them. This is the saved evidence; it may have used pre-round warmup that the export does not contain.
2. **Baseline replay:** starts the explicitly supplied baseline model at the first exported observation, with an empty state.
3. **Candidate replay:** starts the candidate in exactly the same way on the same observed poses. Stance maps the emitted physical hand/family to jab/cross and lead/rear roles afterward.

Both replays retain the original coordinates, timestamps, annotations, completeness assertion and recorded pose timing. Only new recognizer decisions and mapped events replace their old counterparts. There is no end-of-file flush with invented future poses. Inference receives only time, image dimensions, native landmarks and estimator metadata; saved predictions, annotations, internal grid fields and drill labels are not model inputs.

`comparison.json` is written only after all inputs finish and their hashes still match. It records exact recovered, lost and retained reference IDs for the paired strict metric, plus separately named peak-occurrence changes when explicit peaks are complete. Individual reports retain unmatched event IDs, full matches and timing. It never sums multiple pose versions into an apparent independent recording count. A partially failed run may leave diagnostic files but has no completed comparison.

Cold replay can differ from the actual live output because the original grid phase and pre-round observations may be absent. Never substitute a cold baseline for the captured result when reporting prospective performance. A change fitted or selected using these recordings remains a development result, even if its score improves. See [recognition evaluation](recognition-evaluation.md) for the fixed peak protocol and [personal training](personal-training.md) for recording-group separation.

## Diagnosing a missed action

With `--trace`, each replay also writes JSONL snapshots of the actual runtime state after each source observation:

- New causal grid ticks, their observed source timestamp, per-arm validity, and support for `[background, straight, hook, uppercut]`.
- Active temporal candidates, pending candidates, emitted events, and reset epochs.
- Source-gap and image-size resets. Each new grid tick appears once within its epoch.

The trace observes existing state; it does not modify probabilities, thresholds or event acceptance. It is a diagnostic snapshot, not an invented explanation for a rejection. For example, high family support without a finalized event calls for inspecting duration, observed samples, geometric qualification and arbitration. Low support despite plausible visible trajectories calls for inspecting features and training coverage. Neither a high pose score nor high family support certifies anatomical identity or correct form.

The comparison separately reports recognition-only runtime, maximum buffered entries, and unfinished candidates at the end. Emission delay relative to a predicted peak/end is distinct from independent video-reference delay in the evaluator. Recorded pose inference times remain original evidence; replay wall time and recognition-only milliseconds exclude capture, pose inference, transport and display. A fast replay cannot establish real-time webcam performance.

All outputs can contain private motion evidence. Keep datasets, weights, traces and reports under ignored `data/`.
