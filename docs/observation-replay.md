# Replay external native observations

`ml.replay_observations` runs the existing recognizer over saved local-pose video observations, without creating a browser Session or guessing stance. It retains physical `left`/`right` hands and `straight`/`hook`/`uppercut` families. This makes external action diagnosis possible when jab/cross roles, complete labels and exact peaks are unavailable.

Use the existing optional [recognizer environment](personal-recognition.md). No models are downloaded, service settings changed, source files overwritten or accuracy scores calculated. Keep all inputs, weights and outputs under ignored `data/`.

```sh
python -m ml.replay_observations data/external-observations.json \
  --provenance data/external-observations-provenance.json \
  --manifest data/frozen-recognizer/manifest.json \
  --output-dir data/pilot/new-observation-replay \
  --trace
```

For an exact replay of already captured decisions, add `--require-captured-parity`. Every original recognition object must then match, excluding only `inferenceMs`. The first mismatch is retained and fails the run. Use this only when the saved sequence includes the same original cold start and model; missing live warmup or a different model can legitimately produce different outputs. This is software parity, not recognition accuracy.

## Input contract

The input is the local-pose video-observation envelope used by the external inspection harness:

- `sourceId`, `sourceSha256` (the original video hash), `sourceFrameCount` and `modelInfo` from the native service.
- `frames[]` in original presentation order, each containing `sourceFrameIndex0Based`, `t` and `result`.
- Each `result` contains the service's exact `t`, `width`, `height`, `landmarks` and `estimator`. Native landmarks are either empty or the 33-slot service mapping; every point has finite `x`, `y` and native `score`. Scores above one and original coordinates are preserved. Compatibility `visibility` cannot substitute for a missing native score.
- Optional original `result.recognition` stays separate from new decisions. Other source evidence, including original native keypoints and JPEG hashes, remains in the pinned source file.

The recognizer receives only those five frame fields. Saved event decisions, reference annotations, injected grid times and source labels never become frame-level model inputs. The tool accepts no action-reference file and never creates stance-dependent roles, form judgments, masks or training targets. Use [personal replay](personal-replay.md) for valid labeled browser Sessions and their paired accuracy evaluation.

The separate provenance JSON records declarations explicitly:

```json
{
  "purpose": "inspection-only",
  "sourceGroup": "one-original-recording-group",
  "videoSha256": "replace-with-the-64-character-original-video-sha256",
  "timestampSemantics": "native presentation milliseconds, original origin retained",
  "resetBeforeFrameIndices": []
}
```

`sourceGroup` records a supplied grouping, not a verified actor identity or independent-person split. Record unknown identity as unknown in your source audit; do not assign different people merely because filenames differ. The video hash binds this declaration to the observation envelope; this command does not decode or rehash the original video. Pin the original video and its native timestamp map in the acquisition protocol separately. Likewise, `timestampSemantics` is retained as a declaration, not proof of pixel/time synchronization.

Source timestamps must be finite, nonnegative and strictly increasing. Nonzero origins and variable intervals stay exact: no frame-index/FPS reconstruction, time-zero shift, interpolation or frame skipping occurs. Absolute source indices may have gaps, which remain visible; the count describes supplied observations, not proof of full-video coverage. Failed/partial envelopes must first be preserved and, if needed, converted into an explicitly documented separate completed subset.

## Editorial cuts and state

External b-roll may switch actors or viewpoints without a timestamp gap. Review continuity before interpreting a trace. Declare known shot boundaries using sorted, unique `resetBeforeFrameIndices`, referring to actual supplied source indices. A declared boundary clears recognizer state **before** that observation and is recorded in the trace. It must come from source/shot inspection, not punch annotations or a search for better recognition. An explicit reset can break captured parity; that failure is evidence, not a reason to omit the reset silently.

The normal runtime also resets on source gaps over 150 ms and image-size changes. With no declared boundaries, all source observations are consumed continuously. The tool does not detect cuts, select actors, establish continuity or fill missing context. There is no end-of-file flush or extra guard pose. Unfinished candidates remain explicitly unfinished.

## Outputs and failures

The output directory must be new. Inputs, provenance, model manifest, both recognizer weight files and executable helpers are hashed before inference and checked again afterward. The normal model loader enforces the pose/detector hashes, native score policy, feature code and runtime compatibility.

- `protocol.json` and `input-lock.json` describe the run and exact inputs.
- `decisions.jsonl` preserves source indices/times, new recognition, original recognition and any available parity result.
- Optional `trace.jsonl` observes new causal grid ticks, native-source times, uncalibrated family support, reset epochs and active/pending candidates. It does not invent a rejection reason when none is exposed by the runtime.
- `summary.json` is written only after successful processing, input checks and cleanup. It contains physical-hand events, resets, unfinished candidates and recognizer-only timing.
- A failed run retains partial diagnostics and `failure.json` when writable, with no completion summary. Do not overwrite it or silently substitute a later attempt.

Inputs are bounded to 128 MiB and 10,000 observations; provenance/model manifests to 100,000 bytes. Each JSONL output is bounded to 128 MiB. Native source/image association and reviewed annotations remain separate evidence. A matching replay or fast recognition-only runtime establishes neither accurate punches nor real-time webcam throughput.
