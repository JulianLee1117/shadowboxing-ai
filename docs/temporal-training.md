# Experimental causal temporal recognition

`ml/temporal.py` trains a small per-arm straight-action recognizer on saved pose observations. It is a runnable offline research baseline. It does not retrain the pose estimator, validate boxing form, or replace the browser detector. The first bounded experiment overfit its small same-person corpus. A five-session follow-up with the same protocol recovered more actions but still produced many more false events than the deterministic detector. Its weights remain private and are not enabled in the app.

## Run locally

Install the optional reference dependencies in a separate environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements-train.txt
.venv/bin/python -m ml.temporal train \
  data/labels/session-a.json data/labels/session-b.json data/labels/session-c.json \
  --output data/experiments/temporal-run-001
```

The input files must be complete schema 1.0 session exports with anatomical hand labels, explicit stance and `annotationsComplete: true`. Unobservable intervals must be labeled. Each session needs at least 100 frames, 5 seconds, 4 labeled straight actions and both hands represented. Training also requires at least 30 known positive and negative arm-frames. These are software preconditions, not evidence that a dataset is large or diverse enough.

The output directory must not already exist. Existing captures, reports and checkpoints are never overwritten. Keep footage, annotations and generated weights under ignored `data/`; do not commit personal recordings or checkpoints.

Run a checkpoint against a different session:

```sh
.venv/bin/python -m ml.temporal predict \
  --checkpoint data/experiments/temporal-run-001/final-all-development.weights.json \
  --input data/labels/session-next.json \
  --output data/experiments/session-next-prediction.json
python3 -m ml.evaluate data/experiments/session-next-prediction.json \
  --output data/experiments/session-next-evaluation.json
```

Prediction does not read annotations to choose boundaries. Output provenance flags exact captures that were used for training. Final all-session weights have no independent evaluation within their training corpus; cross-validation results belong to different fold checkpoints.

## Fixed initial protocol

The source records protocol version, feature order, source hash, seed, dependency versions, input-file/capture/annotation fingerprints, all split assignments and every hyperparameter before fitting. The initial CPU experiment uses two causal convolutions with 24 channels, width 5 and dilations 1/2; the receptive field is 13 observed samples. Their wall-clock span varies with actual frame cadence. All padding is on the left. The shared network processes each anatomical arm separately and receives explicit physical-hand and lead-hand indicators.

Inputs include aspect-corrected image coordinates centered on the shoulders and scaled by torso length, elbow angle, shoulder-to-wrist reach, measured confidence, missing/observed masks, elapsed source time and backward differences. Invalid coordinates become zero with a separate missing mask. Derivatives require consecutive valid observations; they are never interpolated. Context resets at non-forward timestamps, source gaps over 200 ms and image-dimension changes. A low-confidence frame remains a masked observed input, so the network may use recent history; that is a learned action estimate, not a recovered landmark or proof that an occluded movement occurred.

Normalization and positive-class weighting use training sessions only. Continuous known background is included. `unobservable` intervals and straight-action annotations with unknown physical hand are excluded from the loss. Hook/uppercut/other annotations remain negative examples for this straight-only model. These features cannot establish whether confident landmark predictions are anatomically correct.

With three sessions, each fold trains on one complete session, reports validation loss on another, and evaluates on the third. Each session appears as test exactly once. More sessions leave one for validation and one for test, using the remaining sessions for training. There is no frame-level random split, early stopping, checkpoint selection, hyperparameter sweep or threshold search. The fixed initial fit uses 120 AdamW steps, learning rate0.002, weight decay0.001 and a reproducible seed. Validation loss is diagnostic only.

This is session-held-out exploratory development. If all recordings belong to the same person/day, it cannot establish unseen-person or unseen-environment performance. Whole-session splits also cannot repair label decisions or engineering choices already informed by those sessions. Exact duplicate captures are rejected even when IDs or inference telemetry change; independently re-encoded or altered copies of one recording still require dataset-owner identity control.

## Events and evaluation

Per-arm sigmoid outputs are uncalibrated action scores. Two observed scores at or above 0.6 start a segment; two below 0.4 finalize it, using the first low timestamp as its end and the second as causal emission time. The decoder accepts 120–1800 ms segments, aborts across source gaps and leaves unfinished tails uncounted. Adjacent punches can merge, and guard movement can create false positives; these are measured failures rather than corrected using labels. Probability-maximum time is not a measured physical extension peak. This baseline provides no guard-return or technique-quality judgment.

The existing evaluator performs one-to-one exact class/anatomical-hand matching at temporal IoU≥0.5, with per-class counts and false events per minute. Unknown intervals follow the same evaluator exclusion policy used for deterministic replays. Compare both systems on identical captures, annotations and exclusions. A detector developed using those clips is a development comparator, not equivalent to a newly held-out population test. Report false events as well as matched punches; frame classification accuracy is insufficient.

`protocol.json`, fold weights, fold probability timelines, predicted test sessions, and `cross-validation-report.json` preserve the experiment. `final-all-development.weights.json` trains the identical architecture on all supplied sessions afterward and is explicitly marked without a remaining independent test. Numeric JSON includes weights and preprocessing and can be inspected without unpickling code. There is currently no verified browser inference implementation; do not imply that exporting JSON establishes Python/browser parity.

## Verification and next evidence

Run optional training tests with:

```sh
.venv/bin/python -m unittest ml.tests.test_temporal -v
```

Tests verify causal prefix invariance, future-frame perturbations, missing-coordinate masks, context reset at gaps, unknown-label masking, disjoint session roles, duplicate-capture rejection, event-gap/tail behavior, checkpoint round trips and immutable outputs. They are synthetic software fixtures, not accuracy evidence. The standard-library test suite skips this module when optional training dependencies are absent.

A practical next collection is a few natural 30-second rounds across two or three days, including pauses, adjusting clothing, raising hands, changing guard and other non-punch gestures. Label the original video independently before viewing predictions; explicitly mark uncertain or occluded intervals. Reserve at least one later session untouched for evaluation. This is a targeted starting protocol, with no fixed recording count that guarantees improvement.

Before a learned model can replace the existing detector, collect materially more independent continuous sessions with guard-motion negatives, fast repeated punches, partial returns, different views and actual occlusions. Add manually verified anatomical keypoints around wrong-pose cases. Keep recognition labels separate from coach-adjudicated form labels, then evaluate by held-out session and eventually participant. Keep a new test partition untouched by model selection.

The implementation follows PyTorch's [Conv1d semantics](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Conv1d.html), uses weighted [binary cross-entropy with logits](https://docs.pytorch.org/docs/2.10/generated/torch.nn.BCEWithLogitsLoss.html), and seeds supported deterministic CPU execution. PyTorch cautions that reproducibility is not guaranteed across releases/platforms, so the report records the reference environment and seed. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.10/notes/randomness.html)
