# Reproducible local personal-recognizer training

`ml.train_personal` trains the existing 40-input per-arm causal TCN and writes a **new, opt-in local bundle**. It never downloads models, changes a running service, edits recordings, or activates its output. Recognition still does not grade technique. Keep all training manifests, videos, labels, checkpoints and generated reports in ignored local storage.

This is a new public training protocol, `personal-training-1`. It retains the deployed feature architecture and fixed optimizer recipe. Its audited target masks and runtime gap resets differ from the older private scripts; do not claim it reproduces their exact checkpoint or metrics.

## Inputs

First create an [action-dataset-1 manifest](action-dataset.md) containing **every pose version** you want to audit. Each entry requires:

- The labeled schema 1.0 Session, original local `video`, and verified `videoSha256`.
- Physical hand, stance, participant/day, source recording group, rights declaration and explicit label scope. Model predictions are not reference labels.
- A `train`, `validation` or `test` split. Live and offline poses from the same original recording must have the same source group, split, original video, clock offset, frozen annotations and label scope. Renaming a pose export does not create an independent recording.
- Native RTMPose-M model fingerprints, input-color contract, detector policy and `.55` native-score policy matching the supplied base bundle. Native scores are not calibrated confidence or MediaPipe visibility. Finite raw scores above one remain intact; only the existing feature code clips its numeric inputs.

`sourceGroup` and original video SHA can never cross splits. A video must have exactly one source group. The dataset's stricter `participant-day` or `participant` split policy is also enforced. Honest grouping is still needed for re-encoded or cropped copies; a file hash cannot establish annotation truth or participant identity. Supplied videos fingerprint the operator-declared source association; contradictory attached video hashes fail.

Supply a locally provisioned runtime-compatible base recognizer manifest and its weights/license files. It provides the **fixed external family model and pose contract only**. The new TCN is initialized from the seed; this CLI does not warm-start or update the base checkpoint.

Save a separate training manifest beside your dataset manifest:

```json
{
  "schemaVersion": "personal-training-1",
  "dataset": { "path": "dataset.json", "sha256": "REPLACE_WITH_SHA256" },
  "baseRecognizer": {
    "path": "base-bundle/manifest.json",
    "sha256": "REPLACE_WITH_SHA256"
  },
  "seed": 41729,
  "steps": 1800,
  "entries": [
    {
      "id": "round-a-offline",
      "sessionSha256": "REPLACE_WITH_SESSION_SHA256",
      "cadenceFps": [15, 20, 25, 30]
    },
    {
      "id": "round-a-live",
      "sessionSha256": "REPLACE_WITH_SESSION_SHA256",
      "cadenceFps": [30]
    },
    {
      "id": "reserved-round-b",
      "sessionSha256": "REPLACE_WITH_SESSION_SHA256",
      "cadenceFps": [30]
    }
  ]
}
```

Every dataset entry—including reserved entries—needs an explicit hash and cadence list here. Entry order in the dataset fixes the training sequence order. Paths are local and relative to their own manifest; absolute paths work. Compute SHA-256 from actual bytes, for example `shasum -a 256 path/to/file`.

## Run

Audit does not need Torch or NumPy and creates no files:

```sh
python3 -m ml.train_personal --manifest data/local/train.json --audit-only
```

For training, install the optional local dependencies explicitly in your environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -r ml/requirements-recognizer.txt
.venv/bin/python -m ml.train_personal \
  --manifest data/local/train.json \
  --output data/local/runs/personal-001
```

The output directory must not exist. A run uses CPU, one Torch thread and deterministic algorithms. It refuses malformed labels, mismatched hashes/policies, split leakage, missing class support, nonfinite coordinates/scores, or changed inputs/source code during training. Limits are 128 entries, 6,000 observed frames per entry and grid sequence, 200,000 total input frames, and 10,000 training steps. These are resource limits, not recommended dataset sizes.

Only `train` entries reach the optimizer. Reserved data is audited but never used for class weights, gradients, stopping, checkpoint selection or reported accuracy. The fixed step count selects the last checkpoint. An all-train manifest is permitted as development fitting; it supplies no held-out evidence.

## What is reproduced

The CLI imports the runtime's exact `_networks`, `grid` and `temporal_features`: 40 inputs, 48 channels, causal dilations 1/2/4/8, and four families—background, straight, hook and uppercut—independently for each physical arm. Jab/cross naming remains stance-dependent downstream.

Source-cadence simulation chooses past observations only, then uses the existing 30 Hz hold grid with sample age and new-observation features. It does not interpolate joint coordinates. Gaps over 150 ms and resolution changes start new contexts, matching runtime resets. Targets describe each grid tick; unknown supervision and invalid own-arm observations are excluded from loss.

The fixed recipe is AdamW at `.002`, weight decay `.01`, feature noise standard deviation `.015`, gradient norm clipping at `1`, and square-root inverse-frequency weights normalized to mean one. Odd steps horizontally reflect relative x and x-velocity channels, preserving physical-hand family labels. Deterministic seeds reproduce tensors within the same recorded dependency/platform environment; different versions or hardware are not promised bit-identical outputs.

Targets use shared dataset policy `half-open-uncertainty-first-v1`:

- Intervals are `[startMs, endMs)`. Same-arm overlapping positive labels require adjudication.
- Only explicitly complete **all-family** annotation establishes background. Straight-only or incomplete label scope leaves unlabeled time masked.
- Unknown-hand punches mask both arms. Hand-specific unobservable intervals mask that arm; unknown has priority over positives.
- At least one usable supervised target in every family is required. This only prevents an unsupported output class; it does not establish adequate training coverage.

## Output and review

The new directory contains `model.pt`, the copied fixed external model/license, `manifest.json`, `protocol.json`, `audit.json`, and exact input-manifest snapshots. The protocol records input/annotation/video/source hashes, split membership, cadence lists, observed class counts, seed, steps, losses and dependency versions. Snapshot relative paths refer to the original manifest locations recorded in the protocol; they are evidence, not a portable dataset archive.

Before publishing the manifest, the CLI loads the bundle through the actual `ModelBundle.load` compatibility checks and verifies exact temporal-state parity. The manifest is written last; an interrupted incomplete output must be discarded or inspected, not reused as a successful run. Existing output is never overwritten.

**Do not select a bundle because training loss decreased.** Replay it locally against untouched grouped recordings and use the [strict interval and occurrence diagnostics](recognition-evaluation.md). Retain false detections, unknown intervals and latency results. Once a recording influences fitting or model selection, describe it as development data. The service requires an explicit later `--recognizer /absolute/path/to/new/manifest.json`; this command does not make that change for you.

Software checks, including a tiny synthetic fit, runtime loading, deterministic weights and causal prefix parity:

```sh
.venv/bin/python -m unittest ml.tests.test_train_personal -v
```

The synthetic fixture tests code behavior only. It does not validate boxing recognition, score calibration, identity tracking or coaching.
