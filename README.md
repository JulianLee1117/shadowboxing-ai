# Corner — Shadowboxing AI

A local shadowboxing practice studio for an M3 MacBook Pro and its webcam. Train with guided drills, inspect motion in replay, and collect the evidence needed to build a trustworthy coach.

**Status: working research prototype, September 26, 2026.** Real MediaPipe pose inference runs locally in a worker. Jab/cross recognition is an experimental geometric baseline. Technique corrections and numeric skill scores are not released: this repository has not passed coach-reviewed boxing accuracy or efficacy validation.

## Run locally

Use Node.js 20.19+ (22.12+ recommended) and npm.

```sh
npm ci
npm run models:setup -- --all
npm run dev
```

Open [the local studio](http://127.0.0.1:5173). Models and runtime files are downloaded during setup, then served from your device. Setup without `--all` installs Full only. Exact artifact URLs and SHA-256 hashes are recorded in [the model manifest](model-manifest.json). Weights and copied WASM files are excluded from Git.

Desktop Chromium is the verified browser target. Camera access requires localhost or HTTPS. Safari and mobile-browser inference have not been validated. The responsive interface has been checked at desktop and 390px widths.

## First round

Start with [the five-minute test checklist](docs/first-user-test.md). The studio defaults to one minute of free practice. Model settings and diagnostics are under **More options**; reference labels and telemetry are collapsed in review.

1. Select a drill and stance. The default round is one minute; leave **More options** alone for your first test. **Explore a simulated round** works without a camera; synthetic data is labeled and excluded from real accuracy reports.
2. Enable the camera. Frame your head, shoulders, hips, elbows and hands, including full reach. Improve lighting or try an oblique view if a hand disappears.
3. Raise your left hand and verify that the **L** wrist label follows it. Confirm the setup. Preview mirroring never changes anatomical labels.
4. Turn on **Save round video** if you want original-video replay. It is off by default, and no microphone is requested.
5. Start a round. Drill callouts use available local speech voices and give prompts only. They do not judge technique.
6. Finish and open **Round review**. Scrub and step through frames, reveal experimental detections, add reference labels, export JSON/video, or delete the round.

Imported clips are retained locally with their analysis so labels can be checked against original footage. The import surface states this before analysis. Reanalyzing resets playback and temporal state. Motion-only recordings can inspect model predictions but cannot independently establish whether those predictions were correct.

## Implemented

- Explicit camera lifecycle, permission-error recovery, and immediate track shutdown.
- MediaPipe Full/Lite/Heavy adapter with a worker, one in-flight frame, transfer cleanup, timeouts, and fresh-worker CPU fallback.
- Same-origin assets and a worker network boundary blocking SDK telemetry. Development and preview servers also set a restrictive network CSP.
- Distinct decoded-frame processing with `requestVideoFrameCallback` where available; fallback timing is identified in capture metadata.
- Causal straight-punch candidates with anatomical side/stance mapping and resets for missing joints, time gaps, invalid geometry, and unconfirmed calibration.
- Guided jab, cross, 1–2, and open-capture rounds, local callouts and optional video recording.
- IndexedDB persistence, original-video or skeleton replay, frame stepping, action labels, completeness declaration, JSON/video export and deletion.
- Captured settings, actual execution delegate, timestamps, frame age, inference latency, model provenance and skipped-frame telemetry in exports.
- A Python evaluator with one-to-one temporal matching, precision/recall, class diagnostics, false events per minute, coverage and timing statistics.
- An optional RTMPose/RTMW extraction adapter with explicit model provenance. Real RTM inference is not yet verified on target footage.

## What remains unvalidated

No recorded human boxing sample has been evaluated yet. Runtime tests use synthetic streams, which verify software integration, not pose or punch accuracy. No sustained 20-minute hardware acceptance test or comparison between models on boxing footage has passed. Hooks, uppercuts and other motion can be labeled manually but are not recognized by the live baseline.

Setup confirmation checks tracking, not a correct guard. The detector's score is a heuristic signal, not calibrated confidence. “Return” means movement toward a repetition's starting position, not proof of correct recovery. Front-facing punches and self-occlusion can defeat the projected-geometry baseline.

Cloud review is not connected and no API key is needed. No video leaves the device through this application. The browser database is local storage, not a backup: export recordings you want to preserve. Finishing saves a round; closing or reloading during an unfinished round can lose it.

The next step is multiple consented real sessions plus independently reviewed action labels. Actual technique correction also requires coach-reviewed criteria, examples/counterexamples, visibility rules, and the [release gates](docs/project-plan.md#release-gates).

## Verify

```sh
npm run check
python3 -m unittest discover -s ml/tests -v
npx playwright install chromium
npm run test:e2e
npm audit
```

Browser tests use generated streams and stub physical camera requests. They include actual local model inference and CPU fallback, persistence/export/annotation, camera shutdown, and mobile layout. Tests need local model assets. Synthetic blank-video timings must not be quoted as boxing performance.

For a production build, run `npm run build` and `npm run preview`. Assets are copied into `dist`. Configure equivalent CSP/permissions headers if serving from a different host.

## Evaluate recordings

```sh
python3 -m ml.evaluate /path/to/session.json --output /path/to/report.json
```

Without complete labels the tool reports capture quality only. Complete annotations or an explicit `--annotations-complete` assertion are required for recognition metrics. Synthetic recordings are refused unless diagnostic `--allow-synthetic` is used. See [benchmarking](docs/benchmarking.md) for exact semantics and the optional RTM path.

## Project notes

- [Product and technical plan](docs/project-plan.md)
- [First sprint and completion ledger](docs/first-sprint.md)
- [Implementation and validation status](docs/implementation-status.md)
- [Pose research](docs/research/pose-models.md)
- [Data and evaluation research](docs/research/data-and-evaluation.md)
- [Coaching validity](docs/research/coaching-validity.md)
- [Future video review layer](docs/research/video-review.md)

Raw footage, datasets, credentials, models and test output stay out of Git. Third-party code, weights and data retain their own licenses; see [model notes](public/models/README.md). Research documentation is not a license grant for those resources.
