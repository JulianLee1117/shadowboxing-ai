# Corner — Shadowboxing AI

A local laptop-camera practice app: record a short round, replay your video, and inspect experimental detections for jabs, crosses, lead/rear hooks and lead/rear uppercuts.

**Research prototype, September 26, 2026.** Pose inference and saved-video analysis run on your device. Recognition and technique coaching have not been validated. No technique grades or corrective judgments are enabled.

## Run locally

Use Node.js 20.19+ (22.12+ recommended) and npm.

```sh
npm ci
npm run models:setup -- --all
npm run dev
```

Open [the local app](http://127.0.0.1:5173). Keep the same browser and address to access saved rounds; `localhost` and `127.0.0.1` have separate storage.

Setup downloads model/runtime assets, which are then served locally. Without `--all`, it installs Full only. URLs and hashes are recorded in [the model manifest](model-manifest.json). Desktop Chromium is the current target; saved-video analysis requires supported WebCodecs decoding. Safari and mobile inference have not been validated. See [third-party notices](public/third-party-notices.txt) and [model notes](public/models/README.md).

## Practice and review

1. Keep the default **30 seconds**, choose **Lead hand**, and click **Enable camera**. Frame your head, hips and extended hands. **More options → Practice focus** offers free practice, jab–cross, double jab and double jab–cross prompts.
2. Click **Record round**, then step back during the **8-second countdown**. Recording starts automatically, including when tracking is uncertain. No second click is needed.
3. Let the timer finish or click **Stop & save**. Review opens and the camera is released. Camera rounds save video and tracking locally; no microphone is requested.
4. Watch the large replay first. Use **Focus video** for fullscreen viewing, **0.5×**, frame stepping, **Show tracking** and **Show detections** to inspect individual movements. Detections show the current punch name and physical hand on the video, including in focus mode. The six count cards, punch timeline and event list appear below the player; click a punch to replay its onset. Predicted L/R labels and visibility scores do not prove hand identity.
5. A newly completed round with video starts a separate local analysis. You can keep watching, follow progress, or **Cancel analysis**. **Original results remain the default**; use **Show video analysis** / **Show original** to compare. A fresh pass can improve or worsen particular detections.

Older recordings offer **Analyze recording** on demand. Compatible saved reports load from local storage. Analysis uses native decoded video frames and timestamps, processes them sequentially with a fresh Full worker, and preserves the original video, tracking, detections and reference labels. It has limits of 185 seconds, 5,550 processed frames, 250 MB, 4K pixels and four minutes of processing; ordinary source rates through 60 fps are preserved, with higher rates sampled. A reached duration/frame/time limit produces an explicitly partial report when usable evidence exists.

Review groups observed punches into **jab–cross (1-2)**, **double jab (1-1)**, **double jab–cross (1-1-2)** and **jab–cross–jab (1-2-1)**. These are experimental sequences, not form grades or proof that the requested drill was performed. Uncertain tracking can block a combination while leaving the original punch events intact. See [recognition events and tracking diagnostics](docs/recognition-events.md).

**More options** also contains Full/Lite/Heavy selection, tracking display, countdown sound, **Open video** and **Try demo**. Imported clips retain their original file; **Analyze clip** starts their first pass. The demo is synthetic and excluded from real accuracy reports. Older motion-only rounds remain readable. **Recheck detections**, when offered in original mode, applies current counting rules to saved poses without rerunning the model; **Use saved detections** restores their original events.

See [the short first-test guide](docs/first-user-test.md). Export **Evidence JSON** and **Export video** for a backup; derived video-analysis reports have their own export. Browser storage is not a backup, and an unfinished round can be lost on reload. Save failures are shown so an in-memory result can still be exported.

## What works, and what remains experimental

The app has local worker inference with CPU fallback, recording/replay, separate cached video analysis, immutable detection rechecks, combination grouping, uncertainty flags, reference labeling and exports. Full remains the live default. The worker restricts external requests, and development/preview servers apply a restrictive CSP. No cloud review, analytics, API key or voice coaching is connected; footage is not uploaded.

The current detector, `projected-six-punch-v7-supported-rise`, uses projected motion and observed recovery. Straights retain independent arm gating, supported brief peaks, partial-return repetitions and a bounded path-detour check. Hooks and uppercuts add experimental directional bent-arm rules; this is not a trained six-punch model. Names and notation follow the selected lead hand, while saved events retain anatomical left/right identity. The large practice view shows recent detected punch names and a running total; **Focus view** expands the practice area. Live labels report detections, not technique advice.

Class support does not establish reliable recognition. Hooks can remain entirely missed in real footage when the active arm disappears or its projected path does not match the rules. Depth-directed movements, rapid punches, body-level variants and overlapping arms are difficult. Foreshortening and confidently misplaced joints can defeat recognition; relaxed arm lowering can produce false counts. Counting a repetition does not certify a complete guard return. Tracking diagnostics never verify identity, swap hands or repair coordinates. Fullscreen falls back to an enlarged page view when the browser cannot enter native fullscreen.

Local RTM pose challengers place some visibly misplaced right-wrist/elbow estimates more accurately, but have not established a live replacement for Full. A runnable [temporal training experiment](docs/temporal-training.md) produces excessive false events on the small development corpus; its weights are **not enabled in the app**. No model comparison, training experiment or second video pass is claimed as validated accuracy, generalization, or coaching benefit. Private footage and numeric experiment reports stay out of Git. See [implementation status](docs/implementation-status.md) and [next decisions](docs/iteration-plan.md).

The [six-punch dataset tools](docs/action-dataset.md) prepare physical-hand/stance labels, preserve unknown targets and check split leakage for a future learned recognizer. A separate [coaching-review workflow](docs/coaching-review.md) collects independent human criterion labels and agreement without model predictions or automatic grades. Neither tool supplies trained browser weights or technique advice. The [current external-data audit](docs/research/boxing-datasets-2026-09.md) records verified access and reuse gaps. The [short capture protocol](docs/six-punch-capture.md) describes useful hook/uppercut examples and non-punch movements.

## Verify and evaluate

```sh
npm run check
python3 -m unittest discover -s ml/tests -v
npx playwright install chromium
npm run test:e2e
npm audit
```

Browser tests use generated streams and stub physical-camera requests. They verify actual local inference and software behavior, not boxing accuracy. Optional temporal tests need their separate dependencies. Local assets are required. To inspect a production build, run `npm run build` and `npm run preview`; equivalent security headers are needed when serving elsewhere.

```sh
python3 -m ml.diagnose /path/to/session.json --output data/diagnosis.json
python3 -m ml.evaluate /path/to/session.json --output data/evaluation.json
npm run benchmark -- /path/to/day1-labeled.json /path/to/day2-labeled.json \
  --output-dir data/pilot/runs/new-experiment
```

`diagnose` reports tracking and timing. `evaluate` reports recognition metrics only with complete reference labels or an explicit completeness assertion. `benchmark` preserves inputs and compares saved/current detections on unchanged poses, including recovered/lost actions and false detections. These are development tools; see [benchmarking](docs/benchmarking.md).

## Development workflow

This project uses direct commits and pushes to **`main`**, as requested by its owner; do not open a pull request for routine changes. Review the diff, run the relevant checks above, then commit and push the reviewed source, tests and documentation to `main`. Keep recordings, personal exports, datasets, model weights and generated reports in ignored locations. Inspect staged files before committing; never stage private artifacts as part of a broad add.

## Project notes

- [Product and technical plan](docs/project-plan.md)
- [First sprint](docs/first-sprint.md)
- [Pose research](docs/research/pose-models.md)
- [Optimization sources](docs/research/optimization-update.md)
- [Data and evaluation](docs/research/data-and-evaluation.md)
- [Coaching validity](docs/research/coaching-validity.md)
- [Future video review](docs/research/video-review.md)

Research plans include capabilities beyond the prototype. Recordings, personal exports, datasets, credentials, model binaries and test output stay out of Git. No sustained 20-minute acceptance result is claimed.
