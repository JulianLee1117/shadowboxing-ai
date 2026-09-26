# Corner — Shadowboxing AI

A local shadowboxing practice app for a laptop webcam: record a short round, replay the original video, and compare it with experimental jab/cross detections.

**Status: research prototype, September 26, 2026.** MediaPipe pose inference runs locally. Counting uses projected-motion heuristics; accurate recognition and technique coaching have not been validated. No technique grades or corrective judgments are enabled.

## Run locally

Use Node.js 20.19+ (22.12+ recommended) and npm.

```sh
npm ci
npm run models:setup -- --all
npm run dev
```

Open [the local app](http://127.0.0.1:5173). Use the same browser and address to retain access to saved rounds: `localhost` and `127.0.0.1` have separate browser storage.

Models and runtime files download during setup, then are served from your device. Setup without `--all` installs Full only. Artifact URLs and SHA-256 hashes are recorded in [the model manifest](model-manifest.json); weights and copied WASM files stay out of Git. Desktop Chromium is the current target. Safari and mobile-browser inference have not been validated.

## Record your first round

Practice defaults to **30 seconds** of free practice with the left hand leading. See [the short first-test checklist](docs/first-user-test.md).

1. Choose **Lead hand**, then **Enable camera**. Frame your head through hips, leaving room for both arms to extend.
2. Click **Record round** near the laptop, then step back. Recording starts automatically after **8 seconds**, even if tracking is uncertain. No second click or setup checkbox is required.
3. Practice normally. **L / R** are predicted anatomical hand labels; check they follow your physical hands. Amber means uncertain tracking. Visibility alone does not verify hand identity or technique.
4. Let the timer finish or click **Stop & save**. The app opens **Review** and releases the camera. Camera rounds save video and tracking locally; no microphone is requested.
5. Watch the original video first. Use **0.5×** speed and frame stepping to inspect movement. **Show tracking** and **Show detections** expose the model separately. Labels and **Export & details** are collapsed below replay.

**More options** contains tracking visibility, countdown sound, Full/Lite/Heavy models, **Open video**, and **Try demo**. Models can be changed while the source is off. Imported clips retain their original video locally; **Analyze clip** starts their analysis. The demo is synthetic and excluded from real accuracy reports.

Older motion-only rounds remain available in **Saved rounds**. Updates preserve their original detections. **Recheck detections** applies new counting rules to their saved tracking in memory; **Use saved detections** restores the original view. Updated analysis exports separately and does not rerun the pose model. A skeleton without source video cannot independently validate pose or punch accuracy. If the browser reports that recording is unavailable, that round may contain tracking only.

## Current recognition limitation

Independent arm gating prevents an uncertain guarding hand from resetting the opposite arm's detector. Cross recognition remains unvalidated: an incorrectly tracked wrist or a foreshortened view can still fail the projected-motion checks. A high visibility score does not guarantee the correct hand was located.

The detector checks the outward path separately from recovery, accepts a brief observed peak only with supporting neighboring frames, and acquires repeat starts from confirmed returns or observed flexed reversals. A bounded path check tolerates one isolated tracking detour without altering raw landmarks or peak evidence. Existing labeled rounds now support repeatable regression comparisons; fresh sessions are needed to evaluate a frozen version. See [the implementation ledger](docs/implementation-status.md) and [next iteration decisions](docs/iteration-plan.md).

## Implemented

- Local MediaPipe Full/Lite/Heavy inference in a worker, one in-flight frame, timeouts and CPU fallback.
- Same-origin assets, a worker network boundary blocking external SDK requests, and restrictive development/preview CSP headers.
- An eight-second recording countdown, independent arm visibility checks, anatomical hand/lead mapping and experimental straight-punch events.
- Local video and pose storage, automatic review, half-speed playback/stepping, versioned detection rechecks, reference labels, JSON/video export and per-round deletion.
- Capture settings, actual delegate, timestamps, timing telemetry, model provenance and detector version in new exports.
- Local replay and batch comparison tools, labeled event evaluation and separate pose diagnostics. Optional RTMPose/RTMW extraction has run on target footage and retains explicit model/preprocessing/runtime provenance; Full remains the live default.

Hooks, uppercuts and other actions can be labeled manually but are not recognized by the live baseline. Front-facing punches and self-occlusion can defeat projected geometry. Heuristic scores are not calibrated confidence, and returning toward a starting position is not proof of correct guard recovery.

Cloud review is not connected and no API key is required. The app does not upload footage. Browser storage is not a backup: export video and JSON you want to preserve. Closing or reloading during an unfinished round can lose it. No sustained 20-minute acceptance test, held-out recognition accuracy or coach-reviewed efficacy result is claimed.

## Verify

```sh
npm run check
python3 -m unittest discover -s ml/tests -v
npx playwright install chromium
npm run test:e2e
npm audit
```

Automated browser tests use generated streams and stub physical camera requests. They exercise actual local inference and software behavior; synthetic timings are not boxing-performance measurements. Local model assets are required. For a production build, run `npm run build` and `npm run preview`; configure equivalent CSP/permissions headers when serving elsewhere.

## Inspect exported rounds

```sh
python3 -m ml.diagnose /path/to/session.json --output data/diagnosis.json
python3 -m ml.evaluate /path/to/session.json --output data/evaluation.json
npm run benchmark -- /path/to/day1-labeled.json /path/to/day2-labeled.json \
  --output-dir data/pilot/runs/new-experiment
```

`diagnose` reports per-arm visibility, projected geometry and timing, without claiming accuracy. Add `--include-timeline --window-ms 1000` for detailed tracking inspection. `evaluate` requires complete reference labels, or an explicit `--annotations-complete` assertion, for recognition metrics; otherwise it reports capture quality only. Both reject synthetic sessions unless `--allow-synthetic` is explicitly selected for software checks. See [benchmarking](docs/benchmarking.md) for matching rules and the optional RTM path.

`benchmark` requires complete independent labels, preserves input sessions, refuses an existing output directory, and compares saved/current detections on unchanged poses. It reports recovered and lost reference actions as well as total matches and unmatched detections. This is a development regression check, not held-out accuracy or coaching validation.

## Project notes

- [Product and technical plan](docs/project-plan.md)
- [First sprint and completion ledger](docs/first-sprint.md)
- [Implementation and validation status](docs/implementation-status.md)
- [Pose research](docs/research/pose-models.md)
- [Optimization sources and priorities](docs/research/optimization-update.md)
- [Data and evaluation research](docs/research/data-and-evaluation.md)
- [Coaching validity](docs/research/coaching-validity.md)
- [Future video review layer](docs/research/video-review.md)

Research plans describe intended capabilities beyond this prototype. Raw footage, personal exports, datasets, credentials, model binaries and test output stay out of Git. Third-party resources retain their own licenses; see [model notes](public/models/README.md).
