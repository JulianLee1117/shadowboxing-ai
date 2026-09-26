# Implementation and validation status

September 26, 2026. This ledger describes the current app, separately from the proposed targets in the project plan.

## Current workflow

Corner opens with its camera off. **Practice** defaults to a 30-second free-practice round and a left lead hand. **Enable camera** starts a local preview; **Record round** starts an eight-second countdown so the person can step away from the laptop. Capture begins when the countdown expires even if neither arm is assessable. No setup checkbox or second start click is required.

Camera rounds record video and tracking locally with no microphone. Completion opens **Review** and releases the capture source. A recording failure is reported and can leave a motion-only round. Imported clips retain the original file; **Analyze clip** starts analysis directly. **More options** contains the model selector, tracking display, countdown sound, video import and synthetic demo.

Review opens original video without model overlays where footage is available. **Show tracking** and **Show detections** expose those layers separately. Reference labels, export and technical details are collapsed below playback. Earlier IndexedDB rounds, including motion-only captures, remain readable. Stored events are not recomputed when the detector changes; new sessions include a detector version. Half-speed replay and frame stepping support inspection. Older rounds expose **Recheck detections** under **Show detections**: it runs the current counting rules over saved tracking, without rerunning pose or modifying the recording. Updated results are temporary and have a separate provenance-bearing export. **Use saved detections** restores the original view.

MediaPipe inference runs in a module worker using local assets. The worker restricts fetch to same-origin because the SDK contains telemetry code; dev/preview servers add a restrictive CSP. No analytics or cloud-review service is connected.

Completed rounds retain unmirrored pose frames, source-relative timestamps, original detections, labels, capture settings, model provenance and available timing measurements. Video is stored as a local Blob and exported separately from JSON. Save failures are surfaced, and an in-memory round remains exportable. Storage is scoped to browser profile and origin; `localhost` and `127.0.0.1` are different stores.

## Detector behavior and current limitations

The detector is an experimental projected extension/recovery heuristic, not a trained six-punch classifier. Physical hand identity comes from anatomical landmark indices; lead/rear mapping comes from the selected lead hand. Mirroring affects only the preview.

Arm observability is now independent: an uncertain guarding wrist no longer erases the opposite arm's history. Each arm still requires its own visible shoulder/elbow/wrist and shared shoulder/hip anchors for torso normalization. Timing gaps, invalid geometry and missing active-arm evidence still reset the relevant state. Tracking indicators describe observability, not correct technique. Predicted depth is not substituted for unavailable image evidence.

The current detector measures straightness only along the outward path to the observed peak. Including return travel in that ratio incorrectly penalized quick retractions. A single qualifying extension sample now requires coherent observed neighbors on both sides of the peak; it cannot pass as an isolated spike. An accepted event's confirmed return can establish readiness for the next stroke without imposing a second rest period. Initial acquisition, missing tracking, peak geometry and complete recovery remain guarded.

Recorded development clips are reviewed independently of detection output to distinguish observable actions from tracking and counting failures. These clips guide changes, so before/after recovery is a development regression result, not held-out recognition accuracy. Provisional assistant action labels are not coach-verified form judgments. Personal recordings, detailed measurements, annotations and diagnostic reports stay outside Git.

A controlled local Full/Heavy experiment used identical decoded frames and a fixed detector. Full remains the default after that development comparison. Re-encoded offline frames, initialization and headless timing differ from live camera capture; the experiment does not establish sustained latency or a universal model ranking. Artifacts and detailed measurements remain private.

Projected geometry can still miss genuine punches because of foreshortening, short visible excursion or an incorrect wrist track. A pose model may place a wrist on the wrong physical hand even with high visibility confidence. Lighting and occlusion are possible contributors; the source video and tracking must be compared before assigning a cause. The current system does not repair hand swaps or infer invisible punch peaks.

The next test uses a fresh round containing separated slow punches, comfortable faster pairs, a double-jab sequence and an idle interval. Maintain the same stance and a view with both arms visible; compare the original video before exposing detections. This checks generalization beyond the clips used for debugging. See [the short checklist](first-user-test.md).

## Timing and lifecycle

- Camera processing uses `requestVideoFrameCallback` where supported; fallback timing is identified in capture metadata.
- Only one inference request is in flight. New decoded frames encountered while busy are skipped and counted.
- `frameAgeMs` begins when the application observes a decoded frame. It excludes sensor exposure and the full display pipeline. `inferenceMs` measures the worker model call.
- UI p95 uses nearest rank; Python tools use interpolation, so small-sample summaries can differ.
- Events store causal `detectedAtMs` separately from observed start/peak/end for comparison with independent annotations.
- Camera/demo deadlines and the displayed timer use a monotonic clock independent of pose arrivals; file analysis uses source time. Round duration is frozen at finalization entry. In-flight inference drains with a bounded wait. An explicit stop releases the camera and cancels inference immediately, potentially omitting the last pending frame.
- Imported clips decode and warm the model while paused. Reanalysis resets playback and temporal state. `videoOffsetMs` identifies the replay origin in retained footage; playback is bounded to the analyzed round and stale pose overlays are hidden across gaps.
- Hiding the page cancels a pending countdown or ends an active round. Stop and unmount release camera/worker resources. Reloading an unfinished round is not a guaranteed save operation.

## Verification and diagnostic tools

Unit tests cover aspect-correct geometry, anatomical hand mapping, temporal resets, independent-arm gating, observed peak support, quick repeat cycles, missing active-arm evidence, immutable rechecks, synthetic demo behavior and countdown timing. Python tests cover one-to-one event matching, incomplete labels, synthetic exclusions, cross-model landmark mapping and per-arm diagnostics. Consult current test output for exact results.

Browser integration tests use generated streams and stub physical camera requests. They exercise local model inference, CPU fallback and capture/review/storage behavior. These establish software integration, not boxing accuracy. Earlier uploaded-video tests also checked repeated analysis, original-file retention and exported byte preservation against a production build; the simplified interface has its own regression checks.

`python3 -m ml.diagnose` reports per-joint confidence/clipping, per-arm coverage, projected angles/reach, estimated world-angle comparisons and timing. Its optional timeline helps locate failures; it does not label actions or score technique. `python3 -m ml.evaluate` produces recognition metrics only with an explicit completeness assertion for reference labels. Both reject synthetic sessions by default. No personal recording is included in automated test fixtures.

## Remaining evidence and features

1. Capture original video across multiple days and views, including idle movement and natural transitions. Independently label all actions, physical hands and boundaries before revealing model suggestions.
2. Separate pose-model failures from temporal-detector failures on the same labeled clips. Extend the initial Full/Heavy comparison to independent sessions and a provenance-audited RTM or body-guided hand model before choosing a replacement.
3. Run sustained hardware testing. No 20-minute acceptance result is claimed.
4. Train and evaluate a temporal classifier with sufficient data before expanding recognition to hooks, uppercuts and combinations. Heuristic scores remain uncalibrated.
5. Obtain coach-reviewed criteria, examples and visibility rules before enabling corrective technique advice. No guard, power or biomechanics grades are provided.
6. Validate a mixed-skill cohort with held-out people before broader claims. Personal success does not establish universal coverage or learning benefit.
7. Add selected-clip cloud review only after its evidence and critique evaluation are ready. Current application flows do not upload footage.

The optional RTM extractor has tested coordinate mapping and provenance checks, but its external inference stack has not been run on target footage. It remains a research adapter. Substantive research documents describe the broader plan; they are not claims that those capabilities have shipped.
