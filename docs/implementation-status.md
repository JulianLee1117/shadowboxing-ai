# Implementation and validation status

September 26, 2026. This ledger describes the current app, separately from the proposed targets in the project plan.

## Current workflow

Corner opens with its camera off. **Practice** defaults to a 30-second free-practice round and a left lead hand. **Enable camera** starts a local preview; **Record round** starts an eight-second countdown so the person can step away from the laptop. Capture begins when the countdown expires even if neither arm is assessable. No setup checkbox or second start click is required.

Camera rounds record video and tracking locally with no microphone. Completion opens **Review** and releases the capture source. A recording failure is reported and can leave a motion-only round. Imported clips retain the original file; **Analyze clip** starts analysis directly. **More options** contains the model selector, tracking display, countdown sound, video import and synthetic demo.

Review opens original video without model overlays where footage is available. **Show tracking** and **Show detections** expose those layers separately. Reference labels, export and technical details are collapsed below playback. Earlier IndexedDB rounds, including motion-only captures, remain readable. Stored events are not recomputed when the detector changes; new sessions include a detector version.

MediaPipe inference runs in a module worker using local assets. The worker restricts fetch to same-origin because the SDK contains telemetry code; dev/preview servers add a restrictive CSP. No analytics or cloud-review service is connected.

Completed rounds retain unmirrored pose frames, source-relative timestamps, original detections, labels, capture settings, model provenance and available timing measurements. Video is stored as a local Blob and exported separately from JSON. Save failures are surfaced, and an in-memory round remains exportable. Storage is scoped to browser profile and origin; `localhost` and `127.0.0.1` are different stores.

## Detector behavior and current limitations

The detector is an experimental projected extension/recovery heuristic, not a trained six-punch classifier. Physical hand identity comes from anatomical landmark indices; lead/rear mapping comes from the selected lead hand. Mirroring affects only the preview.

Arm observability is now independent: an uncertain guarding wrist no longer erases the opposite arm's history. Each arm still requires its own visible shoulder/elbow/wrist and shared shoulder/hip anchors for torso normalization. Timing gaps, invalid geometry and missing active-arm evidence still reset the relevant state. Tracking indicators describe observability, not correct technique. Predicted depth is not substituted for unavailable image evidence.

A motion-only pilot exposed a coupling defect: opposite-arm confidence loss could reset an otherwise observable punching arm. Independent arm gating fixes that dependency, but does not establish accurate cross recognition. Projected reach and path checks can still reject a real cross when its wrist is tracked incorrectly. Thresholds were not tuned to the pilot.

No original video or independent action labels were available for that pilot. A pose model may place a wrist on the wrong physical hand even with high visibility confidence; landmarks alone cannot identify whether lighting, occlusion or another model failure caused that error. Recognition accuracy and technique cannot be established from such an export. Personal recordings, detailed pilot measurements and local diagnostic reports stay outside Git.

The next test is **two recorded 30-second rounds**, with the preferred opposite three-quarter view first and the original view second. Keep the same lead hand and stance while changing body angle. Start each round by raising the physical left hand, then right hand, followed by five slow jabs and five slow crosses with pauses. Compare actual video with the overlay to check hand identity before inspecting counts. This tests two views without assuming a lighting explanation; see [the short checklist](first-user-test.md).

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

Unit tests cover aspect-correct geometry, anatomical hand mapping, temporal resets, independent-arm gating, missing active-arm evidence, synthetic demo behavior and countdown timing. Python tests cover one-to-one event matching, incomplete labels, synthetic exclusions, cross-model landmark mapping and per-arm diagnostics. Consult current test output for exact results.

Browser integration tests use generated streams and stub physical camera requests. They exercise local model inference, CPU fallback and capture/review/storage behavior. These establish software integration, not boxing accuracy. Earlier uploaded-video tests also checked repeated analysis, original-file retention and exported byte preservation against a production build; the simplified interface has its own regression checks.

`python3 -m ml.diagnose` reports per-joint confidence/clipping, per-arm coverage, projected angles/reach, estimated world-angle comparisons and timing. Its optional timeline helps locate failures; it does not label actions or score technique. `python3 -m ml.evaluate` produces recognition metrics only with an explicit completeness assertion for reference labels. Both reject synthetic sessions by default. No personal recording is included in automated test fixtures.

## Remaining evidence and features

1. Capture original video across multiple days and views, including idle movement and natural transitions. Independently label all actions, physical hands and boundaries before revealing model suggestions.
2. Separate pose-model failures from temporal-detector failures on the same labeled clips. Compare Full/Heavy and a provenance-audited RTM model before choosing a replacement.
3. Run sustained hardware testing. No 20-minute acceptance result is claimed.
4. Train and evaluate a temporal classifier with sufficient data before expanding recognition to hooks, uppercuts and combinations. Heuristic scores remain uncalibrated.
5. Obtain coach-reviewed criteria, examples and visibility rules before enabling corrective technique advice. No guard, power or biomechanics grades are provided.
6. Validate a mixed-skill cohort with held-out people before broader claims. Personal success does not establish universal coverage or learning benefit.
7. Add selected-clip cloud review only after its evidence and critique evaluation are ready. Current application flows do not upload footage.

The optional RTM extractor has tested coordinate mapping and provenance checks, but its external inference stack has not been run on target footage. It remains a research adapter. Substantive research documents describe the broader plan; they are not claims that those capabilities have shipped.
