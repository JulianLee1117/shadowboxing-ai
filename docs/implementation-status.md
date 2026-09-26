# Implementation and validation status

September 26, 2026. This ledger is distinct from the proposed targets in the project plan.

## Working end-to-end

Corner starts with its camera off and requests video only after an explicit action. MediaPipe inference runs in a module worker with local assets. The SDK contains telemetry code, so the worker restricts fetch to same-origin; dev/preview servers supply an additional restrictive CSP. No analytics or cloud-review service is included.

Completed rounds retain unmirrored pose frames, source-relative timestamps, experimental straight-punch events, manual labels, capture settings, model provenance and available timing measurements. Optional camera recordings and imported original clips are stored as local blobs. Camera recording starts only when its toggle is enabled for a round. A save failure is surfaced, and exports remain available from the in-memory round.

Anatomical wrist labels are shown on the overlay. Mirroring does not change the pixels given to the model. Setup confirmation is manual and distinct from correct technique. The motion engine detects projected extension/recovery separately for each hand and abstains when required landmarks or temporal context fail. It is not a trained six-punch classifier.

## Timing and lifecycle

- Camera frames use `requestVideoFrameCallback` when supported. Animation-frame fallback is marked in capture metadata.
- Inference is bounded to one request. New decoded frames encountered while busy are skipped and counted.
- `frameAgeMs` starts when the application observes a decoded frame. It excludes sensor exposure and the full display pipeline.
- `inferenceMs` measures the worker model call. UI p95 uses nearest rank; the CLI uses interpolation, so small-sample summaries can differ.
- Events store `detectedAtMs` separately from their observed start/peak/end for comparison with independently annotated boundaries.
- Round duration is frozen at finalization entry. In-flight inference drains with a bounded wait. A privacy stop releases the camera and cancels inference immediately, potentially omitting the last pending frame.
- Imported clips decode and warm the model while paused, so first-inference startup does not consume the clip. Reanalysis resets playback and temporal state and warms the fresh worker. `videoOffsetMs` identifies the replay origin in retained footage.
- Camera/worker resources are released on stop and unmount. Active rounds end when the page becomes hidden. Reloading an unfinished round is not a guaranteed save operation.

## Automated verification

Tests cover aspect-correct geometry, physical-side/stance mapping, temporal reset, uncalibrated/occluded rejection, duplicate prevention, synthetic demo integration, event matching, incomplete labels, synthetic exclusions and conservative cross-model joint mapping.

Browser tests run the actual Full model and local WASM with generated streams, CPU fallback, zero observed external requests, explicit startup, camera denial/shutdown, persistence, export, label creation/removal, completeness updates, deletion and responsive layout. Uploaded-video tests check retention and repeated analysis. These establish software behavior only; consult current test output for exact results.

The uploaded-video workflow also passed against the compiled production build and its worker assets, including repeated analysis, nonempty timestamped frames, original-video byte preservation, JSON export and deletion. Desktop and 390px layouts were visually inspected.

The short-clip integration test forces a failed GPU initialization and verifies actual CPU fallback inference, avoiding a software-rendered CI GPU throughput requirement. A separate runtime test exercises normal delegate selection. Neither test is a target-hardware performance benchmark.

The production build is typechecked and dependencies audited. Test captures are synthetic; no physical camera or private recording was used during implementation.

## Remaining evidence and features

1. Record real consented sessions on multiple days, including idle motion, natural transitions, view variation and unassessable repetitions.
2. Independently annotate action identity, physical hand and boundaries. Labels created after seeing model suggestions can be biased and are not blinded ground truth.
3. Compare Full/Heavy and a provenance-audited RTM model on identical footage; measure wrist/elbow failures and downstream events.
4. Run sustained hardware testing. No 20-minute acceptance result is claimed.
5. Obtain coach-reviewed criteria and labels before enabling corrective technique advice. The UI provides drill intent and callouts, not guard/power/biomechanics grades.
6. Train/evaluate a temporal classifier with sufficient data, then expand to hooks, uppercuts and combinations. Heuristic scores are not calibrated confidence.
7. Add selected-clip cloud review after the evidence and critique evaluation are ready. Current exports provide the foundation without uploads.
8. Validate a mixed-skill cohort with held-out people before broader claims. Personal success does not establish universal coverage or learning benefit.

The optional RTM extractor has tested coordinate mapping and provenance checks, but its external inference stack has not been run on target footage. It is a research adapter, not a validated second live backend.
