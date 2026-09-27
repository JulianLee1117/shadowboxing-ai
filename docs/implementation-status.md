# Implementation and validation status

September 26, 2026. This ledger describes shipped code, separately from the targets in [the project plan](project-plan.md).

## Practice and replay

Practice opens with the camera off, a **30-second** round, free practice and a left lead hand. **Enable camera → Record round** starts an eight-second countdown; recording begins without another click, even when tracking is uncertain. **More options** holds practice-focus prompts, model selection, tracking display, countdown sound, video import and the synthetic demo. Prompts do not influence recognition.

The practice area now uses most of the available window. **Focus view** expands it, while large recent-punch names, physical hand, total and timer remain readable. Per-arm tracking stays visible during a round; when neither arm is assessable, the display says **Tracking unclear** and recording continues. Review has a larger player, focus mode, six punch count cards and a clickable event timeline.

Camera rounds record local video and poses without microphone access. Completion opens Review and releases the camera. The timer/deadline is independent of pose arrivals. Stop, navigation and unmount release resources; hiding the page cancels a pending countdown or ends an active round. Recording/save failures are surfaced. Imported clips retain the original file. Earlier motion-only rounds remain readable.

Review supports original-video playback, half speed, frame stepping, separate tracking/detection overlays, reference labels and exports. **Original results remain the default**. A separate analysis is available through **Show video analysis**, with **Show original** restoring the saved evidence. **Recheck detections**, when offered in original mode, applies current rules to existing poses; it does not rerun pose inference or overwrite the capture.

## Separate local video analysis

Newly completed rounds with video start an analysis automatically; older recordings offer **Analyze recording**. Progress and cancellation are visible, and replay remains available while analysis runs. Navigation invalidates pending callbacks and releases processing resources.

The decoder uses Mediabunny/WebCodecs to read native frames with decoded media timestamps. A fresh MediaPipe Full worker processes them sequentially. This replaces the earlier seek-position approach: no pose interpolation, JPEG round trip or estimated seek timestamp is used. Normal source rates through 60 fps retain their native frames; higher-rate sources are sampled. Source timestamps, decoder/runtime/model provenance, cadence and completeness are recorded.

Analysis is bounded by **185 seconds**, **5,550 processed frames**, **250 MB**, **4K pixel count**, and **four minutes of processing**. The first applicable cap governs; for example, a 60 fps clip can reach the frame cap before the duration cap. Unsupported decoding reports an error. Duration/frame/time truncation is labeled partial when frames are available; partial results do not replace original results by default.

Reports are cached separately in IndexedDB and checked against video fingerprint, stance, timing and analysis version. Source video, poses, original detections and annotations are preserved. Reports export separately, and deleting a round also removes its derived report. Cache/save failures are surfaced. Browser/profile/origin changes can hide stored rounds, so export video and evidence JSON for a backup.

A fresh pass can produce different or worse results. The exact-decoding implementation is being compared against saved captures; no accuracy gain is claimed.

## Recognition and uncertainty

The live baseline is MediaPipe Full plus `projected-six-punch-v7-supported-rise`. It recognizes experimental jab/cross events and adds hook/uppercut outputs using anatomical hand/stance mapping, projected direction, supported bent-arm geometry and observed recovery. Class support is not demonstrated reliable six-punch recognition: real hooks remain missed when joints disappear or the model estimates the wrong path. Fresh pose passes can change which actions qualify. Mirroring only affects display. Independent arm gating prevents an uncertain guarding hand from resetting the opposite arm, while shared torso anchors remain necessary.

A brief qualifying peak requires supporting observations. Confirmed flexed retraction and observed reversals can establish a repeat's origin, including partial returns between fast repetitions. A static reference can follow coherently observed inward guard motion. Small rejected movements retain bounded acquisition observations, without inheriting their rejected peak or a ready state. Spatial guard return remains a separate observation at detection time. Outbound path measurement can omit one tightly constrained detour while retaining strict observed peaks; it does not change raw landmarks or repair identity. Missing active-arm evidence and timing/framing discontinuities reset the relevant history.

An accepted full stroke can support a shorter-travel repeat by the same hand within 650 ms of its peak. The repeat still needs its own supported straightening, movement, path and recovery, and must reach the full-stroke extension range. Repeat-only events cannot prolong or move this reference. The isolated-stroke extension threshold is unchanged. This recovered additional fast second jabs in the development clips without adding unmatched events in that comparison; it is not a guarantee for unseen rounds.

Review now groups observed events into **1-2**, **1-1**, **1-1-2** and **1-2-1**, with deterministic nonoverlapping membership. Peak order allows overlapping punch/recovery intervals. Guard return is not a form or sequence grade. Uncertain intervals can prevent grouping without deleting source punch events. The streaming interface records causal finalization and supports a future event consumer; no speech is connected.

Tracking-trust diagnostics flag missing observations, acquisition periods, gaps and selected trajectory inconsistencies. They never swap hands, fabricate coordinates or verify anatomical identity. A diagnostic trusted state means only that these checks found no inconsistency; sustained confident errors can remain. See [event semantics and limitations](recognition-events.md).

## Experiments and evidence

Full, Heavy, Holistic, RTMPose-M and RTMW-L have been exercised locally. On selected failed-cross frames, both RTM models place the right wrist/elbow closer to the visible joints than Full. This is useful coordinate evidence, not a validated replacement pipeline. RTM scores are not calibrated to MediaPipe visibility, and its person detector can produce false extra boxes. Full IMAGE mode recovered some jabs while losing crosses in the new recordings, so it does not replace VIDEO mode. Comparisons require matched inputs, timestamps and explicit preprocessing. Runtime measurements from short offline runs are not sustained live-performance validation.

The optional Python extractor supports explicit Core ML execution with CPU node fallback and saved execution profiles. CPU remains its default. It refuses unavailable or entirely CPU execution when Core ML was requested. This makes Mac experiments faster to evaluate locally; the browser still uses Full, and no model-specific RTM confidence policy or native live bridge is enabled. See [extraction and profiling](benchmarking.md#optional-rtmpose--rtmw-extraction).

The [causal temporal training pipeline](temporal-training.md) trains and evaluates a small offline recognizer. Both the initial experiment and its fixed-protocol five-session follow-up produced too many false events to justify enabling the weights. It remains research code, not the app default or an independent generalization result. Private captures, provisional action labels, checkpoints and detailed metrics stay outside Git.

Five provisionally labeled recordings support regression testing and failure diagnosis. The recovery revision increased matched actions on both saved live poses and fresh video poses, with remaining false background gestures and shortened-boundary mismatches. A held extension can also be misread when the pose itself suggests an early return. Once a clip informs changes, it is development data. A later-day session evaluated against a frozen version is still needed; coach-reviewed technique labels are separate from action labels. No corrective guard, power or biomechanics advice is enabled.

## Verification and limits

Two standard-library research CLIs extend the data workflow. `ml.action_dataset` audits source/participant/day splits, fingerprints evidence, preserves physical hand and stance-dependent six-punch identity, and prepares per-arm family targets with explicit unknown masks. Incomplete jab/cross annotation cannot establish multiclass background. No multiclass weights are trained or enabled. `ml.coaching` prepares prediction-blind local human-review packets for two draft guard criteria, validates eligibility/evidence, and compares independent reviews with abstention coverage. It never prefills form judgments or issues corrections. See [dataset preparation](action-dataset.md), [coaching review](coaching-review.md), and the [external-source catalog](research/boxing-datasets-2026-09.md).

Tests cover detector resets/peak/repeat behavior, combinations, tracking intervals, immutable reports, native-frame decoding, analysis cancellation/cache lifecycle, recording/review, and evaluation/training software. Browser tests use generated video and stub physical camera requests. They establish integration behavior, not boxing accuracy; consult current test output for exact results.

Live capture allows one inference request in flight and reports skipped frames. `frameAgeMs` begins at the browser's observed frame callback and excludes sensor exposure; model-call time is separate. Punch events retain causal finalization timestamps. Saved-video cadence is source cadence, not processing throughput.

Model assets and processing remain local. Development/preview CSP and worker request restrictions prevent external SDK requests; no analytics, cloud-review API or microphone is connected. See [third-party notices](../public/third-party-notices.txt). No 20-minute acceptance test, unseen-person accuracy or coaching-efficacy result is claimed. Next work is listed in [iteration decisions](iteration-plan.md); user testing is in [the short guide](first-user-test.md).
