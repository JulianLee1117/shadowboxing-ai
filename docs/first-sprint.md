# First sprint: prove the camera can support the coach

Proposed duration: 3–5 working days for the lab, followed by 1–2 weeks of capture, annotation and comparison. These are estimates. This document is a backlog; none of these implementation tasks is complete.

Target: M3 Max MacBook Pro, 36 GiB, built-in webcam. Initial user: beginner familiar with basic techniques. Personal usefulness comes first; generalization is a separate cohort evaluation.

## Sprint outcome

A local app records only when asked, plays the same clip through candidate pose models, shows synchronized raw/skeleton replay, measures sustained runtime, and exports evidence for manual annotation. We finish with a reasoned model/view choice and a list of criteria we can or cannot assess.

No movement criticism is user-facing until its criterion has a reviewed definition and passes the canonical gates in [the project plan](project-plan.md#release-gates). Developer findings remain explicitly unvalidated.

## Ordered implementation tickets

| ID | Task | Done when |
|---|---|---|
| LAB-01 | Create React/TypeScript local camera shell | Explicit start/stop; camera permission failure handled; raw/mirrored preview mapping correct; camera shuts down on stop |
| LAB-02 | Add capture diagnostics | Requested and delivered settings shown; real timestamps, distinct frames, gaps, processing duration, and queue depth recorded |
| LAB-03 | Add local-only recording and replay | Explicit recording; export/delete; frame-step or accurately timestamped playback; no cloud dependency; original timeline retained |
| LAB-04 | Add common pose adapter and MediaPipe worker | Pinned bundle/hash; landmarks align after resize/mirroring; main UI remains responsive; raw and filtered data separate |
| LAB-05 | Add offline RTMPose/RTMW comparison | Same decoded inputs and common joint subset; crop transforms recorded; CPU/Core ML path actually reported; no implicit downloaded-data upload |
| LAB-06 | Add review annotations and event labels | Mark action/hand, interval, visible joints, unknown motion and unsupported criteria; annotation exports retain original-session IDs |
| LAB-07 | Implement replay benchmark report | Metrics grouped by session/view/speed; event and point evidence retained; uncertainty, coverage and timing included |
| LAB-08 | Perform view/model decision review | Choose supported view per criterion; document failures; decide whether a more expensive model is justified |

Dependencies: LAB-01 → LAB-02/03 → LAB-04; LAB-05 can start on consented saved clips after the adapter contract exists; LAB-06/07 depend on reproducible recordings. A coach can define the first rubric while the camera lab is built.

## Small capture protocol

First capture a short setup pilot to choose framing. Aim to see the head and fully extended hands, with feet included when evaluating footwork. Avoid declaring a fixed distance or angle correct for every room. Compare front, oblique and side on a comfortable simple drill and record what becomes hidden in each view.

Then collect multiple separate sessions, initially about 200–400 annotated events in total, including non-punch activity and transitions. This amount tests the pipeline and exposes failure modes; it is not enough to certify broad accuracy. Hold out later-day sessions before any window extraction or threshold fitting.

| Block | Content | Purpose |
|---|---|---|
| Identity/setup | Declared stance, named hand raise, comfortable extension | Anatomical mapping, scale, field of view |
| Background | Rest, walk into frame, adjust guard/clothes, pause, wave | Avoid calling every hand movement a punch |
| Straights | Isolated jab and cross at comfortable slow and normal speeds | First event/phase recognition task |
| Sequence | 1–2, double jab, pauses, deliberately different safe sequence | Avoid recognizing the requested combination by assumption |
| Other motion | Familiar hooks/uppercuts, slips/rolls/steps if comfortable | Unknown/out-of-scope handling; label these separately |
| Acquisition failures | Natural partial occlusion, poor framing, moved camera, ordinary lighting differences | Validate abstention and setup feedback |

Use coach demonstrations for defined fault examples where possible. Do not ask the beginner to force unfamiliar or exaggerated movements to manufacture errors. Confirm what a coach can actually see from the test view before creating technique labels. Do not use an AI critique as its own evaluation ground truth.

The user has not confirmed coach access. Without it, proceed with capture quality, action labeling, timing, and candidate observations. Keep claims about correction validity provisional. Broader release requires independent coaching review.

## Benchmark design

Compare MediaPipe Full/Heavy and one RTMPose body/feet model first. Add RTMW only if its extra detail can support a selected criterion or resolve an observed failure. Do not benchmark ten large models before learning what is wrong with the first two.

Use the same saved input for offline comparisons. Match preprocessing/crops when possible, and disclose pipeline differences when each model requires its own detector. Normalize keypoint error by a fixed reference scale annotated from the clip; do not let each model's erroneous body scale change its own denominator.

Manually annotate a spread of clearly visible wrist/elbow/ankle points, oversampling peak movement, self-occlusion transitions, and model disagreement. Mark genuinely hidden points unavailable; annotation is not a license to guess depth. Audit anatomical swaps and crop failures explicitly.

Produce:

- Per-joint visible-point error and error by phase/view/speed.
- Left/right swaps, lost tracks, and out-of-frame crop failures per minute.
- Model confidence versus observed error; raw confidence cannot be compared as if every model calibrates it identically.
- Runtime p50/p95, frame age, frame loss, queue depth and memory after warm-up and during 20 minutes of sustained use.
- A short gallery of the most consequential failures, with source timestamps.
- Later, action event precision/recall and critique validity at useful coverage using the canonical project-plan protocol.

If exact sensor timestamps are unavailable, label the latency origin as the observed frame callback or decode time. Do not call that quantity sensor-to-display latency. Run models separately to avoid unfair resource contention, then measure the winning complete user workflow.

## Initial criterion candidates

Ask a coach to review two visible criteria first: return of the punching hand to a selected guard region after an isolated straight, and position of the non-punching hand during that drill. Add cadence relative to an explicitly prescribed drill target as a descriptive measurement. Do not equate normal timing variation with bad technique.

For each criterion, deliver a one-page specification: intended skill/drill, accepted styles, view/visibility requirements, temporal interval, features and units, preliminary threshold, examples/counterexamples, exemptions, cue wording, and abstention reasons. Only after labeling and evaluation should it become a released correction.

## Necessary verification

Use meaningful tests around behavior that can corrupt feedback: mirrored/anatomical mapping, aspect-correct geometry, timestamp gaps, track reset, missed/extra/double-counted events, unknown actions, and a missing required joint suppressing a finding. Add integration replay checks using consented fixtures kept outside Git when private.

Verify offline and live feature parity. A noncausal smoother or full-clip model cannot be substituted into the live benchmark. Compare original PyTorch outputs and exported inference results before accepting an ONNX/Core ML model. Exercise camera permission rejection, stop/restart, offline operation, and deletion.

No unit-test suite or inference result exists yet; these are acceptance requirements for the implementation sprint.

## Decision record at sprint end

Record exact package versions, model URLs and hashes, licenses, camera/browser settings, supported view(s), benchmark split IDs, per-condition metrics and confidence intervals, failed criteria, and next experiment. The decision can be “MediaPipe is sufficient,” “RTM materially improves the supported cue,” or “capture does not support this cue.” Each is useful progress.

Only after this decision should implementation expand into a polished guided coaching experience. Early visual design should serve framing, evidence review, and usable cues; model validity is the critical dependency.
