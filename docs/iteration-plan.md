# Next iteration decisions

Reviewed September 27, 2026. The immediate priority is consistent recognition on a new round. The wider product goals remain in [the project plan](project-plan.md).

**Data priority correction:** the user's recordings are examples of the target webcam domain, not a standard of correct boxing. Prioritize external skilled demonstrations plus explicitly assessed mistakes/acceptable variations for form, and varied skilled/imperfect actions for recognition. [The revised acquisition plan](research/expert-reference-data.md) separates these roles. The user does not need to supply perfect technique or label their own videos as the coaching foundation.

## Current product loop

Choose a lead hand and duration, enable the camera, then start the countdown and step back. The skeleton, persistent punch history and running count provide live feedback. Finishing opens the original video and live results immediately. A full video pass is optional; it is useful for diagnosis, not a required step or a promise of better recognition.

The optional native RTMPose service and personal causal recognizer are running locally. They recognize all six stance-dependent punch identities. The current model uses seven development recordings; fitted results do not establish transfer. MediaPipe remains available as the default for installations without local weights. [Implementation status](implementation-status.md) describes the shipped paths and limits.

## Current next gate

The latest round exposes two distinct problems: patchy right elbow/wrist observations on most rear hooks, and missed short/fast jabs even when those joints are tracked. The persistent log and 150 ms arbitration hold improve feedback delivery; they do not fix those recognition errors. [The latency audit](research/detection-latency-2026-09.md) retains the failed exact-parity gate and reviewed boundary tradeoff.

1. On existing development footage, mark visible wrist/elbow pixels and physical hand for curved-action and guard windows. Compare one fixed observation challenger (pose or short RGB-plus-pose), measuring localization, false guard counts and whole-event outcomes together.
2. Separately diagnose causal phase/cycle behavior for fully tracked short and repeated straights. Freeze a protocol before fitting; do not change pose gates to mask classifier errors.
3. Build the external skilled-reference and counterexample collection. Use **Correct punches** as an optional labeling interface for source recordings; user cards can correct action identity without certifying good form. Form examples need explicit context and assessment, and mixed-combination windows may be not applicable. Add missed moments and ordinary non-punch movement. Selected windows do not certify complete annotation or exact action bounds.
4. Keep an untouched later-day round for the next frozen candidate. Report occurrence, interval coverage, false counts, observed pose cadence and actual delay separately. Evaluate one coaching cue only after its visible hand/head evidence and context rubric are reliable.

The first labels are supervised review material, not reinforcement learning and not an immediate model update. [The form roadmap](research/form-coaching-roadmap-2026-09.md) explains why genuine pose throughput, clear source pixels and guarded abstention matter more than a nominal 30 fps label.

## Overnight progress and next experiment

The public BoxingWeb archive is now local: 50 match videos and matching annotations. All 50 JSONs and one representative six-family video have been inspected. The audit found invalid intervals, uncertain label provenance in one round, and athlete overlap across the supplied splits; raw row counts are not clean training-label counts. See the [acquisition evidence](research/expert-reference-data.md). This gives us concrete diverse actions to inspect, but fight effectiveness is not a technique grade and the media terms remain unresolved.

A bounded multi-person RGB+pose inspection run on the sample confirms why the existing single-person webcam extractor cannot simply ingest fight footage. The detector sees opponents, officials and spectators. The [inspection CLI](rgb-pose-inspection.md) retains source frame indices, decoded timestamps, raw scores, all person instances and hashes. Associate an annotation with a reviewed actor before creating training targets. The next useful representation comparison is a small reviewed RGB-plus-pose versus pose-only experiment, with actor identity, source timing, unknown intervals and source grouping fixed first.

Two tempting local adjustments were rejected before activation: a time-based straight-punch sample requirement recovered no additional strict matches across 17 development views, and one-thread ONNX inference improved median saved-JPEG service time only 2.24%, below its predeclared 10% gate. No current recognition weights or service defaults changed. Detailed evidence and continuation priorities are in [the overnight ledger](overnight-progress.md).

The user-facing correction flow now says **Correct punches**. Action labels save and resume without a guard answer; form observations are optional and explicitly provisional. This reduces labeling friction without making the user's execution a good-form reference.

## What the earlier rounds tested

Two fresh rounds were recorded after the previous bundle was frozen. Preserve their original video, per-frame native decisions and model fingerprint before any re-analysis. Label recognizable actions from video before consulting predictions, including imperfect punches, ordinary transitions and uncertain intervals.

Both captures sustained roughly 20 pose observations per second, with no gaps longer than 150 ms. The low-count round continued receiving arm observations after detections stopped. Processing every source-video frame also retained its low count. This directs the next investigation toward the pose representation, classifier and event decoder rather than treating another full pass or a larger live frame queue as the solution. A high native joint score still does not establish the correct joint location.

Keep the initial results as prospective evidence for the frozen model. Once a recording informs a representation, decoder or training decision, its later results are development evidence. Do not relabel missed punches as background, move reference boundaries to improve matching, or count an ambiguous interval as a success.

Separate action identity from interval coverage. The strict interval evaluation can reject a correctly named punch when the output covers only its brief peak instead of the labeled load/hold/return. The implemented [peak-occurrence diagnostic](recognition-evaluation.md) uses a fixed ±250 ms window, physical hand and family, with one-to-one matches. It requires explicit independent peaks and leaves strict interval scores unchanged. Its tolerance is frozen before a future untouched test; its results on these already-inspected clips are development diagnostics. Provisional labels are action judgments, not technique assessments, and upward arm-motion lookalikes need coach review.

Focused image/pose comparisons found no gross active-hand swap in the missed curved actions. An opposite-arm feature probe affected the repeated-straight misclassification but did not recover all missed curves. One fixed training experiment omitted opposite elbow/wrist offsets and derivatives in half of training sequences, using only the original seven recordings. It recovered some new actions but increased false detections and misses in older rounds, so it was rejected. Production weights and recognition thresholds remain unchanged. Keep that negative result; the next experiment needs broader trajectory coverage and controlled evaluation, not repeated tuning of this same pair of rounds.

## Work completed in this iteration

- [Personal training](personal-training.md) now runs from pinned video/session hashes and explicit recording, participant and day groups. Unknown intervals never become background targets. Model bundles are immutable and activation is separate.
- [Paired native replay](personal-replay.md) compares captured evidence, an explicitly supplied baseline and a candidate. It reports recovered/lost reference identities and passive classifier/decoder traces. Inference never reads reference labels or saved decisions.
- Both new recordings are now available to development training. Their original prospective results and frozen positive labels remain intact. Earlier straight-only clips were reviewed for all families in separately derived training views, including unknown masks where arms leave view. This additional review was informed by failure diagnosis, not blind validation.
- Bounded expanded-data, initialization and rehearsal experiments compare old and new recording groups. Recovering new punches while adding false guard counts is a regression tradeoff, not automatic acceptance. Failed candidates and their protocols remain retained locally.
- The expanded-data candidates were rejected. The fully reviewed public-CLI fit recovered new actions but regressed older mixed rounds. Rehearsal preserved more earlier decisions, but added real guard/arm-lowering counts; peak-time evaluation exposed regressions that improved interval scores alone hid. The active checkpoint remains the earlier seven-recording fit.
- The observed-cycle decoder fix is shipped with unchanged weights. It prevents a completed first straight from being discarded when the model remains continuously active across repetitions. Replays with current weights remain unchanged; the fix alone does not resolve the newest round's missed curves.
- Forty successive imported-file rounds completed in about 21 minutes with every native session closed and no app errors or external requests. Processing averaged about 20 observations per second, with bounded observed delay and roughly steady service memory. Counts still varied on repeated playback of the same clip. This is pipeline stress evidence, not a sustained foreground-webcam or independent recognition acceptance result.

## Next acceptance checks

| Priority                                        | Work                                                                                                                                                          | Evidence required before shipping                                                                                                                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Validate the frozen version on a fresh round | Record comfortable singles, quick repetitions, hooks/uppercuts and deliberate non-punch guard movements. Review video before predictions.                     | Report hand/family occurrence, interval timing, false idle counts and delayed detections separately. Preserve the whole round as held-out evidence until that assessment is complete. |
| 2. Broaden background and posture coverage      | Add distinct continuous sessions with relaxed arm lowering, guard changes, turning and varied trajectories. Group every pose version with its original video. | Preserve ordinary-round matches and reduce actual non-punch counts. Do not tune thresholds repeatedly on the same pair of clips or mistake boundary mismatches for new motions.       |
| 3. Complete webcam performance acceptance       | Measure successive foreground webcam rounds, including callback-to-inference delay, skipped observations, cleanup and detection delay.                        | Report real webcam measurements separately from the repeated imported-file stress run. Do not use source-video fps as processing throughput.                                          |
| 4. Add one useful coaching cue                  | With a boxing coach, define one visually observable criterion and acceptable variants, then label examples and counterexamples independently.                 | High cue precision and explicit abstention on unobservable views; punch classification alone never becomes a form grade.                                                              |

Keep a small untouched later-day check before claiming a new personal model is more reliable. Broader support requires additional people, stances, backgrounds and camera angles held out by person/session. More clips of one recording or synthetic views of one motion are not independent evaluation samples.

## When to change model families

First establish whether a miss comes from bad coordinates, a poor representation or temporal decoding. A larger pose model cannot repair an event decoder, and a new classifier cannot recover an unobserved trajectory merely by receiving a higher confidence score.

If visible pixels contain useful evidence that the keypoints lose, compare a short-window RGB-plus-pose model offline. [PoseC3D](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html) and the [official MMAction2 implementation](https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/posec3d) are concrete representation baselines, not ready-made boxing coaches. Their benchmark scores do not transfer to continuous six-punch recognition, and an offline clip model must be checked for future-frame use, latency and Mac runtime cost before live deployment. Do not add a large video model solely because a demo looks convincing.

External data work should target continuous actions and non-punch movement with clear physical-hand labels and usable media rights. [The source audit](research/boxing-datasets-2026-09.md) identifies candidate datasets and outstanding access/rights questions. No new dataset license, private-video upload or outreach is implied by this plan.

## Next user-facing test

Use the two new recordings for this investigation before asking for more. Check saved-round removal, Undo and restore with disposable test data; keep actual user recordings intact during development. The next recognition check should be one ordinary fresh round against a frozen candidate, with comfortable singles, quick repetitions and a few seconds of non-punch movement. [The short test guide](first-user-test.md) explains the controls.

Voice can consume the existing finalized detection events later. Start with an optional brief callout or combo prompt once recognition is consistent; technique corrections require their separate validation. Voice should not make an uncertain detector sound authoritative.
