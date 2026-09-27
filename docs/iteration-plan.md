# Next iteration decisions

Reviewed September 26, 2026. The immediate priority is reliable evidence from the current app; the broader goals remain in [project-plan.md](project-plan.md).

## What shipped

Keep MediaPipe Full live. The app now prepares a separate local analysis of new completed recordings, using native WebCodecs frames/timestamps and sequential Full inference. Progress, cancellation, cached reports and original/analysis comparison are available. **Original results remain the default**: rerunning a video is not guaranteed to improve recognition.

Observed-retraction/reversal acquisition and a bounded path-detour check address specific detector failures. The current revision follows coherent inward guard shifts, separates partial-return repetitions and preserves recent acquisition observations after unsupported small gestures. An accepted full stroke provides a time-limited extension reference for shorter-travel same-hand repeats, without lowering the isolated-stroke requirement. It improves development-set recognition without repairing erroneous landmarks. Review groups four supported combinations and exposes uncertainty intervals. Practice-focus prompts keep testing simple without biasing recognition. These features are experimental and do not grade technique; see [recognition events](recognition-events.md).

The [temporal training pipeline](temporal-training.md) is runnable, but both its initial and five-session experiments produced excessive false events. Its weights are not enabled in the browser. RTMPose-M and RTMW-L improve selected visibly misplaced right-wrist/elbow estimates, making pose replacement worth pursuing. They have not established a drop-in live replacement: confidence semantics, person selection and input/runtime differences matter. Detailed measurements and recordings remain private.

## Decisions before another model change

The [six-punch data preparer](action-dataset.md) and [independent coaching review](coaching-review.md) now make the next evidence collection runnable. Add the missing lead/rear hooks and uppercuts plus explicitly reviewed background before extending the causal model to per-arm families and event boundaries. Use [two short capture rounds](six-punch-capture.md) to start; v7 adds experimental hook/uppercut rules, but reliable recognition of those actions is still unresolved. The current blocker is often missing or incorrectly placed active-arm joints, so collecting more punch labels alone cannot repair the pose input. The [external-source audit](research/boxing-datasets-2026-09.md) identifies BoxingVI and ShadowPunch as access/rights leads, not imported training data. Correctness of technique requires separate coach judgments.

| Remaining problem                                   | Evidence to collect                                                                   | Next experiment                                                                           |
| --------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| A confident wrist is in the wrong place             | Original frames with manually marked wrists/elbows and visible/ambiguous states       | Extend the promising RTM joint comparisons to a fixed full-clip and runtime evaluation    |
| False extra person boxes prevent analysis           | Verified target-person regions and background examples                                | Validate explicit person selection/tracking before judging another pose model             |
| RTM native scores fail MediaPipe-oriented gates     | Manually checked joint locations and visibility across native score ranges            | Freeze and validate a model-specific observation policy before comparing recognition      |
| Faster native inference needs a live path           | Full-clip profiles, coordinate differences, warm-up and sustained frame delivery      | Test a local Core ML capture loop before adding a browser/native bridge                   |
| Natural motion defeats fixed rules                  | Continuous action/background labels, including imperfect returns and fast repetitions | Retrain the temporal baseline after adding independent sessions and negatives             |
| Relaxed arm lowering looks like a straight          | Non-punch gestures plus real body-level straights in several views                    | Test direction and torso-relative features without assuming every low fist is invalid     |
| A new video pass changes counts                     | Identical source timestamps, model/runtime provenance and frozen action labels        | Compare recovered, lost and false events; do not choose a default from total counts alone |
| Recognition is adequate but correction is requested | Coach-labeled criteria, counterexamples and view limitations                          | Validate one observable cue before enabling technique feedback                            |

Do not train a pose network from scratch for this pilot. More landmarks or higher model confidence does not establish anatomical correctness. Do not tune confidence thresholds to make one clip produce the intended punch count.

## Short development loop

1. Review and label source video before showing predictions. Count recognizable imperfect actions, preserve non-punch movement, and mark genuinely unobservable intervals.
2. Freeze the detector/model/settings and run the complete development set. `npm run benchmark -- <labeled sessions> --output-dir <new private directory>` compares counting rules on unchanged poses; separate video-analysis reports preserve fresh-inference provenance.
3. Trace misses to tracking, onset, peak, path, recovery or grouping. Add a meaningful regression for the proposed mechanism, then recheck lost matches and false events as well as gains.
4. Evaluate the frozen version on a later-day session that has not informed changes. Once that session is used to revise the system, treat it as development data and reserve another.

Use existing clips before requesting another recording. The next user test after a frozen update should be small: [one natural 30-second round](first-user-test.md), with paced singles, comfortable quick sequences and a short idle ending. Repeat on a later day when practical. More recordings are useful only when independently labeled and kept separate by session; adjacent-window random splits leak similar evidence.

## Before enabling a learned model or coaching

Add materially more continuous sessions, especially guard changes, clothing adjustments, pauses, fast repeats and genuine occlusions. Keep action labels separate from technique judgments. Freeze preprocessing and event decoding before evaluating; frame/clip classification scores cannot replace continuous-round false-event and missed-action measurement.

For broader claims, hold out people and environments. For technique feedback, obtain a qualified boxing coach's criterion labels and abstention rules. Personal success does not establish learning benefit or universal coverage. Sustained hardware testing and later selected-clip cloud review remain separate work; current application flows upload nothing.

Research references and reuse restrictions are documented in [optimization sources](research/optimization-update.md), [pose research](research/pose-models.md) and [third-party notices](../public/third-party-notices.txt). No external project score is validation of this coach.
