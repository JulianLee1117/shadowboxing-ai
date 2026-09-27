# Six-punch recognition and observable technique feedback

Research checked 2026-09-26. This is a proposed labeling and validation contract, not evidence that the app can already grade boxing technique. It extends [coaching validity](coaching-validity.md); the [master release gates](../project-plan.md#release-gates) remain canonical. No private recordings or participant results are included.

## Decision

Expand the action vocabulary independently of form assessment. Start technique work with two coach-reviewed criteria, `guard_recovery` and `non_punching_hand_guard`, in isolated straight-punch drills with a selected high guard. Build independent review packets now; keep automated corrective advice disabled until each criterion passes validation. A beginner selects a training goal, not whether their own technique is correct.

The current straight-punch detector's `guardReturn` records spatial return toward that repetition's origin by `detectedAtMs`. That origin can itself be a poor guard. Its `endMs` marks confirmed retraction, which can precede complete recovery. Neither field is a technique label or a suitable substitute for reviewing the subsequent video.

## Evidence and its limits

The [IBA-hosted coaching manual](https://www.iba.sport/wp-content/uploads/2019/01/AIBA-Coach-Regulations-Manual_WEB_2019_01-1.pdf), printed pp. 41–55, distinguishes lead/rear straights, hooks and uppercuts, including head/body variants. Its basic instruction includes protecting with the other hand and returning after delivery. Its later sections cover combinations, feints and different distances. These support a contextual rubric, not one universal skeleton template. The manual is foundational material, not newly published research.

[England Boxing's coaching handbook, Part 1](https://www.englandboxing.org/wp-content/uploads/2022/03/EB_Boxing-Coaching-Handbook-Part-1_v8-002.pdf), notably the punch instruction and observation/feedback sections, also supports guard control and focused coaching. We propose the observable criteria below; the federation has not validated our thresholds, software or advice.

The boxing-specific [Lahkar et al. validation study](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2022.939980/full) used three elite boxers, ten synchronized video cameras and an optical reference system. Joint-angle agreement varied, with particular limitations for rotations and the wrist. This does not establish single-webcam accuracy for fast punches.

The current [MediaPipe output documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) defines landmark visibility as likelihood of visibility in the image and supplies model-estimated image/world coordinates. Visibility is not verified hand identity or localization accuracy. Our engineering implication: high visibility must not override contradictory pixels or justify anatomical-angle claims.

[FitAQA v2](https://arxiv.org/abs/2608.08736v2), revised September 17, 2026, separates visual perception, correctness judgment and temporal grounding across 30 bodyweight exercises. Its evaluations identify perception and error localization as continuing problems for multimodal language models. This is a recent preprint about fitness, not boxing validation. Its useful design lesson is to preserve inspectable evidence before generating advice; changing to a newer general video model does not establish coaching validity.

## Six-punch taxonomy

Store action family, anatomical hand and stance at the event. Derive lead/rear language from those fields; preview mirroring changes presentation only.

| Canonical action | Left hand leads / orthodox | Right hand leads / southpaw |
| ---------------- | -------------------------- | --------------------------- |
| `lead_straight`  | Left jab                   | Right jab                   |
| `rear_straight`  | Right cross                | Left cross                  |
| `lead_hook`      | Left hook                  | Right hook                  |
| `rear_hook`      | Right hook                 | Left hook                   |
| `lead_uppercut`  | Left uppercut              | Right uppercut              |
| `rear_uppercut`  | Right uppercut             | Left uppercut               |

Suggested representation: `family: straight|hook|uppercut`, `hand: left|right|unknown`, and `role: lead|rear|unknown`. Preserve `other_motion`, `unknown_action` and `unobservable` outcomes. Unknown stance need not erase a visible left hook; it prevents asserting lead/rear. Stance switches need time-local stance labels. Head/body intent is a separate attribute, unknown by default in shadowboxing. Requested drills and optional punch numbers are never action ground truth.

Recognition hypotheses should use the temporal delivery and recovery, not a single pose: outward straight trajectory, lateral curved delivery, or upward/forward delivery from below. These descriptions do not translate reliably into screen-axis rules. Toward-camera straights can appear short; a hook can cross the face; short hooks, body uppercuts and hybrid trajectories can be ambiguous. Label ambiguity rather than forcing six classes. Feints, parries, arm lowering/re-guarding and ordinary reaching are essential negative examples.

The manual's basic hook example uses an elbow near a right angle. That does not justify requiring every hook to have a 90-degree elbow, especially in a 2D projection. Keep action recognition and any future hook-specific coaching rubric separate.

## First criteria and view requirements

All entries below are **labeling candidates**, initially limited to coach-confirmed isolated jab/cross practice in the selected beginner high-guard context. Hooks and uppercuts can receive action labels now; these first criterion versions are not applicable to them. Advanced styles require their own reviewed context, not harsher scoring against a beginner template.

| ID / status                             | Proposed observable question                                                                 | Required evidence and exclusions                                                                                                                                                                                                                                                                                                               |
| --------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `non_punching_hand_guard` / first       | Did the other hand remain within the chosen guard region during this stroke?                 | Identifiable non-striking hand and head reference throughout the relevant delivery. Prefer a tested front or oblique view where that hand stays visible. Exclude another punch, intentional parry/block, alternate guard or deliberate transition. A hidden far hand is unobservable.                                                          |
| `guard_recovery` / first                | After the isolated stroke, did the striking hand visibly recover to the chosen guard region? | Visible hand/head and post-stroke video until recovery or a contextual stopping point. Evaluate actual video after the detector's endpoint. If the clip, occlusion or next purposeful action prevents judgment, mark unobservable; do not turn missing recovery evidence into a fault. An overlapping combination requires a different rubric. |
| `preparatory_pullback` / later research | Was there a distinct, unnecessary visible pullback before an isolated straight?              | Clear pre-launch guard and delivery in a view that exposes the direction. A coach must distinguish an extra preparation from preceding recovery, feint, pivot or an intended setup. Do not apply a straight-punch rule to hook/uppercut loading.                                                                                               |

The guard region needs a coach-reviewed reference for the chosen drill. Normalizing it to visible head/torso size is a measurement proposal, not proof the user's initial pose is correct. A wrist landmark is only a proxy for a hand: verify the hand region in RGB before treating a wrist entering a polygon as a successful guard.

Start with visible position, not “too slow.” Recovery timing can be descriptive; judging speed requires a prescribed drill target, stable phase definitions and measured timing repeatability. A suitable eventual cue is: “Your rear hand moved below the selected guard during three of the five jabs I could assess.” It needs evidence for every claimed repetition and a coach-approved corrective template.

## Abstention is criterion-specific

A frame can support punch identity while failing to support guard position. Track assessability over each criterion's critical interval, not by one whole-round visibility percentage. A side view might reveal extension while concealing the other hand; a front view might reveal guard while compressing extension. Test front, oblique and side views on the actual laptop height and lens tilt; there is no universally optimal angle.

Withhold a judgment when identity is uncertain, the hand is clipped/blurred/hidden, the apparent wrist lies on the forearm, a gap covers the relevant phase, the action or stance is ambiguous, or the context is unsupported. A model's confidence cannot establish that a hand was observed. Preserve separate image evidence, action certainty, measurement reliability and criterion judgment.

Do not derive force, impact quality, pressure under a foot, balance margin, muscle activation or injury risk from these cues. Do not grade fist/wrist alignment, anatomical elbow angle, shoulder rotation or chin protection from sparse pose alone. Apparent 2D rotation is a projection measurement; a plausible reconstructed body is not a measured force or validated boxing biomechanics.

## Independent coach-label contract

An action label answers **what happened**. A criterion label answers **whether one contextual requirement was assessable and met**. A feedback decision answers **whether a correction should be delivered**. Keep all three records separate from model predictions and original session evidence.

Each local review packet should contain:

- Schema/rubric versions, source-video SHA-256, source session ID, recording offset and timestamp units.
- Independently annotated action ID, family, physical hand, stance/context, and source-time interval. Preserve uncertain boundaries and any stance changes.
- A blank record per proposed criterion, reviewer ID, context eligibility, view adequacy, evidence interval, rationale and judgment. Initialize every record to `not_reviewed`; do not prefill it from detector results.
- Separate `feedbackWarranted: yes|no|uncertain|not_reviewed`. A visible deviation can be too minor, context-dependent or low priority to warrant a cue.
- Optional observation measurements with explicit units and algorithm/model versions. These are model evidence, not coach labels or calibrated probabilities.

Judgment semantics:

| Judgment         | Meaning                                                                                            |
| ---------------- | -------------------------------------------------------------------------------------------------- |
| `pass`           | Eligible, adequately observed, and this criterion was met; not a global endorsement of form.       |
| `fail`           | Eligible, adequately observed deviation under this rubric; not automatically a useful correction.  |
| `ambiguous`      | Available evidence admits materially different judgments or the rubric is insufficiently specific. |
| `unobservable`   | The criterion could apply, but decisive visual evidence is unavailable.                            |
| `not_applicable` | Action, drill, guard/style or deliberate competing action falls outside this rubric.               |
| `not_reviewed`   | No independent judgment yet; never count it as pass or failure.                                    |

Illustrative review record; field spelling can follow the packet implementation:

```json
{
  "actionId": "reference-rep-04",
  "criterionId": "non_punching_hand_guard",
  "rubricVersion": "isolated-high-guard-v0.1",
  "reviewerId": "coach-A",
  "contextEligibility": "eligible",
  "viewAdequacy": "adequate",
  "judgment": "fail",
  "feedbackWarranted": "yes",
  "evidenceIntervalMs": [12300, 12500],
  "rationale": "The identifiable rear hand moved below the reviewed guard region during delivery; no competing action was visible."
}
```

These are hypothetical labels, not a measured example. Preserve individual reviewers' records and any later adjudication. An agreement summary must show eligibility/view disagreements and unobservable rates, not compute an apparently strong binary score after silently discarding difficult cases.

## Validation sequence

1. Have two qualified coaches independently review raw video without model overlays or predictions. Establish action references separately from criterion judgments, then adjudicate disagreements while retaining both originals. Test whether the rubric itself yields useful agreement before training a classifier.
2. Include acceptable style variants, visible deviations, both stances, body-level punches, fast combinations, partial returns, feints, defensive actions and mundane arm movements. Occluded examples need explicit unobservable labels. Do not ask a beginner to self-certify form or reproduce potentially harmful faults.
3. First validate hand/head localization and phase evidence against the video. Then measure criterion judgments and final emitted cues independently. A better event count is not evidence of better critique.
4. Split by person/session before extracting clips. For personal adaptation, reserve a future day and changed capture conditions; describe that narrower result honestly. Broader claims require additional people and supported skill/style groups.
5. Report each criterion by view and hand: corrective-claim precision with uncertainty, missed deviations, false cues per minute, eligible-event coverage and all-event coverage. Track coach-rated usefulness separately. Use session/participant-aware confidence intervals and publish abstention reasons.

The existing release protocol proposes criterion precision of at least 95%, a one-sided 95% lower confidence bound of at least 90%, and at least 70% eligible-event coverage, among other gates. These remain unachieved targets; passing on a few personal clips would not establish general coaching accuracy. A label's uncertainty cannot be eliminated by asking a language model to choose more confidently.

## Smallest implementation foundation now

Build the planned local `ml/coaching.py` packet generator, validator and agreement report around **explicit reference annotations plus the original video's verified SHA-256**. No uploads or automated advice are needed. Candidate metadata can include all six action labels while making these two rubric versions inapplicable outside isolated straights. Keep packet labels separate from detector events and saved-video analysis; preserve originals and refuse mismatched source fingerprints.

Meaningful tests should cover source mismatch, blank labels remaining blank, pass/fail requiring eligible and adequate evidence, invalid/out-of-range evidence intervals, distinct abstention outcomes, independent reviewer preservation and physical-side mappings in both stances. Add a regression ensuring current `guardReturn` cannot populate a technique judgment. Agreement reporting should not require every action to have a forced binary answer.

This yields concrete coach-review data and a testable rubric without adding live controls or asking the user to approach the laptop again. The next implementation step is a validated observation extractor for one criterion, followed by calibration and a controlled cue policy. A language model may later phrase accepted structured findings; it must not invent additional faults, quantities or biomechanical causes.
