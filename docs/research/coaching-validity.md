# Coaching validity and product rubric

Research checked 2026-09-26. Proposed product: a personal-first shadowboxing coach on an M3 MacBook Pro webcam, with local live analysis and optional cloud review of selected clips. The initial user is a beginner who knows the basics; the longer-term ambition spans experience levels. This document separates published evidence from proposed engineering and coaching decisions. Numerical product targets below are release proposals, not achieved results or published boxing standards. The [master project plan](../project-plan.md#release-gates) is the canonical release protocol; this document explains its coaching-validity requirements.

## Core recommendation

Build a narrow, evidence-producing coach before expanding the action vocabulary. The first useful system should recognize isolated lead and rear straight punches in a guided drill, identify a few visible deviations from an explicitly selected beginner high-guard drill, and show the exact repetition that supports each correction. A fluent description of boxing is not sufficient evidence that a system correctly observed a particular punch.

Treat these as separate questions with separate acceptance tests:

1. Was a movement visible enough to assess?
2. What action occurred, with which hand, and when?
3. Was a specified observable criterion satisfied in this drill?
4. Is a correction useful, appropriate for this person, and supported by the observed sequence?
5. Does practicing with that correction improve independent coach ratings over time?

Correctly answering one does not establish the others. In particular, action-recognition accuracy does not establish critique accuracy.

### Experience and individual adaptation

Build the first curriculum around the user's stated level: a beginner with basic familiarity. Onboarding should let the user select experience, goals, stance, and drill/guard intent; do not infer skill level or an appropriate style from appearance. Offer an editable profile, not a permanent classification.

Broader usefulness requires more than changing the tone of an explanation. Beginners can use demonstrations, isolated drills, simpler cues, and slower guided sets. Experienced users need optional demonstrations, flexible drills, fewer interruptions, and rules that respect intentionally different guards and combinations. Recognition can generalize across these contexts before every critique does. A user should be able to use recording, event timelines, and supported drills even when a particular style has no validated critique rubric.

Personalization should adapt body scale, supported view, chosen coaching goal, and feedback frequency. It should not redefine every habitual movement as correct or compare every boxer with one ideal skeleton. A reference guard needs a reviewed rubric; an initial self-recording alone is not a correctness label.

“Useful across skill levels” is a product direction, not evidence of universal coverage. Before broader release, evaluate both novices and experienced boxers, with acceptable style variants and independently labeled coach disagreements. State supported populations, views, techniques, and drill contexts in the product. Do not promise reliable critique for every person, mobility pattern, or style before those conditions have been tested; offer capture and descriptive modes where assessment is unsupported.

## What the evidence supports

| Evidence | Established result | Implication for this project |
| --- | --- | --- |
| [England Boxing coaching handbook, Part 1](https://www.englandboxing.org/wp-content/uploads/2022/03/EB_Boxing-Coaching-Handbook-Part-1_v8-002.pdf), printed pp. 67, 73–75, 133–134 | The beginner material includes guard recovery, control of the non-punching hand, and coordinated movement. It recommends observation before intervention and limited, specific feedback. | Choose a declared drill and a few coach-reviewed criteria. Feedback frequency and visual criteria below are our proposed implementation, not a reproduction of the handbook. |
| [Lahkar et al., 2022, boxing markerless validation](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2022.939980/full) | Three elite boxers were recorded with 10 synchronized video cameras at 60 Hz and 12 optical cameras at 300 Hz. Segment velocities agreed more strongly than joint angles; internal/external rotations were relatively weaker. | This is evidence of potential, not validation of a single laptop camera. Do not transfer its accuracy estimates to our pipeline or claim precise shoulder/wrist rotation from sparse landmarks. |
| [Stefański et al., 2024, single-camera punch detection](https://www.mdpi.com/1099-4300/26/8/617) ([PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11353713/)) | The authors report 95% balanced accuracy but 49% F1 for punch frames and 97% F1 for non-punch frames in a boxing-bout detection task. | Headline summary metrics do not establish usable punch-event performance. Report event precision/recall, per-class performance, false events per minute, and technique-critique validity separately. Bout footage is a different domain from webcam shadowboxing. |
| [OpenCap Monocular, March 2026 preprint](https://arxiv.org/abs/2603.24733) | This single-smartphone method was validated for walking, squatting, and sit-to-stand. The reported rotational mean absolute error is 4.8 degrees for those evaluated tasks. | An interesting future comparison for 3D reconstruction, not evidence that its estimates are accurate for fast punches. Its physics and learned priors must be tested on boxing before coaching use. |
| [MotionBench, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Hong_MotionBench_Benchmarking_and_Improving_Fine-grained_Video_Motion_Understanding_for_Vision_CVPR_2025_paper.pdf) | The benchmark specifically tests detailed motion comprehension because coarse video understanding does not sufficiently evaluate it. | Benchmark any video-language model on our actual event and critique tasks. A modern model's general video score is not a boxing-coaching credential. This paper does not establish the performance of models released after its evaluation. |
| [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) and [Geifman and El-Yaniv, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) | Classifier scores can be miscalibrated; selective prediction trades coverage for lower error. | Learn calibration and rejection thresholds on separate boxing validation data. Report both accuracy and coverage. An arbitrary detector confidence or a language model's claimed confidence is not a probability that advice is correct. |

The following camera and rubric proposals are engineering hypotheses to validate. They are deliberately narrower than full biomechanical assessment.

## An observability contract

Every criterion needs a required view, required visible body regions, supported movement context, and a precise statement of what it measures. The UI should describe an unavailable criterion as “not assessed,” not assign it zero or infer that it was satisfactory.

| Candidate capability | Measurement we can attempt | Necessary conditions | Initial status |
| --- | --- | --- | --- |
| Lead/rear straight recognition | Temporal hand/arm trajectory, body-relative motion, RGB context and declared stance | Visible start and return; verified anatomical left/right; recognized view | MVP in guided drills |
| Punch count and instructed sequence | Matched action events and order | A background/unknown class; separated event boundaries; no counts based only on the command given | MVP |
| Guard recovery | Visible hand returns to a coach-approved guard region after an isolated punch | Face/hand both visible; selected high-guard rubric; recovery window not interrupted by another deliberate action | MVP candidate |
| Non-punching hand displacement | The resting hand visibly moves away from its expected region during the other hand's straight punch | Far-side hand not hidden; no parry, feint, defensive transition, or alternate guard | MVP candidate |
| Cadence and combo pauses | Time between recognizable events | Monotonic capture timestamps and known frame losses | MVP descriptive metric; “too slow” requires a prescribed target |
| Large preparatory hand motion | Visible opposite-direction hand motion before a straight punch | Correct event boundary; appropriate camera view; exclude a preceding recovery or legitimate feint | Later, after a coach-defined telegraphing rubric |
| Stance consistency and gross crossing of feet | Projected foot locations and changes relative to the person's calibrated stance | Entire feet visible; stable camera; camera-angle-specific validation | Later; do not equate projected geometry with balance |
| Hooks and uppercuts | Temporal class and hand | View supports arc/depth cues; wrist and elbow tracked through occlusion | Later guided six-punch vocabulary, before free mode |
| Slips, rolls, steps and pivots | Visible movement category and timing | Full relevant body visible and separately labeled movements | Later; movement recognition does not prove an incoming punch was evaded |
| Hip/shoulder timing | Relative timing of visible segment orientation proxies | Stable view, adequate temporal resolution, boxing-specific reference validation | Research only initially |
| Chin position, elbow flare, wrist alignment | Detailed head/arm/hand orientation | Sufficient face/hand pixels and validated 3D or view-specific measurement | Do not launch from generic sparse-pose heuristics |

Do not infer the following from this MVP: punch force or power, pressure under each foot, true center of mass or stability margin, impact quality, accuracy against a real opponent, tactical success, injury likelihood, muscle activation, or physiological fatigue. Shadowboxing has no measured contact event, and a plausible 3D body reconstruction is still a model-based estimate. Call a directly measured timing trend “cadence changed,” not “you are fatigued.”

Use normalized projected distances for comparisons within a supported view; do not label them centimeters. Do not display sparse 2D elbow angles as anatomical 3D joint angles. An improvement smaller than repeatability error should be reported as inconclusive.

## Camera acquisition on the actual MacBook

Start with the existing webcam and measure what it delivers. Requested resolution and frame rate are not proof of sustained capture performance. Record effective frame intervals, exposure/blur proxies, person size, occlusion, dropped frames, inference delay, and thermal behavior during a complete round.

Proposed setup flow:

1. Ask for stance selection: orthodox, southpaw, or unsure. Handedness does not uniquely determine stance. Let the user correct the interpretation later.
2. Verify anatomical side by having the user raise a named hand. A mirrored preview must not invert labels in analysis, recordings, or coaching language. Store the transform explicitly.
3. Guide the user to frame the head, both hands at full reach, and feet if the drill requires feet. Use an on-screen silhouette and visible margins rather than an assumed distance in meters.
4. Have the user perform a slow test extension and a few comfortable repetitions. Flag clipping, repeated hand disappearance, blur, or extreme camera tilt before evaluating technique.
5. Compare front, approximately three-quarter, and side views on the same short drill. A laptop's screen angle and low lens position matter. Choose the view that best supports the selected criterion on this hardware and room; do not declare 45 degrees universally optimal.
6. Save the camera/drill combination as a supported setup. Recalibrate after moving the laptop or materially changing the boxer's position.

Practical hypotheses to test:

- A front view may show both hands and guard changes well but compress the trajectory of a punch directed toward the lens.
- A side view may expose extension and recovery well while hiding the far-side hand.
- A three-quarter view may offer a useful compromise, but the far-side hand can still disappear near the face.
- A bright, even scene and a stable laptop mount may matter more than selecting a larger model if fists blur or leave the image.
- Higher captured frame rate can help event timing if exposure and inference throughput remain adequate. At 30 fps, frames are about 33 ms apart; at 60 fps, about 17 ms. Do not report event timing with millisecond precision that capture cannot justify.

One camera can support separate front-view and side-view drill blocks. Two separate recordings are not synchronized multi-view capture and cannot be triangulated as if they describe the same punch. Consider a synchronized phone camera only if the single-camera validity tests demonstrate a material gap.

The boxer should not need to look sideways at the screen during a repetition. Show the demonstration and setup first, use brief audio between sets, and reserve detailed overlays for replay.

## Product modes and teaching loop

### Guided fundamentals first

Initial vocabulary: `lead_straight`, `rear_straight`, `guard`, `other_motion`, `unobservable`. Start with single repetitions, then a lead–rear straight combination. Support both stances. Use semantic names internally; punch numbers should be a display convention, since numbering varies between coaching systems.

The instruction is context, not ground truth. If the user is asked for a jab but throws a hook, the system must not “recognize” a jab because it was expected. The recognizer needs explicit unexpected-action and unknown outcomes. A mismatch should suppress technique advice that assumes the requested action occurred.

Proposed teaching loop:

1. Show a short coach-approved demonstration and explain one observable goal.
2. Capture a short set, such as 5–10 comfortable repetitions, under a validated camera setup.
3. Confirm which repetitions were assessable and which action was observed.
4. Give at most one prioritized correction and, where supported, one specific positive observation.
5. Show one replay with the relevant time window highlighted.
6. Repeat the drill with the same criterion, then compare assessable repetitions using the same camera/rubric version.

Feedback should be scheduled between repetitions or sets rather than narrating every landmark fluctuation. Choose a cue only after repeated clear evidence, except for setup failures where immediate correction is useful. Frequency, minimum evidence count, and cooldowns are tunable UX choices to validate with the user and a coach.

### Six punches, combinations, then free shadowboxing

Add lead/rear hooks and uppercuts after each passes event-recognition tests. Then test sequence recognition and recovery rules across combinations. A combo is not simply isolated-punch rules concatenated: hand recoveries and the next attack can overlap, so their valid evaluation windows differ.

Free mode adds stance switches, body shots, feints, long guard, defense, pivots, unusual tempo, and improvised combinations. It should initially provide a timeline, counts with uncertainty, and selected high-confidence observations. Do not run a beginner high-guard rule against every style. Ask for drill/guard intent when it matters, or abstain.

Avoid a single “boxing skill score.” Prefer “8 of 10 visible jabs returned to the selected guard region” plus evidence and coverage. Progress should separate adherence to the selected drill from broader boxing competence.

## Critique rules and abstention

Implement a rule registry; each entry should specify:

- Version, name, supported techniques, stance/guard assumptions, and coaching reviewer.
- Required landmarks or image regions and supported camera views.
- Event phase and exclusions: for example, a hand should not be required at guard while it is legitimately punching.
- Features, thresholds, uncertainty treatment, minimum duration, and minimum repeated evidence.
- Calibrated validity threshold; separate values for observing a deviation and deciding it merits feedback.
- A short allowed cue and a linked replay region.
- Known exceptions, unsupported contexts, and abstention reasons.

Keep four confidence concepts distinct: image/track quality, event identity, measurement reliability, and correctness of the proposed critique. A joint's visibility score is not a probability that a critique is right. Multiplying these scores does not create a calibrated probability unless that composition is validated.

Before releasing a correction, require all relevant gates:

1. Supported task and view.
2. Required regions observed across the critical phase.
3. Accepted action event and phase boundaries.
4. Observable rule supported by consistent measurements.
5. No contextual exemption or detected contradictory evidence.
6. Calibrated policy predicts sufficiently low error at useful coverage.

Quality failures should yield actionable setup feedback: “Your rear hand is hidden in this view; turn slightly and repeat.” Unknown technique should yield an unknown event. Neither should be converted into a low form score.

A video-language model can draft a concise explanation from accepted structured observations and selected clips. It should not silently add new faults. Any proposed additional visual finding remains an unverified candidate until it passes its own validation path. Feedback text should refer to evidence IDs; a validator should reject unsupported quantities, causal claims, and contradictions.

## Concrete feedback example and schema

The following is an illustrative object, not a measured session or trained model result. The numerical values demonstrate a data contract. The threshold is hypothetical and must be fitted with coach labels.

```json
{
  "schema_version": "0.1",
  "session_id": "example-only",
  "mode": "guided",
  "drill": "isolated_lead_straight_high_guard",
  "stance": "orthodox",
  "view": "validated_front_oblique",
  "event": {
    "id": "rep-04",
    "observed_action": "lead_straight",
    "requested_action": "lead_straight",
    "start_ms": 12100,
    "extension_ms": 12400,
    "end_ms": 12900,
    "action_score": 0.97,
    "score_calibration_id": "illustrative-calibrator"
  },
  "observation": {
    "id": "obs-04",
    "criterion": "non_punching_hand_guard_displacement",
    "rubric_version": "high-guard-v0.1",
    "body_side": "right",
    "time_window_ms": [12200, 12500],
    "required_regions_visible": true,
    "observed_drop_torso_units": 0.19,
    "hypothetical_threshold_torso_units": 0.15,
    "measurement_space": "normalized_image_projection",
    "repeated_in_assessable_reps": 3,
    "assessable_reps": 5,
    "total_reps": 7,
    "exemptions_detected": [],
    "abstain_reason": null
  },
  "decision": {
    "status": "feedback_candidate",
    "policy_version": "illustrative-unvalidated",
    "critique_probability": null,
    "release_eligible": false
  },
  "proposed_feedback": {
    "evidence_ids": ["obs-04"],
    "observation": "Your rear hand dropped during 3 of the 5 jabs I could assess.",
    "cue": "On the next set, keep the rear hand in your selected guard while you jab.",
    "replay_event_id": "rep-04"
  }
}
```

For a validated pipeline, the cue can be released after the policy gate. Until then, show it in developer/coach review. An appropriate abstention for a different repetition is: “I could not assess your rear-hand position because it was hidden.” Do not say “Your shoulder rotation cost you 20% power”; this pipeline observes neither that quantity nor its cause.

## Validation before claiming accuracy

### Reference labels

Recruit at least two qualified boxing coaches for the rubric and an adjudicator for disagreements before expansion beyond a personal prototype. Have them mark visible evidence independently, unaware of model outputs. Preserve disagreements and `not_observable` labels. Coaches should label action, phase boundaries, visible criterion, severity/actionability, and whether the proposed cue is supported and useful.

Do not automatically turn every expert-style variation into an error. Include acceptable alternatives and exaggerated, coach-demonstrated deviations. Do not ask novices to perform potentially harmful mistakes to create training data. Obtain consent and appropriate recording rights; public demonstration videos are not automatically licensed training material.

For this personal-first project, begin with multiple sessions of the same user across different days and realistic lighting. Report this as personal performance only. Later datasets need multiple participants, experience levels, body proportions, clothing, camera arrangements, skin tones, stance sides, and motion speeds. Separate training, threshold/calibration, and test sets by person; where evaluating personalization, use a future-session holdout and state the narrower scope. Split source recordings before extracting overlapping clips.

### Proposed initial acceptance gates

The following align with the [canonical master release targets](../project-plan.md#release-gates). They deliberately prioritize few false corrections at useful coverage. No metric has yet been measured for this repository. Protocol changes require a documented rationale and a new version; earlier drafts of this supporting note are not alternate targets.

| Gate | Proposed requirement | Evaluation detail |
| --- | --- | --- |
| Visibility | At least 95% abstention on curated unassessable examples | Hand/face occlusion, clipping, blur, partial body, second person, view change, mirror transform, and dropped-frame cases; also report mistaken abstention on valid examples |
| Guided straight-punch recognition | Precision at least 95% and recall at least 90% on supported views | One-to-one event matching at temporal intersection-over-union at least 0.5; wrong-side labels fail; include idle/unknown motion and report each supported punch/stance condition |
| Critique validity | Precision point estimate at least 95%, with one-sided 95% lower confidence bound at least 90% for each released criterion | Score all emitted corrective claims against independent coach evidence labels; an asserted claim without observable support is not a valid correction. Account for participant/session clustering and label uncertainty |
| Useful coverage | Assess at least 70% of otherwise eligible events for each enabled criterion | Declare eligibility before model inference using the supported protocol and independent labels; also report all-event coverage so model rejections cannot shrink the denominator |
| Nuisance corrections | At most 1 false corrective cue per 5 minutes of realistic use | Evaluate final emitted cues after cooldown and repetition policy; also audit raw finding precision so cue suppression cannot conceal a broken detector |
| Event response | p95 event-finalization delay at most 350 ms after the annotated event end | Use live causal replay and explicit boundary definitions. Separately report deliberation, queue, speech-start, and acquisition delays; no claim that inference time equals complete cue latency |
| Session reliability | A 20-minute session on the target M3 MacBook without increasing inference backlog; p95 processed-frame age at most 150 ms | Document the timing start point and actual camera cadence; measure sustained effective frame rate, inference throughput, lost-frame rate, memory, and thermal behavior |
| Review explanations | Every factual movement claim traceable to validated evidence | Automatic reference checks plus blinded coach review; audit unsupported-claim rate and usefulness separately |
| Personal progress signal | Evidence of improvement retained in a later matched-view session | Independent coach-rated outcomes; promising personal evidence, not causal or population-wide efficacy proof; see the study design below |

For combination recognition, report exact sequence match, edit distance, and event timing; include wrong order, missing/extra punches, deliberate pauses, and unrequested actions. Freeze a numerical combination gate when that feature's protocol is specified, rather than adopting an unrelated action-classification threshold.

Passing average gates does not establish every subgroup. Restrict a feature to tested conditions if it fails a stance, view, speed, or device condition. Choose enough independent examples to make confidence intervals informative; a few attractive demo clips cannot establish a 95% claim. Reuse a frozen test set for comparability but maintain a fresh holdout to avoid repeated benchmark tuning.

### Does the coaching improve skill?

Recognition and critique gates establish technical feasibility. They do not prove learning. A later study should compare guided practice with and without personalized feedback, use blinded coach ratings on unseen recordings, and test retention after the feedback is removed. Keep practice duration comparable. For a single user, run repeated baseline–practice–retest blocks and describe the result as personal evidence rather than a population-wide efficacy claim.

## Decisions to preserve during implementation

- Keep raw evidence and decisions separable so changing a rubric does not require re-running capture.
- Version model, feature extraction, camera transform, calibration, rubric, and feedback template in every session.
- Keep local recordings opt-in and provide deletion/export. Optional cloud review should visibly identify the selected clip and purpose.
- Show what was assessed and why the system abstained. Avoid encouraging the user to optimize a generic skeleton score.
- Require direct observation for a technique claim and a coach-reviewed rule for a correction.
- Improve camera observability and labels before adding model size when failure analysis points to missing evidence.
