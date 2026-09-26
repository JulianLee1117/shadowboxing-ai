# Data, temporal recognition, and evaluation

Research checked: 2026-09-26. Initial test environment: Julian's M3 Max MacBook Pro and built-in webcam. Julian is a beginner learning basic techniques; the intended product should serve broader users. Start with local live analysis and optional cloud review of selected clips. Published findings are linked; data volumes, model choices, and acceptance targets below are proposed engineering decisions, not demonstrated performance. The [master plan's release gates](../project-plan.md#release-gates) are canonical.

## The central distinction

A system can recognize a jab while being wrong about whether that jab was well executed. Train and evaluate these as separate problems:

1. **Visibility:** Is the required evidence observable from this camera in this interval?
2. **Event recognition:** What happened, with which physical hand and lead/rear role, and when?
3. **Execution assessment:** Which specific, observable property met or missed a coach-defined criterion?
4. **Coaching:** Is there enough evidence to issue this cue now, and does acting on it help?

There is no verified public dataset in this review that supplies all four for laptop-webcam shadowboxing. Domain-specific collection and coach review are the highest-value investments. A large video model does not remove those requirements.

## Boxing data that actually exists

| Resource | Verified contents and access | Appropriate role and limitations |
|---|---|---|
| **BoxingVI**, Nov 2025 | 6,915 clips, six punch classes, temporal boundaries, AlphaPose-derived 2D poses. Source footage includes shadowboxing, bag work and sparring. Paper reports 18 athletes / 20 videos; split is 5,513 training and 1,402 validation clips. | Best candidate for initial six-punch recognition experiments. No labeled technical faults, trustworthy 3D ground truth, or coaching evaluation. Small identity pool. [Paper](https://arxiv.org/html/2511.16524v1) |
| **BoxMAC**, Dec 2024 | 15 professional boxers, over 60,000 frames, 13 actions; 2,314 clips for multilabel video classification. Includes six punches, stance, slip, block, guard, duck, clinch and no-action. Four PTZ cameras around a ring. | Useful taxonomy and future defensive-action research. Frame labels can describe both boxers together; this differs from solo event recognition. Paper splits by video, which does not establish separation by person. A working official download and reuse license were not verified in this review. [Paper](https://arxiv.org/html/2412.18204v1) |
| **Boxing Punch Detection with Single Static Camera**, 2024 | Authors publish a Kaggle dataset link; stated use is noncommercial academic teaching/research and nonprofit research. | Useful external punch/no-punch benchmark. Match footage and random sample splits do not establish cross-athlete home-webcam accuracy. Use the authors' release, not an unverified mirror with different terms. [Paper and data statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11353713/) |
| **BoxMind**, Jan 2026 preprint | BoxingWeb/BoxingStudio contain 80 manually annotated rounds and 10.9K atomic events. | Strong architectural evidence for event-first pose-plus-RGB analysis. It reports punch detection F1 0.783 at temporal IoU 0.5; technique F1 0.725 and mean attribute F1 0.700. It explicitly operates offline. Dataset/code release and reuse permission were not verified. These are match-analysis results, not form-critique accuracy. [Paper](https://arxiv.org/html/2601.11492v1) |

### BoxingVI access audit

The [official repository](https://github.com/Bikudebug/BoxingVI) resolves and links a [Google Drive dataset folder](https://drive.google.com/drive/folders/1Vyl8twJQ1qkqEPwhvfsrJsJ8nLQ92uoy). Its README describes `Annotation_files`, `RGB_videos`, `Skeleton_data`, with spreadsheet rows `(start_frame, end_frame, punch_class)`. The visible repository root contains `README.md` and `Figure`; no license file or executable training baseline was visible. The Drive link resolved, but actual downloadable files and their terms were not inspected. Do not equate a working link with a completed dataset audit.

The paper's availability statement says the distributed dataset contains YouTube links, boundaries and categories rather than copyrighted video. This differs from the README's RGB folder description. The paper also labels split groups S1–S20 as subjects while reporting 18 athletes. Before relying on its split, map every source video to actual identities and check whether any athlete appears on both sides. Its poses are model outputs, not measured ground truth. The paper has no reported recognition experiment validating its proposed coaching applications. [Paper](https://arxiv.org/html/2511.16524v1)

**Decision:** useful optional research bootstrap, not a critical dependency. Begin collecting consented personal footage immediately; preserve the ability to train without uncertain third-party media. Before importing a release, record source/version, exact available files, license, redistribution rules, identities, class balance, and missing source-video rate. Author contact is a later user-authorized action; no one was contacted during this research.

## Research that informs the design

**V-JEPA 2.1 is a current video-encoder candidate.** Meta's official repository dates its release to March 16, 2026 and lists 80M, 300M, 1B and 2B checkpoints. Most repository code is MIT, with specified Apache-2.0 portions. It provides frozen-feature probing workflows and warns that its `decord` dependency needs an alternative on macOS. Start with the 80M or 300M encoder for offline representation experiments; measure local inference and conversion separately. A foundation encoder needs boxing labels and a task head. [Official repository](https://github.com/facebookresearch/vjepa2), [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482)

**VideoPrism is a useful second encoder, not an automatic winner.** Google released public Base/Large checkpoints in June 2025 and a frozen-backbone classification example in March 2026. The [official repository](https://github.com/google-deepmind/videoprism) uses JAX/Flax; the [Google Base model card](https://huggingface.co/google/videoprism-base-f16r288) declares Apache-2.0. [Transformers' current implementation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/videoprism.md) documents PyTorch classes and notes its June 2026 contribution. Benchmark exported or PyTorch behavior before choosing it for this Mac. Large published benchmark scores do not establish reliable punch boundaries or technical fault detection.

**Temporal locality matters.** [OnlineTAS](https://arxiv.org/abs/2411.01122) explicitly distinguishes streaming from whole-video segmentation and addresses oversegmentation. Its [repository](https://github.com/QingZhong1996/OnlineTAS) still showed only README/license and “Coming soon” during this review. Use its ideas, not a presumed ready-to-run implementation. Classical [MS-TCN](https://pages.iai.uni-bonn.de/gall_juergen/download/jgall_TCN_cvpr19.pdf) uses acausal convolutions; copying its offline benchmark into a live system can silently introduce future-frame access.

**Pose representations deserve an ablation.** [PoseConv3D](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html) represents estimated skeletons as heatmap volumes and reports robustness to pose noise on its benchmarks. That is an option for offline comparison against coordinate-based temporal models. It is not evidence that every new pose model or monocular 3D reconstruction improves boxing recognition.

**Execution labels are their own data product.** [Fitness-AQA](https://github.com/ParitoshParmar/Fitness-AQA) has actual form-error assessment for squat, overhead press and row, but requires access request and is explicitly noncommercial. [EgoExo-Fitness](https://github.com/iSEE-Laboratory/EgoExo-Fitness) combines action/substep boundaries, technical-keypoint verification, comments and quality scores; dataset access has a separate agreement even though repository code is Apache-2.0. These are useful annotation designs, not boxing training sets.

**Recent quality benchmarks support evidence-first assessment.** [FineGym-AQA / FineGrade, CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_FineGrade_A_Rule-Consistent_Scoring_Framework_for_Fine-Grained_Action_Quality_Assessment_CVPRF_2026_paper.html) separates temporal parsing and rule-consistent scoring. Gymnastics has official scoring rules; shadowboxing requires an explicit coach-agreed rubric rather than borrowing gymnastics scores.

**A general multimodal model should not be the live form judge.** [FitAQA, August 2026 preprint](https://arxiv.org/html/2608.08736v1), evaluates 2,219 exercise videos with 38 form errors across 30 exercises. Its best reported quality-perception Q-MAcc is 54.3%; best temporal grounding Recall@0.7 is 27.0%. Providing ground-truth perception substantially improves judgment. These are fitness, not boxing results, but they support separating visual evidence from verbal reasoning. Treat selected-clip VLM review as a separately tested assistant and use structured measured evidence for live cues. Dataset release is promised in the abstract; do not assume available training data.

## A concrete data collection sequence

### Personal feasibility: prove repeatability across days

Begin with **200–400 diverse events across three or more days** to establish the capture and annotation pipeline. Expand toward roughly **8–12 sessions of 8–12 minutes** for personal validation; these are starting estimates, not a promise of adequate model accuracy. Use the real built-in camera and actual room. Keep complete continuous rounds, not only isolated punches. Begin with the user's declared stance, a defined camera orientation, jabs, crosses, jab-cross, idle, and background movement. Add hooks and uppercuts after the initial distinctions and boundaries work. A beginner's habitual execution must not become the target standard through calibration.

Each session should include:

- Isolated repetitions at comfortable slow and normal pace, with full recovery and natural variation.
- Repeated same-hand punches, short combinations, pauses, interrupted/partial punches, and deliberate changes in rhythm.
- Hard negatives: adjusting clothing, touching the face, wiping sweat, reaching toward the laptop, stepping, and casual gestures. Label feints explicitly; decide whether they are a separate supported event or an abstention class.
- Free shadowboxing without a script. Drill prompts are useful metadata, but the prompt must not become the observed-action label.
- Brief setup failures: cropped feet/hands, movement near the edge, backlighting, camera-distance change and partial self-occlusion. These train visibility gates, not technique judgments.

Use warm-up/comfortable repetitions; there is no need to deliberately stress joints or exaggerate bad mechanics. A qualified coach can annotate naturally occurring faults, demonstrate approved examples, and decide which distinctions are meaningful from the selected view. Personal self-labeling is adequate for obvious action identity in many clips; it is not a substitute for independent ground truth for technique quality.

Divide whole sessions before extracting clips: approximately 60% development/training, 20% validation, 20% locked test, with at least two later days in the test portion. Sessions from the same recording, synchronized alternate views, augmented copies and overlapping windows always stay together. Collect a new untouched session after tuning. This estimates performance on **Julian on later days in supported conditions**, not other people.

Record a small optional reference set with a synchronized second camera or coach observation to diagnose what the laptop view misses. The product still runs with one camera; extra evidence is for ground-truth checking, not secretly supplied at test time.

### Personal coaching: approve one cue at a time

Begin with two or three narrowly worded candidates such as visible return of the punching hand to the calibrated guard region, non-punching hand leaving its guard region, or observable combo order/timing. A coach must specify contexts in which each cue is valid. Do not turn a visible hand displacement into an unsupported claim about injury risk, power, impact, balance or tactical error.

For each candidate, collect independent visible-positive, visible-negative, ambiguous and unobservable examples. A useful initial annotation target is **50–100 held-out clear examples per polarity per cue across multiple sessions**, followed by natural uninterrupted rounds; rare errors may require more collection. These numbers give a feasibility estimate, not production-grade certainty. Include valid technique/style variation and a mix of examples near the decision boundary. An intentionally staged fault alone does not establish recognition of real mistakes.

### First expansion gate: cohort validation with new people

The personal stage is a convenient development environment, not the intended population. Design the recruitment, participant metadata, consent and rubric during personal validation. Once capture is stable and a coach has approved the first criteria, run a small external usability/labeling pilot; do not wait for a highly personalized six-punch system before checking other people. The first expansion milestone is an athlete-held-out cohort study, before broad product claims or a public beta.

Plan roughly **20–30 consenting adults** for that initial study, then budget a larger cohort based on failures and confidence intervals. Include novices and experienced participants, different stances/body proportions/clothing, several rooms, light levels and cameras. Evaluate beginner guidance and calibration usability explicitly: a person must be able to reach an assessable setup without a developer arranging the camera. This initial adult cohort does not establish suitability for children or every mobility/technique variation. Obtain separate permission for capture, training, retention and any sharing. Keep sensitive/raw video out of Git.

Hold out entire people for the population test, with additional held-out room/device tests. Calibration footage for a test person may only perform the declared runtime calibration; do not tune a model on their test labels. Report personalized performance separately from cold-start performance. Small cohorts cannot support broad subgroup equality claims; show uncertainty and do not hide unsupported slices in an aggregate score.

## Annotation contract

Store physical hand (`left/right`) separately from boxing role (`lead/rear`), plus the stance interval and mirror transform. A reflected preview must not swap anatomical labels. Define event onset, extension/turnaround and recovery/end precisely in the annotation guide, including what happens when the next punch begins before recovery. Support overlapping left/right event intervals rather than forcing one mutually exclusive frame class.

Minimum per-event fields:

```json
{
  "session_id": "session_001",
  "subject_id": "personal_001",
  "event_id": "event_0042",
  "time_s": {"start": 12.10, "turnaround": 12.37, "end": 12.70},
  "physical_hand": "left",
  "role": "lead",
  "technique": "straight",
  "stance": "orthodox",
  "view": "calibrated_oblique",
  "visibility": {"punching_wrist": "visible", "other_wrist": "occluded"},
  "execution_labels": [
    {"criterion": "non_punching_hand_guard", "judgment": "unobservable"}
  ],
  "annotator_ids": ["coach_a", "coach_b"],
  "rubric_version": "0.1",
  "adjudication": "not_required"
}
```

For each criterion use `pass`, `fail`, `ambiguous`, or `unobservable`, with evidence interval, severity if reliable, and a brief rationale. “Not annotated” is a distinct missing-data state. Do not convert unknowns into negative examples. Record pose model/version, per-joint confidence, dropped-frame timestamps, image resolution, camera orientation, session/drill and consent provenance separately.

Have two coaches independently label a shared subset, blind to model output. Measure disagreement before adjudication; preserve it. Prefer label-specific agreement and confusion tables plus an agreement statistic appropriate to label prevalence. If humans cannot reliably assess a cue from the supported view, narrow the cue, request a better view, or omit it. A single scalar “boxing score” obscures this problem.

## Model comparison that is worth running

Use identical training data and locked splits for these baselines:

1. **Transparent kinematics + state machine:** normalized positions, velocities and distances, visibility masks, short temporal smoothing. Establish the simplest achievable event and cue behavior.
2. **Small causal temporal model:** temporal convolution network (TCN) over normalized pose plus confidence, optional velocities/bone vectors; separate left/right event heads, boundaries and technique heads. Preserve actual timestamps. Use past-only normalization/filtering for the live path.
3. **Pose graph/heatmap model:** determine whether richer skeleton representation improves errors enough to justify complexity; do not assume it will.
4. **Frozen video encoder + small head:** V-JEPA 2.1 Base first, then VideoPrism Base or a larger encoder on selected offline clips. Compare RGB-only, pose-only and fused predictions. Start frozen before fine-tuning an enormous model on a tiny corpus.
5. **Selected-clip multimodal review:** blinded assessment from raw clip alone versus clip plus structured measurements. Require an evidence interval and an abstain option; measure unsupported claims.

A clip model can inspect a completed punch and produce useful delayed feedback, but that is different from streaming anticipation. Explicitly declare future context/lookahead. Replay tests must reveal each frame only at its original timestamp and include buffering, smoothing and decoding delay. Offline denoising with future frames must never appear in live latency claims.

Treat torso normalization as a modeling choice: preserve absolute camera-normalized coordinates too, because root-centered pose alone can erase useful footwork/displacement. Augment brightness, scale, missing joints and realistic timing; apply reflections only with correct anatomical and stance relabeling. Do not force tempo or body proportions to match one expert exemplar.

## Evaluation and canonical release gates

Measure each stage as well as end-to-end performance. “Accuracy” without the task, split and denominator is not useful. The table mirrors the [master plan](../project-plan.md#release-gates); it does not introduce an alternative set of acceptance thresholds. All targets are proposed, unmeasured, and must be frozen before evaluation.

| Dimension | Canonical initial proposed gate | Measurement |
|---|---|---|
| Sustained live operation | 20-minute session without growing inference backlog; p95 processed-frame age ≤150 ms | Actual delivered camera cadence; document measurement start point on this Mac |
| Guided jab/cross events | Precision ≥95%, recall ≥90% on supported views | One-to-one matching at tIoU ≥0.5; wrong-side labels fail; include idle/unknown segments |
| Event response | p95 event-finalization delay ≤350 ms after annotated event end | Live causal replay; report boundary definition and full distribution |
| Individual critique | Precision point estimate ≥95%, with one-sided 95% lower bound ≥90% | Coach-labeled supported findings; analyze session clustering and label uncertainty |
| Assessment coverage | Assess ≥70% of otherwise eligible events for each enabled criterion | Declare eligibility before inference; report unconditional all-event coverage too |
| Nuisance corrections | ≤1 false corrective cue per 5 minutes of realistic use | Emitted cues after smoothing and cooldown, not raw rule decisions |
| Bad visibility | ≥95% abstention on curated unassessable examples | Also report mistaken abstention on valid footage |
| Review explanations | Every factual movement claim traceable to validated evidence; audit unsupported-claim rate | Automated reference checks plus blinded coach review |
| Personal progress signal | Evidence of improvement retained in a later matched-view session | Coach-rated criterion outcomes; promising personal evidence, not causal or population-wide efficacy proof |

Alongside these gates, report diagnostic metrics: per-class event and technique F1, macro-F1, hand/stance confusion, false events per minute, missed/double counts, boundary error, unknown-class rejection, and conditional classification accuracy on correctly detected events. Combo diagnostics include exact observed sequence match, insertion/deletion/substitution rates, edit distance, repetition count and inter-punch timing error. Report guided and free-form sequences separately. These metrics guide error analysis; broader classes need an explicitly frozen extension protocol rather than inheriting a jab/cross approval.

Thresholds must be finalized with the coach and user before the locked evaluation. Higher precision is worth lower coverage for critique, but 100% precision from almost never speaking is not useful: always show coverage and recall. Report participant/session counts and confidence intervals, using participant-level or session/block-level bootstrap rather than pretending adjacent frames are independent. A tiny personal test may be unable to establish the required one-sided 95% lower bound of 90%; gather additional independent rounds before claiming that gate is passed.

Distinguish processed-frame age from actual sensor-to-overlay delay, which may not be measurable from a browser. Report p50/p95 for each available timing quantity and its start/end definitions. The canonical 350 ms gate concerns event finalization after annotated event end; cue scheduling, speech onset/duration and cooldown are separate measurements. A correction intentionally deferred until after a combination is not an event-detector latency failure. Declare any lookahead and never use full-video refinement in a live causal benchmark.

Every release test needs a **false-confidence suite**: partial body, hand occlusion, mirror mismatch, unsupported camera angle, no boxer, unrelated motions, unseen technique, fast overlapping punches, repeated jabs, dropped frames, low light and an unsupported stance. Desired behavior can be “adjust camera” or “cannot assess”; confident invented critique is a failure.

Finally, demonstrate benefit with a personal repeated-session experiment: alternate matched drills with and without an enabled cue, and have a coach blindly assess later clips. Separate immediate compliance from retained improvement. A higher internal score after coaching is circular evidence if the same model gives both the cue and the score.

## Recommended first experiment

Collect three short sessions on different days, annotate jab/cross/idle with event boundaries, and evaluate a causal pose baseline on the third day. Add one coach-approved guard-return criterion with explicit visibility labels. Inspect every false cue, not only aggregate accuracy. If the webcam view cannot support that criterion consistently, fix capture/view guidance before increasing model size. This yields a concrete go/no-go decision for the product's core promise at low cost.
