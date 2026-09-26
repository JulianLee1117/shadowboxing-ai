# Next iteration decisions

Reviewed September 26, 2026. This is the implementation sequence for improving the existing local coach; the broader product goals remain in [project-plan.md](project-plan.md).

## What to optimize now

Keep MediaPipe Full in the live browser worker. Paired local experiments have now exercised Full, Heavy, Holistic, RTMPose-M and RTMW-L. The alternatives did not justify a default switch. Holistic added processing without sufficient useful hand rescue; the tested RTM CPU pipeline exposed model-specific confidence differences and false additional person boxes. These are pipeline findings on development recordings, not universal model rankings. Detailed runs, model hashes, video comparisons and provisional labels remain private.

The present failure mechanisms need separate fixes:

| Failure | Evidence needed | Next intervention |
| --- | --- | --- |
| A recognizable repeat never starts a candidate | Observed flexion/reach history and reference state | Reacquire from an observed local return, including when the hand never becomes motionless |
| One confident wrist estimate teleports away and back | Source video plus neighboring observed joint positions | Bound a robust path estimate; never let the rejected point manufacture a peak |
| Several frames place the hand on the wrong arm or hide it | Anatomical wrist/elbow labels and occlusion labels from video | Compare region-based tracking or a second pass on those frames; maintain unknown when evidence is absent |
| Motion is visible but style differs from rigid rules | Continuous event labels, background actions and repetition boundaries | Train a small temporal recognizer after session-separated data exist |
| Recognized action needs a correction | Coach-labeled criteria, visibility and counterexamples | Validate each cue independently before enabling corrective advice |

Neither faster inference nor a more confident model score resolves a wrongly located hand. Re-decoding a clip can also change tracking: all pose challengers must see the same images and timestamps as their baseline.

## The repeatable development loop

1. Freeze independent video labels before showing predictions. Keep unclear intervals explicit. Mark imperfect but recognizable actions as actions; recognition must not require ideal form.
2. Run `npm run benchmark -- <labeled sessions> --output-dir <new private run>`. Review lost matches and false detections as well as recovered actions. Keep the old reports intact.
3. Trace each remaining miss to tracking, onset/reference acquisition, peak evidence, path geometry or recovery. Inspect video for the specific proposed mechanism.
4. Change one mechanism, add a meaningful adversarial regression, then run the complete labeled set and software checks. Avoid repeatedly changing numeric gates until one recording scores well.
5. Use later-day recordings as held-out evaluation only after freezing the version. Once a clip informs changes, it becomes development data.

The browser stays simple: practice, review, optional detection overlay and recheck. Research controls and model experiments belong in local tools. No cloud processing or additional user setup is needed for this loop.

## Training data and the next model

Do not train a pose network from scratch for this pilot. First annotate anatomical wrists/elbows on the existing failure intervals, with visible/occluded/ambiguous states. This gives a direct test of tracking quality that a punch counter cannot supply. Preserve continuous non-punch movement, incomplete returns, open hands and ordinary stance changes instead of creating only neatly isolated positive clips.

The next learned component should be a compact causal temporal recognizer over torso-normalized observations, actual time deltas, missingness/confidence and arm-motion features, with per-arm action/background/unknown outputs. Compare it with the frozen rules using the same event matching; do not use clip accuracy as a substitute for continuous-round recall and false events. Export/runtime work should follow a demonstrated offline gain.

Plan an initial personal pilot across several days and views, including front/oblique framing, both stance choices and comfortable speed variation. The number of sessions should grow until learning curves and held-out results stabilize; no fixed punch count guarantees adequacy. Existing same-person clips support engineering diagnosis, not generalization to other athletes. For a broader release, hold out entire people, rooms and sessions; a random split of adjacent windows leaks nearly identical evidence between train and test.

For form critique, obtain a qualified boxing coach's labels for a small, explicit set of observable criteria, plus examples where the camera cannot decide. Pose-derived straightness or return distance is not automatically a correct technique grade. Action labels and critique labels must remain separate.

## External work worth borrowing from

- [FightFlow](https://github.com/krishmula/fightflow), a 2026 course project, reports stronger results for pose sequences than its image baseline and a substantial score reduction when switching to video-level splits. It is useful as an inspectable temporal-feature and evaluation reference. Its four action classes and reported clip results do not validate this webcam coach; the inspected repository did not establish a reusable data/weight license, so nothing was incorporated.
- [BoxingVI](https://arxiv.org/abs/2511.16524) describes 6,915 labeled punch clips from sparring video. This is a potential action-pretraining source, subject to access and footage terms. Sparring differs from solo webcam practice, and the paper's action labels do not supply form-critique ground truth.
- [rtmlib](https://github.com/Tau-J/rtmlib) is now represented by an actual local extraction adapter with explicit weight provenance and preprocessing. Future changes must address person selection, confidence calibration and runtime together; increasing landmarks alone did not solve the tested pipeline.

No external footage, checkpoint or project score is being presented as validation of Corner. The fastest useful next step is the measured failure-driven loop above, followed by a learned temporal model when its data can support a fair comparison.
