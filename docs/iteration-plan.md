# Next iteration decisions

Reviewed September 26, 2026. The immediate priority is consistent recognition on a new round. The wider product goals remain in [the project plan](project-plan.md).

## Current product loop

Choose a lead hand and duration, enable the camera, then start the countdown and step back. The skeleton, punch name and running count provide live feedback. Finishing opens the original video and live results immediately. A full video pass is optional; it is useful for diagnosis, not a required step or a promise of better recognition.

The optional native RTMPose service and personal causal recognizer are running locally. They recognize all six stance-dependent punch identities. The current model uses seven development recordings; fitted results do not establish transfer. MediaPipe remains available as the default for installations without local weights. [Implementation status](implementation-status.md) describes the shipped paths and limits.

## What the new rounds test

Two fresh rounds were recorded after the previous bundle was frozen. Preserve their original video, per-frame native decisions and model fingerprint before any re-analysis. Label recognizable actions from video before consulting predictions, including imperfect punches, ordinary transitions and uncertain intervals.

Both captures sustained roughly 20 pose observations per second, with no gaps longer than 150 ms. The low-count round continued receiving arm observations after detections stopped. Processing every source-video frame also retained its low count. This directs the next investigation toward the pose representation, classifier and event decoder rather than treating another full pass or a larger live frame queue as the solution. A high native joint score still does not establish the correct joint location.

Keep the initial results as prospective evidence for the frozen model. Once a recording informs a representation, decoder or training decision, its later results are development evidence. Do not relabel missed punches as background, move reference boundaries to improve matching, or count an ambiguous interval as a success.

Separate action identity from interval coverage. The current strict interval evaluation can reject a correctly named punch when the output covers only its brief peak instead of the labeled load/hold/return. Preserve that score and describe those cases explicitly. Before a future untouched test, define a separate one-to-one peak-time/hand/family criterion for live counting alongside interval overlap for replay timing; do not choose its tolerance after inspecting the new test results. Provisional labels are action judgments, not technique assessments, and upward arm-motion lookalikes need coach review.

Focused image/pose comparisons found no gross active-hand swap in the missed curved actions. An opposite-arm feature probe affected the repeated-straight misclassification but did not recover all missed curves. One fixed training experiment omitted opposite elbow/wrist offsets and derivatives in half of training sequences, using only the original seven recordings. It recovered some new actions but increased false detections and misses in older rounds, so it was rejected. Production weights and recognition thresholds remain unchanged. Keep that negative result; the next experiment needs broader trajectory coverage and controlled evaluation, not repeated tuning of this same pair of rounds.

## Priorities and acceptance checks

| Priority | Work | Evidence required before shipping |
| --- | --- | --- |
| 1. Explain misses | Compare the source image, active-arm coordinates, temporal family support and event rejection reason for every missed action. Separate short detected fragments from absent detections. | A reproducible diagnosis from the original captured evidence and a fixed full-frame pass, with unchanged labels. |
| 2. Improve trajectory and posture coverage | Check whether the active-arm trajectory is represented correctly across orientations and guard positions. Test inactive-arm dependence instead of assuming it explains the failure. Evaluate a bounded augmentation or representation change using earlier complete recordings. | Preserve ordinary-round matches, lower misses on varied arm/guard conditions, and avoid new counts from phone handling, guard adjustments or idle movement. Freeze the experiment before comparison; keep failed candidates. |
| 3. Make training reproducible | Turn the retained personal training protocol into an explicit CLI with dataset hashes, recording/person/day groups, background/unknown masks and immutable candidate bundles. | A fresh run reproduces preprocessing and evaluation. No neighboring-window train/test split, no automatic activation of newly trained weights. |
| 4. Test actual session performance | Measure successive foreground webcam rounds and a sustained session, including callback-to-inference delay, skipped observations, resource cleanup and event latency. | Report real webcam measurements separately from isolated imported-file playback. Do not use source-video fps as processing throughput. |
| 5. Add one useful coaching cue | With a boxing coach, define one visually observable criterion and acceptable variants, then label examples and counterexamples independently. | High cue precision and explicit abstention on unobservable views; punch classification alone never becomes a form grade. |

Keep a small untouched later-day check before claiming a new personal model is more reliable. Broader support requires additional people, stances, backgrounds and camera angles held out by person/session. More clips of one recording or synthetic views of one motion are not independent evaluation samples.

## When to change model families

First establish whether a miss comes from bad coordinates, a poor representation or temporal decoding. A larger pose model cannot repair an event decoder, and a new classifier cannot recover an unobserved trajectory merely by receiving a higher confidence score.

If visible pixels contain useful evidence that the keypoints lose, compare a short-window RGB-plus-pose model offline. [PoseC3D](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html) and the [official MMAction2 implementation](https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/posec3d) are concrete representation baselines, not ready-made boxing coaches. Their benchmark scores do not transfer to continuous six-punch recognition, and an offline clip model must be checked for future-frame use, latency and Mac runtime cost before live deployment. Do not add a large video model solely because a demo looks convincing.

External data work should target continuous actions and non-punch movement with clear physical-hand labels and usable media rights. [The source audit](research/boxing-datasets-2026-09.md) identifies candidate datasets and outstanding access/rights questions. No new dataset license, private-video upload or outreach is implied by this plan.

## Next user-facing test

Use the two new recordings for this investigation before asking for more. Check saved-round removal, Undo and restore with disposable test data; keep actual user recordings intact during development. The next recognition check should be one ordinary fresh round against a frozen candidate, with comfortable singles, quick repetitions and a few seconds of non-punch movement. [The short test guide](first-user-test.md) explains the controls.

Voice can consume the existing finalized detection events later. Start with an optional brief callout or combo prompt once recognition is consistent; technique corrections require their separate validation. Voice should not make an uncertain detector sound authoritative.
