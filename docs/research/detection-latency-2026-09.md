# Detection latency: current limits and next steps

Research checked September 27, 2026. Decoder `observed-personal-cycles-v5` uses a 150 ms arbitration hold with unchanged weights. This is an intentional recognition-timing revision: it can change event boundaries, not just their delivery time.

## Where the delay comes from

The personal TCN uses past context. Its 31-sample receptive field does **not** require waiting for one second of future video. Recognition still needs evidence that a movement happened and ended. In the [current decoder](../../ml/recognizer.py):

- A temporal family run closes after 100 ms of nonmatching evidence. Emission then waits until the later of that decision and the last active timestamp plus 150 ms. These waits overlap; they are not always additive.
- Geometric straight proposals require at least 66 ms after their observed end and 150 ms after their peak. They share the 150 ms post-end arbitration hold.
- Arbitration chooses between overlapping geometric and temporal straights and suppresses nearby duplicates. It is part of recognition, not just a display timer.
- A continuous same-family run may last up to 1.8 seconds. The observed-cycle preservation fix can rescue a completed first straight at that limit, but it does not make that event immediate.

Peak-to-emission delay also includes the observed follow-through. A held punch can therefore appear substantially later than a quick punch even when frame processing is fast. Unavailable observations and a stopped recording cannot supply missing recovery evidence.

Keep four measurements separate: processing time, browser callback-to-result age, source-event-to-emission delay, and result-to-display delay. None alone establishes sensor-to-display latency. The [evaluation protocol](../recognition-evaluation.md) defines delays against fixed video-reference peaks and ends as well as predicted boundaries.

## The fixed 150 ms experiment

One development comparison changed only the arbitration hold from 300 to 150 ms. It used the same weights, gates, source timestamps and frozen annotations on 17 older saved pose views. Several views belong to the same recording; they are not 17 independent tests. Both variants used the same explicit grid origin without reconstructing unexported camera warmup.

Unchanged events commonly arrived about 130–170 ms earlier. However, seven views failed the predeclared requirement that ordered event content remain identical except for emission time. Earlier emission sometimes selected a temporal straight before a later geometric proposal, changing its start/end, support score and event ID. Some events near the recording's end became deliverable before capture stopped.

This comparison did **not** show a loss of existing strict matches or an increase in scored false events. The largest observed start change was 96 ms, end change 200.1 ms, and support-score change about 0.0415. Replaced proposals kept their physical-hand label, family and observed peak. That does not prove anatomical tracking correctness, and scores from different proposal sources are not calibrated confidence.

The original exact-array gate failed and was retained as a failed result. A subsequent review explicitly considered different event boundaries acceptable if strict matches and fixed peak-occurrence behavior held, causal/reset checks passed, and added tail outputs did not reveal clear false actions. Eligible hand/family/peak inputs stayed identical on 16 views; the remaining view gained a final straight at its already-frozen reference peak. Extra curved tail detections aligned with an earlier video-only observation but remain excluded because the concurrent other-hand action is ambiguous. They were not relabeled to improve metrics.

The already-frozen 150 ms candidate was then evaluated once on a newly recorded, independently video-labeled round. Its ordered event content was identical to the 300 ms replay except for emission, with a median improvement of about 133 ms and unchanged strict and peak-occurrence results. Existing observation/reset invariants, five actual-data causal-prefix checks and invalid-clock checks passed. That evidence supported the versioned v5 change under the reviewed decision rule; it does not retroactively pass the original exact-array gate. No model fitting or timing sweep used the new round.

## Practical direction

Keep a persistent, promptly updated log of confirmed events so a short-lived popup does not hide feedback. That improves visibility without pretending the detector decided earlier.

For a future faster interaction, separate a **tentative movement preview** from a confirmed punch. A preview needs its own stable ID and explicit provisional state, must be cancellable, and must not increment totals, form combinations, or trigger technique correction. Measure first-correct-preview delay, cancellations, duplicate cues and false previews per minute. Raw family support is not calibrated confidence.

The larger recognition improvement would be a dedicated causal boundary or motion-phase predictor, rather than requiring the family classifier to turn into background. Train and evaluate it against reviewed phase/boundary targets, retain overlapping left/right actions, and test fast repeats, held extensions and guard adjustments. This is a proposed architecture change, not an implemented or validated feature. Faster inference alone cannot remove the evidence needed to distinguish those movements.

## Primary research

- [Streaming Detection of Queried Event Start, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/b689b90ddf2b47d3103decabe6d47446-Paper-Datasets_and_Benchmarks_Track.pdf) distinguishes computation latency from observation latency and evaluates false alerts in chronological order. Its task concerns language-queried egocentric starts; the useful lesson here is measurement and alert semantics, not a ready boxing model.
- [ActionSwitch, ECCV 2024](https://arxiv.org/abs/2407.12987) distinguishes framewise predictions from completed instances, supports overlapping actions and trains conservative state transitions to reduce fragmentation. It motivates separate cycle boundaries and family evidence.
- [Online Temporal Action Localization with Memory-Augmented Transformer, ECCV 2024](https://arxiv.org/abs/2408.02957) combines past memory for starts with current evidence for ends. It is a relevant boundary-modeling reference; its benchmarks do not establish six-punch accuracy or Mac webcam performance.
- [Distilling Offline Action Detection Models into Real-Time Streaming Models, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Patel_Distilling_Offline_Action_Detection_Models_into_Real-Time_Streaming_Models_WACV_2026_paper.pdf) studies causal attention and caching to eliminate repeated computation. Our bounded TCN tail already addresses a related cost. Reported gains for their models cannot be transferred to this app or interpreted as reduced observation requirements.
- [OZ-TAL, May 2026 preprint](https://arxiv.org/abs/2605.09976) explores online zero-shot localization using vision-language models. It is a transfer-learning research lead, with no established evidence here for precise physical-hand punches or low-latency M3 deployment.

The proposed application changes are engineering inferences from these sources and the local implementation. None of these papers validates this coach's technique critiques.
