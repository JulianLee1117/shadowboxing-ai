# Optional personal punch recognition

The local pose service can run an explicitly enabled personal recognizer alongside RTMPose. It combines a fixed external family model with a small personal temporal model. It outputs physical hand, motion family, observed event times, and model support. The browser maps straight punches to jab/cross and curved punches to lead/rear roles using stance.

This is experimental action recognition. It does not verify anatomical identity, classify good technique, measure impact, or establish that an observed punch was performed correctly. The current personal model has fitted development evidence only. Whole-recording leave-out experiments still fail on unfamiliar mixed-punch views, so a fresh recording remains necessary before making an independent accuracy claim.

## What runs

The fixed model is a two-layer, 128-unit LSTM with 16 inputs and four outputs, using the checkpoint from [ACM40960/Boxing](https://github.com/ACM40960/Boxing/tree/c9d94ffa0ae24b8065c6c40f5f9f8dc453619676). Its jab/cross outputs are combined as straight-family support. The project is published under the [MIT license](https://github.com/ACM40960/Boxing/blob/c9d94ffa0ae24b8065c6c40f5f9f8dc453619676/LICENSE), copyright 2025 Jayaganeshan Thanga Kumar and Kabilesh Sekar. The exact checkpoint and license hashes are recorded in the local bundle. Upstream camera, cooldown, and technique-feedback code is not executed, and upstream accuracy claims are not treated as validation here.

Straight proposals use observed arm extension and bounded prominence/recovery evidence. Current own-arm confidence remains necessary. Normalization may reuse a recently observed scalar torso length for at most 250 ms when the far-side torso becomes hidden. This does not reconstruct a wrist, elbow, shoulder, or identity. Large shoulder displacement clears that normalization support.

The personal model is a shared per-arm causal temporal convolutional network: 40 inputs, 48 channels, four residual blocks with kernel 3 and dilations 1, 2, 4, 8, then four outputs: background, straight, hook, uppercut. Its receptive field is 31 grid samples. Normalization operates across channels at each time, so a later frame cannot change an earlier prediction. The temporal model proposes all three punch families and learns background from observed guard adjustments, transitions and idle periods. Its curved proposals no longer require the external model to agree: that veto rejected clearly observed uppercuts in the newest recording. Geometric straight proposals remain available, but also require temporal straight support; overlapping straight proposals are arbitrated before emission.

Learned straight peaks must satisfy the same basic projected reach, elbow-angle and wrist-height limits as geometric straight peaks. This prevents a bent-arm forehead cover from becoming an extra cross merely because the temporal label remains straight. These are action filters, not measurements of correct punching technique.

## Features and training protocol

Features contain seven observed joint offsets relative to the active shoulder, scaled by projected torso length; five native score features; their time-normalized finite differences; source-sample age; and a distinct-observation indicator. Native scores are preserved in source evidence. Their feature values are bounded for the checkpoint's numerical contract; they are not converted into calibrated visibility probabilities.

The temporal model uses a causal 30 Hz grid. Each grid tick may hold the latest already observed pose and records its age. No future frame or interpolated joint is used. Held ticks do not count as additional observations: accepting a curved event requires multiple distinct source samples spanning observed time. A missing finite trajectory cannot be replaced by a fabricated peak.

The frozen development training run used reviewed continuous action intervals, real idle/transition background, and explicit unknown masks. Ambiguous or unobservable intervals were conservatively masked for both arms rather than taught as background. No drill label supplies an action label. Physical hand remains separate from stance-dependent role.

The current fit uses seven personal recordings, including both saved-live and full-frame pose versions of the newest recording. These versions are one recording, not independent samples. Whole-recording leave-out experiments exclude every version of the omitted recording; no random neighboring-window split is used. Those checks still show substantial misses and false events on excluded mixed-punch recordings. Final fitted results are development regression evidence, not prospective accuracy or evidence for another person. The newest recording’s original model failure and pre-fit reference labels are preserved separately.

The reproducible development configuration uses seed 41729, 1,800 AdamW steps, learning rate 0.002, weight decay 0.01, gradient norm clipping at 1, square-root inverse-frequency class weighting, and feature noise with standard deviation 0.015. Horizontal reflection augments relative image-plane x offsets and their velocities without changing family or physical-hand labels. Cadence augmentation uses observed-source selections at 15, 20, 25, and 30 fps followed by causal sample hold on the 30 Hz grid. This tests reduced cadence without inventing extra captured frames. Real motion blur and future lighting/view changes remain separate uncertainties.

The exact development script, protocol, source hashes, and tensor checkpoint are retained locally. A general public training CLI is not shipped yet. The [action dataset tooling](action-dataset.md) audits provenance, label scope, rights declarations and split leakage; it does not automatically reproduce or enable this personal model. Future training must preserve those checks and reserve complete fresh recordings before fitting or tuning.

## Serving contract

The optional [local service](local-pose.md) loads a local bundle only when requested. The repository contains runtime code and tests; personal weights and recordings are not distributed through Git. A locally provisioned manifest must pin:

- The personal and external checkpoint hashes and fixed architecture.
- Feature and runtime source hashes, protocol version, and bundle fingerprint.
- Pose/detector checkpoint hashes, input color contract, and native score policy.
- Training provenance and external license/source attribution.

Weights load as tensor-only state dictionaries into fixed local architectures. Missing dependencies, changed files, incompatible pose metadata, or missing configured recognition output cause explicit failure. There is no silent switch to another detector.

Each service session owns independent bounded history. New sessions, timestamp reversals, gaps over 150 ms, and image-size changes reset state. Every frame returns matching recognizer provenance and zero or more decisions. Event boundaries and projected peaks use source-observation timestamps; `detectedAtMs` is the current consumed frame time. Event support is uncalibrated and must not be displayed as a correctness percentage.

Recognition waits for temporal evidence. Arbitration holds an observed completed event for at least 300 ms; classification and observed recovery can add delay, and an active candidate may span up to 1.8 seconds. Brief invalid-arm intervals retain classification state for at most 250 ms without generating coordinates, advancing observed event endpoints or proving recovery. Peaks must come from an actual source observation meeting the model’s elbow/wrist score gate. Fast same-family straights can split at observed return troughs; a missing interval cannot supply the trough. Downstream combinations distinguish event time from delivery time. Missing end-of-round evidence is not fabricated to finish a candidate.

Inference evaluates the causal network only on its required recent context, while preserving the longer observed history needed for segmentation. The context optimization was checked against full-history probabilities and event outputs before decoder changes. It reduces computation without adding lookahead. The decoder version, training protocol and runtime/feature hashes are pinned with the checkpoint; a changed bundle has a new fingerprint even though the wire-protocol recognizer ID stays the same.

Saved round poses do not include every native warmup observation or necessarily preserve the original causal-grid phase. Starting a new recognizer at the first exported pose can therefore change boundaries and decisions. Stored-frame experiments must identify that initialization choice; use the original per-frame emitted decisions as capture evidence and verify changes through the actual browser pipeline. Real-time imported-file playback exercises that pipeline but is distinct from a new webcam recording.

Run the dependency-light contract and lifecycle checks with:

```sh
python3 -m unittest ml.tests.test_recognizer -v
```

The frozen runtime was also checked through exact per-frame replay, prefix causality, reduced cadence, resets, held-sample negatives, source/hash validation, and event matching. Short local execution measurements do not establish the full 20-minute sustained-performance target. [Recognition events](recognition-events.md), [benchmarking](benchmarking.md), and the [third-party notices](../public/third-party-notices.txt) describe related contracts and limitations.
