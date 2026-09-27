# Optional personal punch recognition

The local pose service can run an explicitly enabled personal recognizer alongside RTMPose. It combines a fixed external family model with a small personal temporal model. It outputs physical hand, motion family, observed event times, and model support. The browser maps straight punches to jab/cross and curved punches to lead/rear roles using stance.

This is experimental action recognition. It does not verify anatomical identity, classify good technique, measure impact, or establish that an observed punch was performed correctly. The personal curved-punch model has development evidence only; a fresh mixed recording remains necessary before making an independent accuracy claim.

## What runs

The fixed model is a two-layer, 128-unit LSTM with 16 inputs and four outputs, using the checkpoint from [ACM40960/Boxing](https://github.com/ACM40960/Boxing/tree/c9d94ffa0ae24b8065c6c40f5f9f8dc453619676). Its jab/cross outputs are combined as straight-family support. The project is published under the [MIT license](https://github.com/ACM40960/Boxing/blob/c9d94ffa0ae24b8065c6c40f5f9f8dc453619676/LICENSE), copyright 2025 Jayaganeshan Thanga Kumar and Kabilesh Sekar. The exact checkpoint and license hashes are recorded in the local bundle. Upstream camera, cooldown, and technique-feedback code is not executed, and upstream accuracy claims are not treated as validation here.

Straight proposals use observed arm extension and bounded prominence/recovery evidence. Current own-arm confidence remains necessary. Normalization may reuse a recently observed scalar torso length for at most 250 ms when the far-side torso becomes hidden. This does not reconstruct a wrist, elbow, shoulder, or identity. Large shoulder displacement clears that normalization support.

The personal model is a shared per-arm causal temporal convolutional network: 40 inputs, 48 channels, four residual blocks with kernel 3 and dilations 1, 2, 4, 8, then four outputs: background, straight, hook, uppercut. Its receptive field is 31 grid samples. Normalization operates across channels at each time, so a later frame cannot change an earlier prediction. Personal straight predictions do not replace the fixed straight-proposal stage. A personal hook/uppercut proposal additionally needs majority curved-family support from the fixed model.

## Features and training protocol

Features contain seven observed joint offsets relative to the active shoulder, scaled by projected torso length; five native score features; their time-normalized finite differences; source-sample age; and a distinct-observation indicator. Native scores are preserved in source evidence. Their feature values are bounded for the checkpoint's numerical contract; they are not converted into calibrated visibility probabilities.

The temporal model uses a causal 30 Hz grid. Each grid tick may hold the latest already observed pose and records its age. No future frame or interpolated joint is used. Held ticks do not count as additional observations: accepting a curved event requires multiple distinct source samples spanning observed time. A missing finite trajectory cannot be replaced by a fabricated peak.

The frozen development training run used reviewed continuous action intervals, real idle/transition background, and explicit unknown masks. Ambiguous or unobservable intervals were conservatively masked for both arms rather than taught as background. No drill label supplies an action label. Physical hand remains separate from stance-dependent role.

Training used whole recordings, not random neighboring-window splits. One personal mixed recording supplied the current fit; other complete recordings were excluded from gradient fitting. Those recordings were already known development material and helped assess the hybrid, so their transfer checks are not an independent generalization result. No held-out curved-punch claim is made.

The reproducible development configuration uses seed 41729, 600 AdamW steps, learning rate 0.002, weight decay 0.01, gradient norm clipping at 1, square-root inverse-frequency class weighting, and small feature noise. Cadence augmentation uses observed-source selections at 15, 20, 25, and 30 fps followed by causal sample hold on the 30 Hz grid. This tests reduced cadence without inventing extra captured frames. Real motion blur and future lighting/view changes remain separate uncertainties.

The exact development script, protocol, source hashes, and tensor checkpoint are retained locally. A general public training CLI is not shipped yet. The [action dataset tooling](action-dataset.md) audits provenance, label scope, rights declarations and split leakage; it does not automatically reproduce or enable this personal model. Future training must preserve those checks and reserve complete fresh recordings before fitting or tuning.

## Serving contract

The optional [local service](local-pose.md) loads a local bundle only when requested. The repository contains runtime code and tests; personal weights and recordings are not distributed through Git. A locally provisioned manifest must pin:

- The personal and external checkpoint hashes and fixed architecture.
- Feature and runtime source hashes, protocol version, and bundle fingerprint.
- Pose/detector checkpoint hashes, input color contract, and native score policy.
- Training provenance and external license/source attribution.

Weights load as tensor-only state dictionaries into fixed local architectures. Missing dependencies, changed files, incompatible pose metadata, or missing configured recognition output cause explicit failure. There is no silent switch to another detector.

Each service session owns independent bounded history. New sessions, timestamp reversals, gaps over 150 ms, and image-size changes reset state. Every frame returns matching recognizer provenance and zero or more decisions. Event boundaries and projected peaks use source-observation timestamps; `detectedAtMs` is the current consumed frame time. Event support is uncalibrated and must not be displayed as a correctness percentage.

Recognition waits for temporal evidence. Arbitration holds an observed completed event for 800 ms; a curved candidate may itself span up to 1.8 seconds. This is not immediate voice feedback. Downstream combination recognition must account for late decisions and distinguish event time from delivery time. Missing end-of-round evidence is not fabricated to finish a candidate.

Run the dependency-light contract and lifecycle checks with:

```sh
python3 -m unittest ml.tests.test_recognizer -v
```

The frozen runtime was also checked through exact per-frame replay, prefix causality, reduced cadence, resets, held-sample negatives, source/hash validation, and event matching. Short local execution measurements do not establish the full 20-minute sustained-performance target. [Recognition events](recognition-events.md), [benchmarking](benchmarking.md), and the [third-party notices](../public/third-party-notices.txt) describe related contracts and limitations.
