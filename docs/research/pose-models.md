# Pose and motion perception research

Research checked: **2026-09-26**. Initial target: the user's **M3 Max MacBook Pro with 36 GiB unified memory and built-in webcam**, personal use first, local live inference and optional cloud review of selected clips. Hardware was identified during planning; no camera capture, model installation, inference benchmark, or accuracy experiment has been performed.

This document distinguishes **source facts** from **engineering recommendations**. Published pose benchmarks establish candidate quality, not boxing-coaching accuracy. No reviewed model supplies a validated boxing critique engine out of the box.

## Decision

**Recommendation:** run a small, identical-input bake-off before selecting the production pose model. Start with MediaPipe Full/Heavy in a browser worker and RTMPose-m with feet / RTMW-m or -l through local ONNX Runtime. Keep a common landmark-and-evidence interface so a better model can replace the first choice. Use SAM 3D Body only as an optional review/annotation experiment after the real-time baseline works. Do not make a high-resolution foundation model or full 3D mesh a prerequisite for useful coaching.

The live system should use visible 2D evidence and temporal events as its most dependable inputs. Monocular 3D is a helpful hypothesis, especially for relative orientation; it is not measured depth or proof of biomechanically correct movement. Richer output is not automatically more accurate output.

## Practical shortlist

| Candidate | Verified capability | Proposed role on this Mac | Principal uncertainty |
| --- | --- | --- | --- |
| MediaPipe Pose Landmarker Full and Heavy | Browser API; 33 landmarks, image coordinates and inferred world coordinates; Lite/Full/Heavy bundles | First working camera loop and benchmark baseline | Wrist localization during punches, self-occlusion, guard covering the face |
| RTMPose-m, 26-point body/feet variant | ONNX deployment available through rtmlib; body and foot points | Lean local alternative for torso, arms and footwork | Whether its boxing wrist/ankle errors improve enough to justify a local inference service |
| RTMW-m and RTMW-l | Whole-body 133-point pose; m/l/x family | Accuracy challenger; -m first, -l if target latency permits | Small hands and fists remain difficult even with more landmarks |
| RTMW3D | Monocular whole-body 3D; current rtmlib includes Wholebody3d | Secondary 3D experiment after 2D comparison | Depth reliability, export correctness and target-hardware cost |
| Apple Vision 3D body pose | 17 joints, prominent person only, macOS 14+ | Optional native baseline if building a macOS wrapper | Fewer extremity details; native integration cost; no established boxing performance |
| SAM 3D Body | Promptable single-image full-body mesh, including hands and feet | Selected-clip review or annotation assistance, initially GPU-hosted if used | Temporal stability, compute cost, setup portability, no boxing-specific validation |

Capability sources: [MediaPipe overview](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker), [rtmlib](https://github.com/Tau-J/rtmlib), [RTMW paper](https://arxiv.org/abs/2407.08634), [Apple Vision documentation](https://developer.apple.com/documentation/vision/identifying-3d-human-body-poses-in-images), [SAM 3D Body repository](https://github.com/facebookresearch/sam-3d-body). Proposed roles and uncertainties are our engineering judgments.

### MediaPipe: best starting integration, not a proven final accuracy winner

**Source facts.** Google's current Tasks API is designed for on-device pose applications and supplies 33 landmarks, visibility and presence information. The Web API's `detect()` and `detectForVideo()` calls are synchronous; Google explicitly recommends workers to avoid blocking the interface. World coordinates are documented in meters, hip-centered. [Web API guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/web_js)

The older BlazePose GHUM model card, dated **2021-04-16**, explicitly excludes applications requiring metric-accurate depth and describes degradation from poor light, motion, overlapping faces and unfavorable face orientation. It identifies Apache 2.0 licensing. Its reported phone FPS should not be reused as a current browser or M3 Max benchmark. [Google model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf)

**Recommendation.** Compare Full and Heavy on exactly the same clips. Heavy should earn its extra compute through lower boxing-specific error. Keep Lite as a fallback only if needed. Do not interpret the `worldLandmarks` unit label as a calibrated speed measurement. Do not treat visibility as calibrated probability that a coaching statement is correct.

### RTMPose / RTMW: strongest practical challenger

**Source facts.** The RTMW paper was submitted **2024-07-11** and describes m/l/x sizes, multi-dataset training and 2D/3D whole-body variants. Its reported 70.2 COCO-WholeBody mAP is a generic keypoint benchmark, not punch recognition or form critique. [RTMW paper](https://arxiv.org/abs/2407.08634)

Current upstream rtmlib provides ONNX models, body/feet, whole-body and whole-body 3D APIs without the full MMCV/MMPose training dependency stack. Its `device='mps'` setting maps to ONNX Runtime's `CoreMLExecutionProvider` when available, falling back to CPU. This is a Core ML execution route, not proof that PyTorch MPS is used. [rtmlib code](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/base.py)

**Recommendation.** Benchmark a 26-point RTMPose model first when the target cues depend on wrists, elbows, shoulders, hips and feet. Benchmark RTMW-m and -l when finer hand/foot landmarks improve a defined cue. Many of the 133 points are face/hand details that may add little to the initial drill set. Compare both model latency and full processed-frame age using the documented measurement start point; a detector, crop transform or JPEG/WebSocket hop can erase an apparent model speed advantage.

Use an expanded person region that includes fully extended arms. Reusing a crop or detector result too aggressively can cut off fast wrists. Re-detect on tracking failure or unsafe crop margins. Pin checkpoint URLs and hashes; rtmlib documents an author-managed Hugging Face mirror for OpenMMLab files. [rtmlib deployment and model documentation](https://github.com/Tau-J/rtmlib)

### Newer 3D and high-resolution options

**SAM 3D Body.** The official repository dates checkpoint release to **2025-11-19**. The arXiv paper was submitted **2026-02-17** and appeared at CVPR 2026. The released backbone options are 631M and 840M parameters. Models predict the MHR representation and accept optional keypoint/mask prompts. The official install path uses Python, PyTorch and Detectron2, with gated checkpoint access. [Repository](https://github.com/facebookresearch/sam-3d-body), [paper](https://arxiv.org/abs/2602.15989), [installation](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md)

**Judgment:** credible high-quality review candidate, but neither a turnkey browser component nor a validated real-time Mac coach. A 36 GiB machine having space for weights does not establish usable latency or dependency compatibility. Evaluate on selected difficult clips before attempting an Apple-porting project. A mesh can look plausible while the hidden wrist is wrong.

**SAM-Body4D.** A **2025-12-09** paper specifically identifies temporal inconsistency and occlusion failures when single-image SAM 3D Body is applied frame by frame, and proposes a video pipeline. This supports evaluating temporal stability separately from still-image accuracy. It does not establish real-time suitability for this laptop. [Paper](https://arxiv.org/abs/2512.08406)

**Sapiens2.** Meta released it on **2026-04-24**. The official pose task predicts 308 keypoints; available task checkpoints start at 0.4B, whereas a smaller 0.1B backbone also exists. Standard input is 1024×768, and the quick start uses CUDA. This is a credible current high-resolution comparator. [Official repository](https://github.com/facebookresearch/sapiens2), [pose task](https://github.com/facebookresearch/sapiens2/blob/main/docs/POSE.md)

**Judgment:** do not adopt Sapiens2 as the default teacher or product dependency. In addition to compute cost, its custom license expressly prohibits use “for biometric processing”; applicability to this application must be resolved before use. This is an unresolved license constraint, not a conclusion that all sports use is prohibited. [Sapiens2 license, section 1(b)(vi)](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)

**ViTPose++.** The original implementation and current rtmlib support make it a useful 2D accuracy comparator, especially offline. It should be selected only if it improves the relevant wrist/elbow/ankle errors on our clips; published COCO accuracy alone does not justify adding another runtime. [Official implementation](https://github.com/ViTAE-Transformer/ViTPose), [rtmlib model support](https://github.com/Tau-J/rtmlib)

**GVHMR.** The official project is a video/world-motion recovery method from SIGGRAPH Asia 2024, with TPAMI 2026 noted in its repository. It supports a static-camera path. Its license restricts ordinary use to educational, research and nonprofit purposes, with commercial use requiring separate contact. It is useful background for world-space motion research, not a default future commercial dependency. [Project](https://github.com/zju3dv/GVHMR), [license](https://github.com/zju3dv/GVHMR/blob/main/LICENSE)

**InstantHMR.** An additional author-maintained project checked on this date provides a compact 224×224 ONNX MHR regressor, 70 keypoints and a Core ML option. The author reports roughly 5 ms for the model alone on an RTX 4070; that is neither independent verification nor Mac performance. Code is Apache 2.0, weights use the SAM license. Its source is SAM 3D Body dataset labels, not evidence of boxing accuracy. Treat this as an experimental candidate after the established baselines, not a production recommendation. No initial release date was verified in this review. [Author repository](https://github.com/mohamdev/InstantHMR)

## Deployment on the M3 Max

**Source facts.** ONNX Runtime's official macOS packages can include the Core ML execution provider, which can use Apple CPU, GPU and Neural Engine facilities. Actual support depends on build, operators, shapes and graph partitioning. The documentation recommends verifying available providers; selecting Core ML does not guarantee that every operation runs on the Neural Engine. [Core ML execution-provider documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)

ONNX Runtime Web also offers WebGPU execution. It is a distinct deployment target that requires testing supported operators, browser behavior and the exported graph. [ONNX Runtime WebGPU documentation](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)

**Recommended comparison paths:**

1. Browser camera → worker → MediaPipe Full/Heavy → timestamped landmarks.
2. Local capture or local service → ONNX Runtime CPU and Core ML variants → RTMPose/RTMW → the same landmark schema.
3. Only if justified, export the winning local model to ONNX Runtime Web or a native Core ML wrapper. Avoid doing native packaging, browser export and CUDA infrastructure simultaneously.

Do not assume 60 FPS from the built-in camera. Query actual device capabilities and negotiated settings, then measure distinct-frame timestamps and drops. A request for 60 FPS is not evidence of 60 captured frames. Preserve real timestamps through every transform.

Use the [master plan's canonical gate](../project-plan.md#release-gates): **a 20-minute session without growing inference backlog, with p95 processed-frame age ≤150 ms** at the actual camera cadence. These are proposed acceptance targets, **not achieved measurements**. Document the timing origin: use a sensor/capture timestamp only when the capture path supplies a reliable one; otherwise label the result callback-to-result or another precisely defined processing interval. A callback-origin measurement does not include unknown upstream camera delay. Convert timestamps to a common clock before calculating durations; media time and application monotonic time must not be subtracted as if they shared an origin.

Measure model execution separately from full pipeline age, with warm-up and sustained statistics. Aim to preserve all usable frames for event timing, but if a model cannot keep up, process the newest frame and log every skipped frame instead of producing delayed live critique. Reassess event validity around gaps, and tune thresholds against the resulting sampling rate. A low processed-frame age achieved by discarding most frames does not establish successful motion capture.

## Why boxing needs its own evidence set

**Source evidence.** AthleticsPose (**2025-07-17**) found substantial domain-transfer problems for real athletics, sensitivity to camera view/subject scale and difficulty estimating high-speed kinematic indicators. This is evidence from athletics, not boxing; it supports the need for a boxing-specific benchmark rather than providing a boxing accuracy estimate. [AthleticsPose paper](https://arxiv.org/abs/2507.12905)

**Engineering analysis.** A boxing guard hides joints, crossed arms can swap identities, a jab toward the lens changes little in 2D despite substantial depth motion, and a small full-body crop gives a fist few pixels. Motion blur can destroy information before inference. A larger model cannot reconstruct missing visual evidence with certainty. Full-body footwork and fine hand position also compete for image resolution.

Use initial setup coaching to improve the observation: sufficient light, a stable elevated camera, full-body framing with arm-extension margin, and a tested oblique camera view. Treat front/oblique/side as benchmark conditions; do not assert one angle is universally best. If a specific correction requires another view, ask for a short side-view drill rather than fabricate a confident diagnosis. An external higher-frame-rate camera or second synchronized view is an upgrade path if the built-in camera becomes the measured bottleneck.

### Small first benchmark

Record a consented pilot on this actual camera: idle guard, relaxed hands, jab, cross, lead/rear hook, uppercut, 1–2, 1–2–hook, slips, rolls, step-in/out, pivots and natural transitions. Include slow demonstrations and realistic speed, both stances when feasible, front/oblique/side, ordinary and improved lighting, sleeves, and bare hands versus wraps/gloves if those are intended use conditions. Separate “not visible / cannot judge” from “incorrect technique.”

Keep a session-disjoint holdout for this personal coach. When expanding beyond one person, use person-disjoint splits. Adjacent frames from the same recording must not be spread across training and test sets.

| Measure | Why it matters |
| --- | --- |
| Wrist, elbow and ankle error, normalized to a fixed per-clip torso scale | More relevant than an average dominated by easy face points |
| Left/right swaps and tracking loss per minute | Swaps can reverse jab/cross labels and false-trigger critique |
| Error during fastest movement phases and self-occlusion | Average still-frame quality hides the hardest coaching moments |
| Event peak/onset/return timing against labeled video | Smoothing and sparse inference can move the event in time |
| Valid-evidence coverage at a fixed error bound | A useful model should be right when it speaks and know when it cannot see |
| Warm and sustained p50/p95 processed-frame age over 20 minutes, skipped frames, memory | Confirms the whole pipeline works on the actual machine; state whether timing starts at capture or callback |
| Downstream false critiques per round | Final selection must improve coaching, not just landmarks |

Manual 2D annotation can validate visible points; it cannot supply true hidden depth. For 3D validation, use synchronized calibrated views or another appropriate reference capture. Do not promote predictions from a second model to ground truth merely because it is larger.

### Minimal pose-output contract

Keep capture timestamp, source frame size, crop transform, model/checkpoint hash, anatomical left/right joint names, raw 2D points, optional 3D hypotheses, each model's raw confidence fields, and missing/occluded flags. Add calibrated quality and uncertainty later as separate fields. Store raw and filtered trajectories separately. Display mirroring must never change anatomical left/right labels.

Use modest causal filtering for live display; retain a timing-preserving path for event detection and an optional noncausal offline path for review. Excessive smoothing can make a bad measurement look stable and delay punch peaks. Never fill a long wrist occlusion and then criticize the invented path as if observed.

## License and reproducibility record to keep with every checkpoint

| Component | Verified licensing position on research date | Required implementation record |
| --- | --- | --- |
| MediaPipe code / BlazePose GHUM model card | Apache 2.0 in repository/model card | Exact Tasks bundle, version, hash, model card and notices; confirm bundle-specific provenance |
| MMPose / rtmlib | Apache 2.0 repositories; author RTMPose weight mirror labeled Apache 2.0 | Exact detector and pose weights, their cards and notices; do not infer every third-party dataset/weight license from the wrapper |
| ViTPose | Apache 2.0 code | Weight and dataset provenance for chosen variant |
| SAM 3D Body | Custom SAM license for code and checkpoints; checkpoint access gated | License copy, accepted checkpoint terms, MHR/optional-component terms |
| Sapiens2 | Custom license with relevant use restrictions | Resolve applicability before evaluating or integrating |
| GVHMR | Educational/research/nonprofit use; separate commercial permission | Exclude from commercial critical path unless terms change or permission is obtained |
| InstantHMR | Apache code; SAM-licensed weights according to author | Separate code, checkpoint, MHR and detector records |

License sources: [MediaPipe code](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE), [BlazePose model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf), [MMPose](https://github.com/open-mmlab/mmpose), [RTMPose author weight card](https://huggingface.co/Tau-J/RTMPose/blob/main/README.md), [ViTPose license](https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE), [SAM license](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE), [Sapiens2 license](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md), [GVHMR license](https://github.com/zju3dv/GVHMR/blob/main/LICENSE), [InstantHMR licensing notes](https://github.com/mohamdev/InstantHMR#license).

All linked primary sources were checked on 2026-09-26. Repository `main` links can change. At implementation, pin commit IDs, package locks, checkpoint hashes and the exact license texts used; reported benchmark results must include those versions and the actual capture settings.
