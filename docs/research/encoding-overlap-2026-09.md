# Overlap frame encoding with local inference

The live local pipeline now prepares the next JPEG while the current frame is being processed by the native service. The bounded saved-frame benchmark on the M3 passed its preset throughput and latency gate. This improves frame delivery; it does not establish better punch recognition or measured webcam FPS.

## Behavior and ownership

Previously the pump could capture a new bitmap during inference, but JPEG encoding waited until the previous request finished. Encoding and inference were therefore mostly sequential. The recorded laptop round had median/p95 encoding times of 7.4/10.2 ms, making this a measurable opportunity.

The pump now owns a closeable prepared frame. `LocalPoseClient.prepareImage` immediately captures source pixels, retains their timestamp, and encodes in the existing worker. `detectPrepared` consumes that exact JPEG once. The pipeline allows one encoder, one HTTP request and one waiting frame. Newer callbacks replace waiting frames; callbacks arriving during encoding are skipped rather than captured later under an old timestamp. There is no queue of old video references and no parallel native inference.

Prepared frames are bound to their client/session and source timestamp. Closed, consumed, foreign and retimed frames are rejected. Stopping disposes encoding, aborts the request, clears waiting images and closes the session. Round boundaries discard pending evidence, including a capture that finishes after discard. Offline analysis and warmup retain the serial convenience API. The MediaPipe path retains bitmap handoff.

The JPEG settings, pixels, native model, recognition weights, native score floor and recognition decoder are unchanged. The visible skeleton and persistent punch history receive every delivered result through the existing path.

Timing remains explicit: `captureMs + encodeMs + requestMs` is active-work `totalMs`; `queueMs` records encoded waiting time; `preparedAgeMs` includes it. The hook's callback-to-result `frameAgeMs` remains the primary end-to-end delivery measure. Historical active-work totals must not be compared with queue-inclusive ages.

## Fixed production benchmark

The private `data/pilot/encoding-overlap-production-v1/` protocol reused 180 hashed lossless frames from a six-second F segment. Installed headless Chrome used the actual public client and pump, existing Core ML detector/pose service, and active v5 recognizer on this M3. Images were decoded before timing. No camera or microphone was accessed.

Six trials used the fixed order serial, overlap, overlap, serial, serial, overlap, with a fresh native session each time. Source callbacks followed native approximately 30 Hz timestamps. JPEG worker warmup added no model observations. Hashing occurred after each measured trial. No candidate fit, extraction, browser test or other benchmark ran concurrently. The 800 MiB decoded-cache limit, monotonic delivery, exact source pixels and maximum of one HTTP request were checked.

| Pair | Serial processed FPS | Overlap processed FPS | Relative gain | Change in p95 frame age |
| ---- | -------------------: | --------------------: | ------------: | ----------------------: |
| 1    |                22.70 |                 25.87 |       +13.97% |                −7.03 ms |
| 2    |                23.01 |                 25.56 |       +11.08% |                −2.58 ms |
| 3    |                22.73 |                 25.40 |       +11.75% |                −3.42 ms |

Median paired throughput gain was **11.75%**, with **3.42 ms lower p95 frame age**. All three pairs improved both measures. The frozen gate required at least 10% median paired throughput gain, no median p95-age increase, and exact shared-input pixel/pose parity. It passed. The earlier handoff prototype independently measured 12.79%; the table above is the finished production implementation.

Independent audit found exact JPEG/native-pose equality in all **695 repeated paced comparisons** across 180 distinct source frames. Two additional, untimed-for-gating passes delivered all 180 observations in identical order, serial versus one prepared frame ahead. All JPEGs, poses and recognition content matched exactly, including six emitted events per pass; timing telemetry was excluded. Different sampled observations in a paced run can legitimately produce different recognition events. These are deterministic integration checks, not additional recognition accuracy tests.

This short, already-known recording excludes sensor capture, live video decoding and varying room/light conditions. It is not a claim of 25 FPS webcam performance, new punch recall, calibrated confidence or trustworthy form critique. Actual webcam throughput and fast-punch behavior still require a fresh user round.

## Reproduction and regression checks

Production protocol SHA256: `84dcc4a1fd6536ca9fcd74e2e33286b1d224268b296eab2cfaf4313bcb8def6e`.
Input-lock SHA256: `4c4e5a0921b88eb87f0d948805567af12197ccc7426433483dff789263c6989f`.
Completion SHA256: `e021199514caa2993fa6e7d1fa2cb20419f40bbda69855ecaec88e70a38c30e0`.

The production protocol's `sourceFiles` pins were checked before and after execution. Its inherited prototype `sourceHashes` field is historical context, not the production source identity. Both attempts and their immutable outputs remain local; do not rerun over either directory. The original prototype imports the earlier pump API, so its pinned source revision is required to reproduce it.

Ten new lifecycle tests cover immediate capture, overlapping encoding, one request, bounded ownership, timestamps, single-use frames, replacement, round boundaries and disposal. The full frontend check passes 263 tests, 14 tool tests, formatting, TypeScript and a production build. All 50 browser tests pass. Browser coverage includes demo mode, visible 20 Hz skeleton delivery, round start/end, saved review, imports and the actual local MediaPipe worker; synthetic test fixtures do not access the user's camera.

Next user test: a fresh ordinary 20–30 second round containing singles, quick jab/cross pairs, both hooks and uppercuts, and a short guard-only pause. Check that the skeleton remains responsive, punch history persists, and finishing opens the original replay. Compare recorded frame age/cadence and action errors separately; a throughput gain alone is insufficient to declare the recognition problem solved.
