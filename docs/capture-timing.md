# Capture timing and actual frame delivery

New rounds record source delivery separately from pose processing, so the next webcam test can distinguish observed video delivery from inference throughput. This adds no camera access, image storage or model work; measurements run only during an ordinary user-started round.

## Distinct measurements

| Field                               | Meaning                                                                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `capture.requestedFps`              | Requested camera setting, not a measurement.                                                                                    |
| `capture.trackSettingsFps`          | Frame rate reported by the camera track's settings. This is not measured callback delivery or sensor exposure.                  |
| Legacy `capture.deliveredFps`       | Earlier releases used this name for the same track setting. Preserve it in old evidence; do not reinterpret it as measured FPS. |
| `capture.sourceCadence.mediaFps`    | Unique observed source timestamps per source-media second, across the first–last observed span.                                 |
| `capture.sourceCadence.callbackFps` | The same observations per elapsed browser callback second. Playback rate or scheduling can make this differ from media cadence. |
| `measuredFps`                       | Processed pose observations per source-media second, across their first–last span.                                              |
| `frames[].frameAgeMs`               | Browser source callback to returned pose result; includes waiting, encoding and inference, but starts after sensor exposure.    |

The review's existing **Export & details** area shows **Video delivery rate** only when real video-frame callbacks were used and a finite observed rate exists. The main practice view, skeleton and persistent punch history are unchanged. The full evidence JSON retains both clocks and coverage details.

`source-cadence-1` records unique observation count, duplicate/invalid/regressing callback counts, first/last round-relative media times, clock spans, and source-gap statistics. Rates use `(observations − 1) / span`; a missing or single observation yields null rather than zero. A clock regression makes aggregate rates unavailable. Statistics retain at most 20,000 gap samples; a partial gap distribution is explicitly marked, while counts and maximum gap continue. These counts concern accepted timestamp observations, not all camera exposures.

Source measurements occur **before** the pose pump can skip or replace a frame. They start fresh for each round and stop at the same closing boundary. Demo rounds do not fabricate capture evidence. The animation-frame fallback is labeled `animationFrame-estimate`; polling `currentTime` cannot establish unique decoded camera frames, so its rate is not shown as video delivery in review.

Offline video analysis plans from camera track settings when available, retaining the legacy fallback for old rounds. It does not cap original-video decoding to slow pose processing or missed browser callbacks. Native decoded video timestamps remain the evidence for actual saved-video cadence.

## Interpreting the next webcam round

Compare track settings, observed source delivery, processed pose cadence and frame age separately. A requested 30 FPS stream is not proof of 30 delivered source observations. A better pose rate does not imply improved hand identity or punch recall. Gaps, lighting-related exposure, motion blur, inference cost and event-finalization delay are different potential limits.

The [encoding overlap benchmark](research/encoding-overlap-2026-09.md) improved saved-frame delivery by 11.75%; these new measurements enable the actual webcam follow-up without recording unattended. Use one ordinary 20–30 second round with fast straights, curves and guard-only movement. Export the original evidence for paired diagnosis; preserve the saved video and original decisions.

## Recording clock limitation

Source callback timing does not itself prove exact alignment with MediaRecorder's encoded video origin. The app's existing camera mapping starts the round from preview `currentTime` and records shortly afterward. No new guessed offset is applied to old videos, labels or skeletons. Precise joint-localization judgments need established pixel/time correspondence as well as visible anatomy.

A fixed synthetic frame-ID audit completed three four-second recordings in installed Chrome 154, with start phases of 0, 11 and 23 ms. It reproduced the inspected recorder API ordering without mounting the hook or accessing a camera. Exact decoded image IDs were joined to source callback times and native integer saved-video PTS, without a lag search. The 265 uniquely matched interior frames had median `saved PTS − round-relative callback media time` of **−12.484, −1.463 and −1.504 ms**. Within-trial ranges were under one millisecond. The first saved image in each trial was cached from **7.862, 32.505 and 41.681 ms before** round start; duplicate callbacks with identical ID/time let a separate supplement resolve these three images too. A first-frame correction would therefore not describe the interior frames.

This establishes that exact zero is not guaranteed in the synthetic setup. It does **not** establish the historical offset of J, a one- or two-frame shift, or a universal physical-camera correction. The fixed generator requested 30 FPS but delivered about 22; this limitation is preserved, not replaced by a more favorable rerun. All 268 decoded images, original records, ID/PTS analyzer and supplement remain in ignored `data/pilot/recorder-clock-audit-v1/`. Protocol SHA256: `ee2bfbebdc65f0e4de7e0ae0dc19288641fdcff8f8402803fcffaa5f44c07e3c`. No production clock or original recording was changed.
