# Overnight continuation — September 27, 2026

Read this first on each scheduled continuation. This pass ends by **10 AM America/Los_Angeles on September 27, 2026**; after that summarize, do not start another experiment. Inspect Git and running work before doing anything. Do not repeat completed comparisons or duplicate downloads.

## Authorization and product target

- Commit and push directly to **main**, without PRs. Preserve unrelated work.
- M3 MacBook Pro webcam; local live recognition, large video, visible skeleton, persistent punch history, simple tasteful UI with no dropdown arrows. Accurate six-punch recognition precedes voice and trustworthy form critique.
- The user's imperfect technique is target-domain/action evidence, **not correct-form supervision**. Prioritize varied skilled/imperfect action data and separately assessed skilled references, faults and acceptable variations.
- Keep personal video, external media, generated features and weights under ignored `data/`. Preserve original sessions, annotations, detections and video. No private uploads, outreach, agreements or spending. Do not capture camera/microphone while unattended.
- Run bounded experiments with written fixed gates. Retain failed evidence. Never promote a candidate just because it increases counts or fits familiar recordings.

## Current live app

`npm run dev:mac` remains running in exec session `40236`, log `/tmp/corner-studio-v5-final.log`. Vite is `http://127.0.0.1:5173`; native pose is `http://127.0.0.1:8765`. Health needs `Origin: http://127.0.0.1:5173`. Check listeners/processes before starting another service; another project uses IPv6 Vite.

Ignored `data/local-studio.json` activates RTMPose RGB/CoreML with minimum native score 0.55 and `data/pilot/recognition-runtime-v5/bundle/manifest.json`. Bundle fingerprint: `a821d37e52129c311d7aa78256522e0c6c10a6670f52a6692b61605eb25efd61`. **No recognition weights, pose gates or runtime settings changed in this pass.**

Heartbeat `improve-shadowboxing-overnight` is active hourly for seven occurrences in this thread. It requires the Mac on and Codex running. A bounded `caffeinate -i -t 25200` runs in exec session `1691` to prevent idle sleep; no permanent settings changed. Do not start another copy.

## Current pass

- Review now opens with **Correct punches**. Action labels save/count/resume without requiring a guard judgment. New `local-coach-review-2` records explicitly identify user observations as provisional; legacy v1 records retain their original provenance. Reselecting the same action correctly exits edit mode. No form score or automatic training is added.
- Final `npm run check` passed: 253 unit tests, 14 Node tool tests, formatting/typecheck/production build (`/tmp/corner-overnight-final-check.log`). Full browser suite passed 50 tests (`/tmp/corner-overnight-full-e2e.log`); after the final wording/reselection fix, the three affected review tests passed again (`/tmp/corner-overnight-final-review.log`). The earlier eight-test review/storage run passed too. Actual Chrome review was visually checked without writing labels; all 11 user rounds remain. The page is left on Practice with camera off.
- External acquisition and annotation audit are complete for this bounded pass. No active download remains. The reusable [RGB+pose inspection CLI](rgb-pose-inspection.md) is implemented with resource limits, finite-value checks, input/model/helper hashes and explicit coverage. All 186 Python tests passed in the optional runtime; the standard-library environment also passed with 11 optional tests skipped. Thirteen focused tests cover the adapter.
- The final public CLI smoke processed three sample-video frames and nine unassigned people. `data/pilot/external-rgb-pose-inspection-v1/public-cli-final-smoke/inspection.json` matches the final script SHA-256 `9851c69dd2612a7a85761e929aa4dc85a06f86cd8095242b47fa67a77b72b361`, reports `frame_limit`/incomplete coverage, and retains finite numeric tensors. No current live-model source changed.
- UI/provenance changes are committed as `e222486` (`Simplify punch corrections and keep form observations provisional`). The external-data audit, inspection CLI and this ledger are in the following commit. Check current Git and exact-head GitHub Actions before continuing; do not duplicate these commits. All three agents finished their bounded tasks; no experiment or download is left running.

## External data acquired locally

`data/external/boxingweb-inspection-2026-09-27/` contains the full public BoxingWeb archive: 5,190,972,710 bytes, SHA-256 `9ce5052f55f9c256dd118c6c58ce3e3c32c6d74e11ae34b4bb8842fb50fd2ba8`. Safe listing found 50 MP4, 50 JSON and 50 PKL files (40 training rounds, 10 test rounds). All 50 JSON annotations and one 24 MB, 84-second sample video extracted. **No pickle loaded; reuse terms remain unverified; not cleared training data.** The 18-frame six-family contact sheet confirms match footage with opponents, officials, cuts and occlusion; it does not validate every action or form. Source and acquisition evidence are preserved beside the archive; do not redownload it.

There are 6,872 raw punch rows, 6,556 positive-duration intervals and 6,522 passing the author duration filter. One 107-row test round has unresolved confidence-field/model/human provenance; conservatively quarantine it. That leaves 6,433 structurally duration-valid rows across 49 files, not certified human labels. Seven literal athlete names overlap release splits. `sample-normalized-annotations.json` preserves all 63 sample rows (62 punches) and joins 18 observed native PTS frames plus 60 decoder-relative frames by source hash/index; missing mappings stay null, and actor tracks/stance/peaks remain unassigned. Its SHA-256 is `a6591f6b3769fadf756fb08f58be7e395366d3ed2631a902a36870afa0dd2c29`.

`data/pilot/external-rgb-pose-inspection-v1/` retains the original inspection adapter, seven tests and two real-video smokes. The opening 30 frames yield 317 person instances; the hook-window 30 frames yield 97. Numeric RGB crops, causal image differences, native joint scores and coordinates are retained without action labels, tracking identities or pickle loading. OpenCV timestamps are explicitly decoder-relative, not verified native PTS. These are pipeline evidence, not recognition accuracy. Do not overwrite these v1 artifacts when running the public adapter.

`data/external/coachme-bx-audit-2026-09-27/` holds CoachMe labels (163 train/41 test, 612 original plus 612 augmented comments). Only jabs/crosses; no original RGB acquired, no verified person split, no direct equivalent of our guard criterion. See [the source audit](research/expert-reference-data.md).

## Completed negative experiments — do not repeat

1. `data/pilot/recognition-time-support-v1/`: replaced seven contiguous straight samples with five distinct observations spanning at least 200 ms, maximum gap 100 ms. All 17 older A–I strict match sets and TP/FP/FN unchanged; 16 event arrays identical. G-lowfps changed two boundaries and delayed one event. Zero recovered actions fails the fixed promotion gate. Seven causal/cadence tests passed. No J run, no activation.
2. `data/pilot/onnx-thread-audit-v1/`: three alternating trials on 25 F JPEGs, default ONNX threads versus one intra-op thread. Exact pose outputs identical. Default median 28.89 ms vs 28.24 ms (2.24% gain, below fixed 10% gate); candidate p95 41.07 ms vs baseline 29.80 ms. Saved-JPEG microbenchmark excludes browser/capture/recognizer and overlapped other development activity; do not generalize the tail difference. Gate failed; service unchanged.
3. Existing RTMW-L and MediaPipe Heavy/Full/Holistic comparisons are already exhausted on D/E/F. No consistent rear-arm improvement and RTMW was slower. Do not repeat those same comparisons without a new hypothesis and input evidence.

## Evidence that determines next work

Latest personal round J is already inspected and therefore development evidence. Original capture: 30 s, 20 detections, 23 scored actions. Strict matching 10 TP/10 FP/13 FN; fixed peak-occurrence diagnostic 13/7/10. Only two of eight rear hooks occurrence-matched. Seven hooks had weak native elbow/wrist support; some fast jabs were fully tracked but missed. Source video 30 fps, observed pose cadence about 18.8 fps. Current weights fit seven personal recordings; transfer remains weak. Never call J an untouched new test.

J median/p95 timing (ms): capture 0.0/0.1; decode 2.0/2.27; person detection 23.94/33.34; pose 8.89/14.83; recognizer 5.73/8.09; encoding 7.4/10.2; request 43.6/59.5; service 40.36/55.52; total 51.2/67.1. Person detection is the largest measured service cost. Previous detector-caching candidates already failed accuracy checks. Nominal video fps and shorter UI delay do not prove recognition improvement.

## Next priorities

1. Use the already acquired sample; do not start with another literature list or download. Complete its native frame-to-PTS map and audit a small fixed set of the six-family windows against visible pixels. Record uncertain actor/hand/boundaries explicitly. Fight broadcast footage offers action variation; it does not automatically define correct form or match a solo webcam.
2. Make the first reviewed actor-to-instance associations and visible wrist/elbow references for those bounded windows, using source video/frames and frozen hashes. Preserve all unassigned instances and native scores. Never attach a fight label to the largest person, screen side or a guessed lead hand. Do not train on unverified labels/terms. In parallel, the existing local A–I action data can support a fixed RGB+pose representation experiment without treating its execution as correct form.
3. Use visible-pixel/pose discrepancies to choose one fixed observation or short RGB+pose challenger. Evaluate false guard counts and retained/recovered event identities on all development views, along with latency. Avoid more threshold fishing.
4. Improve UX only where a demonstrated flow is confusing. Preserve large replay, visible skeleton, immediate original-video review and persistent punch history. Use synthetic fixtures for destructive tests; never label/delete real user recordings during verification.
5. Keep action recognition, observable form evidence, and usefulness of correction as separate acceptance checks. Do not add confident live advice or voice while form reliability is unestablished.

No new user recording is needed to unblock data inspection. The next useful user test is one ordinary fresh round only after a frozen recognition candidate meets its development gate; include singles, quick repeats, curves and a few seconds of guard changes without punches.
