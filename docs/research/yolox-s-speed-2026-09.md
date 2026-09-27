# Smaller person detector — bounded M3 comparison

**Keep the active YOLOX-M detector.** YOLOX-S reduced predecoded-image native observation time by a median paired **19.815%**, but the experiment stopped when a strict interval match changed on recording B. Both models still emitted that cross at the same peak time. This is a boundary-sensitive failed gate, not demonstrated loss of the physical punch. A separately frozen follow-up later preserved B's range-compatible actions, then failed on new unmatched hand/family errors in the difficult rear-hook recording. Remaining acceptance checks are incomplete.

No production code, active manifest, weights, score gates or service settings changed. No model was fitted. The candidate and all evidence remain local under ignored `data/pilot/yolox-s-speed-v1/`.

## Fixed comparison

One official [YOLOX-S model-zoo artifact](https://github.com/Tau-J/rtmlib#model-zoo) was acquired, with the same 640-pixel BGR detector preprocessing, 0.7 person threshold, exactly-one-person policy, RTMPose-M RGB model and 0.55 native-joint floor. Core ML with CPU partitions was verified by executed provider profiles. A private compatibility manifest binds the new detector hash; the original v5 bundle remains untouched.

The protocol and 7,527 input/source/model pins were frozen before inference and verified afterward. One process ran with a 20-minute cap, finishing in 166.263 seconds. No retries, model sweep or threshold search. Stage 1 ran while other agents held decoding, tests, builds and inference; stage 2 timing is not a speed result.

## Stage 1: speed and person availability passed

The same 180 preselected existing images—20 uniformly indexed per A–I recording—were decoded before timing. Model initialization and 20-image warmup were excluded. Three alternating M→S, S→M, M→S pairs measured detector, pose and result construction, excluding JSON serialization and file writes:

| Pair | M median (ms) | S median (ms) | Reduction | M p95 (ms) | S p95 (ms) |
| ---- | ------------: | ------------: | --------: | ---------: | ---------: |
| 1    |        26.988 |        21.605 |   19.946% |     36.635 |     22.480 |
| 2    |        26.878 |        21.900 |   18.522% |     28.678 |     22.523 |
| 3    |        26.992 |        21.644 |   19.815% |     27.770 |     22.427 |

The median paired improvement is approximately 5.35 ms, with median paired p95 difference −6.156 ms. Each pass has the same 177 single-person and three zero-person observations; no per-image count mismatch or nonfinite native coordinate/score. Independent recalculation reproduced the gate results. Peak process RSS of 900,759,552 bytes includes both detector/pose pairs and decoded images, so it is not per-model memory.

This **excludes camera exposure/callback delivery, decoding, browser JPEG preparation, HTTP, temporal recognition and rendering**. It is neither a webcam FPS claim nor a measured improvement to the live app. All trials, including the first M trial's higher tail, remain in the report.

## Stage 2: strict boundary trade, then stop

Full native A and B series (850 and 784 images per condition) were processed using identical saved pixels and source times. Neither recording gained a new own-arm support-loss run longer than 100 ms inside the compared action windows. A's native and simulated 20 Hz replays preserve 12 TP / 0 FP / 0 FN and all matched reference IDs.

B-native scores **20 TP / 1 FP / 1 FN** for both detectors, retaining 19 of the original matched IDs. One is lost and one recovered:

| Reference           | M right-straight interval / peak (ms) | S right-straight interval / peak (ms) | Strict interval IoU          |
| ------------------- | ------------------------------------- | ------------------------------------- | ---------------------------- |
| `b-5`, 6410–7080    | 6469–6768 / 6571                      | 6404–6804 / 6571                      | .446269 → .582840; recovered |
| `b-16`, 17410–17810 | 17373–17673 / 17510                   | 17408–17607 / 17510                   | .601831 → .490050; lost      |

The selected peak, physical hand and family are identical for both decisive actions. S's shorter `b-16` interval overlaps the existing reference by 197/402, just below the frozen 0.5 cutoff. Both outputs have 21 events and the same ordered hand/family sequence; maximum absolute differences across corresponding events are 67 ms start, 36 ms selected peak, 100 ms end and 101 ms emission. This prediction-to-prediction comparison does not establish reference-based occurrence equivalence for every action.

The declared gate required retaining every previously matched reference ID, so the attempt correctly stopped. An independent standard-library rescore reproduced the loss/recovery. Original intervals and unknown masks remain unchanged; there are no complete scalar reference peaks for B, so a complete peak-occurrence score was withheld.

**B at 20 Hz, C–I full replays, J rear-hook retention and browser delivery were not run.** The passing speed screen cannot replace those missing checks. Do not resume this failed attempt or relabel its outcome as a pass.

## Follow-up requirements

A separate video-only reference audit subsequently reviewed every native frame within all 21 existing action windows, with 150 ms of context on either side and a full-clip overview. It preserves 21 terminal ranges, one defensible scalar peak and 20 unknown scalar peaks; held extension and adjacent plausible terminal frames must not be converted into midpoint peaks. The author read no predictions, but knew existing annotation notes before the protocol. This familiar recording remains provisional development evidence, not blind validation or correct-form supervision.

Independent checks verified all 784 source-image hashes, original video/session/annotation hashes, exact native times and all reference IDs/bounds. Three action sheets plus three adjacent full-resolution frames received a separate visual spot check, not a complete expert relabeling. The source labels and unknown mask remain untouched. New references SHA256: `4478653b14acc26074a839ac8b85ea65dc98c3dd329311e17afdda91cb491d49`; independent audit: `777fa85aafe8daafa4caad6e7d9d2c9834dcafad121a9865351f7ba51378c939`, in `data/pilot/view-b-peak-audit-v1/`. This does not change the earlier failed gate.

Any later promotion attempt needs a new frozen protocol that separately measures punch occurrence, hand/family errors, guard false positives, interval boundaries and actual delivery cost. Define its policy before candidate inference; include remaining recordings, both cadences and J's known difficult rear hooks. Reuse saved candidate outputs when compatible rather than repeating completed model calls. A revised interpretation cannot retroactively pass this experiment.

Model archive SHA256: `e8e8f51cb9c70ed1c3781ed12f1debf918d78cf94638554f1a467f704f627349`; ONNX: `332e09ea9696e3401049b6c5314851db020b6430f85da159e65f37034ab3aee8`. Code licenses do not by themselves establish trained-checkpoint redistribution terms; the [upstream checkpoint clarification](https://github.com/open-mmlab/mmpose/issues/3271) remains unresolved. No dataset training or weight redistribution clearance is claimed.

Protocol SHA256: `c9a51d8e9ca24cb7ef699cf0260c8c66ffc3d426b2aa4f1e5dab4bf42d0dd844`. Private files retain all six timing passes, model/provider profiles, full A/B native observations, completed causal replays, raw B event differences, independent audits and unchanged-pin completion. Process PID 41690 / exec 4909 ran 07:02:40–07:05:26 PDT; log `/tmp/corner-yolox-s-speed-v1.log`. No experiment process remains.

## Separately frozen follow-up: useful boundary clarification, new error tradeoff

A second attempt, `data/pilot/yolox-s-followup-v2/`, froze a new policy before scoring or new inference. It preserved the first attempt and its failed strict gate. The new [terminal-range diagnostic](../terminal-range-evaluation.md) keeps B's reviewed uncertainty intact; A/F use original strict matches, while C/D/E/G/H/I/J require complete original scalar evidence. Every baseline match must remain, unmatched counts may not increase within a physical-hand/family, and new unmatched occurrences or ambiguous associations block acceptance. Unknown exclusions are checked separately. No model activation is part of the run.

Input preparation verified all source pixels, native clocks, provider policies and cache lineage. A/B observations and completed replays were reused; G–I baseline observations were prepared for reuse. Older C–F caches did not establish the current static Core ML detector configuration, so fresh paired observations would be required. J received one complete 899-frame lossless extraction with its original source PTS, not a timing shift or synthetic frame-rate conversion. The plan fixed A→B→J→C… I so known difficult rear hooks were checked early. G's reference identity adapter verifies the original label-source hash and video identity before changing only the association to its derived session ID; original reference values remain untouched.

The attempt completed in **85.120 seconds**, with all **8,552 pins unchanged**, and stopped at **J-native**:

| Check                         | M baseline                                      | S candidate                                                         | Finding                                                                      |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| A, native and simulated 20 Hz | 12 strict matches, no unmatched predictions     | Same IDs and counts                                                 | Reused control retained                                                      |
| B, native and simulated 20 Hz | 21 terminal-compatible references               | Same 21, no unmatched/excluded predictions or ambiguous assignments | Range compatibility retained; not scalar accuracy                            |
| J, native scalar occurrence   | 13 matches / 6 unmatched / 10 missed references | 14 / 6 / 9                                                          | All matched IDs retained, one additional rear hook; error-family gate failed |
| J, native strict interval     | 11 / 8 / 12                                     | 12 / 8 / 11                                                         | Separately retained interval diagnostic                                      |

B-native still has the original strict boundary trade. At simulated 20 Hz, its strict counts change from 20/1/1 to 21/0/0; this does not establish new physical punches or better webcam delivery.

On J, the candidate adds unmatched **right-straight** and **left-uppercut** events near 2.7 s and 22.7 s. Two previously unmatched right-hook events disappear from the unmatched set, one through a newly matched reference. The aggregate unmatched count stays six, but the changed errors violate the prespecified per-family and new-occurrence gate. The corresponding full-native baseline matches only **one of eight rear-hook references**, versus two for S; these are saved-PNG cold replays, distinct from the earlier captured live round. Neither result establishes reliable rear-hook recognition.

No new >100 ms own-arm support-loss run appeared in J's original action intervals. Person availability improved from 892 to 897 single-person observations out of 899, but neither more boxes nor score-supported joints certify correct localization. Only **1,798 fresh full-frame pose observations** and ten fixed warmup observations were needed; A/B inference and the earlier speed screen were not repeated. Native and 20 Hz B recognition reused exact saved poses.

**J at 20 Hz, C–I new recognition checks and browser delivery were not run after the failure. Keep YOLOX-M active.** There was no retry, threshold search, fit or deployment. This attempt's timings include model setup, decoding, recognition, traces and file operations and may overlap other work; they are not a speed comparison. The earlier isolated 19.815% observation-time result remains the only paired detector timing evidence.

Protocol SHA256: `18493405d2f51898c22a77ca849a3263b55330b784598f2bed83e4fcf9faf8c1`; input lock: `ff9dcb1eb2a3a667385c111a3416cc80fed4452a03dada2d8d9c88c5ca66b68c`. PID 47448 / exec 96794, 08:17:07–08:18:28 PDT; log `/tmp/corner-yolox-s-followup-v2.log`. Partial-output safeguards, raw native observations, passive decision traces, both metrics and the failed result remain local and ignored.

An independent stored-output audit reproduced every saved strict/peak metric and all four B range diagnostics, and verified the first-failure stop and all 8,552 pins. Its bounded source check used exactly two decisive associations and six existing J PNGs: the new right straight lies inside the already reviewed right-hook action, while the new left uppercut coincides with raised guard/body settling rather than a clear independent uppercut. These provisional pixel judgments strengthen the error finding without changing labels. Audit SHA256: `164d7efe904e1f044e32acf5407a94dbea46be99148de83a3fff016596d968b7`, in `data/pilot/yolox-s-followup-v2/independent-result-audit.json`. No new inference was needed.
