# Arm-local RGB-motion experiment — September 27, 2026

**Rejected; active v5 stays unchanged.** Giving each arm its own image-motion descriptor lost previously recognized hooks and uppercuts at both tested cadences. One fixed candidate was fitted. No additional crop, threshold, seed or label search followed the result; no candidate was activated.

This tests the arm-local hypothesis proposed after the [global RGB pilot](rgb-motion-pilot-2026-09.md). It reuses the same nine private, exactly paired image/pose recordings. A–G train the candidate; H/I are already-known development recordings excluded from fitting; J is excluded entirely. All nine recordings depict one person on one day. Imperfect punches remain valid action examples, not correct-form supervision.

## Occurrence results first

Cells are **matched / unmatched predictions / missed references**. The fixed occurrence diagnostic uses one-to-one physical-hand/family matching and an observed peak within ±250 ms of the frozen reference. C/D/E/G/H/I supply 119 eligible references. A/B/F lack complete explicit peak references and remain withheld; they are still included in strict interval scoring below.

| Fixed-peak scope                         |    Active v5 | Reused zero-RGB control | Arm-local RGB |
| ---------------------------------------- | -----------: | ----------------------: | ------------: |
| Native frames, eligible six recordings   |  110 / 5 / 9 |            103 / 5 / 16 |  105 / 5 / 14 |
| Simulated 20 Hz, eligible six recordings | 109 / 4 / 10 |            104 / 2 / 15 |  101 / 5 / 18 |
| Native frames, H/I only                  |   17 / 3 / 7 |             13 / 1 / 11 |   13 / 2 / 11 |
| Simulated 20 Hz, H/I only                |   20 / 2 / 4 |              17 / 0 / 7 |   14 / 3 / 10 |

Compared with active v5, native occurrence matches lose G `g12` (right hook), `g19` (right uppercut), and H `h10`/`h11` (right hooks), `h14`/`h15` (left uppercuts), `h17` (right uppercut). Gains are only G `g17` (right uppercut) and I `i04` (right cross). At 20 Hz, losses are G `g18` (right hook), `g19`, and the same five H references plus `h13` (right uppercut); there are no occurrence gains.

The unchanged strict metric also requires temporal IoU ≥0.5. It scores 175 action references across A–I; a strict miss can reflect timing, family, hand or missing detection.

| Strict interval scope     |     Active v5 | Reused zero-RGB control | Arm-local RGB |
| ------------------------- | ------------: | ----------------------: | ------------: |
| Native frames, A–I        | 155 / 15 / 20 |           150 / 11 / 25 | 147 / 19 / 28 |
| Simulated 20 Hz, A–I      | 153 / 17 / 22 |           150 / 10 / 25 | 147 / 16 / 28 |
| Native frames, H/I only   |   11 / 9 / 13 |             11 / 3 / 13 |    9 / 6 / 15 |
| Simulated 20 Hz, H/I only |  12 / 10 / 12 |             13 / 4 / 11 |   11 / 6 / 13 |

Both cadences lose earlier strict matches `f-18` (left hook), `f-23` (right uppercut), and G `g17`, `g19`, `g28`; native cadence also loses `g12`. Per-recording unmatched strict predictions increase in F, G and I. Lower aggregate unmatched counts in one scope cannot compensate for lost matched identities.

The prespecified gate required H/I strict improvement over both baselines, no lost A–G strict matches, no per-recording unmatched increase, retention of every eligible occurrence match, and at least two H/I occurrence gains over both baselines at both cadences. It fails decisively. An independent audit recomputed all 18 reports, checked unchanged labels and observations, and confirmed exact decision arithmetic. No extra pixel review was needed to reject. Unmatched predictions are not automatically evidence of invented physical actions; this report does not relabel them as guard hallucinations.

## Frozen representation and control

Each arm gets the existing 40 pose features plus 96 image features. The current observed shoulder/elbow/wrist define a bounding rectangle with a fixed margin of 8% of image height. Current and previous delivered frames must each have exactly one accepted person and finite, in-frame own-arm anchors meeting the unchanged experimental native score policy. Otherwise that arm's image descriptor is zero. The ordinary pose-valid mask and decoder gates remain unchanged.

The **same current rectangle** crops both images, resized to 64×64. Signed and absolute RGB differences are mean-pooled into a 4×4 grid. Delivery selection happens before computing differences separately for 15/20/25/30 Hz training cadences. The causal 30 Hz grid holds observed descriptors; it does not interpolate joints or pixels. Native and simulated 20 Hz evaluation use identical delivered poses across comparators. These are pixel differences, not optical flow or semantic arm segmentation. A crop can include the other arm/background or omit the preceding hand position outside its fixed margin.

The one candidate uses the same 136-input, 48-hidden-channel causal network, seed 41729, 1,800 steps, AdamW settings, augmentation, labels and frozen v5 event decoder as the previous experiment. The control checkpoint is reused only after proving all 28 ordered zero-RGB training tensors, targets, masks and class counts equal; seeded initial weights, full-shape noise/reflection RNG consumption, optimizer settings and Torch 2.10.0/NumPy 2.2.6 versions also match. No second control was fitted. Ten focused tests passed, including anatomical crop assignment, reflection, prefix causality, missing anchors and selection before differences. Independent review cleared the fit only after checking all 209 locked file hashes.

The rejected global descriptor remains a reference result: eligible occurrence totals were 110/11/9 native and 106/13/13 at 20 Hz. Moving the same coarse representation into arm-local crops did not solve its generalization problem on these known recordings.

## Cost, provenance and next decision

Per-recording medians for both arms' crop/resize/difference/pooling range 0.95–1.82 ms per native observation; p95 values range 1.70–2.54 ms. PNG hash/decode medians are separately 4.44–7.05 ms. The single-thread fit took 14.58 seconds. These are preparation/training measurements, excluding camera capture, pose, transport, recognition and display; they are not a live throughput result.

All sources, features, the one checkpoint, 18 candidate replays, baseline comparison outputs, tests and independent audits remain ignored under `data/pilot/arm-rgb-motion-v1/`. Original references and uncertainty intervals—including the earlier F boundary case—are unchanged. A pre-fit evaluation-ID correction is separately recorded with the preceding source and lock retained; it changes no feature, fit or scoring rule. The checkpoint requires private 136-feature inputs and is not a portable runtime bundle.

- Protocol SHA256: `5d2f757a25ec17d5eeb1b9f354b1d7503e06840369bb92fcb3bc66690ed1d180`.
- Input-lock SHA256: `8d7a37fe58ec52321053ec84424459f1c4bf2cc57cc95049c5f85a8421b614e8`.
- Checkpoint SHA256: `6867e03fb7868e2ac2970d936f956af11fb57bcf8755a3980bf6a6c9474609e6`.
- Decision SHA256: `38d9501fc71c098cfbd767b80c2b6f262f36fa735cd6cad5125984a1041d8418`.

Keep v5 and stop this descriptor family for now. Both lost F actions had valid arm descriptors throughout their labeled intervals, so missing anchors alone cannot explain the failure. The evidence does not isolate crop geometry, appearance sensitivity or limited supervision as the sole cause. The next useful decision is whether a semantically richer visual representation is justified by independently checked actor/arm identity and explicit background examples. For external match footage, resolve actor-to-label association and reuse terms before training. Any later fit needs new prespecified evidence and retention gates; these familiar H/I results are not a fresh test set, and none of this validates technique critique.
