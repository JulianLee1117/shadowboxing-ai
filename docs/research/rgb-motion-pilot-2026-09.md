# Fixed RGB-motion experiment — September 27, 2026

**Rejected; the live v5 model remains active.** Coarse image differences changed event timing and recovered some matches, but increased unmatched punch occurrences and lost an existing match. Neither candidate was activated. All videos, features, checkpoints and detailed outputs remain ignored under `data/pilot/rgb-motion-fusion-v1/`.

## Frozen comparison

Two models were trained once each with seed 41729 and 1,800 steps. Both use the same causal 136-input, 48-hidden-channel network, optimizer, supervision masks and unmodified v5 decoder. The control receives 40 pose features plus 96 zeros. Fusion receives the same pose features plus signed and absolute RGB differences pooled into a 4×4 grid. The existing active model is an additional baseline on identical pose observations.

A–G are training recordings; H/I are already-known development recordings excluded from these fits. J is excluded entirely. All recordings are one person and day. Imperfect punches are action evidence, not correct-form examples. These results cannot establish generalization, coaching quality or unseen-person accuracy.

Lossless images and pose observations join by exact source timestamps and hashed bytes. A–F coordinates and raw native scores match the canonical series exactly. G/H/I are separately derived paired views; they are not claimed equal to historical live observations. Original videos, action labels and uncertainty intervals are unchanged.

For each 15/20/25/30 Hz training cadence, delivered observations are selected before calculating image differences. Both images use the same current expanded person box. First frames, missing/ambiguous people and long gaps produce zero descriptors. There is no future-frame interpolation, label-selected crop or final decoder flush. The native arm-observation gate remains intact; appearance cannot bypass an unobserved wrist in this experiment.

A separate amendment froze simulated 20 Hz evaluation before training/results. It uses differences between delivered images, not subsampled native-frame differences. Nine focused tests and independent input, split, native-score and causality audits passed.

## Results and rejection

Each cell is **matched / unmatched predictions / missed references**. Strict scoring requires physical hand/family agreement and temporal IoU ≥0.5. H/I remain development evidence, not an untouched test set.

| Strict scoring scope |    Active v5 | Pose control |   RGB fusion |
| -------------------- | -----------: | -----------: | -----------: |
| Native frames, A–G   |  144 / 6 / 7 | 139 / 8 / 12 |  144 / 7 / 7 |
| Native frames, H/I   |  11 / 9 / 13 |  11 / 3 / 13 |   19 / 8 / 5 |
| Simulated 20 Hz, A–G | 141 / 7 / 10 | 137 / 6 / 14 | 140 / 7 / 11 |
| Simulated 20 Hz, H/I | 12 / 10 / 12 |  13 / 4 / 11 |   20 / 8 / 4 |

Strict gains do not necessarily mean newly recognized punches. A separately frozen occurrence diagnostic uses physical hand/family and peak timing within ±250 ms, with one-to-one matching. It is available for C/D/E/G/H/I; A/B/F have incomplete or absent explicit peaks, which remain unknown.

| Occurrence scoring, same six recordings |    Active v5 | Pose control |    RGB fusion |
| --------------------------------------- | -----------: | -----------: | ------------: |
| Native frames                           |  110 / 5 / 9 | 103 / 5 / 16 |  110 / 11 / 9 |
| Simulated 20 Hz                         | 109 / 4 / 10 | 104 / 2 / 15 | 106 / 13 / 13 |

The preset gate required H/I strict improvement over both baselines, preservation of older matched identities, and no increased unmatched events. It failed:

- G loses reference `g28` under both cadences.
- I adds unmatched events: native 1→3 and simulated 20 Hz 1→4 under strict scoring.
- H's native strict matches rise 8→14, but occurrence matches fall 14→13 and unmatched occurrences rise 2→6. At 20 Hz, occurrence matches fall 16→14 and unmatched occurrences rise 2→5. This is not six newly recognized actions.
- F recovers one jab at native cadence and retains all 20 earlier strict matches. Its additional scored unmatched event is an uncertainty-boundary effect: an existing ambiguous right action starts 1 ms before the frozen exclusion interval. Independent review of four source frames confirms a real sweeping action, with family unresolved. It is not evidence of a newly invented guard action. The gate and labels remain unchanged.

Both models fit the small training set to low loss. That did not establish useful event recognition. No threshold, seed or model search followed these results. The evidence rejects this specific coarse descriptor and fit; it does not establish that RGB representations in general cannot help.

A bounded visual audit of 18 source frames gives a more specific failure taxonomy. Two H right-hook detections accompany actual left-arm punches: the right arm adjusts guard or moves with the body. Another H detection follows a relaxed open-hand drop/swing and return to guard; it crosses an existing uncertainty interval, which stays unchanged. Several other H peak failures are real same-hand/family actions with late selected peaks. In I, two extra left-hook detections align with labeled upward bent-arm actions, while another follows post-action arm lowering and approach toward the laptop. These are provisional visual observations, not new gold labels or correct-form judgments. Both timing errors and guard/wrong-family errors matter; extra counts cannot be dismissed as annotation boundaries alone.

## Cost and reproducibility

Image crop/resize/difference/pooling medians ranged 2.63–5.22 ms per observation across the recordings, with p95 values 5.06–7.33 ms. These figures exclude image decoding, capture, pose, recognition, transport and display. Cached replay and simulated delivery do not measure webcam throughput. The existing recorded live median/p95 end-to-end timing remains 51.2/67.1 ms, with observed pose cadence around 18.8 Hz; no new camera capture occurred.

The exact checkpoints are private research artifacts requiring 136-channel inputs, not deployable 40-feature runtime bundles. The experiment retains the original native and 20 Hz protocols, input lock, all 54 replays and reports, two checkpoints and a decision manifest. An optional peak-reference identity check initially stopped G evaluation: its reference named the original captured session while the paired artifact had a derived ID. A separately hashed resume adapter verified the original session ID and matching video SHA before adapting only that association. Raw references, peaks, labels, completed A–F outputs and original failure logs are preserved.

Protocol SHA256: `2b7c889641ad453caf4c27f260b346b50f7d52c975a8752c280655d8763061ae`.
Input-lock SHA256: `4051d8842dbdb0bf95a54351a8ade4ce9c2636e72deb9e5edf9f53299471b8c8`.
20 Hz amendment SHA256: `e22a0ea4772d0cb8a9fad3b78aca56ef7d8785076083a179302c408e1dd1dae9`.

## Next decision

Reuse these exact image/pose pairs and negative results. The bounded H/I audit supports testing an arm-specific visual representation: both arm classifiers currently receive the same global image-motion descriptor, and opposite-arm motion can accompany false counts. This is a hypothesis, not a proven cause. Define a new fixed protocol before fitting; an arm-local feature must use observable landmarks and causal crops, preserve missing-arm abstention, and measure its added cost. Any new supervision must be a separately versioned, reviewed collection with explicit negative and unobservable cases. Keep the frozen references unchanged.

For external match footage, the completed [actor audit](expert-reference-data.md) makes actor-to-label association a prerequisite: one visible boxer can be missing while the opponent is the only detected person. A fixed comparison of detector crops with independently reviewed actor boxes can test that limitation without training or changing the live model. External terms remain unresolved, and match success is not correct-form supervision. Another descriptor or pretrained visual encoder should be selected only with a stated failure mechanism and a new fixed protocol, not a repeated threshold sweep on these results.
