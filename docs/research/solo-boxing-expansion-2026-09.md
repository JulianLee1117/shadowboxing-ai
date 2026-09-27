# Additional solo boxing footage — September 27, 2026

Four additional public-source videos are now acquired locally: **228,926,191 bytes and 22,348 native frames**. Three contain useful candidate solo views from additional apparent performer/source groups. They broaden the [earlier four demonstrations](solo-boxing-data-2026-09.md), but are not yet a labeled six-punch corpus or correct-form references. No inference, training or model selection used these new videos.

The fixed acquisition bound was six clips / 300 MB. Original media, source pages, download lineage, hashes, full native timestamp maps and review artifacts remain ignored in `data/external/solo-boxing-expansion-2026-09-27/`. An independent audit checked the source-specific declarations, media hashes, Commons original-file SHA1 and all 15 HLS segment slices. No earlier footage was downloaded again.

## Source and suitability audit

| Source                                                                                                                                   | Declaration and attribution                                                                                                                             | Candidate footage                                                                                                                                    | Limits                                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Ramsay boxing training](https://www.dvidshub.net/video/940612/b-roll-mig-marine-trains-boxing-match), 2024-10-12                        | Individual PUBLIC DOMAIN declaration; I Marine Expeditionary Force; video by Lance Cpl. Fabian Ortiz; caption names Kymani A. Ramsay.                   | Barehand frontal work at native frames 3288–3720 (54.855–62.062 s); gloved frontal work at 5088–5688 (84.885–94.895 s), with guard/duck transitions. | Multiple exercisers, bags, pads, sparring, cuts and cropped body parts elsewhere. Not every shot is the named boxer. The unacquired alternate camera, DVIDS 940611, is the same actor/day group.                                                                      |
| [Fort Jackson training](https://www.dvidshub.net/video/1015971/fort-jackson-boxing-training-b-roll), 2026-07-22                          | Individual PUBLIC DOMAIN declaration; Nathan Clinebelle / Fort Jackson Public Affairs; caption names Michael Mathews.                                   | Frontal/oblique shadowboxing, slips and guard movement, sampled at frames 360–900 (14.079–32.097 s).                                                 | Edited shots, mirrors, camera motion, blur and a feet-only cut near the end of the sampled window. Exact shot interiors still need annotation.                                                                                                                        |
| [Maïva Hamadouche portrait](<https://commons.wikimedia.org/wiki/File:(_Objectif_Tokyo_2020_)_Portrait_Ma%C3%AFva_Hamadouche.webm>), 2021 | CC BY 3.0 Unported; Saint-Quentin-en-Yvelines – Communauté d’agglomération. Commons records verification of the original YouTube license on 2021-08-21. | Oblique full-body ring shadowboxing sampled at frames 2180–2400 (87.207–96.007 s), including guard and extensions.                                   | Edited portrait, interviews, bags, ropes, subtitles and watermark. Earlier samples show relaxed hands/footwork; guarded action appears around 90.607 s. Some actions are slow/held; publication timestamps do not establish capture speed or sports-form correctness. |
| [USA Boxing practice](https://www.dvidshub.net/video/149230/usa-boxing-practice-with-army-olympic-coaches-b-roll), 2012-07-13            | Individual PUBLIC DOMAIN declaration; Keith Smith / U.S. Army Installation Management Command.                                                          | Acquired and screened, but excluded from the clean solo set.                                                                                         | Mostly multiple athletes/bag work and interviews. Offscreen interview hands cannot supply observable guard negatives.                                                                                                                                                 |

These are **screening windows**, not complete action labels or exact edit boundaries. Dense review used 200 ms samples, which cannot enumerate all fast punches. Right-hook and left-uppercut coverage remains unestablished. A mirror reflection is not a second independently observed athlete; screen position is not anatomical hand identity.

The [DVIDS use notice](https://www.dvidshub.net/about/copyright) retains privacy/publicity, possible third-party interests and non-endorsement restrictions. The Commons file's [CC BY 3.0 terms](https://creativecommons.org/licenses/by/3.0/) require attribution, license information and indication of changes. Neither declaration establishes every depicted-person clearance or validates technique. No agreements were accepted and no media was redistributed.

## Exact media and clocks

| Local source ID     |       Bytes | Native frames | Time base | First / last native PTS (s) |
| ------------------- | ----------: | ------------: | --------- | --------------------------- |
| `ramsay-ortiz`      | 117,252,839 |         5,779 | 1/60000   | 0 / 96.3963                 |
| `fort-jackson-2026` |  36,772,988 |         4,252 | 1/90000   | 2.066733333 / 143.908433333 |
| `maiva-2021`        |  50,571,962 |         5,456 | 1/1000    | .007 / 218.207              |
| `usa-boxing-2012`   |  24,328,402 |         6,861 | 1/2997    | 0 / 228.895562229           |

All maps are complete with increasing PTS and positive durations. These are original **published-media clocks**, not proof of sensor cadence, exposure or absence of retiming. Preserve nonzero origins. The [dataset catalog](../../ml/datasets/catalog-2026-09.json) records each original media SHA256.

Fort Jackson uses the official 720p HLS rendition: 15 unencrypted transport-stream segments concatenated byte-for-byte in playlist order. Each segment's URL, hash and byte offset, the playlist and the full concatenation hash are saved. There was no remux, transcode or clock rewrite. The full 174 MB MP4 was not acquired because it would exceed the aggregate bound; equivalence to its clock is not asserted.

Initial OpenCV frame-count estimates were incorrect for TS/WebM. Those overview attempts failed their count assertions; the failures were retained. Final review used the full FFmpeg native counts and verified complete decoded coverage. No media was redownloaded to repair those estimates.

## Exact shot boundaries within two fixed windows

A subsequent video-only pass reviewed all **762 native frames** in the already declared Fort Jackson 360–900 and Hamadouche 2180–2400 windows. It found three hard cuts, including a second Hamadouche cut missed by sparse screening. This adds continuity references, not punch or form labels:

| Source       | Last frame before / first after cut | Native PTS before / after (s) | Change                                            |
| ------------ | ----------------------------------- | ----------------------------- | ------------------------------------------------- |
| Fort Jackson | 857 / 858                           | 30.661966667 / 30.695333333   | Upper-body view to feet-only view                 |
| Hamadouche   | 2190 / 2191                         | 87.607 / 87.647               | Earlier shot to relaxed footwork                  |
| Hamadouche   | 2263 / 2264                         | 90.527 / 90.567               | Relaxed footwork to separate guarded oblique shot |

Fort Jackson **360–857** and Hamadouche **2264–2400** are candidate continuous interiors for later action/guard annotation. Hamadouche **2191–2263** is separately reviewable relaxed-hand/footwork context, not automatically negative training truth. Fort Jackson **858–900** is excluded from arm-action review because the arms are outside view; it is not a guard-negative interval. Hamadouche **2180–2190** is only a short censored tail.

The fixed scope can start or end inside a shot; it does not certify unreviewed action onset/recovery or earlier warmup. Preserve these limits, reset at selected starts/cuts, and keep all shots from the same performer/session together. No adjacent byte-identical decoded frames were found, but that does not establish sensor FPS or exclude editorial retiming.

Private `data/pilot/external-shot-audit-v1/` preserves protocol, complete reviewed-frame indices/pixel hashes, six full-resolution cut-edge images, source-clock spans and immutable shot manifest SHA256 `e72e68b0ddd777a281cebb450211d666e83a1ed72f368bebab1d6c9a1a020176`. These exact cuts supersede only the earlier approximate brackets within this scope. No model outputs selected the shots.

Independent verification checked source/file hashes, all 762 native time records, complete shot partitions and all six cut-edge images, without repeating the full visual review. Audit SHA256: `6d5da2c0f4ef47d3ddda900c9f1bcae2862c446992f3a05f9d434c71c1b55b09`. The separate `group-aliases.json` maps the manifest's group spellings to the catalog's canonical source events; apply it before any split. These aliases are the same groups, not additional people. Mapping SHA256: `5819c67a222f98f5bee122e2e8c35d75f42ab742869678f5a68771dd2526121f`.

## Annotation work before a model baseline

1. Freeze exact continuity boundaries, actor identity and eligible shot interiors from pixels. All views/encodings of the same performer-day/event stay together. Reserve source groups before prediction.
2. Review every native frame within those interiors for physical hand, family, uncertain transitions and observable non-punch motion. Leave ambiguity unknown. Do not infer negatives from unsampled time, interviews or invisible hands.
3. Preserve terminal ranges where an action has no defensible single peak frame. Action labels, correct-form judgments and joint-localization references are separate tasks.
4. Only then run the unchanged baseline, with declared cut resets and exact timestamps. The [observation replay tool](../observation-replay.md) can diagnose saved native observations without guessing stance or manufacturing jab/cross labels.

Do not choose shots because the current model recognizes them. These additions support a broader fixed evaluation, not a claim of recognition improvement. The next representation study remains conditional on reviewed labels, guard/background coverage and adequate source separation.

Acquisition SHA256: `33c3bb2ac7f00ae1b5b317a2909f380eb170d8c1b2cc8df91693f7efef94e20d`; screening: `43191cd4d4f4388e98cd02b7e006f0efc8c7b1611c55d15455bc1e656a185368`; final completion: `2dd38b6610eda918ff9ce91ad56f2f86a0be5bc2d87790bc46130eb579fcf391`; independent source audit: `ab3146deef366212107e0c4828e0cc7d2445e28b32c7d4eadcd367b74de01758`.

A separate immutable `erratum.json` corrects the earlier Ramsay creator/athlete wording without overwriting those records: Ortiz is the videographer, Ramsay the captioned athlete. Both official camera sources belong to `ramsay-2024-10-12-escondido`. Erratum SHA256: `60818be96c3c2f5c5703c7ee9ce40d1356c080af6ea7075b3e6fbf829105d375`; the table and catalog above apply that correction.
