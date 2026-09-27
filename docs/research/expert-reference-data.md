# Skilled references, imperfect punches, and webcam evaluation

Updated September 27, 2026 after the user clarified that their recorded technique should not define the standard. External boxing footage is a priority for the next data iteration. Personal recordings remain useful, but collecting more of one person's repetitions is not a substitute for skilled examples or independently assessed technique.

## Three different roles for footage

| Collection                               | What it teaches or tests                                                                     | Required labels                                                                                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Varied boxing actions                    | Recognize six punches across skilled and imperfect execution, combinations, views and people | Physical hand, family, stance/role, event timing, visible evidence and genuine non-punch intervals                                                |
| Technique references and counterexamples | Judge one specific form criterion in its intended drill                                      | Skilled demonstrations, acceptable variations, mistakes, context, visibility and the correction warranted; no whole-clip assumption of perfection |
| Actual laptop rounds                     | Measure transfer, false counts, tracking failures and useful feedback in the supported setup | Independent action references; technique judgments only when reviewed explicitly; untouched later-day/person evaluation                           |

A poorly executed cross is still a cross. A professional boxer can make a mistake or deliberately change guard. A successful punch in a match does not certify beginner shadowboxing form. Keep action identity, execution quality, and tactical context separate. Never mark an entire user's recording as bad, or an entire professional recording as correct.

The existing swiping/card UI is a labeling tool for any suitable imported recording. The user's own cards are optional recognition corrections and personal judgments, not the required source of expert technique. They do not train a model automatically.

## External sources checked

The [source catalog](../../ml/datasets/catalog-2026-09.json) and [acquisition audit](boxing-datasets-2026-09.md) retain detailed access findings.

- **BoxingVI:** priority six-punch recognition source. Its [author release](https://github.com/Bikudebug/BoxingVI) describes intervals and estimated poses; it does not label correct versus faulty execution. Resolve original video availability and terms before an RGB training import.
- **BoxingWeb / BoxMind:** a new [official repository](https://github.com/gouba2333/BoxingWeb) links a public package with video and event annotations. The public listing was inspected: one `boxingweb.rar` archive, 5,190,972,710 bytes, with no separately downloadable small sample. The archive was not acquired in this bounded inspection. It is a strong match-recognition lead; its referenced LICENSE is absent and source-media training terms remain unverified. Hand/family/target/effectiveness labels are not a solo form rubric.
- **ShadowPunch:** promising fast-action source with physical-hand labels. Its [paper](https://openreview.net/pdf?id=Jq8HYNZG9s) is relevant, but a usable release remains unverified. It cannot be counted as acquired training data.
- **CoachMe BX:** additional form-feedback lead. The [official repository](https://github.com/MotionXperts/MotionExpert) publishes jab/cross JSON annotations and estimated 22-joint pose pickles, including reference motions. A pinned JSON inspection found 163 training and 41 test entries, 612 original comments and 612 augmented comments. No raw RGB appears in the published dataset directory. The repository declares Apache-2.0; this does not establish access to absent source videos. The [paper](https://arxiv.org/html/2509.11698v1) identifies college boxing-team annotators, a 60 fps collection, and instruction-generation evaluation. This is a useful starting reference, not validated webcam coaching or six-punch coverage. Augmented wording is not independent evidence; the JSON lacks explicit participant IDs, so person separation is unverified.
- **Boxing jab skeleton dataset:** the [author's Kaggle metadata](https://www.kaggle.com/api/v1/datasets/view/denuwang/boxing-jab-skeleton-dataset) reports `licenseName: Unknown`. Do not rely on another paper's accuracy claim as evidence of reusable, representative form data.

Only CoachMe's public JSON metadata/labels, README and license were acquired in this update, under ignored `data/external/coachme-bx-audit-2026-09-27/`, pinned to revision `450375b593b1466de201fe5885ebf2916604d382` with file hashes and a derived count audit. No RGB footage was downloaded, no pickle executed, and no model trained or activated. Some comments discuss properties the current webcam cannot substantiate; preserve their provenance rather than copying them directly into live advice.

## Recommended first collection

Use clear, continuous solo drills from several skilled boxers as the reference foundation. Prefer front and three-quarter views with the full relevant body visible; retain original speed, source timestamps and actual frame rate. Edited tutorials can inform the rubric, but demonstrations must be separated from talking, cuts, slow motion, and intentionally incorrect examples. Match highlights are a later robustness source because opponents, occlusion and tactical choices complicate form labels.

Useful teaching references include [England Boxing's coach-led shadowboxing series](https://www.englandboxing.org/news_articles/england-boxing-shadow-boxing-drills-week-seve/) and [John Pullman's jab lesson](https://pullmansboxing.com/videos/blog/2021/2/12/episode-2-the-jab-boxing-training-technique-amp-drills). These are viewing/rubric references; a reusable video grant was not verified. The creator-owned [Commons heavy-bag analysis](https://commons.wikimedia.org/wiki/File:Boxing_Heavy_Bag_Analysis.webm) explicitly declares CC BY 4.0, but is an edited slow-motion analysis with no established expert form labels. It can be a localization experiment, not the requested technique standard or native-speed timing benchmark.

Start with a deliberately small pilot: **3–5 skilled demonstrators, all six punches, about 10 isolated repetitions per punch and several short natural combinations per person**, across supported views. This is an engineering starting point, not a statistically sufficient training set. Include ordinary guard adjustments, feints and arm lowering. At least one separate person/day is reserved before extracting clips; views, adjacent repetitions and augmentations of a source stay in the same split.

For the first form criterion, collect isolated high-guard straights with the other hand clearly visible. Pair acceptable examples with naturally occurring or safely demonstrated guard deviations, acceptable variations, and unobservable cases. Label the specific evidence and whether a correction would be useful. More clean repetitions alone cannot teach the decision boundary between a mistake, a different style and a hidden hand.

Use an existing creator-owned collection with suitable terms where available. If the needed source videos are unavailable, a small consented recording session with skilled boxers is the proposed fallback; no outreach, booking or payment has been made. The acquisition brief is the same for both routes, so finding more public links does not become the entire project.

## Next experiment

1. Acquire and inventory one usable external RGB source, retaining creator, terms, person/session grouping and timing. Inspect pixels and annotations before a full import.
2. Run the existing pose pipeline on that footage. Audit wrist/elbow localization on hooks and uppercuts; source poses from a different model are not interchangeable ground truth.
3. Compare a frozen existing-data baseline against external-data-plus-local-development training. Report six-punch events, hand errors, non-punch false counts and latency on separate held-out laptop rounds. Keep form judgments out of the action-class target.
4. In parallel, use reference/counterexample judgments to validate the narrow guard criterion. Gate any live cue on observable evidence and criterion performance, not similarity to one champion's skeleton.

The immediate request to the user is ordinary product testing when a candidate is ready. They do not need to record perfect technique or manually create the entire coaching dataset.
