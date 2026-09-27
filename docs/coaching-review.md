# Local coaching labels and independent review

## In-app coach cards

Saved original videos now offer **Correct punches**. These collect personal judgments; a participant can identify an imperfect punch without certifying its form. Skilled references and assessed counterexamples are the priority in the [data plan](research/expert-reference-data.md). Confirm physical-hand/action identity; an optional observation then asks about the non-punching hand during an isolated high-guard straight. Progress and resume follow action labels, so a skipped guard question never prevents an action correction from counting as reviewed. A detected event supplies only a navigation marker. **Find a missed punch** adds a manual window from the full video. Hooks, uppercuts, non-punches and uncertain identities receive no automatic form rubric.

Answers persist separately on the session, with source/video hashes, stance, window bounds and proposal provenance. New records attribute the answers to `local-user` with expertise `user_review`, and explicitly mark form judgments `provisional_not_correct_form_references`. The UI supports local playback, notes, Undo and export. Failed writes keep the card uncommitted and offer retry, draft export or discard; concurrent coaching edits are rejected, and annotation writes preserve the latest coaching labels.

New exports use **`local-coach-review-2`**, with `exportState` distinguishing the current review from an unsaved attempt. Existing v1 records remain readable and editable without rewriting their historical `coach_self_reported` authorship; that field never verified expertise. Both versions are selected user-authored clip judgments, not a complete punch inventory, independently validated action boundaries, or a direct input to the CLI schema below. “Can't tell” currently combines visual ambiguity and insufficient evidence; the richer CLI separates them. Mapping to a training dataset requires a deliberate source/context/boundary review. Labels never activate a model automatically. Further independent review is useful before broader corrective-coaching claims; agreement alone is not proven coaching benefit.

## Independent review packets

`ml/coaching.py` prepares a local review packet from human action annotations and the original video. It collects evidence for a draft rubric; it does not evaluate form, issue advice, or turn detector output into ground truth. The first two criteria are `guard_recovery` and `non_punching_hand_guard`, restricted to an independently identified isolated straight in an agreed high-guard drill. See the [criterion specification](research/technique-feedback-spec.md).

Before collecting ratings, a qualified boxing coach must define the intended guard and acceptable variations for that drill. A participant can annotate what action occurred, but that does not certify its technical quality. The packet includes six-punch action identities where labels exist; hooks and uppercuts deliberately have no form criteria in this first rubric. Unknown-hand actions are excluded with an explicit ID/reason list. Stance must be known and constant throughout the source.

## Prepare and review

Keep source video, labels and generated packets under ignored `data/`. The CLI has no external dependencies, downloads or uploads.

```sh
python3 -m ml.coaching prepare data/round-labeled.json \
  --video data/round.webm --output data/reviews/round-001
```

The output contains `packet.json`, `review-template.json` and a short guide. Output directories must be new. Packet targets come exclusively from supplied human action annotations. Predictions, pose coordinates, model scores and automatic judgments are omitted. The video is referenced by filename, byte length and SHA-256; it is not copied. Verify that the operator supplied the correct video: legacy exports do not independently bind their poses to those video bytes, and the tool does not decode video to establish that association.

Give each reviewer the same original video and packet, without the model overlay or another person's ratings. Copy the template to separate files and enter pseudonymous reviewer IDs and self-reported expertise. Every criterion starts `not_reviewed`; no labels are prefilled as correct.

For each action and criterion, record:

- `context`: `eligible`, `not_applicable`, or `unknown`.
- `view`: `adequate`, `inadequate`, or `unknown` for this criterion.
- `judgment`: `pass`, `fail`, `ambiguous`, `unobservable`, `not_applicable`, or `not_reviewed`.
- `outcomeObserved`: whether the criterion's actual outcome was observed.
- `feedbackWarranted`: `yes`, `no`, `uncertain`, or `not_reviewed`. A visible deviation does not automatically warrant interrupting practice.
- `evidence`: `{ "startMs": ..., "endMs": ... }` and a written `rationale` for every reviewed rating.

Evidence times use session milliseconds; video playback time equals session time plus the packet's `videoOffsetMs`. The ±800 ms navigation window is not a recovery deadline. Inspect more of the source whenever necessary and cite an interval anywhere within its duration. If the next purposeful action, occlusion or recording end prevents a judgment, mark that uncertainty instead of inventing a failed return. Neither the detector's event end nor `guardReturn` certifies technique.

```sh
python3 -m ml.coaching check data/reviews/round-001/packet.json \
  data/reviews/round-001/reviewer-a.json
python3 -m ml.coaching compare data/reviews/round-001/packet.json \
  data/reviews/round-001/reviewer-a.json data/reviews/round-001/reviewer-b.json \
  --output data/reviews/round-001/agreement.json
```

`check` requires every template row to remain present, explicit evidence for reviewed judgments, and eligible context/adequate view/observed outcome before pass or fail. A correction marked `yes` requires an independently supported fail. The validation cannot verify whether a reviewer actually watched the video, has coaching expertise, or judged correctly.

`compare` requires distinct reviewer IDs and reports judgment counts, confusion among jointly reviewed ratings, context/view/outcome/feedback disagreements, and agreement/kappa only for mutually decisive pass/fail pairs. It includes decisive coverage, so excluding difficult cases cannot silently appear as complete agreement. Degenerate kappa is `null`, never perfect by default. Agreement is not accuracy or proof that feedback improves boxing. Adjudication remains a separate human task; source reviews are never overwritten.

## Before any live correction

Review the rubric with a qualified coach, collect independent labels and disagreements, then evaluate each proposed cue against adjudicated outcomes on untouched recordings. Include acceptable guard variants, deliberate defensive actions and genuinely hidden hands. Freeze eligibility and abstention rules before evaluating cue precision and coverage. Establish usefulness of the correction separately from whether its underlying movement was visible.

Force, impact, unseen wrist alignment, balance and universal boxing proficiency are outside this rubric. No live technique feedback or voice behavior is enabled by these tools.
