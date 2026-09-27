# Next test: live delivery, punch history and original replay

## 1. Check feedback from boxing distance

Open [Practice](http://127.0.0.1:5173), choose your lead hand, keep **30s**, and enable the camera. Tracking is on: blue means the estimated physical left hand, orange the right. **Record round** gives you eight seconds to step back; no second click is needed.

Throw a few comfortable singles, repeated jabs, a short combination, hooks and uppercuts. The **Detected / Newest first** panel keeps recent detections, including repeated punches. Each row has its hand, number/name, detection sequence and source time. A **+N** accent marks an arriving batch; names appear together. Pause briefly: the history should stay readable. Tracking warnings should not erase it.

Leave the last few seconds without punching. Look for false counts during guard changes or hand lowering. **Stop & save**, or let the timer end. The camera turns off and original video opens immediately; no second analysis is required.

Check these three things:

- Can you read the latest names from your stance without moving toward the laptop?
- Do consecutive punches get distinct rows quickly enough to follow?
- Do physical L/R match your hands, and which movements were missed or counted incorrectly?

In the saved round's **Export & details**, note **Video delivery rate** and **Pose rate**. The first measures delivered video callbacks; the second measures processed tracking observations. Older rounds or unsupported browsers may have no delivery measurement. Keep this separate from recognition: smoother tracking does not prove that more punches were correctly identified. The evidence export preserves frame-age timing for diagnosis. One ordinary 20–30 second mixed round at your usual practice pace, with a short guard-only finish, is enough for this check.

## 2. Optionally correct a few existing clips

Your recorded technique is not the reference standard. Imperfect punches still help test recognition; skilled external examples and assessed counterexamples are the priority for learning form. You do not need to label your own technique to unblock that work. See the [data plan](research/expert-reference-data.md).

In **Saved rounds**, choose a recording and click **Correct punches**. The original video appears without prediction overlays.

1. Play the clip, use **0.5×** if useful, and inspect the marked moment. Confirm the physical hand and action, or choose **Not a punch** / **Can't tell**.
2. A straight offers an **optional** observation about the **other hand**: **Guard held**, **Needs work**, **Can't tell**, or **Not this drill/style**. Use **Next** to skip it; your action label is already saved. The first two mean an isolated high-guard straight with the hand clearly visible. A combination, intentional defense or different guard can be not applicable.
3. Use **Undo** for a misclick. **Next** can leave a card unanswered. Answers save locally; reload once to check they remain.
4. Use **Find a missed punch**, scrub the full video, and **Review this moment** to add something the detector missed. Adding a window does not label it automatically.

If you want to help correct recognition, start with 10–15 useful examples. Don't force guard judgments on mixed-action clips or assume a correctly named punch has correct form. No new technique demonstration is required from you.

Labels are supervised review material, not an automatic model update or a technique score. Proposal windows are not verified action boundaries or a complete punch inventory. No video is uploaded.

## Replay and backup

Closing punch review returns to the original replay with tracking available. **Re-run analysis** remains optional; different counts alone do not prove improvement. Saved-round remove buttons support **Undo** and **Recently deleted**.

**Export labels** saves coaching judgments. Under the normal review's **Export & details**, keep **Evidence JSON** and **Export video** together for a backup. Browser storage stays local and can be cleared by the browser.
