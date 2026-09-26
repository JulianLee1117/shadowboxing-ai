# Next test: two 30-second rounds

This checks hand identity before punch counts. A pose model may place the R wrist on the physical left hand; landmarks alone cannot establish whether lighting, occlusion or another model error caused it. High visibility confidence can still accompany incorrect tracking.

## Keep the lead hand unchanged

Open [Practice](http://127.0.0.1:5173), choose your usual **Lead hand**, and leave the round at **30 seconds**. Use that same lead-hand selection and boxing stance for both rounds. Change your body's angle relative to the camera; do not switch stance just because you turn.

- **Round A:** start with the preferred opposite view. Use a comfortable three-quarter angle that keeps both arms visible; do not turn your back to the camera.
- **Round B:** repeat from the original view for comparison.

For each round, click **Enable camera**, then **Record round**. You have **8 seconds** to step back. Recording starts automatically even if L / R tracking is uncertain. Video and tracking stay local, with no microphone.

## Repeat the same sequence

Once recording starts:

1. Raise your **physical left hand** for about one second, then your **physical right hand** for about one second.
2. Settle into your usual guard.
3. Throw **5 slow, comfortable jabs**, with a brief reset between punches.
4. Throw **5 slow, comfortable crosses**, with a brief reset between punches.
5. Relax for any remaining time.

Keep your normal punch path; do not stretch to satisfy the counter. Note how many punches you actually complete if time runs out.

## Compare video with tracking

When each round finishes, **Review** opens and the camera is released. Confirm **Saved on this device** and that the webcam indicator turns off. For Round B, choose **New round** and repeat without changing **Lead hand**.

Watch the original video first, then enable **Show tracking**. During the opening hand raises, check whether L follows the physical left hand and R the physical right. During crosses, check whether R follows the correct wrist, stays near the other hand, or disappears. Then enable **Show detections** and compare counts with what the video shows. Tracking confidence and the skeleton alone cannot establish correctness.

Under **Export & details**, preserve both **Evidence JSON** and **Export video** for each round. JSON does not include the footage. Keep the two views separate.

## Report back

```text
Lead hand (same in both rounds):
Round A, opposite view — actual jabs/crosses: __/__ ; detected: __/__
Round B, original view — actual jabs/crosses: __/__ ; detected: __/__
Does L/R follow the correct physical hand in each view?
During crosses, does R follow the other hand, stay bent, or disappear?
Both videos saved, replay aligned, and camera off afterward?
```

The coupling bug between arms is fixed, but cross recognition remains unresolved. These two recorded views are a diagnostic comparison, not an accuracy validation.
