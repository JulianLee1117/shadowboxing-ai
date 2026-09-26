# Next test: one fresh one-minute round

Use a new recording after the detector update. The earlier clips helped find bugs; recounting them cannot establish performance on new movement.

## Setup

Open [Practice](http://127.0.0.1:5173), keep your usual **Lead hand**, and choose **1 minute**. Use the view with your chest partly facing the camera and both arms visible. Keep your head, hips and extended hands in frame throughout the round. Put the main light in front of you where practical, rather than a bright window behind you. Keep your normal stance and punch path.

Click **Enable camera**, then **Record round**. The eight-second countdown gives you time to step back. Recording starts automatically; you do not need to reach the laptop again.

## Sequence

1. Raise your physical left hand, then your right, so their identity is clear in the video.
2. Throw **3 slow jabs**, then **3 slow crosses**, with a brief reset between punches.
3. Throw **3 jab–cross pairs** at a comfortable normal speed, resetting between pairs.
4. Throw **3 jab–jab–cross sequences**, resetting between sequences but keeping the two jabs linked naturally.
5. Spend the remaining time in guard, shifting position and relaxing your hands without throwing a punch.

If you complete the sequence, the reference count is **12 jabs and 9 crosses**. Note any extra, omitted or interrupted action rather than assuming the intended count is the actual count. Faster means your comfortable normal speed; there is no need to alter your motion to satisfy the counter.

## Review

Let the round finish. Confirm **Saved on this device** and that the webcam indicator turns off. Watch the original video first, using **0.5×** speed or the frame buttons when useful. Then enable **Show tracking** to check whether L/R follows the correct physical hands, including when they overlap. Finally enable **Show detections** and check the separate punches, linked sequences and idle interval.

For an older saved round, **Recheck detections** compares the current counting rules with its saved results. It uses existing tracking and cannot fix a pose error. **Use saved detections** restores the original results. Neither action overwrites the video or saved evidence.

## Report back

```text
Completed all sequences, or actual changes:
Detected jabs / crosses:
Which sequence missed or added punches:
Did L/R stay on the correct physical hands?
Any counts during the final idle interval?
Video saved and camera off afterward?
```

Video and tracking remain local. Keep both **Evidence JSON** and **Export video** if making a backup. These are recognition checks; technique critique still needs separate coach-reviewed evidence.
