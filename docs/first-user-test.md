# Next test: one natural 30-second round

First try **Recheck detections** on an existing round to compare the updated rules using exactly the same saved tracking. **Use saved detections** restores the original result. This does not fix a misplaced wrist or alter the recording.

For the next live check, use one fresh recording after the update. There is no target detection count to satisfy; keep your normal stance and movement.

## Record

Open [Practice](http://127.0.0.1:5173), keep **30 seconds**, and choose your usual **Lead hand**. Leave **Practice focus** on free practice for this mixed test. Keep your head, hips and extended hands in view, with light in front of you where practical.

Click **Enable camera**, optionally turn on **Focus view**, then **Record round** and step back during the eight-second countdown. No second click is needed. Check that the timer, punch name and count are readable from your stance. Tracking warnings should remain visible during recording.

- Try two comfortable examples each of **jab, cross, lead hook, rear hook, lead uppercut and rear uppercut**, with a brief natural reset between actions. Stop the sequence early if needed; this is not a required count.
- If time permits, add a few faster punches or a short mixed sequence at your usual pace. Note which names appear, which punches are missed, and whether either arm becomes uncertain.
- Spend the last few seconds without punching: relax and lower your hands, then return to guard. Check for false counts during this portion.

This is a short sample, not a rigid script. Note extra, missed or interrupted actions rather than assuming the intended sequence happened. On a later day, repeat one similar round with the version held fixed; the earlier debugging footage cannot establish fresh-session performance.

## Review

Confirm the camera indicator turns off and **Saved on this device** appears. Watch the original video first; **0.5×** and frame stepping help inspect fast movement. Then use **Show tracking** to check whether L/R follows your physical hands, and **Show detections** to inspect singles and combinations. Uncertainty flags identify evidence to inspect, not proven mistakes or technique errors.

A separate local video analysis starts for the new completed recording. You can watch while it runs or choose **Cancel analysis**. **Original results stay selected**; when ready, use **Show video analysis** and **Show original** to compare. Different counts do not automatically mean better counts. Older saved recordings offer **Analyze recording** on demand. Motion-only rounds can use **Recheck detections** when available; that reuses saved poses.

## Report back

A short note is enough:

```text
Were the video, timer and live punch names easy to see from your stance?
Which punch types were missed or mislabeled, and roughly when:
Did tracking follow the correct hands during overlap or fast punches?
Any counts during the idle ending?
Video saved, camera off, and local analysis completed or canceled?
```

Keep **Evidence JSON** and **Export video** together for a backup; **Export video analysis** saves the separate derived report. Everything stays local. The four experimental combination labels and their limitations are described in [recognition events](recognition-events.md); technique critique still requires separate coach-reviewed evidence.
