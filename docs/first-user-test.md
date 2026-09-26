# First test: about five minutes

This checks camera setup, experimental punch detection, and replay. We are not evaluating technique critique yet. Missed detections do not mean your technique is wrong.

## 1. Set up

Open **Training studio → Practice**, leave **Free practice** and the **1-minute** duration selected, then click **Enable camera**. Choose your stance: orthodox means your left hand leads; southpaw means your right hand leads.

Keep your head, hips, elbows, and hands visible, including your full reach. Raise your physical left hand and confirm that the overlay’s **L** follows it, even with a mirrored preview. Confirm the checkbox, then click **Use this setup**.

Turn on **Save round video** for this test so you can inspect the actual recording. It stays on this device. Leave **More options** alone.

## 2. Perform one short round

Click **Start round**:

1. Hold your guard without punching for **10 seconds**. Target: no punch detections.
2. Throw **5 comfortable jabs**.
3. Throw **5 comfortable crosses**.
4. Throw **3 jab–cross combinations**.

Pause about two seconds between repetitions or combinations. Keep your normal technique; do not change it just to make the counter respond. Click **Finish round** when done, or let the timer finish.

Target: **8 jabs and 8 crosses**, matching your performed punches. Detection accuracy remains unverified.

## 3. Check the result

Confirm the round saved locally. Click **Stop camera** and verify that the webcam indicator turns off. Open **Round review**, select the round, and use **Show experimental detections** to inspect its counts and events.

Play and scrub the video. Check that the skeleton follows the visible movement and disappears when tracking has a substantial gap.

Only if punches toward the camera hide an arm, repeat from a slightly angled position with both hands visible. Report the two views separately.

## Report back

```text
Stance and camera angle:
Detected jabs / crosses: __ / __ (performed 8 / 8)
False detections during the first 10 seconds:
Left/right labels correct? Any tracking loss or noticeable lag?
Video saved and replay aligned?
Camera indicator off after Stop camera?
One confusing or broken thing:
```
