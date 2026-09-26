"""Conservative COCO body-to-MediaPipe index mapping (not a 3D conversion)."""
import math

# COCO body order is shared by the first 17 COCO-WholeBody points when
# rtmlib's to_openpose=False. Eyes/ears are deliberately not approximated.
COCO_TO_MEDIAPIPE = {0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
                     11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}


def map_body_landmarks(keypoints, scores, width: int, height: int) -> list[dict]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if len(keypoints) not in (17, 133) or len(scores) != len(keypoints):
        raise ValueError("Expected exactly 17 COCO or 133 COCO-WholeBody points; OpenPose order is unsupported")
    landmarks = [{"x": 0.0, "y": 0.0, "visibility": 0.0, "presence": 0.0} for _ in range(33)]
    for source, target in COCO_TO_MEDIAPIPE.items():
        x, y = float(keypoints[source][0]), float(keypoints[source][1])
        score = float(scores[source])
        if not all(math.isfinite(v) for v in (x, y, score)):
            continue
        # Store raw score as visibility for schema interoperability only; its
        # meaning is model keypoint confidence, not calibrated MP visibility.
        landmarks[target] = {"x": x / width, "y": y / height,
                             "visibility": max(0.0, min(1.0, score))}
    return landmarks
