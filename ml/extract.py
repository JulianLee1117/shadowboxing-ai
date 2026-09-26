"""Optional RTMPose/RTMW saved-video extraction with explicit local model provenance.

This module never downloads weights. It requires local ONNX files and a manifest.
Run ``python3 -m ml.extract --help``; install ml/requirements-extract.txt separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

from .evaluate import distribution
from .landmarks import COCO_TO_MEDIAPIPE, map_body_landmarks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("family") not in ("rtmpose-body", "rtmw-wholebody"):
        raise ValueError("Manifest family must be rtmpose-body or rtmw-wholebody")
    for key in ("detector", "pose"):
        item = manifest.get(key, {})
        local_path = item.get("path")
        if not isinstance(local_path, str) or "://" in local_path:
            raise ValueError(f"{key}.path must point to a local ONNX file; automatic downloads are disabled")
        model_path = (path.parent / local_path).resolve()
        if not model_path.is_file() or model_path.suffix.lower() != ".onnx":
            raise ValueError(f"Missing local {key} ONNX file: {model_path}")
        if not isinstance(item.get("sourceUrl"), str) or not item["sourceUrl"].startswith("https://"):
            raise ValueError(f"Record the original HTTPS artifact URL in {key}.sourceUrl")
        if not isinstance(item.get("inputSize"), list) or len(item["inputSize"]) != 2 or any(type(x) is not int or x <= 0 for x in item["inputSize"]):
            raise ValueError(f"{key}.inputSize must be [width, height]")
        digest = sha256(model_path)
        if item.get("expectedSha256") and digest != item["expectedSha256"]:
            raise ValueError(f"{key} SHA-256 does not match the manifest")
        item["path"] = str(model_path)
        item["sha256"] = digest
        item["sha256Meaning"] = "local_file_fingerprint_not_independent_authenticity_verification"
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assume-cfr", action="store_true", help="Explicitly use decoded frame index / reported FPS when the source is known constant-frame-rate")
    parser.add_argument("--max-frames", type=int, help="Smoke-test only; the output is marked truncated")
    args = parser.parse_args(argv)
    cap = None
    try:
        manifest = load_manifest(args.manifest)
        if not args.video.is_file():
            raise ValueError(f"Video does not exist: {args.video}")
        protected = [args.video, args.manifest, *(Path(manifest[k]["path"]) for k in ("detector", "pose"))]
        if args.output.resolve() in [p.resolve() for p in protected]:
            raise ValueError("Output must not overwrite an input artifact")
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        try:
            import cv2
            from rtmlib import Body, Wholebody
        except ImportError as error:
            raise ValueError("Install the optional pinned ml/requirements-extract.txt into an isolated environment first") from error
        installed = {name: version(name) for name in ("rtmlib", "numpy", "opencv-python", "opencv-contrib-python", "onnxruntime")}
        if installed["rtmlib"] != "0.0.16":
            raise ValueError("This adapter targets rtmlib==0.0.16; use the pinned environment")
        solution = Body if manifest["family"] == "rtmpose-body" else Wholebody
        model = solution(det=manifest["detector"]["path"],
                         det_input_size=tuple(manifest["detector"]["inputSize"]),
                         pose=manifest["pose"]["path"],
                         pose_input_size=tuple(manifest["pose"]["inputSize"]),
                         to_openpose=False, backend="onnxruntime", device="cpu")
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise ValueError("OpenCV could not decode this video")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if args.assume_cfr and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("Cannot use --assume-cfr: source FPS is unavailable")
        frames = []
        counts = {"zeroPeople": 0, "multiplePeople": 0, "singlePerson": 0}
        raw_origin = None
        previous_t = None
        while args.max_frames is None or len(frames) < args.max_frames:
            success, pixels = cap.read()
            if not success:
                break
            if args.assume_cfr:
                timestamp = len(frames) * 1000 / fps
            else:
                raw_t = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if not math.isfinite(raw_t) or raw_t < 0:
                    raise ValueError("Decoder timestamp unavailable; verify a CFR source before explicitly using --assume-cfr")
                if raw_origin is None:
                    raw_origin = raw_t
                timestamp = raw_t - raw_origin
                if previous_t is not None and timestamp <= previous_t:
                    raise ValueError("Decoder timestamps are not strictly increasing; verify a CFR source before explicitly using --assume-cfr")
            previous_t = timestamp
            height, width = pixels.shape[:2]
            started = time.perf_counter()
            keypoints, keypoint_scores = model(pixels)
            inference_ms = (time.perf_counter() - started) * 1000
            person_count = len(keypoints)
            # No identity guessing: this single-person benchmark abstains when
            # the detector produces multiple people, including in mirrors.
            landmarks = []
            if person_count == 1:
                expected_points = 17 if manifest["family"] == "rtmpose-body" else 133
                if len(keypoints[0]) != expected_points:
                    raise ValueError(f"Model returned {len(keypoints[0])} points; manifest expects {expected_points}")
                landmarks = map_body_landmarks(keypoints[0], keypoint_scores[0], width, height)
                counts["singlePerson"] += 1
            else:
                counts["zeroPeople" if person_count == 0 else "multiplePeople"] += 1
            frames.append({"t": timestamp, "width": width, "height": height,
                           "landmarks": landmarks, "inferenceMs": inference_ms})
            if len(frames) % 100 == 0:
                print(f"Processed {len(frames)} frames", file=sys.stderr)
        if not frames:
            raise ValueError("No video frames decoded")
        gaps = [b["t"] - a["t"] for a, b in zip(frames, frames[1:])]
        duration = frames[-1]["t"] + (statistics.median(gaps) if gaps else 0)
        result = {"artifactType": "research-pose-series", "schemaVersion": "1.0",
                  "createdAt": datetime.now(timezone.utc).isoformat(),
                  "source": {"path": str(args.video.resolve()), "sha256": sha256(args.video)},
                  "modelManifest": manifest, "runtime": {"packages": installed,
                      "python": platform.python_version(), "platform": platform.platform(),
                      "backend": "onnxruntime", "deviceRequested": "cpu"},
                  "timestampMode": "assumed_cfr_frame_index" if args.assume_cfr else "opencv_pos_msec_relative_to_first_frame",
                  "durationMsEstimate": duration, "lastFrameDurationIsEstimated": True,
                  "truncatedByFrameLimit": args.max_frames is not None and len(frames) == args.max_frames,
                  "mapping": {"inputOrder": "COCO17" if manifest["family"] == "rtmpose-body" else "COCO_WholeBody133",
                              "outputOrder": "MediaPipe33_partial_body_only",
                              "availableIndices": sorted(COCO_TO_MEDIAPIPE.values()),
                              "missingPointRepresentation": "zero_coordinate_and_zero_confidence",
                              "confidenceSemantics": "RTM_keypoint_score_not_calibrated_MediaPipe_visibility",
                              "depth": "unavailable_no_z_or_world_landmarks"},
                  "personCounts": counts, "inferenceMs": distribution([f["inferenceMs"] for f in frames]),
                  "frames": frames,
                  "limitations": ["Pose extraction only: no punch recognition, coaching, or camera validation.",
                                  "This artifact is not a browser Session; do not relabel model as full/lite/heavy.",
                                  "Detector-plus-pose timing includes first-call warm-up; excludes video decoding.",
                                  "CPU requested; no Core ML, MPS, browser, or real-time claim."]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {len(frames)} frames to {args.output}")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 2
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
