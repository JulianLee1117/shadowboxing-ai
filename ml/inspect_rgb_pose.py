"""Inspection-only local RGB + pose tensor adapter. No training or identity inference.

Every detected person is retained. Instance indices are frame-local, never tracks.
No annotation side, jersey color, handedness or stance is guessed from an image.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from .extract import (
    configure_detector_score_policy,
    detector_postprocessing,
    load_manifest,
    native_keypoints,
    session_providers,
    sha256,
)

VERSION = "inspection-rgb-pose-native-v2"
RGB_SIZE = 112
MAXIMUM_FRAMES = 300
MAXIMUM_INSTANCES = 1000
MAXIMUM_INSTANCES_PER_FRAME = 32
MAXIMUM_FEATURE_BYTES = 64 * 1024 * 1024
MAXIMUM_DECODED_PIXELS = 4096 * 2160
MAXIMUM_DECODED_FRAMES = 3600
MAXIMUM_DECODE_SECONDS = 120


def pin_inputs(paths):
    return {str(Path(path).resolve()): sha256(Path(path)) for path in paths}


def verify_pinned_inputs(pins):
    for path, expected in pins.items():
        if sha256(Path(path)) != expected:
            raise ValueError("An input or adapter source changed during extraction")


def validate_video_declaration(provenance, digest):
    if "videoSha256" in provenance and provenance["videoSha256"] != digest:
        raise ValueError("Provenance videoSha256 does not match the input video")


def validate_resource_budget(frames, instances):
    """Bound retained tensors; temporary decoder/compression copies need extra RAM."""
    rgb_bytes = RGB_SIZE * RGB_SIZE * 3
    payload_bytes = frames * (rgb_bytes * 3 + 8) + instances * (
        rgb_bytes + 17 * 3 * 4 + 8 * 4 + 4
    )
    if (
        frames > MAXIMUM_FRAMES
        or instances > MAXIMUM_INSTANCES
        or payload_bytes > MAXIMUM_FEATURE_BYTES
    ):
        raise ValueError("Inspection feature limit exceeded; use a shorter interval")
    return payload_bytes


def bounded_box(box, width, height, margin=0.15):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Finite xyxy person box required")
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Person box must have positive area")
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    bounds = (
        max(0, math.floor(x1 - dx)),
        max(0, math.floor(y1 - dy)),
        min(width, math.ceil(x2 + dx)),
        min(height, math.ceil(y2 + dy)),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Person box does not intersect this source frame")
    return bounds


def person_observations(model, pixels, pose_color_order):
    import numpy as np

    boxes = np.asarray(model.det_model(pixels))
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("Detector boxes must be finite Nx4 coordinates")
    if len(boxes) > MAXIMUM_INSTANCES_PER_FRAME:
        raise ValueError(
            "Too many people for bounded inspection; use a tighter source clip"
        )
    if len(boxes) == 0:
        return boxes, np.empty((0, 17, 2)), np.empty((0, 17))
    pose_pixels = pixels[:, :, ::-1] if pose_color_order == "RGB" else pixels
    points, scores = model.pose_model(pose_pixels, bboxes=boxes)
    points, scores = np.asarray(points), np.asarray(scores)
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError("Pose coordinates and native scores must be finite")
    if len(points) != len(boxes) or len(scores) != len(boxes):
        raise ValueError("Each detector box must have exactly one pose output")
    if points.shape[1:] != (17, 2) or scores.shape[1:] != (17,):
        raise ValueError("This fixed feature contract requires COCO17 RTMPose")
    return boxes, points, scores


def rgb_crop(pixels, box):
    import cv2

    height, width = pixels.shape[:2]
    crop = bounded_box(box, width, height)
    x1, y1, x2, y2 = crop
    rgb = cv2.cvtColor(pixels[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    # Actual source pixels resized for an RGB branch, not generated joint data.
    return cv2.resize(rgb, (RGB_SIZE, RGB_SIZE), interpolation=cv2.INTER_AREA), crop


def decode_samples(path, *, start_ms=0, duration_ms=1000, max_frames=30, coverage=None):
    import cv2

    coverage = {} if coverage is None else coverage
    coverage.update(terminationReason="interrupted", intervalComplete=False)
    cap = cv2.VideoCapture(str(path))
    previous = None
    delivered = 0
    decoded_index = -1
    started = time.monotonic()
    try:
        if not cap.isOpened():
            raise ValueError("Could not decode local video")
        while delivered < max_frames:
            if (
                decoded_index + 1 >= MAXIMUM_DECODED_FRAMES
                or time.monotonic() - started > MAXIMUM_DECODE_SECONDS
            ):
                raise ValueError(
                    "Decode work limit exceeded; create a shorter local source clip"
                )
            success, pixels = cap.read()
            if not success:
                coverage["terminationReason"] = "decoder_end_or_read_failure"
                break
            decoded_index += 1
            height, width = pixels.shape[:2]
            if height <= 0 or width <= 0 or height * width > MAXIMUM_DECODED_PIXELS:
                raise ValueError("Decoded image exceeds inspection pixel limit")
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if (
                not math.isfinite(t)
                or t < 0
                or (previous is not None and t <= previous)
            ):
                raise ValueError("Nonmonotonic/missing decoder timestamps")
            previous = t
            if t < start_ms:
                continue
            if t >= start_ms + duration_ms:
                coverage.update(
                    terminationReason="interval_boundary_observed",
                    intervalComplete=True,
                )
                break
            delivered += 1
            yield decoded_index, t, pixels
        else:
            coverage["terminationReason"] = "frame_limit"
        coverage.update(
            decodedFrames=decoded_index + 1,
            deliveredFrames=delivered,
            lastDecodedMs=previous,
            truncatedByFrameLimit=delivered >= max_frames,
        )
    finally:
        cap.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-ms", type=float, default=0)
    parser.add_argument("--duration-ms", type=float, default=1000)
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args(argv)
    if not args.video.is_file() or not args.provenance.is_file():
        raise ValueError("Existing local video and provenance JSON are required")
    if (
        not all(math.isfinite(x) for x in (args.start_ms, args.duration_ms))
        or not 0 <= args.start_ms <= 120_000
        or not 0 < args.duration_ms <= 30_000
        or not 1 <= args.max_frames <= MAXIMUM_FRAMES
    ):
        raise ValueError(
            "Inspection limit: start0..120s, duration<=30s and1..300frames"
        )
    if args.provenance.stat().st_size > 100_000:
        raise ValueError("Inspection provenance exceeds its size limit")
    provenance = json.loads(args.provenance.read_text())
    if (
        provenance.get("purpose") != "inspection-only"
        or not isinstance(provenance.get("sourceGroup"), str)
        or not provenance["sourceGroup"]
    ):
        raise ValueError(
            "Provenance must declare inspection-only purpose and original-recording sourceGroup"
        )
    if args.output.exists():
        raise ValueError("Output exists; choose a new immutable experiment directory")
    pins = pin_inputs(
        [
            args.video,
            args.provenance,
            args.manifest,
            Path(__file__),
            Path(__file__).with_name("extract.py"),
            Path(__file__).with_name("landmarks.py"),
        ]
    )
    validate_video_declaration(provenance, pins[str(args.video.resolve())])
    try:
        import cv2
        import numpy as np
        from rtmlib import RTMPose, YOLOX
    except ImportError as error:
        raise ValueError(
            "Install optional ml/requirements-extract.txt in an isolated environment"
        ) from error
    manifest = load_manifest(args.manifest)
    pins.update(
        {manifest[key]["path"]: manifest[key]["sha256"] for key in ("detector", "pose")}
    )
    if manifest["family"] != "rtmpose-body":
        raise ValueError("Pinned inspection contract supports RTMPose COCO17 only")
    if version("rtmlib") != "0.0.16":
        raise ValueError("This adapter requires pinned rtmlib0.0.16")
    model = SimpleNamespace(
        det_model=YOLOX(
            manifest["detector"]["path"],
            model_input_size=tuple(manifest["detector"]["inputSize"]),
            backend="onnxruntime",
            device="cpu",
        ),
        pose_model=RTMPose(
            manifest["pose"]["path"],
            model_input_size=tuple(manifest["pose"]["inputSize"]),
            to_openpose=False,
            backend="onnxruntime",
            device="cpu",
        ),
    )
    configure_detector_score_policy(model.det_model)
    # Explicit reuse of existing native observation contract; all persons retained.
    model.det_model.score_thr = 0.7
    providers = session_providers(model)
    if any(value != ["CPUExecutionProvider"] for value in providers.values()):
        raise ValueError("Inspection requires explicit CPU sessions")
    args.output.mkdir(parents=True, exist_ok=False)
    scene, deltas, crops, points_all, scores_all = [], [], [], [], []
    frame_index, boxes_all, crop_bounds, timestamps, frame_records = [], [], [], [], []
    previous_rgb = None
    started = time.perf_counter()
    coverage = {}
    samples = decode_samples(
        args.video,
        start_ms=args.start_ms,
        duration_ms=args.duration_ms,
        max_frames=args.max_frames,
        coverage=coverage,
    )
    with closing(samples):
        for index, (decoded_index, t, pixels) in enumerate(samples):
            height, width = pixels.shape[:2]
            if height <= 0 or width <= 0 or height * width > MAXIMUM_DECODED_PIXELS:
                raise ValueError(
                    "Inspection supports positive frames up to4096x2160 pixels"
                )
            validate_resource_budget(len(scene) + 1, len(crops))
            rgb = cv2.resize(
                cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB),
                (RGB_SIZE, RGB_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            delta = (
                np.zeros_like(rgb, dtype=np.int16)
                if previous_rgb is None
                else rgb.astype(np.int16) - previous_rgb.astype(np.int16)
            )
            previous_rgb = rgb
            scene.append(rgb)
            deltas.append(delta)
            timestamps.append(t)
            inference_start = time.perf_counter()
            boxes, points, scores = person_observations(
                model, pixels, manifest["pose"]["inputColorOrder"]
            )
            validate_resource_budget(len(scene), len(crops) + len(boxes))
            instances = []
            for local, (box, xy, score) in enumerate(zip(boxes, points, scores)):
                crop, bounds = rgb_crop(pixels, box)
                crops.append(crop)
                points_all.append(xy / np.array([width, height]))
                scores_all.append(score)
                frame_index.append(index)
                boxes_all.append(
                    np.asarray(box) / np.array([width, height, width, height])
                )
                crop_bounds.append(
                    np.array(bounds) / np.array([width, height, width, height])
                )
                instances.append(
                    {
                        "localIndex": local,
                        "tensorRow": len(crops) - 1,
                        "identity": None,
                        "nativeKeypoints": native_keypoints(xy, score, width, height),
                    }
                )
            # Full source image hash binds the decoder output; no RGB pixels leave disk.
            import hashlib

            frame_records.append(
                {
                    "decodedFrameIndex": decoded_index,
                    "decoderReportedMs": t,
                    "width": width,
                    "height": height,
                    "decodedBgrSha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "instanceCount": len(instances),
                    "instances": instances,
                    "rawDetectorObservations": model.det_model.last_detection_observations,
                    "observationExtractionMs": (time.perf_counter() - inference_start)
                    * 1000,
                }
            )
            if index in (0, args.max_frames // 2, args.max_frames - 1):
                cv2.imwrite(str(args.output / f"source-{index:04d}.jpg"), pixels)
    if not scene:
        raise ValueError("No decoded samples in requested inspection interval")
    verify_pinned_inputs(pins)
    np.savez_compressed(
        args.output / "features.npz",
        decoder_ms=np.asarray(timestamps, dtype=np.float64),
        scene_rgb=np.asarray(scene, dtype=np.uint8),
        causal_scene_delta=np.asarray(deltas, dtype=np.int16),
        instance_frame_index=np.asarray(frame_index, dtype=np.int32),
        person_rgb=np.asarray(crops, dtype=np.uint8).reshape(-1, RGB_SIZE, RGB_SIZE, 3),
        native_xy=np.asarray(points_all, dtype=np.float32).reshape(-1, 17, 2),
        native_score=np.asarray(scores_all, dtype=np.float32).reshape(-1, 17),
        bbox_xyxy=np.asarray(boxes_all, dtype=np.float32).reshape(-1, 4),
        crop_xyxy=np.asarray(crop_bounds, dtype=np.float32).reshape(-1, 4),
    )
    post = detector_postprocessing(model)
    post["personSelection"] = "all_above_threshold_unassigned_frame_local_instances"
    output = {
        "version": VERSION,
        "purpose": "inspection-only",
        "trainingEligible": False,
        "missingBeforeTraining": [
            "verified media training rights",
            "reviewed person identity mapping across frames",
            "verified annotation indexing and frame-to-time mapping",
            "recording/subject grouped split",
            "explicit background and unknown scope",
        ],
        "input": {
            "videoSha256": pins[str(args.video.resolve())],
            "videoFile": str(args.video.resolve()),
            "provenanceSha256": pins[str(args.provenance.resolve())],
            "provenance": provenance,
            "requestedStartMs": args.start_ms,
            "requestedDurationMs": args.duration_ms,
            "maximumFrames": args.max_frames,
        },
        "timestampPolicy": "OpenCV CAP_PROP_POS_MSEC strictly increasing plus zero-based decoded frame index; container/native PTS origin not independently verified. No CFR-index estimate, seek, interpolation or mirror",
        "modelManifest": manifest,
        "configuredProviders": providers,
        "detectorPostprocessing": post,
        "resourceLimits": {
            "maximumFrames": MAXIMUM_FRAMES,
            "maximumDecodedFramesIncludingPreroll": MAXIMUM_DECODED_FRAMES,
            "maximumDecodeLoopSeconds": MAXIMUM_DECODE_SECONDS,
            "maximumDecodedPixels": MAXIMUM_DECODED_PIXELS,
            "maximumTotalInstances": MAXIMUM_INSTANCES,
            "maximumInstancesPerFrame": MAXIMUM_INSTANCES_PER_FRAME,
            "maximumRetainedTensorBytes": MAXIMUM_FEATURE_BYTES,
            "memoryMeaning": "Numeric payload ceiling, not process RSS; decoder/compression and arrays have bounded extra copies",
        },
        "featureContract": {
            "rgbOrder": "RGB",
            "rgbSize": [RGB_SIZE, RGB_SIZE],
            "personCropMarginFraction": 0.15,
            "nativePose": "COCO17 normalized to full image dimensions, raw native scores unclamped",
            "identity": "none: local box order never represents tracks or annotated boxers",
            "causalDelta": "current minus previous sampled scene pixels; firstdelta0; decoder_ms retained for time normalization",
            "labelsIncluded": False,
        },
        "scriptSha256": pins[str(Path(__file__).resolve())],
        "pinnedInputHashes": pins,
        "coverage": {
            **coverage,
            "firstDeliveredDecoderMs": timestamps[0],
            "lastDeliveredDecoderMs": timestamps[-1],
        },
        "featuresSha256": sha256(args.output / "features.npz"),
        "frames": frame_records,
        "packages": {
            name: version(name)
            for name in ("rtmlib", "numpy", "onnxruntime", "opencv-python")
        },
        "extractionWallSecondsExcludingModelLoad": time.perf_counter() - started,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    verify_pinned_inputs(pins)
    (args.output / "inspection.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "frames": len(scene),
                "unassignedInstances": len(crops),
                "featuresSha256": output["featuresSha256"],
                "trainingEligible": False,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
