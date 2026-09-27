"""Optional RTMPose/RTMW saved-video extraction with explicit local model provenance.

This module never downloads weights. It requires local ONNX files and a manifest.
Run ``python3 -m ml.extract --help``; install ml/requirements-extract.txt separately.
CPU is the default. ``--provider coreml`` opts into Core ML with explicit CPU
node fallback and saves execution profiles beside the output. Provider selection
does not establish hardware acceleration, numerical parity, or live throughput.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from types import SimpleNamespace

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
        if key == "pose" and item.get("inputColorOrder", "BGR") not in ("RGB", "BGR"):
            raise ValueError("pose.inputColorOrder must be RGB or BGR")
        digest = sha256(model_path)
        if item.get("expectedSha256") and digest != item["expectedSha256"]:
            raise ValueError(f"{key} SHA-256 does not match the manifest")
        item["path"] = str(model_path)
        item["sha256"] = digest
        item["sha256Meaning"] = "local_file_fingerprint_not_independent_authenticity_verification"
    return manifest


def load_frame_manifest(path: Path) -> tuple[dict, list[tuple[Path, float]]]:
    """Read an ordered, local decoded-image series without inventing cadence."""
    manifest = json.loads(path.read_text())
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Frame manifest must contain a nonempty frames array")
    frames = []
    previous = -1.0
    for item in entries:
        name, timestamp = item.get("file"), item.get("ptsMs")
        if not isinstance(name, str) or "://" in name:
            raise ValueError("Frame file must be a local path")
        image_path = (path.parent / name).resolve()
        if not image_path.is_relative_to(path.parent.resolve()) or not image_path.is_file():
            raise ValueError("Frame file must exist inside the manifest directory")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
            raise ValueError("Frame ptsMs must be finite, nonnegative and strictly increasing")
        previous = timestamp
        frames.append((image_path, float(timestamp)))
    return manifest, frames


def infer_single_person(model, pixels, pose_color_order="BGR"):
    """Do not allow rtmlib's no-box fallback to fabricate a detected person."""
    boxes = model.det_model(pixels)
    if len(boxes) != 1:
        return len(boxes), None, None
    # rtmlib 0.0.16 normalizes channels without reordering. Some exported model
    # pipeline.json files specify to_rgb=true. Make that contract explicit;
    # never change detector color order along with the pose crop.
    pose_pixels = pixels[:, :, ::-1] if pose_color_order == "RGB" else pixels
    points, scores = model.pose_model(pose_pixels, bboxes=boxes)
    if len(points) != 1 or len(scores) != 1:
        raise ValueError("Pose output must contain exactly one detected person")
    return 1, points[0], scores[0]


def session_providers(model) -> dict:
    return {key: getattr(model, attribute).session.get_providers()
            for key, attribute in (("detector", "det_model"), ("pose", "pose_model"))}


def provider_configuration(requested: str, available: list[str]) -> dict:
    """Validate intent before constructing sessions; never alias Core ML to CPU."""
    if requested not in ("cpu", "coreml"):
        raise ValueError("Provider must be cpu or coreml")
    providers = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                 if requested == "coreml" else ["CPUExecutionProvider"])
    missing = [name for name in providers if name not in available]
    if missing:
        raise ValueError(f"Requested provider unavailable: {', '.join(missing)}")
    return {"requested": requested, "providers": providers,
            "providerOptions": [{} for _ in providers],
            "computeConfiguration": "provider_defaults_no_compute_unit_override",
            "hardwareExecution": "not_established_by_provider_selection"}


def configure_coreml_sessions(model, manifest: dict, ort, configuration: dict,
                              profile_directory: Path) -> dict:
    """Use explicit ONNX Runtime sessions rather than rtmlib's fallback alias."""
    creation_ms = {}
    for key, attribute in (("detector", "det_model"), ("pose", "pose_model")):
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str((profile_directory / key).resolve())
        started = time.perf_counter()
        session = ort.InferenceSession(
            manifest[key]["path"], sess_options=options,
            providers=configuration["providers"],
            provider_options=configuration["providerOptions"])
        # CPU nodes in a partitioned Core ML graph are intentional. A runtime
        # error must not silently rebuild/retry the whole session on CPU.
        session.disable_fallback()
        getattr(model, attribute).session = session
        creation_ms[key] = (time.perf_counter() - started) * 1000
        if session.get_providers() != configuration["providers"]:
            raise ValueError(f"{key} session did not register the requested providers: "
                             f"{session.get_providers()}")
    return creation_ms


def summarize_execution_profile(path: Path) -> dict:
    """Count observed provider events, not assumed device placement or FLOPs."""
    events = json.loads(path.read_text())
    if not isinstance(events, list):
        raise ValueError("ONNX Runtime execution profile must contain an event list")
    providers = Counter()
    operators = Counter()
    for event in events:
        if not isinstance(event, dict):
            continue
        args = event.get("args", {})
        if not isinstance(args, dict):
            continue
        provider = args.get("provider")
        if event.get("cat") == "Node" and isinstance(provider, str) and provider:
            providers[provider] += 1
            operators[f"{provider}:{args.get('op_name', 'unknown')}"] += 1
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "executedNodeProviderEvents": dict(providers),
            "providerOperatorEvents": dict(operators),
            "interpretation": "Fused partition events are not operation counts, "
                              "percent compute, specific Apple hardware, or a speed claim."}


def verify_coreml_execution(profiles: dict) -> None:
    if not any(profile["executedNodeProviderEvents"].get("CoreMLExecutionProvider", 0) > 0
               for profile in profiles.values()):
        raise ValueError("Core ML was requested, but execution profiles contain no "
                         "CoreMLExecutionProvider nodes. Refusing an unverified CPU-only result.")


def detector_postprocessing(model) -> dict:
    outputs = [{"name": item.name, "shape": item.shape}
               for item in model.det_model.session.get_outputs()]
    embedded_nms = bool(outputs and outputs[0]["shape"][-1] == 5)
    # Pinned rtmlib's five-column output branch ignores score_thr/nms_thr:
    # NMS already happened in ONNX, and its Python filter is fixed at >0.3.
    return {"outputShapes": outputs, "embeddedNms": embedded_nms,
            "configuredScoreThreshold": model.det_model.score_thr,
            "effectivePostNmsScoreThreshold": .3 if embedded_nms else model.det_model.score_thr,
            "nmsThreshold": "embedded_in_fingerprinted_onnx_graph" if embedded_nms else model.det_model.nms_thr}


def native_keypoints(points, scores, width: int, height: int) -> list[dict | None]:
    """Preserve native-order observations separately; never substitute fingers for wrists."""
    return [{"x": float(point[0]) / width, "y": float(point[1]) / height,
             "score": float(score)}
            if all(math.isfinite(float(v)) for v in (*point[:2], score)) else None
            for point, score in zip(points, scores)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assume-cfr", action="store_true", help="Explicitly use decoded frame index / reported FPS when the source is known constant-frame-rate")
    parser.add_argument("--max-frames", type=int, help="Smoke-test only; the output is marked truncated")
    parser.add_argument("--frames-manifest", type=Path, help="Use ordered local images with explicit ptsMs for paired-model experiments; video remains source provenance")
    parser.add_argument("--provider", choices=("cpu", "coreml"), default="cpu",
                        help="ONNX Runtime provider (default: cpu). coreml requires CoreMLExecutionProvider, allows CPU node fallback, and writes verified execution profiles to OUTPUT.profiles; it makes no real-time or GPU/ANE claim")
    args = parser.parse_args(argv)
    cap = None
    model = None
    profiles_finalized = False
    profile_directory = None
    try:
        manifest = load_manifest(args.manifest)
        if not args.video.is_file():
            raise ValueError(f"Video does not exist: {args.video}")
        protected = [args.video, args.manifest, *(Path(manifest[k]["path"]) for k in ("detector", "pose"))]
        image_manifest = None
        image_frames = None
        if args.frames_manifest:
            if args.assume_cfr:
                raise ValueError("--assume-cfr cannot replace explicit frame-manifest timestamps")
            image_manifest, image_frames = load_frame_manifest(args.frames_manifest)
            protected.extend([args.frames_manifest, *(item[0] for item in image_frames)])
        if args.output.resolve() in [p.resolve() for p in protected]:
            raise ValueError("Output must not overwrite an input artifact")
        if args.output.exists():
            raise ValueError("Output already exists; choose a new path to preserve previous results")
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        try:
            import cv2
            import onnxruntime as ort
            from rtmlib import RTMPose, YOLOX
        except ImportError as error:
            raise ValueError("Install the optional pinned ml/requirements-extract.txt into an isolated environment first") from error
        installed = {name: version(name) for name in ("rtmlib", "numpy", "opencv-python", "opencv-contrib-python", "onnxruntime")}
        if installed["rtmlib"] != "0.0.16":
            raise ValueError("This adapter targets rtmlib==0.0.16; use the pinned environment")
        provider_config = provider_configuration(args.provider, ort.get_available_providers())
        if args.provider == "coreml":
            profile_directory = args.output.with_name(args.output.name + ".profiles")
            # Preserve prior profiles as well as prior extraction artifacts.
            profile_directory.mkdir(parents=True, exist_ok=False)
        # Construct explicit tools: Body's filename-based RTMO auto-selection
        # can replace a local path containing "rtmo" with a remote default.
        # Both supported families use the RTMPose SimCC tool; point count is
        # checked against the manifest below.
        model = SimpleNamespace(
            det_model=YOLOX(manifest["detector"]["path"],
                            model_input_size=tuple(manifest["detector"]["inputSize"]),
                            backend="onnxruntime", device="cpu"),
            pose_model=RTMPose(manifest["pose"]["path"],
                               model_input_size=tuple(manifest["pose"]["inputSize"]),
                               to_openpose=False, backend="onnxruntime", device="cpu"))
        providers = session_providers(model)
        if any(value != ["CPUExecutionProvider"] for value in providers.values()):
            raise ValueError(f"Unexpected execution providers for the CPU protocol: {providers}")
        session_creation_ms = None
        if args.provider == "coreml":
            session_creation_ms = configure_coreml_sessions(
                model, manifest, ort, provider_config, profile_directory)
            providers = session_providers(model)
        cap = None if image_frames else cv2.VideoCapture(str(args.video))
        if cap is not None and not cap.isOpened():
            raise ValueError("OpenCV could not decode this video")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap is not None else None
        if args.assume_cfr and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("Cannot use --assume-cfr: source FPS is unavailable")
        frames = []
        counts = {"zeroPeople": 0, "multiplePeople": 0, "singlePerson": 0}
        raw_origin = None
        previous_t = None
        decoded_hashes = []
        pose_color_order = manifest["pose"].get("inputColorOrder", "BGR")
        while args.max_frames is None or len(frames) < args.max_frames:
            if image_frames is not None:
                if len(frames) == len(image_frames):
                    break
                image_path, timestamp = image_frames[len(frames)]
                pixels = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if pixels is None:
                    raise ValueError(f"Cannot decode frame image: {image_path}")
                decoded_hashes.append({"file": str(image_path.relative_to(args.frames_manifest.parent.resolve())), "sha256": sha256(image_path)})
                success = True
            else:
                success, pixels = cap.read()
            if not success:
                break
            if image_frames is not None:
                pass  # Preserve the explicit source PTS, including its origin.
            elif args.assume_cfr:
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
            person_count, keypoints, keypoint_scores = infer_single_person(model, pixels, pose_color_order)
            inference_ms = (time.perf_counter() - started) * 1000
            # No identity guessing: this single-person benchmark abstains when
            # the detector produces multiple people, including in mirrors.
            landmarks = []
            native_points = []
            if person_count == 1:
                expected_points = 17 if manifest["family"] == "rtmpose-body" else 133
                if len(keypoints) != expected_points:
                    raise ValueError(f"Model returned {len(keypoints)} points; manifest expects {expected_points}")
                landmarks = map_body_landmarks(keypoints, keypoint_scores, width, height)
                native_points = native_keypoints(keypoints, keypoint_scores, width, height)
                counts["singlePerson"] += 1
            else:
                counts["zeroPeople" if person_count == 0 else "multiplePeople"] += 1
            frames.append({"t": timestamp, "width": width, "height": height,
                           "landmarks": landmarks, "inferenceMs": inference_ms,
                           "nativeKeypoints": native_points})
            if len(frames) % 100 == 0:
                print(f"Processed {len(frames)} frames", file=sys.stderr)
        if not frames:
            raise ValueError("No video frames decoded")
        gaps = [b["t"] - a["t"] for a, b in zip(frames, frames[1:])]
        duration = frames[-1]["t"] + (statistics.median(gaps) if gaps else 0)
        execution_profiles = None
        if args.provider == "coreml":
            execution_profiles = {
                key: summarize_execution_profile(Path(getattr(model, attribute).session.end_profiling()))
                for key, attribute in (("detector", "det_model"), ("pose", "pose_model"))}
            profiles_finalized = True
            verify_coreml_execution(execution_profiles)
        result = {"artifactType": "research-pose-series", "schemaVersion": "1.0",
                  "createdAt": datetime.now(timezone.utc).isoformat(),
                  "source": {"path": str(args.video.resolve()), "sha256": sha256(args.video)},
                  "modelManifest": manifest, "runtime": {"packages": installed,
                      "python": platform.python_version(), "platform": platform.platform(),
                      "extractorSha256": sha256(Path(__file__)),
                      "backend": "onnxruntime", "deviceRequested": args.provider, "sessionProviders": providers,
                      "providerConfiguration": provider_config,
                      "registeredProviderOptions": {
                          key: getattr(model, attribute).session.get_provider_options()
                          for key, attribute in (("detector", "det_model"), ("pose", "pose_model"))},
                      "sessionCreationMs": session_creation_ms,
                      "executionProfiles": execution_profiles,
                      "profilingEnabledDuringInference": execution_profiles is not None,
                      "detectorInputColorOrder": "BGR", "poseInputColorOrder": pose_color_order,
                      "detectorPostprocessing": detector_postprocessing(model)},
                  "timestampMode": "decoded_image_manifest_source_pts" if image_frames is not None else "assumed_cfr_frame_index" if args.assume_cfr else "opencv_pos_msec_relative_to_first_frame",
                  "durationMsEstimate": duration, "lastFrameDurationIsEstimated": True,
                  "truncatedByFrameLimit": args.max_frames is not None and len(frames) == args.max_frames,
                  "mapping": {"inputOrder": "COCO17" if manifest["family"] == "rtmpose-body" else "COCO_WholeBody133",
                              "outputOrder": "MediaPipe33_partial_body_only",
                              "availableIndices": sorted(COCO_TO_MEDIAPIPE.values()),
                              "missingPointRepresentation": "zero_coordinate_and_zero_confidence",
                              "confidenceSemantics": "RTM_keypoint_score_not_calibrated_MediaPipe_visibility",
                              "nativeKeypoints": "original_model_order_normalized_xy_and_unclamped_score_separate_from_body_mapping",
                              "depth": "unavailable_no_z_or_world_landmarks"},
                  "personCounts": counts, "inferenceMs": distribution([f["inferenceMs"] for f in frames]),
                  "frames": frames,
                  "limitations": ["Pose extraction only: no punch recognition, coaching, or camera validation.",
                                  "This artifact is not a browser Session; do not relabel model as full/lite/heavy.",
                                  "Detector-plus-pose timing includes first-call warm-up; excludes video decoding.",
                                  "Configured providers and observed node execution do not establish a particular GPU/ANE, numerical parity, browser support, or real-time performance."]}
        if image_frames is not None:
            result["decodedImages"] = {"manifestPath": str(args.frames_manifest.resolve()),
                                       "manifestSha256": sha256(args.frames_manifest),
                                       "manifest": image_manifest, "processedFileHashes": decoded_hashes,
                                       "decoder": "OpenCV imread BGR; cross-decoder pixel equivalence is not independently established"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {len(frames)} frames to {args.output}")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 2
    finally:
        if cap is not None:
            cap.release()
        if profile_directory is not None and model is not None and not profiles_finalized:
            # Finish whatever sessions were initialized even when extraction
            # fails. Diagnostic profiles never turn a failed run into success.
            for attribute in ("det_model", "pose_model"):
                try:
                    getattr(model, attribute).session.end_profiling()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
