"""Evaluate version 1.0 browser exports without treating demonstrations as evidence.

Run ``python3 -m ml.evaluate --help`` from the repository root.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

JOINTS = {"nose": 0, "leftShoulder": 11, "rightShoulder": 12,
          "leftElbow": 13, "rightElbow": 14, "leftWrist": 15,
          "rightWrist": 16, "leftHip": 23, "rightHip": 24,
          "leftKnee": 25, "rightKnee": 26, "leftAnkle": 27, "rightAnkle": 28}
DEFAULT_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24)
PUNCH_LABELS = ("jab", "cross", "hook", "uppercut")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value: Any, name: str, minimum: float = 0) -> float:
    if not finite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear interpolation between neighboring ranks; null for no samples."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values: list[float]) -> dict:
    return {"count": len(values), "p50": percentile(values, .5),
            "p95": percentile(values, .95), "max": max(values) if values else None}


def temporal_iou(a: dict, b: dict) -> float:
    intersection = max(0, min(a["endMs"], b["endMs"]) - max(a["startMs"], b["startMs"]))
    union = max(a["endMs"], b["endMs"]) - min(a["startMs"], b["startMs"])
    return intersection / union if union > 0 else 0


def match_events(predictions: list[dict], annotations: list[dict], threshold: float = .5) -> list[tuple[int, int]]:
    """Maximum-cardinality bipartite matching, exact label and physical hand.

    Candidate edges are considered by descending IoU, then annotation index.
    Augmenting paths prevent a greedy high-IoU match from losing a valid pair.
    This maximizes match count, not the sum of IoUs among equally large matches.
    """
    edges = []
    for prediction in predictions:
        candidates = [i for i, truth in enumerate(annotations)
                      if prediction["label"] == truth["label"]
                      and prediction["hand"] == truth["hand"]
                      and temporal_iou(prediction, truth) >= threshold]
        edges.append(sorted(candidates, key=lambda i: (-temporal_iou(prediction, annotations[i]), i)))
    p_to_g: dict[int, int] = {}
    g_to_p: dict[int, int] = {}
    for start in range(len(predictions)):
        queue = deque([start])
        seen_p = {start}
        via_g: dict[int, int] = {}
        free_g = None
        while queue and free_g is None:
            p = queue.popleft()
            for g in edges[p]:
                if g in via_g:
                    continue
                via_g[g] = p
                if g not in g_to_p:
                    free_g = g
                    break
                next_p = g_to_p[g]
                if next_p not in seen_p:
                    seen_p.add(next_p)
                    queue.append(next_p)
        if free_g is not None:
            g = free_g
            while True:
                p = via_g[g]
                previous_g = p_to_g.get(p)
                p_to_g[p] = g
                g_to_p[g] = p
                if previous_g is None:
                    break
                g = previous_g
    return sorted(p_to_g.items())


def interval_union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def contained(event: dict, intervals: list[tuple[float, float]]) -> bool:
    return any(a <= event["startMs"] and event["endMs"] <= b for a, b in intervals)


def scores(tp: int, fp: int, fn: int, exposure_ms: float) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "falseEventsPerMinute": fp * 60000 / exposure_ms if exposure_ms > 0 else None}


def validate_session(session: Any, allow_synthetic: bool = False) -> dict:
    if not isinstance(session, dict) or session.get("schemaVersion") != "1.0":
        raise ValueError("Expected a browser session object with schemaVersion '1.0'")
    if not isinstance(session.get("id"), str) or not session["id"]:
        raise ValueError("Session id must be a nonempty string")
    if session.get("source") not in ("camera", "file", "demo"):
        raise ValueError("Session source must be camera, file, or demo")
    if session.get("model") not in ("lite", "full", "heavy", "synthetic"):
        raise ValueError("Session model must match schema 1.0: lite, full, heavy, or synthetic")
    synthetic = session.get("source") == "demo" or session.get("model") == "synthetic"
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic/demo sessions are excluded. Use --allow-synthetic only for software checks.")
    duration = number(session.get("durationMs"), "durationMs")
    for field in ("frames", "events", "annotations"):
        if not isinstance(session.get(field), list):
            raise ValueError(f"{field} must be an array")
    for field in ("events", "annotations"):
        ids = set()
        for item in session[field]:
            if not isinstance(item, dict):
                raise ValueError(f"Every {field} entry must be an object")
            if not isinstance(item.get("id"), str) or not item["id"] or item["id"] in ids:
                raise ValueError(f"{field} IDs must be nonempty and unique")
            ids.add(item["id"])
            start = number(item.get("startMs"), f"{field}.{item['id']}.startMs")
            end = number(item.get("endMs"), f"{field}.{item['id']}.endMs")
            if start >= end or end > duration + 1:
                raise ValueError(f"{field}.{item['id']} must have start < end <= durationMs")
            labels = PUNCH_LABELS if field == "events" else (*PUNCH_LABELS, "other", "unobservable")
            hands = ("left", "right") if field == "events" else ("left", "right", "unknown")
            if item.get("label") not in labels or item.get("hand") not in hands:
                raise ValueError(f"{field}.{item['id']} has invalid label or physical hand")
            if field == "events" and "detectedAtMs" in item:
                number(item["detectedAtMs"], f"events.{item['id']}.detectedAtMs")
    previous = -1.0
    for i, frame in enumerate(session["frames"]):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{i}] must be an object")
        t = number(frame.get("t"), f"frames[{i}].t")
        if t < previous or t > duration + 1:
            raise ValueError("Frame timestamps must be nondecreasing and within session duration")
        previous = t
        number(frame.get("inferenceMs"), f"frames[{i}].inferenceMs")
        if "frameAgeMs" in frame:
            number(frame["frameAgeMs"], f"frames[{i}].frameAgeMs")
        if not isinstance(frame.get("landmarks"), list):
            raise ValueError(f"frames[{i}].landmarks must be an array")
        for dimension in ("width", "height"):
            if number(frame.get(dimension), f"frames[{i}].{dimension}") <= 0:
                raise ValueError("Frame image dimensions must be positive")
    return session


def joint_visible(landmarks: list, index: int, threshold: float) -> bool:
    if index >= len(landmarks) or not isinstance(landmarks[index], dict):
        return False
    joint = landmarks[index]
    if not all(finite(joint.get(axis)) and 0 <= joint[axis] <= 1 for axis in ("x", "y")):
        return False
    confidences = [joint[key] for key in ("visibility", "presence") if key in joint]
    return bool(confidences) and all(finite(c) and threshold <= c <= 1 for c in confidences)


def pose_coverage(frames: list[dict], required_joints: tuple[int, ...], threshold: float) -> dict:
    visible = [[joint_visible(f["landmarks"], j, threshold) for j in required_joints] for f in frames]
    count = sum(all(row) for row in visible)
    return {"kind": "processed_frame_landmark_coverage_not_criterion_coverage",
            "requiredJoints": list(required_joints), "minimumConfidence": threshold,
            "processedFrames": len(frames), "assessableFrames": count,
            "fraction": count / len(frames) if frames else None,
            "perJointFraction": {str(j): sum(row[i] for row in visible) / len(frames) if frames else None
                                 for i, j in enumerate(required_joints)}}


def evaluate_session(session: dict, *, annotations_complete: bool = False,
                     allow_synthetic: bool = False, threshold: float = .5,
                     labels: tuple[str, ...] = ("jab", "cross"),
                     required_joints: tuple[int, ...] = DEFAULT_JOINTS,
                     min_confidence: float = .5) -> dict:
    validate_session(session, allow_synthetic)
    if "annotationsComplete" in session and not isinstance(session["annotationsComplete"], bool):
        raise ValueError("annotationsComplete must be an explicit boolean")
    annotations_complete = annotations_complete or session.get("annotationsComplete") is True
    frames = session["frames"]
    timestamps = [f["t"] for f in frames]
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    positive_gaps = [gap for gap in gaps if gap > 0]
    median_gap = statistics.median(positive_gaps) if positive_gaps else None
    result = {"sessionId": session["id"], "source": session["source"], "model": session.get("model"),
              "modelManifest": session.get("modelManifest"), "stance": session.get("stance"),
              "drill": session.get("drill"), "recordedAt": session.get("createdAt"),
              "durationMs": session["durationMs"],
              "synthetic": session["source"] == "demo" or session.get("model") == "synthetic",
              "annotationsCompleteAsserted": annotations_complete,
              "coverage": pose_coverage(frames, required_joints, min_confidence),
              "timing": {"inferenceMs": distribution([f["inferenceMs"] for f in frames]),
                         "frameIntervalMs": distribution(gaps), "duplicateTimestamps": gaps.count(0),
                         "gapsOver2_5xMedian": sum(g > 2.5 * median_gap for g in gaps) if median_gap else 0,
                         "observedProcessedFps": ((len(frames) - 1) * 1000 / (timestamps[-1] - timestamps[0]))
                         if len(frames) > 1 and timestamps[-1] > timestamps[0] else None,
                         "reportedMeasuredFps": session.get("measuredFps"),
                         "reportedSkippedFrames": session.get("skippedFrames"),
                         "processedFrameAgeMs": distribution([f["frameAgeMs"] for f in frames if "frameAgeMs" in f]),
                         "frameAgeDefinition": "browser_frame_callback_to_completed_inference_not_sensor_latency",
                         "eventFinalizationDelayMs": None},
              "eventMetrics": None,
              "warnings": ["Pose coverage measures processed frames only; it does not certify coaching criterion coverage.",
                           "Inference time is not processed-frame age or end-to-end camera latency.",
                           "Recognition metrics do not establish critique correctness or learning benefit."]}
    if result["synthetic"]:
        result["warnings"].append("SYNTHETIC SOFTWARE CHECK: no accuracy or hardware validation claim is permitted.")
    if not annotations_complete:
        result["warnings"].append("Event metrics withheld: complete full-session annotations were not asserted.")
        return result
    annotations = session["annotations"]
    truths = [a for a in annotations if a["label"] in labels and a["hand"] != "unknown"]
    exclusions = interval_union([(a["startMs"], a["endMs"]) for a in annotations
                                 if a["label"] == "unobservable"
                                 or (a["label"] in labels and a["hand"] == "unknown")])
    if any(max(t["startMs"], a) < min(t["endMs"], b) for t in truths for a, b in exclusions):
        raise ValueError("An assessable truth event overlaps an unobservable/unknown-hand interval. Adjudicate annotations first.")
    excluded_predictions = [p for p in session["events"] if contained(p, exclusions)]
    predictions = [p for p in session["events"] if not contained(p, exclusions)]
    exposure = max(0, session["durationMs"] - sum(b - a for a, b in exclusions))
    pairs = match_events(predictions, truths, threshold)
    matched_p = {p for p, _ in pairs}
    matched_g = {g for _, g in pairs}
    class_names = sorted(set(labels) | {p["label"] for p in predictions})
    per_class = {}
    for label in class_names:
        tp = sum(predictions[p]["label"] == label for p, _ in pairs)
        fp = sum(p["label"] == label and i not in matched_p for i, p in enumerate(predictions))
        fn = sum(a["label"] == label and i not in matched_g for i, a in enumerate(truths))
        per_class[label] = scores(tp, fp, fn, exposure)
    f1s = [row["f1"] for row in per_class.values() if row["f1"] is not None]
    result["eventMetrics"] = {**scores(len(pairs), len(predictions) - len(pairs), len(truths) - len(pairs), exposure),
                              "evaluatedExposureMs": exposure, "perClass": per_class,
                              "macroF1OverPresentClasses": statistics.mean(f1s) if f1s else None,
                              "excludedPredictions": [p["id"] for p in excluded_predictions],
                              "exclusionIntervalsMs": exclusions,
                              "unsupportedTruthCount": sum(a["label"] in PUNCH_LABELS and a["label"] not in labels for a in annotations),
                              "unknownHandTruthCount": sum(a["label"] in labels and a["hand"] == "unknown" for a in annotations),
                              "matches": [{"eventId": predictions[p]["id"], "annotationId": truths[g]["id"],
                                           "tIoU": temporal_iou(predictions[p], truths[g]),
                                           "startErrorMs": predictions[p]["startMs"] - truths[g]["startMs"],
                                           "endErrorMs": predictions[p]["endMs"] - truths[g]["endMs"]} for p, g in pairs],
                              "falsePositiveIds": [p["id"] for i, p in enumerate(predictions) if i not in matched_p],
                              "falseNegativeIds": [a["id"] for i, a in enumerate(truths) if i not in matched_g]}
    # Optional future telemetry must use the same source-relative clock as annotations.
    delays = [predictions[p]["detectedAtMs"] - truths[g]["endMs"] for p, g in pairs if "detectedAtMs" in predictions[p]]
    if delays:
        result["timing"]["eventFinalizationDelayMs"] = distribution(delays)
        result["timing"]["eventFinalizationTelemetryCoverage"] = len(delays) / len(pairs)
    result["warnings"].append("Confidence intervals and session/participant independence require a separately designed study; no automatic release-gate pass is asserted.")
    return result


def build_report(sessions: list[dict], **options) -> dict:
    if not sessions:
        raise ValueError("Provide at least one session")
    if any(not isinstance(s, dict) for s in sessions):
        raise ValueError("Each input file must contain one session object")
    if len({s.get("id") for s in sessions}) != len(sessions):
        raise ValueError("Duplicate session IDs would double-count evidence")
    results = [evaluate_session(s, **options) for s in sessions]
    metrics = [r["eventMetrics"] for r in results if r["eventMetrics"] is not None]
    aggregate = None
    if metrics:
        exposure = sum(m["evaluatedExposureMs"] for m in metrics)
        aggregate = scores(sum(m["tp"] for m in metrics), sum(m["fp"] for m in metrics), sum(m["fn"] for m in metrics), exposure)
        aggregate["evaluatedExposureMs"] = exposure
        class_names = sorted({name for m in metrics for name in m["perClass"]})
        aggregate["perClass"] = {name: scores(*(sum(m["perClass"].get(name, {}).get(key, 0) for m in metrics)
                                                    for key in ("tp", "fp", "fn")), exposure) for name in class_names}
    return {"reportVersion": "1.0", "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "synthetic_software_check" if any(r["synthetic"] for r in results)
            else "annotated_benchmark" if len(metrics) == len(results)
            else "partial_annotated_benchmark" if metrics else "capture_quality_only",
            "protocol": {"tIoUThreshold": options.get("threshold", .5),
                         "labels": list(options.get("labels", ("jab", "cross"))),
                         "matching": "maximum_cardinality_exact_label_and_physical_hand",
                         "undefinedRatios": "null",
                         "annotationPolicy": "complete_session_assertion_required",
                         "ignorePolicy": "prediction_fully_contained_in_unobservable_or_unknown_hand_interval",
                         "recallScope": "selected_labels_only; unsupported punches remain negative examples for selected labels"},
            "sessionCount": len(results), "accuracySessionCount": len(metrics), "eventMetrics": aggregate,
            "inferenceMs": distribution([f["inferenceMs"] for s in sessions for f in s["frames"]]),
            "sessions": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Browser session JSON exports (one session per file)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--annotations-complete", action="store_true", help="Assert every target punch and unobservable interval is annotated throughout every session")
    parser.add_argument("--allow-synthetic", action="store_true", help="Software checks only; marks the entire report synthetic")
    parser.add_argument("--tiou", type=float, default=.5)
    parser.add_argument("--labels", default="jab,cross", help="Comma-separated recall scope; default jab,cross")
    parser.add_argument("--min-confidence", type=float, default=.5)
    parser.add_argument("--required-joints", default=",".join(map(str, DEFAULT_JOINTS)), help="MediaPipe indices for processed-frame coverage")
    args = parser.parse_args(argv)
    try:
        if not 0 < args.tiou <= 1 or not 0 <= args.min_confidence <= 1:
            raise ValueError("--tiou must be in (0,1] and --min-confidence in [0,1]")
        labels = tuple(dict.fromkeys(args.labels.split(",")))
        joints = tuple(dict.fromkeys(int(j) for j in args.required_joints.split(",")))
        if not labels or any(label not in PUNCH_LABELS for label in labels):
            raise ValueError("--labels must select jab,cross,hook,uppercut")
        if not joints or any(j < 0 or j > 32 for j in joints):
            raise ValueError("--required-joints must select MediaPipe indices 0..32")
        if args.output.resolve() in [path.resolve() for path in args.paths]:
            raise ValueError("Output must not overwrite an input session")
        sessions = []
        for path in args.paths:
            try:
                sessions.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}: {error}") from error
        report = build_report(sessions, annotations_complete=args.annotations_complete,
                              allow_synthetic=args.allow_synthetic, threshold=args.tiou,
                              labels=labels, required_joints=joints, min_confidence=args.min_confidence)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{report['status']}: {report['sessionCount']} session(s) → {args.output}")
        if report["eventMetrics"]:
            m = report["eventMetrics"]
            print(f"TP={m['tp']} FP={m['fp']} FN={m['fn']} precision={m['precision']} recall={m['recall']}")
        return 0
    except (ValueError, TypeError, OSError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
