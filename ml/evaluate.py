"""Evaluate version 1.0 browser exports without treating demonstrations as evidence.

Run ``python3 -m ml.evaluate --help`` from the repository root.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
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
PEAK_TOLERANCE_MS = 250
PEAK_FIELDS = ("peakMs", "peakTMs", "approximatePeakMs")


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


def punch_family(event: dict) -> str:
    return "straight" if event["label"] in ("jab", "cross") else event["label"]


def explicit_peak(item: dict, name: str) -> float | None:
    """Read an observed source-time peak; never infer one from boundaries."""
    values = [number(item[key], f"{name}.{key}") for key in PEAK_FIELDS if key in item]
    if values and any(value != values[0] for value in values):
        raise ValueError(f"{name} has conflicting explicit peak fields")
    return values[0] if values else None


def reference_peaks(session: dict, reference: dict | None) -> tuple[dict[str, float], dict]:
    annotations = {a["id"]: a for a in session["annotations"]}
    peaks = {}

    def add(item: dict, name: str) -> None:
        if not isinstance(item, dict):
            raise ValueError(f"{name} must be an object")
        peak = explicit_peak(item, name)
        if peak is None:
            return
        identity = item.get("annotationId", item.get("id"))
        if identity not in annotations:
            raise ValueError(f"{name} peak has no matching annotation ID")
        annotation = annotations[identity]
        for key in ("label", "hand", "startMs", "endMs"):
            if key in item and item[key] != annotation[key]:
                raise ValueError(f"{name} changes the stored annotation {key}")
        if not annotation["startMs"] <= peak <= annotation["endMs"]:
            raise ValueError(f"{name} peak must lie within its annotation interval")
        if identity in peaks and peaks[identity] != peak:
            raise ValueError(f"{name} conflicts with another explicit reference peak")
        peaks[identity] = peak

    for annotation in session["annotations"]:
        add(annotation, f"annotations.{annotation['id']}")
    provenance = {"kind": "inline_annotation_peaks", "referenceContentSha256": None}
    if reference is not None:
        if not isinstance(reference, dict):
            raise ValueError("Peak reference must be an object")
        for key in ("sessionId", "sourceSessionId"):
            if key in reference and reference[key] not in (session["id"], session.get("benchmark", {}).get("sourceSessionId")):
                raise ValueError("Peak reference session identity does not match")
        source_hash = session.get("benchmark", {}).get("sourceVideoSha256")
        if source_hash and reference.get("sourceVideoSha256") not in (None, source_hash):
            raise ValueError("Peak reference source video fingerprint does not match")
        arrays = ("annotations", "actionObservations", "approximatePeaks", "peaks")
        if not any(key in reference for key in arrays):
            raise ValueError("Peak reference requires annotations, actionObservations, approximatePeaks or peaks")
        unscored_observations = []
        for key in arrays:
            if key not in reference:
                continue
            if not isinstance(reference[key], list):
                raise ValueError(f"Peak reference {key} must be an array")
            for i, item in enumerate(reference[key]):
                if (key == "actionObservations" and isinstance(item, dict)
                        and item.get("annotationId", item.get("id")) not in annotations):
                    # Raw video notes may include ambiguous actions deliberately
                    # excluded from the frozen scored annotations. Never add them.
                    unscored_observations.append(item.get("annotationId", item.get("id")))
                    continue
                add(item, f"peakReference.{key}[{i}]")
        provenance = {"kind": "explicit_reference_plus_inline_annotation_peaks",
                      "referenceContentSha256": hashlib.sha256(json.dumps(
                          reference, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                      "sourceVideoSha256": reference.get("sourceVideoSha256"),
                      "unscoredReferenceObservationIds": unscored_observations,
                      "labelProvenance": reference.get("labelProvenance", reference.get("status"))}
    return peaks, provenance


def match_peak_events(predictions: list[dict], truths: list[dict], *, identity: bool = True) -> list[tuple[int, int]]:
    """Maximum cardinality within the fixed inclusive source-peak tolerance.

    Inputs contain validated explicit peakMs values. Closest edges are visited
    first, with input-index ties; minimum total timing error is not guaranteed.
    identity=False is only for descriptive unmatched-event confusion analysis.
    """
    edges = [sorted([g for g, truth in enumerate(truths)
                     if abs(prediction["peakMs"] - truth["peakMs"]) <= PEAK_TOLERANCE_MS
                     and (not identity or (prediction["hand"] == truth["hand"]
                          and punch_family(prediction) == punch_family(truth)))],
                    key=lambda g: (abs(prediction["peakMs"] - truths[g]["peakMs"]), g))
             for prediction in predictions]
    p_to_g, g_to_p = {}, {}
    for start in range(len(predictions)):
        queue, seen_p, via_g, free_g = deque([start]), {start}, {}, None
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
                p_to_g[p], g_to_p[g] = g, p
                if previous_g is None:
                    break
                g = previous_g
    return sorted(p_to_g.items())


def peak_timing(matches: list[dict]) -> dict:
    fields = ("peakErrorMs", "absolutePeakErrorMs", "startErrorMs", "endErrorMs",
              "sourcePeakToEmissionMs", "predictedPeakToEmissionMs", "referenceEndToEmissionMs")
    return {key: distribution([m[key] for m in matches if key in m]) for key in fields}


def peak_occurrence_diagnostics(session: dict, predictions: list[dict], truths: list[dict],
                                exposure: float, reference: dict | None, strict: dict) -> dict:
    peaks, provenance = reference_peaks(session, reference)
    missing_truth = [a["id"] for a in truths if a["id"] not in peaks]
    eligible_truths = [{**a, "peakMs": peaks[a["id"]]} for a in truths if a["id"] in peaks]
    eligible_predictions, missing_prediction = [], []
    for prediction in predictions:
        peak = explicit_peak(prediction, f"events.{prediction['id']}")
        if peak is None:
            missing_prediction.append(prediction["id"])
            continue
        if not prediction["startMs"] <= peak <= prediction["endMs"]:
            raise ValueError(f"events.{prediction['id']} peak must lie within its event interval")
        if "detectedAtMs" in prediction and prediction["detectedAtMs"] < prediction["endMs"]:
            raise ValueError(f"events.{prediction['id']} emission precedes its event end")
        eligible_predictions.append({**prediction, "peakMs": peak})
    pairs = match_peak_events(eligible_predictions, eligible_truths)
    matched_p, matched_g = {p for p, _ in pairs}, {g for _, g in pairs}
    matches = []
    strict_pairs = {(m["eventId"], m["annotationId"]) for m in strict["matches"]}
    for p, g in pairs:
        prediction, truth = eligible_predictions[p], eligible_truths[g]
        error = prediction["peakMs"] - truth["peakMs"]
        item = {"eventId": prediction["id"], "annotationId": truth["id"],
                "family": punch_family(truth), "hand": truth["hand"],
                "peakErrorMs": error, "absolutePeakErrorMs": abs(error),
                "startErrorMs": prediction["startMs"] - truth["startMs"],
                "endErrorMs": prediction["endMs"] - truth["endMs"],
                "tIoU": temporal_iou(prediction, truth),
                "alsoStrictMatchedPair": (prediction["id"], truth["id"]) in strict_pairs}
        if "detectedAtMs" in prediction:
            item.update(sourcePeakToEmissionMs=prediction["detectedAtMs"] - truth["peakMs"],
                        predictedPeakToEmissionMs=prediction["detectedAtMs"] - prediction["peakMs"],
                        referenceEndToEmissionMs=prediction["detectedAtMs"] - truth["endMs"])
        matches.append(item)
    unmatched_p = [p for i, p in enumerate(eligible_predictions) if i not in matched_p]
    unmatched_g = [g for i, g in enumerate(eligible_truths) if i not in matched_g]
    complete = not missing_truth and not missing_prediction
    metrics = scores(len(pairs), len(unmatched_p), len(unmatched_g), exposure) if complete else None
    if metrics is not None:
        metrics["evaluatedExposureMs"] = exposure
        metrics["perPhysicalHandFamily"] = {}
        for hand in ("left", "right"):
            for family in ("straight", "hook", "uppercut"):
                selected = lambda item: item["hand"] == hand and punch_family(item) == family
                metrics["perPhysicalHandFamily"][f"{hand}:{family}"] = scores(
                    sum(selected(eligible_truths[g]) for _, g in pairs),
                    sum(selected(p) for p in unmatched_p), sum(selected(g) for g in unmatched_g), exposure)
    confusions = []
    for p, g in match_peak_events(unmatched_p, unmatched_g, identity=False):
        prediction, truth = unmatched_p[p], unmatched_g[g]
        confusions.append({"eventId": prediction["id"], "annotationId": truth["id"],
                           "predictedHand": prediction["hand"], "referenceHand": truth["hand"],
                           "predictedFamily": punch_family(prediction), "referenceFamily": punch_family(truth),
                           "wrongHand": prediction["hand"] != truth["hand"],
                           "wrongFamily": punch_family(prediction) != punch_family(truth),
                           "peakErrorMs": prediction["peakMs"] - truth["peakMs"]})
    return {"protocolVersion": "peak-occurrence-development-v1", "validationStatus": "development_diagnostic_pending_fresh_future_data",
            "status": "scored" if complete else "withheld_missing_explicit_peaks",
            "toleranceMs": PEAK_TOLERANCE_MS, "reference": provenance,
            "matching": "maximum_cardinality_physical_hand_and_family_inclusive_peak_tolerance",
            "familyMapping": "jab/cross -> straight; hook -> hook; uppercut -> uppercut",
            "missingReferencePeakIds": missing_truth, "missingPredictionPeakIds": missing_prediction,
            "referencePeakCoverage": {"available": len(eligible_truths), "required": len(truths)},
            "predictionPeakCoverage": {"available": len(eligible_predictions), "required": len(predictions)},
            "metrics": metrics, "matches": matches, "timingMs": peak_timing(matches),
            "unmatchedPredictionIds": [p["id"] for p in unmatched_p],
            "unmatchedReferenceIds": [g["id"] for g in unmatched_g],
            "confusionAssociations": confusions,
            "confusionInterpretation": "Secondary one-to-one time-only associations among unmatched events; not verified error causes. Never add these to true positives.",
            "timingInterpretation": "Observed reference peak to source-clock emission, not sensor-to-display latency. Partial-reference timing uses available matched peaks only.",
            "scope": "Same full-session predictions, recall labels and exclusion intervals as strict metrics. Does not establish full action timing or correct technique."}


BROWSER_MODELS = ("lite", "full", "heavy", "synthetic", "rtmpose-m", "rtmw-l")


def native_estimator(value: Any) -> bool:
    return (isinstance(value, dict) and value.get("id") in ("rtmpose-m", "rtmw-l")
            and value.get("scoreType") == "simcc" and finite(value.get("minimumScore"))
            and 0 < value["minimumScore"] < 1)


def valid_fingerprint(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_session(session: Any, allow_synthetic: bool = False) -> dict:
    if not isinstance(session, dict) or session.get("schemaVersion") != "1.0":
        raise ValueError("Expected a browser session object with schemaVersion '1.0'")
    if not isinstance(session.get("id"), str) or not session["id"]:
        raise ValueError("Session id must be a nonempty string")
    if session.get("source") not in ("camera", "file", "demo"):
        raise ValueError("Session source must be camera, file, or demo")
    benchmark = session.get("artifactType") == "detector-benchmark-session"
    if benchmark:
        provenance = session.get("benchmark")
        if not isinstance(provenance, dict) or provenance.get("modelId") != session.get("model"):
            raise ValueError("Benchmark model identity must match its provenance")
        if not isinstance(session.get("model"), str) or not session["model"].strip():
            raise ValueError("Benchmark model identifier must be nonempty")
        def valid_hash(value):
            return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        for key in ("inputSessionSha256", "detectorSourceSha256"):
            if not valid_hash(provenance.get(key)):
                raise ValueError(f"Benchmark requires {key}")
        tracking_source = provenance.get("trackingSource")
        if tracking_source not in ("saved-frames", "replacement-pose-series"):
            raise ValueError("Benchmark tracking source is required")
        if tracking_source == "replacement-pose-series":
            for key in ("sourceVideoSha256", "poseSeriesSha256"):
                if not valid_hash(provenance.get(key)):
                    raise ValueError(f"Replacement poses require {key}")
            if not session.get("modelManifest") or not provenance.get("timestampMode"):
                raise ValueError("Replacement pose model and timestamp provenance are required")
        elif session.get("model") not in BROWSER_MODELS:
            raise ValueError("Saved-frame replay must retain the browser model identity")
    elif session.get("model") not in BROWSER_MODELS:
        raise ValueError("Session model must match a supported browser estimator")
    native = session.get("model") in ("rtmpose-m", "rtmw-l")
    recognizer = None
    if native:
        manifest = session.get("modelManifest")
        if (not isinstance(manifest, dict) or manifest.get("id") != session["model"]
                or not native_estimator(manifest.get("estimator"))
                or manifest["estimator"]["id"] != session["model"]):
            raise ValueError("Native browser estimator requires its model identity and score policy")
        weights = manifest.get("modelManifest", {})
        for key in ("detector", "pose"):
            item = weights.get(key, {}) if isinstance(weights, dict) else {}
            digest = item.get("sha256") if isinstance(item, dict) else None
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Native browser estimator requires model weight fingerprints")
        recognizer = manifest.get("recognizer")
        if recognizer is not None:
            if (not isinstance(recognizer, dict)
                    or recognizer.get("protocolVersion") != "shadowbox-recognition-v1"
                    or recognizer.get("recognizerId") != "personal-hybrid-v1"
                    or not all(valid_fingerprint(recognizer.get(key)) for key in
                               ("fingerprint", "checkpointSha256", "externalModelSha256", "poseModelSha256"))
                    or recognizer["poseModelSha256"] != weights["pose"]["sha256"]):
                raise ValueError("Local recognizer model provenance is invalid")
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
    native_decisions = {}
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
        if native and frame.get("estimator") != session["modelManifest"]["estimator"]:
            raise ValueError("Native frame score policy must match the captured model manifest")
        if "estimator" in frame and not native_estimator(frame["estimator"]):
            raise ValueError("Native frame score policy is invalid")
        if "estimator" in frame and session["model"] in ("lite", "full", "heavy", "synthetic"):
            raise ValueError("MediaPipe or synthetic sessions cannot claim native estimator scores")
        decisions = frame.get("recognition")
        if decisions is not None or recognizer is not None:
            if (not isinstance(decisions, dict) or recognizer is None
                    or any(decisions.get(key) != recognizer[key] for key in
                           ("protocolVersion", "recognizerId", "fingerprint"))
                    or decisions.get("state") not in ("warming", "active", "uncertain")
                    or not isinstance(decisions.get("events"), list)):
                raise ValueError("Local frame recognition must match its captured model provenance")
            for decision in decisions["events"]:
                if (not isinstance(decision, dict)
                        or decision.get("hand") not in ("left", "right")
                        or decision.get("family") not in ("straight", "hook", "uppercut")
                        or not isinstance(decision.get("id"), str) or not decision["id"]
                        or not all(finite(decision.get(key)) for key in
                                   ("startMs", "peakMs", "endMs", "detectedAtMs", "score"))
                        or not 0 <= decision["startMs"] <= decision["peakMs"] <= decision["endMs"] <= decision["detectedAtMs"] <= t + .001
                        or decision["startMs"] >= decision["endMs"]
                        or not 0 <= decision["score"] <= 1):
                    raise ValueError("Local recognition must use finite causal round timestamps")
                prior = native_decisions.get(decision["id"])
                if prior is not None and prior != decision:
                    raise ValueError("Local recognition repeated an ID with conflicting evidence")
                native_decisions[decision["id"]] = decision
        for dimension in ("width", "height"):
            if number(frame.get(dimension), f"frames[{i}].{dimension}") <= 0:
                raise ValueError("Frame image dimensions must be positive")
    if recognizer is not None:
        if session.get("stance") not in ("orthodox", "southpaw"):
            raise ValueError("Local recognition requires the captured stance")
        for predicted in session["events"]:
            decision = native_decisions.get(predicted["id"])
            if decision is None:
                raise ValueError("Native prediction has no emitted frame decision")
            role = "lead" if (decision["hand"] == "left") == (session["stance"] == "orthodox") else "rear"
            label = ("jab" if role == "lead" else "cross") if decision["family"] == "straight" else decision["family"]
            if (predicted["hand"] != decision["hand"] or predicted["label"] != label
                    or predicted.get("role", role) != role
                    or any(predicted.get(key) != decision[key] for key in
                           ("startMs", "peakMs", "endMs", "detectedAtMs", "score"))):
                raise ValueError("Native prediction must match its emitted decision and stance mapping")
    return session


def joint_visible(landmarks: list, index: int, threshold: float, estimator: dict | None = None) -> bool:
    if index >= len(landmarks) or not isinstance(landmarks[index], dict):
        return False
    joint = landmarks[index]
    if not all(finite(joint.get(axis)) and 0 <= joint[axis] <= 1 for axis in ("x", "y")):
        return False
    if estimator is not None:
        return (native_estimator(estimator) and finite(joint.get("score"))
                and joint["score"] >= estimator["minimumScore"])
    confidences = [joint[key] for key in ("visibility", "presence") if key in joint]
    return bool(confidences) and all(finite(c) and threshold <= c <= 1 for c in confidences)


def pose_coverage(frames: list[dict], required_joints: tuple[int, ...], threshold: float) -> dict:
    visible = [[joint_visible(f["landmarks"], j, threshold, f.get("estimator")) for j in required_joints] for f in frames]
    count = sum(all(row) for row in visible)
    policies = {json.dumps(f["estimator"], sort_keys=True) for f in frames if "estimator" in f}
    return {"kind": "processed_frame_landmark_coverage_not_criterion_coverage",
            "requiredJoints": list(required_joints), "minimumConfidence": None if policies else threshold,
            "nativeScorePolicies": [json.loads(p) for p in sorted(policies)],
            "processedFrames": len(frames), "assessableFrames": count,
            "fraction": count / len(frames) if frames else None,
            "perJointFraction": {str(j): sum(row[i] for row in visible) / len(frames) if frames else None
                                 for i, j in enumerate(required_joints)}}


def evaluate_session(session: dict, *, annotations_complete: bool = False,
                     allow_synthetic: bool = False, threshold: float = .5,
                     labels: tuple[str, ...] = ("jab", "cross"),
                     required_joints: tuple[int, ...] = DEFAULT_JOINTS,
                     min_confidence: float = .5, peak_reference: dict | None = None) -> dict:
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
              "benchmark": session.get("benchmark"),
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
              "eventMetrics": None, "peakOccurrenceDiagnostics": None,
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
    result["peakOccurrenceDiagnostics"] = peak_occurrence_diagnostics(
        session, predictions, truths, exposure, peak_reference, result["eventMetrics"])
    # Optional future telemetry must use the same source-relative clock as annotations.
    delays = [predictions[p]["detectedAtMs"] - truths[g]["endMs"] for p, g in pairs if "detectedAtMs" in predictions[p]]
    if delays:
        result["timing"]["eventFinalizationDelayMs"] = distribution(delays)
        result["timing"]["eventFinalizationTelemetryCoverage"] = len(delays) / len(pairs)
    result["warnings"].append("Confidence intervals and session/participant independence require a separately designed study; no automatic release-gate pass is asserted.")
    return result


def build_report(sessions: list[dict], *, peak_references: dict[str, dict] | None = None, **options) -> dict:
    if not sessions:
        raise ValueError("Provide at least one session")
    if any(not isinstance(s, dict) for s in sessions):
        raise ValueError("Each input file must contain one session object")
    if len({s.get("id") for s in sessions}) != len(sessions):
        raise ValueError("Duplicate session IDs would double-count evidence")
    if peak_references is not None and (not isinstance(peak_references, dict)
            or any(key not in {s.get("id") for s in sessions} for key in peak_references)):
        raise ValueError("peak_references must map input session IDs to reference objects")
    results = [evaluate_session(s, peak_reference=(peak_references or {}).get(s["id"]), **options) for s in sessions]
    metrics = [r["eventMetrics"] for r in results if r["eventMetrics"] is not None]
    aggregate = None
    if metrics:
        exposure = sum(m["evaluatedExposureMs"] for m in metrics)
        aggregate = scores(sum(m["tp"] for m in metrics), sum(m["fp"] for m in metrics), sum(m["fn"] for m in metrics), exposure)
        aggregate["evaluatedExposureMs"] = exposure
        class_names = sorted({name for m in metrics for name in m["perClass"]})
        aggregate["perClass"] = {name: scores(*(sum(m["perClass"].get(name, {}).get(key, 0) for m in metrics)
                                                    for key in ("tp", "fp", "fn")), exposure) for name in class_names}
    peak_diagnostics = [r["peakOccurrenceDiagnostics"] for r in results]
    peak_aggregate = None
    if all(d is not None and d["metrics"] is not None for d in peak_diagnostics):
        peak_metrics = [d["metrics"] for d in peak_diagnostics]
        exposure = sum(m["evaluatedExposureMs"] for m in peak_metrics)
        peak_aggregate = {**scores(*(sum(m[k] for m in peak_metrics) for k in ("tp", "fp", "fn")), exposure),
                          "evaluatedExposureMs": exposure}
        peak_aggregate["perPhysicalHandFamily"] = {name: scores(
            *(sum(m["perPhysicalHandFamily"][name][k] for m in peak_metrics) for k in ("tp", "fp", "fn")), exposure)
            for name in peak_metrics[0]["perPhysicalHandFamily"]}
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
            "peakOccurrenceDiagnostics": {
                "protocolVersion": "peak-occurrence-development-v1",
                "validationStatus": "development_diagnostic_pending_fresh_future_data",
                "toleranceMs": PEAK_TOLERANCE_MS, "metrics": peak_aggregate,
                "status": "scored" if peak_aggregate is not None else "withheld_incomplete_annotations_or_peaks",
                "scoredSessionCount": sum(d is not None and d["metrics"] is not None for d in peak_diagnostics),
                "timingMs": peak_timing([m for d in peak_diagnostics if d is not None for m in d["matches"]]),
                "timingScope": "available explicitly peaked matched occurrences; may be a subset when aggregate metrics are withheld"},
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
    parser.add_argument("--peak-reference", action="append", type=Path,
                        help="Explicit video-only peak reference; repeat once per input path, in the same order. Does not replace annotations.")
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
        if args.peak_reference and len(args.peak_reference) != len(args.paths):
            raise ValueError("Provide one --peak-reference per input path, in input order")
        if args.output.resolve() in [path.resolve() for path in args.paths + (args.peak_reference or [])]:
            raise ValueError("Output must not overwrite an input session or peak reference")
        sessions = []
        for path in args.paths:
            try:
                sessions.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}: {error}") from error
        if args.peak_reference and any(not isinstance(s, dict) or not isinstance(s.get("id"), str) for s in sessions):
            raise ValueError("Peak references require session objects with string IDs")
        references = {s["id"]: json.loads(path.read_text()) for s, path in zip(sessions, args.peak_reference or [])}
        report = build_report(sessions, peak_references=references, annotations_complete=args.annotations_complete,
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
