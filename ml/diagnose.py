"""Inspect tracking and projected arm motion in a local browser export.

This reports model observations, never punch accuracy or technique correctness.
Run ``python3 -m ml.diagnose round.json --output data/diagnosis.json``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

from .evaluate import JOINTS, finite, percentile, validate_session

ARMS = {"left": (11, 13, 15), "right": (12, 14, 16)}
TORSO = (11, 12, 23, 24)
LEGACY_REQUIRED = (0, 11, 12, 13, 14, 15, 16, 23, 24)
NAMES = {index: name for name, index in JOINTS.items()}


def stats(values: list[float]) -> dict:
    return {"count": len(values), "min": min(values) if values else None,
            "p05": percentile(values, .05), "p50": percentile(values, .5),
            "p95": percentile(values, .95), "max": max(values) if values else None}


def point_at(points: list, index: int) -> dict:
    return points[index] if isinstance(points, list) and index < len(points) and isinstance(points[index], dict) else {}


def point_issues(point: dict, visibility: float, presence: float) -> list[str]:
    issues = []
    if not all(finite(point.get(key)) for key in ("x", "y")):
        issues.append("missing_or_nonfinite_position")
    elif not all(0 <= point[key] <= 1 for key in ("x", "y")):
        issues.append("outside_image")
    if not finite(point.get("visibility")):
        issues.append("missing_visibility")
    elif point["visibility"] < visibility:
        issues.append("low_visibility")
    if "presence" in point and (not finite(point["presence"]) or point["presence"] < presence):
        issues.append("low_or_invalid_presence")
    return issues


def angle(a: tuple, b: tuple, c: tuple) -> float | None:
    u, v = [x - y for x, y in zip(a, b)], [x - y for x, y in zip(c, b)]
    denominator = math.hypot(*u) * math.hypot(*v)
    if denominator < 1e-8:
        return None
    return math.degrees(math.acos(max(-1, min(1, sum(x * y for x, y in zip(u, v)) / denominator))))


def frame_observation(frame: dict, visibility: float, presence: float) -> dict:
    points = frame["landmarks"]
    issues = {NAMES[i]: point_issues(point_at(points, i), visibility, presence) for i in LEGACY_REQUIRED}
    width, height = frame.get("width"), frame.get("height")
    dimensions_ok = finite(width) and finite(height) and width > 0 and height > 0

    def xy(index: int) -> tuple | None:
        point = point_at(points, index)
        if not dimensions_ok or not all(finite(point.get(k)) for k in ("x", "y")):
            return None
        return point["x"] * width / height, point["y"]

    anchors = [xy(i) for i in TORSO]
    scale = None
    if all(p is not None for p in anchors):
        shoulders = tuple((anchors[0][i] + anchors[1][i]) / 2 for i in range(2))
        hips = tuple((anchors[2][i] + anchors[3][i]) / 2 for i in range(2))
        scale = math.dist(shoulders, hips)
    shared_issues = [NAMES[i] for i in TORSO if issues[NAMES[i]]]
    if not dimensions_ok:
        shared_issues.append("invalid_image_dimensions")
    if scale is None or scale < .08:
        shared_issues.append("torso_scale_unavailable_or_too_small")
    arms = {}
    for hand, indices in ARMS.items():
        a, b, c = [xy(i) for i in indices]
        geometry = {"elbowAngle2d": None, "reachTorsoUnits": None,
                    "wristRelativeTorso": None, "upperArmImageHeight": None,
                    "forearmImageHeight": None, "estimatedWorldElbowAngle": None}
        arm_issues = [NAMES[i] for i in indices if issues[NAMES[i]]]
        if a is not None and b is not None and c is not None:
            geometry["elbowAngle2d"] = angle(a, b, c)
            geometry["upperArmImageHeight"] = math.dist(a, b)
            geometry["forearmImageHeight"] = math.dist(b, c)
            if min(geometry["upperArmImageHeight"], geometry["forearmImageHeight"]) < .01:
                arm_issues.append("projected_arm_segment_too_short")
            if scale is not None and scale > 1e-8:
                geometry["reachTorsoUnits"] = math.dist(a, c) / scale
                geometry["wristRelativeTorso"] = [(c[i] - a[i]) / scale for i in range(2)]
        world = frame.get("worldLandmarks") or []
        wp = [point_at(world, i) for i in indices]
        if all(all(finite(p.get(k)) for k in ("x", "y", "z")) for p in wp):
            geometry["estimatedWorldElbowAngle"] = angle(*[tuple(p[k] for k in ("x", "y", "z")) for p in wp])
        arms[hand] = {"jointsVisible": all(not issues[NAMES[i]] for i in indices),
                      "passesTrackingGate": not (shared_issues or arm_issues),
                      "issues": list(dict.fromkeys(shared_issues + arm_issues)),
                      **geometry, "wristSpeedTorsoPerSecond": None}
    legacy_gate = not any(issues.values()) and all(a["passesTrackingGate"] for a in arms.values())
    return {"t": frame["t"], "jointIssues": issues, "torsoScaleImageHeight": scale,
            "sharedIssues": shared_issues, "legacyAllNineGate": legacy_gate, "arms": arms}


def runs(rows: list[dict], predicate, maximum_gap_ms: float) -> list[dict]:
    """Consecutive observed samples, not inferred continuous exposure intervals."""
    result: list[dict] = []
    current = None
    for row in rows:
        if not predicate(row):
            current = None
            continue
        if current is None or row["t"] - current["lastSampleMs"] > maximum_gap_ms:
            current = {"firstSampleMs": row["t"], "lastSampleMs": row["t"], "samples": 1}
            result.append(current)
        else:
            current["lastSampleMs"] = row["t"]
            current["samples"] += 1
    return result


def summarize(rows: list[dict], raw_frames: list[dict]) -> dict:
    count = len(rows)
    fraction = lambda n: n / count if count else None
    summary = {"frames": count,
               "legacyAllNineGateCoverage": fraction(sum(r["legacyAllNineGate"] for r in rows)),
               "joints": {}, "arms": {}}
    for index in LEGACY_REQUIRED:
        name = NAMES[index]
        failures = Counter(issue for row in rows for issue in row["jointIssues"][name])
        summary["joints"][name] = {
            "visibility": stats([p["visibility"] for f in raw_frames
                                 if finite((p := point_at(f["landmarks"], index)).get("visibility"))]),
            "coverage": fraction(sum(not r["jointIssues"][name] for r in rows)),
            "failureCounts": dict(failures)}
    for hand in ARMS:
        arm_rows = [row["arms"][hand] for row in rows]
        usable = [arm for arm in arm_rows if arm["passesTrackingGate"]]
        opposite = "right" if hand == "left" else "left"
        summary["arms"][hand] = {
            "jointCoverage": fraction(sum(a["jointsVisible"] for a in arm_rows)),
            "trackingGateCoverage": fraction(len(usable)),
            "usableFramesBlockedByLegacyGate": sum(r["arms"][hand]["passesTrackingGate"] and not r["legacyAllNineGate"] for r in rows),
            "oppositeArmOnlyBlockFrames": sum(r["arms"][hand]["passesTrackingGate"] and not r["jointIssues"]["nose"] and not r["arms"][opposite]["passesTrackingGate"] for r in rows),
            "geometryAllFiniteSamples": {}, "geometryTrackingGateSamples": {}}
        for field in ("elbowAngle2d", "reachTorsoUnits", "estimatedWorldElbowAngle", "wristSpeedTorsoPerSecond"):
            summary["arms"][hand]["geometryAllFiniteSamples"][field] = stats([a[field] for a in arm_rows if finite(a[field])])
            summary["arms"][hand]["geometryTrackingGateSamples"][field] = stats([a[field] for a in usable if finite(a[field])])
    return summary


def diagnose(session: dict, *, visibility: float = .65, presence: float = .5,
             window_ms: float = 5000, maximum_gap_ms: float = 200,
             include_timeline: bool = False, allow_synthetic: bool = False) -> dict:
    for name, value in (("visibility", visibility), ("presence", presence)):
        if not finite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must lie between 0 and 1")
    for name, value in (("window_ms", window_ms), ("maximum_gap_ms", maximum_gap_ms)):
        if not finite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    validate_session(session, allow_synthetic=allow_synthetic)
    frames = session["frames"]
    rows = [frame_observation(f, visibility, presence) for f in frames]
    gaps = []
    for previous, current in zip(rows, rows[1:]):
        dt = current["t"] - previous["t"]
        gaps.append(dt)
        if not 0 < dt <= maximum_gap_ms:
            continue
        for hand in ARMS:
            a, b = previous["arms"][hand], current["arms"][hand]
            if a["passesTrackingGate"] and b["passesTrackingGate"]:
                b["wristSpeedTorsoPerSecond"] = math.dist(a["wristRelativeTorso"], b["wristRelativeTorso"]) * 1000 / dt
    windows = []
    for bucket in sorted({math.floor(r["t"] / window_ms) for r in rows}):
        indices = [i for i, r in enumerate(rows) if math.floor(r["t"] / window_ms) == bucket]
        windows.append({"startMs": bucket * window_ms, "endMs": min((bucket + 1) * window_ms, session["durationMs"]),
                        **summarize([rows[i] for i in indices], [frames[i] for i in indices])})
    report = {
        "reportType": "pose_observation_diagnostics", "reportVersion": "1.0",
        "sessionId": session["id"], "source": session["source"], "model": session["model"],
        "synthetic": session["source"] == "demo" or session["model"] == "synthetic",
        "durationMs": session["durationMs"], "savedEvents": session["events"],
        "interpretation": [
            "Coverage is a fraction of processed pose frames, not real-world observability or detection accuracy.",
            "2D angles/reach are projected model observations, not anatomical measurements or technique scores.",
            "World landmarks are model estimates, not independent ground truth; they are not used to accept punches here.",
            "Legacy gate reproduces acquisition geometry/visibility rules only, not calibration, temporal state, or event decisions.",
            "High-angle runs are observed samples, not punch events. Video/reference labels are needed to identify actual actions."],
        "settings": {"visibility": visibility, "presence": presence, "windowMs": window_ms,
                     "maximumGapMs": maximum_gap_ms, "minimumTorsoImageHeight": .08,
                     "minimumArmSegmentImageHeight": .01, "highElbowAngleDegrees": 145},
        "timing": {"frameGapMs": stats(gaps), "gapsAboveLimit": sum(g > maximum_gap_ms for g in gaps),
                   "nonIncreasingTimes": sum(g <= 0 for g in gaps),
                   "inferenceMs": stats([f["inferenceMs"] for f in frames if finite(f.get("inferenceMs"))]),
                   "observedFrameAgeMs": stats([f["frameAgeMs"] for f in frames if finite(f.get("frameAgeMs"))])},
        "summary": summarize(rows, frames), "windows": windows,
        "trackingFailureRuns": {hand: runs(rows, lambda r, h=hand: not r["arms"][h]["passesTrackingGate"], maximum_gap_ms) for hand in ARMS},
        "highElbowAngleRuns": {hand: runs(rows, lambda r, h=hand: r["arms"][h]["passesTrackingGate"] and finite(r["arms"][h]["elbowAngle2d"]) and r["arms"][h]["elbowAngle2d"] >= 145, maximum_gap_ms) for hand in ARMS},
    }
    if include_timeline:
        report["timeline"] = rows
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, help="Prefer ignored data/ for personal capture reports")
    parser.add_argument("--window-ms", type=float, default=5000)
    parser.add_argument("--visibility", type=float, default=.65)
    parser.add_argument("--presence", type=float, default=.5)
    parser.add_argument("--maximum-gap-ms", type=float, default=200)
    parser.add_argument("--include-timeline", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.session.resolve():
            raise ValueError("Output must not overwrite the input session")
        report = diagnose(json.loads(args.session.read_text()), visibility=args.visibility,
                          presence=args.presence, window_ms=args.window_ms,
                          maximum_gap_ms=args.maximum_gap_ms, include_timeline=args.include_timeline,
                          allow_synthetic=args.allow_synthetic)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
            print(f"Saved pose diagnostics to {args.output}. This is not an accuracy report.")
        else:
            print(encoded, end="")
        return 0
    except (ValueError, OSError) as error:
        print(f"diagnose: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
