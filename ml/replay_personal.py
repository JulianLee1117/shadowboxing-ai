"""Compare frozen personal recognizers on saved native poses, without changing the app.

This is a cold replay from the first exported observation, not a reconstruction
of camera warmup or an independent test merely because a report was generated.
Original captured decisions, paired replay decisions and labels stay separate.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from .evaluate import build_report, distribution, validate_session
from .recognizer import ModelBundle, MAXIMUM_GAP_MS

LABELS = ("jab", "cross", "hook", "uppercut")
FRAME_INPUTS = ("t", "width", "height", "landmarks", "estimator")
LIMITATIONS = [
    "Cold replay starts at the first exported observation; camera warmup and grid phase may differ.",
    "Original captured predictions are separate evidence, not a paired model baseline.",
    "No final flush, future observations, interpolated joints or reference labels enter inference.",
    "Recorded pose timing is preserved; recognition-only replay timing is reported separately.",
    "Multiple pose versions of one video are correlated views, not independent recordings.",
    "This comparison does not establish performance on a new recording or correct technique.",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_source(path):
    raw = Path(path).read_bytes()
    source = json.loads(raw)
    validate_session(source)
    if source.get("model") not in ("rtmpose-m", "rtmw-l"):
        raise ValueError("Personal replay requires native RTMPose/RTMW observations")
    if source.get("stance") not in ("orthodox", "southpaw"):
        raise ValueError("Personal replay requires an explicit recorded stance")
    frames = source["frames"]
    if not frames:
        raise ValueError("Personal replay requires at least one saved observation")
    if any(b["t"] <= a["t"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Replay frame timestamps must be strictly increasing")
    return source, hashlib.sha256(raw).hexdigest()


def pose_info(source):
    return {k: deepcopy(v) for k, v in source["modelManifest"].items() if k != "recognizer"}


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def reset_reason(previous, current):
    if previous is None:
        return "initial"
    if (previous["width"], previous["height"]) != (current["width"], current["height"]):
        return "dimensions_changed"
    if current["t"] - previous["t"] > MAXIMUM_GAP_MS:
        return "observation_gap"
    return None


def replay(source, bundle, source_hash, *, trace=None):
    """Consume only source poses in order. Labels are copied after inference ends.

    `trace` is a passive observer of actual decoder state. Its snapshots do not
    claim a rejection reason when no explicit reason was exposed by the runtime.
    """
    state = bundle.create_session()
    output_frames, events, costs = [], [], []
    reset_counts = {}
    maximum_buffers = {k: 0 for k in ("raw", "grid", "predictions", "pending", "accepted")}
    previous, last_tick = None, None
    began = time.perf_counter()
    try:
        for saved in source["frames"]:
            # Even a custom exported frame cannot inject old decisions, a grid
            # phase, reference action labels or annotation-dependent features.
            frame = {k: deepcopy(saved[k]) for k in FRAME_INPUTS}
            reason = reset_reason(previous, frame)
            if reason:
                reset_counts[reason] = reset_counts.get(reason, 0) + 1
                last_tick = None
            decision = state.update(frame)
            output_frames.append({**deepcopy(saved), "recognition": decision})
            costs.append(decision["inferenceMs"])
            for event in decision["events"]:
                role = "lead" if (event["hand"] == "left") == (source["stance"] == "orthodox") else "rear"
                label = ("jab" if role == "lead" else "cross") if event["family"] == "straight" else event["family"]
                events.append({**event, "label": label, "role": role,
                               "extension": None, "guardReturn": "unassessable"})
            for key in maximum_buffers:
                maximum_buffers[key] = max(maximum_buffers[key], len(getattr(state, key)))
            if trace is not None:
                ticks = [p for p in state.predictions if last_tick is None or p["t"] > last_tick]
                if ticks:
                    last_tick = ticks[-1]["t"]
                trace(_plain({
                    "sourceTimeMs": frame["t"], "epoch": state.epoch,
                    "reset": reason, "state": decision["state"],
                    "ticks": ticks, "activeCandidates": state.curves,
                    "pendingCandidates": state.pending, "emitted": decision["events"],
                }))
            previous = frame
        elapsed_ms = (time.perf_counter() - began) * 1000
        unfinished = _plain({"activeCandidates": state.curves, "pendingCandidates": state.pending})
    finally:
        state.dispose()

    # Preserve annotations verbatim. They never influence the invocation above.
    output = deepcopy(source)
    output.pop("video", None)
    output.update({
        "id": f"{source['id']}-personal-replay-{source_hash[:12]}-{bundle.model_info['fingerprint'][:12]}",
        "artifactType": "detector-benchmark-session",
        "frames": output_frames, "events": events,
        "detectorVersion": "personal-hybrid-v1:" + bundle.model_info["fingerprint"],
        "modelManifest": {**pose_info(source), "recognizer": deepcopy(bundle.model_info)},
        "benchmark": {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "sourceSessionId": source["id"], "inputSessionSha256": source_hash,
            "detectorSourceSha256": bundle.model_info["runtimeSourceSha256"],
            "recognizerFeatureSourceSha256": bundle.model_info["featureSourceSha256"],
            "recognizerBundleFingerprint": bundle.model_info["fingerprint"],
            "trainingProtocolSha256": bundle.model_info["trainingProtocolSha256"],
            "trackingSource": "saved-frames", "modelId": source["model"],
            "sourceVideoSha256": source.get("benchmark", {}).get("sourceVideoSha256"),
            "initialization": "cold_at_first_exported_observation",
            "sourceBenchmark": deepcopy(source.get("benchmark")),
            "limitations": LIMITATIONS,
        },
    })
    validate_session(output)
    timing = {
        "recognitionOnlyMs": distribution(costs), "replayWallMs": elapsed_ms,
        "recordedFrameCount": len(output_frames), "resetCounts": reset_counts,
        "maximumBufferedEntries": maximum_buffers,
        "unfinalizedAtEnd": unfinished,
        "emittedAfterObservedPeakMs": distribution([e["detectedAtMs"] - e["peakMs"] for e in events]),
        "emittedAfterObservedEndMs": distribution([e["detectedAtMs"] - e["endMs"] for e in events]),
        "timingScope": "recognizer_on_saved_poses_excludes_camera_pose_transport_and_browser",
    }
    return output, timing


def match_changes(before, after):
    if before is None or after is None:
        return None
    old = {m["annotationId"] for m in before["matches"]}
    new = {m["annotationId"] for m in after["matches"]}
    return {"recoveredAnnotationIds": sorted(new - old),
            "lostAnnotationIds": sorted(old - new),
            "retainedAnnotationIds": sorted(old & new),
            "before": {k: before[k] for k in ("tp", "fp", "fn")},
            "after": {k: after[k] for k in ("tp", "fp", "fn")}}


def run(paths, manifest, output_dir, *, baseline_manifest=None, trace=False,
        peak_references=None, label_scopes=None, bundle_loader=ModelBundle.load):
    """Write immutable outputs; comparison.json exists only for a completed run."""
    paths = [Path(p).resolve() for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate input paths would double-count a view")
    scopes = label_scopes or [LABELS]
    if len(scopes) == 1:
        scopes = scopes * len(paths)
    if len(scopes) != len(paths) or any(not scope or set(scope) - set(LABELS) for scope in scopes):
        raise ValueError("Supply one nonempty label scope globally or one per input")
    executable_paths = [Path(__file__), Path(__file__).with_name("evaluate.py")]
    executable_hashes = {str(p.resolve()): digest(p) for p in executable_paths}
    if peak_references is not None and len(peak_references) != len(paths):
        raise ValueError("Provide exactly one peak reference per input session")
    reference_paths = [Path(p).resolve() for p in peak_references] if peak_references else []
    reference_bytes = [p.read_bytes() for p in reference_paths]
    references = [json.loads(raw) for raw in reference_bytes] if reference_bytes else [None] * len(paths)
    reference_hashes = [hashlib.sha256(raw).hexdigest() for raw in reference_bytes]
    sources = [read_source(p) for p in paths]
    manifests = {"candidate": Path(manifest).resolve()}
    if baseline_manifest is not None:
        manifests = {"baseline": Path(baseline_manifest).resolve(), **manifests}
    hashes = {name: digest(path) for name, path in manifests.items()}
    # Load and validate every pose/model pairing before creating output files.
    bundles = {}
    for source, _ in sources:
        key = encode(pose_info(source))
        for name, path in manifests.items():
            if (name, key) not in bundles:
                bundles[name, key] = bundle_loader(path, pose_info(source))
            if bundles[name, key].model_info["fingerprint"] != hashes[name]:
                raise ValueError("Loaded model fingerprint differs from its pinned manifest")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, ((source, source_hash), reference, path) in enumerate(zip(sources, references, paths)):
        prefix = f"{i:03d}"
        options = {"labels": tuple(scopes[i])}
        if reference is not None:
            options["peak_references"] = {source["id"]: reference}
        saved_report = build_report([source], **options)
        write_new(root / f"{prefix}-captured-report.json", saved_report)
        row = {"input": str(path), "inputSha256": source_hash,
               "sourceSessionId": source["id"], "capturedReport": f"{prefix}-captured-report.json",
               "capturedEventMetrics": saved_report["eventMetrics"], "replays": {},
               "recallLabels": list(scopes[i]),
               "scopeNote": "Strict metrics preserve the evaluator's historical rule: unmatched predictions outside recall labels still count. Their presence alone does not prove a nonpunch error when those families were not reviewed; inspect perClass and video.",
               "peakReference": {"path": str(reference_paths[i]), "sha256": reference_hashes[i]}
               if reference is not None else None}
        reports = {}
        for name in manifests:
            bundle = bundles[name, encode(pose_info(source))]
            trace_file = (root / f"{prefix}-{name}-trace.jsonl").open("x") if trace else None
            try:
                derived, timing = replay(source, bundle, source_hash,
                    trace=(lambda item: trace_file.write(encode(item))) if trace_file else None)
            finally:
                if trace_file:
                    trace_file.close()
            options = {"labels": tuple(scopes[i])}
            if reference is not None:
                options["peak_references"] = {derived["id"]: reference}
            report = build_report([derived], **options)
            reports[name] = report
            write_new(root / f"{prefix}-{name}-session.json", derived)
            write_new(root / f"{prefix}-{name}-report.json", report)
            row["replays"][name] = {
                "report": f"{prefix}-{name}-report.json", "eventMetrics": report["eventMetrics"],
                "timing": timing, "modelFingerprint": bundle.model_info["fingerprint"],
                "personalTraining": bundle.model_info.get("personalTraining"),
            }
        if "baseline" in reports:
            row["pairedStrictChanges"] = match_changes(
                reports["baseline"]["sessions"][0]["eventMetrics"],
                reports["candidate"]["sessions"][0]["eventMetrics"])
            occurrence = [reports[n]["sessions"][0].get("peakOccurrenceDiagnostics")
                          for n in ("baseline", "candidate")]
            row["pairedPeakOccurrenceChanges"] = match_changes(*[
                {**d["metrics"], "matches": d["matches"]} if d and d["metrics"] is not None else None
                for d in occurrence])
        rows.append(row)
        print(f"{prefix}: {path.name} replayed", flush=True)
    # A long run must not silently mix source/runtime/model generations.
    for path, (_, expected) in zip(paths, sources):
        if digest(path) != expected:
            raise ValueError("Input changed during replay; no completed comparison was written")
    for name, path in manifests.items():
        if digest(path) != hashes[name]:
            raise ValueError("Model manifest changed during replay")
    for path, expected in zip(reference_paths, reference_hashes):
        if digest(path) != expected:
            raise ValueError("Peak reference changed during replay")
    for bundle in bundles.values():
        if digest(Path(__file__).with_name("recognizer.py")) != bundle.model_info["runtimeSourceSha256"]:
            raise ValueError("Recognizer source changed during replay")
        if digest(Path(__file__).with_name("recognizer_features.py")) != bundle.model_info["featureSourceSha256"]:
            raise ValueError("Feature source changed during replay")
    for path, expected in executable_hashes.items():
        if digest(path) != expected:
            raise ValueError("Replay/evaluation source changed during the comparison")
    result = {"version": "personal-replay-1", "status": "completed_development_comparison",
              "createdAt": datetime.now(timezone.utc).isoformat(),
              "manifests": {n: {"path": str(p), "sha256": hashes[n]} for n, p in manifests.items()},
              "executableSourceHashes": executable_hashes,
              "limitations": LIMITATIONS, "independentRecordingCount": None,
              "views": rows}
    write_new(root / "comparison.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trace", action="store_true", help="Save passive classifier/decoder snapshots as JSONL")
    parser.add_argument("--labels", action="append", required=True,
                        help="Explicit reviewed recall labels, comma separated; once globally or once per input. Does not filter unmatched predictions.")
    parser.add_argument("--peak-reference", action="append", type=Path,
                        help="Independent video peak reference, once per session in input order")
    args = parser.parse_args(argv)
    try:
        run(args.sessions, args.manifest, args.output_dir, baseline_manifest=args.baseline_manifest,
            trace=args.trace, peak_references=args.peak_reference,
            label_scopes=[tuple(dict.fromkeys(value.split(","))) for value in args.labels])
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Personal replay failed: {error}", file=sys.stderr)
        return 2
    print(f"Completed development comparison: {args.output_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
