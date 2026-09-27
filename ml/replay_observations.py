"""Replay saved native observations without inventing stance or action references.

Consumes local-pose video observation envelopes, not browser Sessions or pose
pickles. Writes physical-hand/family decisions and optional passive state traces.
No camera, downloads, pose inference, evaluation, training or model activation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from .evaluate import distribution, native_estimator, valid_fingerprint
from .recognizer import ModelBundle, _verified_path
from .replay_personal import FRAME_INPUTS, _plain, digest, encode, reset_reason, write_new

VERSION = "native-observation-replay-1"
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_FRAMES = 10_000
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
LIMITATIONS = [
    "Cold state starts at the first saved observation; original warmup may be absent.",
    "Physical hands and families remain physical; stance, jab/cross roles and form are not inferred.",
    "Source group, continuity and clock semantics are declared provenance, not verified identity or synchronization.",
    "Only declared shot boundaries, source gaps and size changes reset state; no cut or actor is inferred.",
    "No reference labels, saved decisions, synthetic context or end flush enter inference.",
    "Native joint scores and model support are uncalibrated; neither establishes correct form.",
    "Timing excludes source decode, pose, camera, transport and display; this is not webcam throughput.",
    "This observation-only artifact supplies no recognition accuracy or training eligibility.",
]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path, maximum):
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Input exceeds its byte limit")
    # Reject nonfinite values even in metadata rather than emitting invalid JSON.
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    def decimal(value):
        number = float(value)
        return number if math.isfinite(number) else invalid(value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result
    return json.loads(raw, parse_constant=invalid, parse_float=decimal,
                      object_pairs_hook=pairs), digest_bytes(raw)


def digest_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def read_source(path, provenance_path, *, require_parity=False):
    source, source_hash = read_json(path, MAX_INPUT_BYTES)
    provenance, provenance_hash = read_json(provenance_path, 100_000)
    if not isinstance(provenance, dict) or provenance.get("purpose") != "inspection-only":
        raise ValueError("Provenance must declare inspection-only purpose")
    for key in ("sourceGroup", "timestampSemantics"):
        if not isinstance(provenance.get(key), str) or not 1 <= len(provenance[key]) <= 500:
            raise ValueError(f"Provenance requires an explicit {key}")
    if (not isinstance(source, dict) or not isinstance(source.get("sourceId"), str)
            or not 1 <= len(source["sourceId"]) <= 500):
        raise ValueError("Local-pose observation envelope requires sourceId")
    video_hash = source.get("sourceSha256")
    if not valid_fingerprint(video_hash) or provenance.get("videoSha256") != video_hash:
        raise ValueError("Provenance video SHA-256 must match the observation envelope")
    info = source.get("modelInfo")
    if (not isinstance(info, dict) or info.get("id") not in ("rtmpose-m", "rtmw-l")
            or not native_estimator(info.get("estimator")) or info["estimator"]["id"] != info["id"]):
        raise ValueError("Native pose model and SimCC estimator metadata required")
    frames = source.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError("Require 1..10000 saved observations")
    if type(source.get("sourceFrameCount")) is not int or source["sourceFrameCount"] != len(frames):
        raise ValueError("Declared source frame count must match observations")
    previous_t, previous_index = -1, -1
    indices = set()
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError("Every source row must be an object")
        index, t, frame = row.get("sourceFrameIndex0Based"), row.get("t"), row.get("result")
        if type(index) is not int or index <= previous_index:
            raise ValueError("Native frame indices must be strictly increasing and nonnegative")
        if not finite(t) or t < 0 or t <= previous_t:
            raise ValueError("Source timestamps must be finite, nonnegative and strictly increasing")
        if not isinstance(frame, dict) or set(FRAME_INPUTS) - frame.keys():
            raise ValueError("Saved result is missing a native recognizer input")
        if not finite(frame["t"]) or frame["t"] != t:
            raise ValueError("Row and result timestamps must agree exactly")
        if frame["estimator"] != info["estimator"]:
            raise ValueError("Every native score policy must match modelInfo")
        if any(type(frame[k]) is not int or not 1 <= frame[k] <= 16384 for k in ("width", "height")):
            raise ValueError("Frame dimensions must be bounded positive integers")
        points = frame["landmarks"]
        if not isinstance(points, list) or len(points) not in (0, 33):
            raise ValueError("Native service landmarks must be empty or partial MediaPipe33")
        for point in points:
            if not isinstance(point, dict) or any(not finite(point.get(k)) for k in ("x", "y", "score")):
                raise ValueError("Each native point requires finite x/y/score; visibility cannot replace score")
        if require_parity and not isinstance(frame.get("recognition"), dict):
            raise ValueError("Captured parity requires a recognition record on every frame")
        previous_t, previous_index = t, index
        indices.add(index)
    boundaries = provenance.get("resetBeforeFrameIndices", [])
    if (not isinstance(boundaries, list) or any(type(i) is not int or i not in indices for i in boundaries)
            or boundaries != sorted(set(boundaries))):
        raise ValueError("Declared reset boundaries must be sorted unique observed frame indices")
    if source.get("partial") or source.get("error"):
        raise ValueError("Partial/failed source envelopes need an explicit separate completed subset")
    return source, provenance, source_hash, provenance_hash


def recognition_content(value):
    return {key: deepcopy(item) for key, item in value.items() if key != "inferenceMs"}


class BoundedLines:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8")
        self.bytes = 0

    def write(self, value):
        line = encode(_plain(value))
        size = len(line.encode("utf-8"))
        if self.bytes + size > MAX_OUTPUT_BYTES:
            raise ValueError("Diagnostic output byte limit reached; preserve this partial run")
        self.stream.write(line)
        self.stream.flush()
        self.bytes += size

    def close(self):
        self.stream.close()


def verify_pins(pins):
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError("Pinned input changed during replay; completion withheld")


def run(path, manifest, output_dir, *, provenance_path, trace=False,
        require_parity=False, bundle_loader=ModelBundle.load):
    path, manifest, provenance_path = [Path(p).resolve() for p in (path, manifest, provenance_path)]
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("Output exists; choose a new immutable output directory")
    source, provenance, source_hash, provenance_hash = read_source(path, provenance_path, require_parity=require_parity)
    model, model_hash = read_json(manifest, 100_000)
    if not isinstance(model, dict):
        raise ValueError("Recognizer manifest must be an object")
    # The normal loader also checks compatibility. Pin the actual local weight
    # bytes before load, then verify them again after the entire replay.
    weight_paths = [_verified_path(manifest.parent, model.get(key)) for key in ("checkpoint", "externalModel")]
    helpers = [Path(__file__), Path(__file__).with_name("replay_personal.py"),
               Path(__file__).with_name("evaluate.py"), Path(__file__).with_name("recognizer.py"),
               Path(__file__).with_name("recognizer_features.py")]
    pins = {str(p.resolve()): digest(p) for p in [*helpers, *weight_paths]}
    pins.update({str(path): source_hash, str(provenance_path): provenance_hash, str(manifest): model_hash})
    pose_info = {k: deepcopy(v) for k, v in source["modelInfo"].items() if k != "recognizer"}
    bundle = bundle_loader(manifest, pose_info)
    if bundle.model_info["fingerprint"] != model_hash:
        raise ValueError("Loaded bundle fingerprint differs from pinned manifest")
    for filename, field in (("recognizer.py", "runtimeSourceSha256"), ("recognizer_features.py", "featureSourceSha256")):
        if bundle.model_info[field] != pins[str(Path(__file__).with_name(filename).resolve())]:
            raise ValueError("Loaded executable fingerprint differs from pinned source")
    verify_pins(pins)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "input-lock.json", pins)
    write_new(root / "protocol.json", {
        "schemaVersion": VERSION, "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceId": source["sourceId"], "sourceGroup": provenance["sourceGroup"],
        "provenance": deepcopy(provenance), "sourceVideoSha256": source["sourceSha256"],
        "inputFields": list(FRAME_INPUTS), "maximumFrames": MAX_FRAMES,
        "maximumBytesPerOutput": MAX_OUTPUT_BYTES, "trace": trace,
        "requireCapturedParityExcludingInferenceMs": require_parity,
        "modelInfo": deepcopy(bundle.model_info), "limitations": LIMITATIONS,
    })
    state, decisions, traces = None, None, None
    processed, events, costs, resets = 0, [], [], {}
    previous, last_tick = None, None
    boundaries = set(provenance.get("resetBeforeFrameIndices", []))
    began = time.perf_counter()
    try:
        state = bundle.create_session()
        decisions = BoundedLines(root / "decisions.jsonl")
        traces = BoundedLines(root / "trace.jsonl") if trace else None
        for row in source["frames"]:
            saved = row["result"]
            frame = {k: deepcopy(saved[k]) for k in FRAME_INPUTS}
            reason = reset_reason(previous, frame)
            if row["sourceFrameIndex0Based"] in boundaries:
                state.reset()
                reason = "declared_source_boundary"
            if reason:
                resets[reason] = resets.get(reason, 0) + 1
                last_tick = None
            decision = state.update(frame)
            recorded = saved.get("recognition")
            parity = recognition_content(decision) == recognition_content(recorded) if isinstance(recorded, dict) else None
            decisions.write({"sourceFrameIndex0Based": row["sourceFrameIndex0Based"], "t": frame["t"],
                             "recognition": decision, "capturedRecognition": deepcopy(recorded),
                             "capturedParityExcludingInferenceMs": parity})
            processed += 1
            events.extend(deepcopy(decision["events"]))
            costs.append(decision["inferenceMs"])
            if traces:
                ticks = [p for p in state.predictions if last_tick is None or p["t"] > last_tick]
                if ticks:
                    last_tick = ticks[-1]["t"]
                traces.write({"sourceFrameIndex0Based": row["sourceFrameIndex0Based"],
                              "sourceTimeMs": frame["t"], "epoch": state.epoch, "reset": reason,
                              "state": decision["state"], "ticks": ticks, "activeCandidates": state.curves,
                              "pendingCandidates": state.pending, "emitted": decision["events"]})
            if require_parity and not parity:
                raise ValueError("Captured parity failed; first mismatching decision preserved")
            previous = frame
        verify_pins(pins)
        summary = {
            "schemaVersion": VERSION, "sourceId": source["sourceId"],
            "sourceGroup": provenance["sourceGroup"], "sourceVideoSha256": source["sourceSha256"],
            "inputSha256": source_hash, "frameCount": processed,
            "firstSourceTimeMs": source["frames"][0]["t"], "lastSourceTimeMs": source["frames"][-1]["t"],
            "events": events, "resetCounts": resets, "recognitionOnlyMs": distribution(costs),
            "replayWallMs": (time.perf_counter() - began) * 1000,
            "unfinalizedAtEnd": _plain({"activeCandidates": state.curves, "pendingCandidates": state.pending}),
            "requiredCapturedParityPassed": True if require_parity else None,
            "modelInfo": deepcopy(bundle.model_info), "limitations": LIMITATIONS,
        }
    except Exception as error:
        write_new(root / "failure.json", {"schemaVersion": VERSION, "processedFrames": processed,
                                          "errorType": type(error).__name__, "message": str(error)})
        raise
    finally:
        for writer in (decisions, traces):
            if writer:
                writer.close()
        if state:
            state.dispose()
    # Close output streams and dispose runtime before claiming completion.
    write_new(root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--require-captured-parity", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args.source, args.manifest, args.output_dir, provenance_path=args.provenance,
                     trace=args.trace, require_parity=args.require_captured_parity)
    except (ValueError, OSError, KeyError) as error:
        print(f"Observation replay failed: {error}", file=sys.stderr)
        return 1
    print(f"Replayed {result['frameCount']} observations; {len(result['events'])} physical-hand events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
