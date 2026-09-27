"""Opt-in personal causal recognition. No model downloads or default activation.

The bundled personal checkpoint has development-only curved-punch evidence.
It is not a technique coach, calibrated probability model, or identity verifier.
Weights are pinned and loaded as tensor-only state dictionaries; architecture
and feature code are local and fixed. Runtime dependencies load only on opt-in.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

PROTOCOL_VERSION = "shadowbox-recognition-v1"
RECOGNIZER_ID = "personal-hybrid-v1"
FEATURE_VERSION = "arm-offsets-native-v3"
MAXIMUM_GAP_MS = 150
HISTORY_MS = 4200
LOOKAHEAD_MS = 300
DECODER_VERSION = "observed-personal-cycles-v3"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_path(directory, item):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError("Recognizer weight entry requires a local path and SHA-256")
    path = (directory / item["path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(
            "Recognizer weights must remain inside their manifest directory"
        )
    if not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError("Recognizer weight file is missing or exceeds its size limit")
    if _digest(path) != item.get("sha256"):
        raise ValueError("Recognizer weight SHA-256 mismatch")
    return path


def _validate_pose(manifest, actual):
    required = manifest.get("pose", {})
    estimator = actual.get("estimator", {})
    if actual.get("id") != required.get("id") or estimator != required.get("estimator"):
        raise ValueError("Recognizer pose estimator or native score policy mismatch")
    for key in ("pose", "detector"):
        weight = actual.get("modelManifest", {}).get(key, {})
        if weight.get("sha256") != required.get(key + "Sha256"):
            raise ValueError("Recognizer pose/detector checkpoint mismatch")
        expected_color = required.get(key + "InputColorOrder")
        if expected_color and weight.get("inputColorOrder") != expected_color:
            raise ValueError("Recognizer model input color contract mismatch")
    post = actual.get("detectorPostprocessing", {})
    if post.get("effectivePostNmsScoreThreshold") != required.get(
        "detectorScoreThreshold"
    ):
        raise ValueError("Recognizer person detector score policy mismatch")


def _networks(torch):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, channels, dilation):
            super().__init__()
            self.d = dilation
            self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
            # Retained checkpoint key; normalization below is per-time/channel.
            self.norm = nn.GroupNorm(1, channels)

        def forward(self, x):
            y = self.conv(torch.nn.functional.pad(x, (2 * self.d, 0)))
            y = torch.nn.functional.layer_norm(
                y.transpose(1, 2), (y.shape[1],)
            ).transpose(1, 2)
            return x + torch.nn.functional.relu(y)

    class Temporal(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Conv1d(40, 48, 1)
            self.blocks = nn.ModuleList(
                [Block(48, dilation) for dilation in (1, 2, 4, 8)]
            )
            self.out = nn.Conv1d(48, 4, 1)

        def forward(self, x):
            x = torch.nn.functional.relu(self.input(x))
            for block in self.blocks:
                x = block(x)
            return self.out(x)

    class Family(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(16, 128, 2, batch_first=True, dropout=0.5)
            self.fc = nn.Linear(128, 4)

        def forward(self, x):
            result, _ = self.lstm(x)
            return self.fc(result[:, -1])

    return Temporal(), Family()


class ModelBundle:
    """Shared immutable weights; every HTTP session must have its own state."""

    @classmethod
    def load(cls, manifest_path, pose_model_info):
        manifest_path = Path(manifest_path).resolve()
        if manifest_path.stat().st_size > 100_000:
            raise ValueError("Recognizer manifest exceeds its size limit")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("protocolVersion") != PROTOCOL_VERSION
            or manifest.get("recognizerId") != RECOGNIZER_ID
            or manifest.get("featureVersion") != FEATURE_VERSION
            or manifest.get("architecture") != "causal-arm-tcn-40-48-d1248-v1"
            or manifest.get("decoderVersion") != DECODER_VERSION
        ):
            raise ValueError(
                "Unsupported recognizer protocol, features, or architecture"
            )
        if manifest.get("runtimeSourceSha256") != _digest(__file__) or manifest.get(
            "featureSourceSha256"
        ) != _digest(Path(__file__).with_name("recognizer_features.py")):
            raise ValueError(
                "Recognizer executable feature/runtime fingerprint mismatch"
            )
        training_summary = manifest.get("personalTraining")
        training_hash = manifest.get("trainingProtocolSha256")
        if (
            not isinstance(training_summary, str)
            or not 1 <= len(training_summary) <= 500
        ):
            raise ValueError("Recognizer requires a bounded training summary")
        if (
            not isinstance(training_hash, str)
            or len(training_hash) != 64
            or any(c not in "0123456789abcdef" for c in training_hash)
        ):
            raise ValueError("Recognizer requires its training protocol SHA-256")
        _validate_pose(manifest, pose_model_info)
        temporal_path = _verified_path(manifest_path.parent, manifest.get("checkpoint"))
        family_path = _verified_path(
            manifest_path.parent, manifest.get("externalModel")
        )
        import numpy as np
        import scipy.signal as signal
        import torch
        from .recognizer_features import (
            temporal_features,
            family_features,
            classification_features,
        )

        torch.set_num_threads(1)
        temporal, family = _networks(torch)
        for network, path in ((temporal, temporal_path), (family, family_path)):
            state = torch.load(path, weights_only=True, map_location="cpu")
            if not isinstance(state, dict) or not all(
                torch.is_tensor(value) for value in state.values()
            ):
                raise ValueError("Recognizer checkpoint must contain tensors only")
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Recognizer checkpoint contains nonfinite tensors")
            network.load_state_dict(state, strict=True)
            network.eval()
        instance = cls()
        instance.np, instance.signal, instance.torch = np, signal, torch
        instance.temporal, instance.family = temporal, family
        instance.temporal_features, instance.family_features = (
            temporal_features,
            family_features,
        )
        instance.classification_features = classification_features
        instance.expected_estimator = manifest["pose"]["estimator"]
        instance.model_info = {
            "protocolVersion": PROTOCOL_VERSION,
            "recognizerId": RECOGNIZER_ID,
            "fingerprint": _digest(manifest_path),
            "featureVersion": FEATURE_VERSION,
            "runtimeSourceSha256": _digest(__file__),
            "featureSourceSha256": _digest(
                Path(__file__).with_name("recognizer_features.py")
            ),
            "checkpointSha256": _digest(temporal_path),
            "externalModelSha256": _digest(family_path),
            "poseModelSha256": manifest["pose"]["poseSha256"],
            "developmentOnly": True,
            "scoreSemantics": "uncalibrated_model_support",
            "packages": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
            },
            "personalTraining": training_summary,
            "decoderVersion": DECODER_VERSION,
            "trainingProtocolSha256": training_hash,
            "maximumUncertainArmRetentionMs": 250,
            "timeGridHz": 30,
            "maximumNormalizationCacheMs": 250,
            "maximumContextMs": HISTORY_MS,
            "arbitrationDelayAfterEventEndMs": LOOKAHEAD_MS,
        }
        return instance

    def create_session(self):
        return RecognizerSession(self)

    def temporal_probabilities(self, frames):
        values, valid = self.temporal_features(frames)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(
                self.temporal(self.torch.tensor(values)), 1
            ).numpy()
        return probabilities, valid

    def family_probabilities(self, features):
        x = features[:25]
        x = self.np.pad(x, ((0, max(0, 25 - len(x))), (0, 0)))
        with self.torch.inference_mode():
            values = self.torch.softmax(
                self.family(self.torch.tensor(x[None], dtype=self.torch.float32)), 1
            )[0].numpy()
        return self.np.array([values[0] + values[1], values[2], values[3]])

    def straight_proposals(self, frames):
        """Bounded prominence proposals. No annotation/drill/stance input."""
        np, signal = self.np, self.signal
        features, geometry, scores = self.family_features(frames)
        times = np.array([frame["t"] for frame in frames])
        dt = float(np.median(np.diff(times)))
        separation = max(3, int(180 / dt))
        window = max(5, int(1100 / dt) // 2 * 2 + 1)
        proposals = []
        for h, hand in enumerate(("left", "right")):
            valid = np.isfinite(geometry[:, h]).all(1) & (
                scores[:, [h, 2 + h, 4 + h]].min(1) >= 0.55
            )
            edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
            for lo, hi in zip(edges[::2], edges[1::2]):
                if hi - lo < 7:
                    continue
                reach = signal.medfilt(geometry[lo:hi, h, 4], 3)
                peaks, properties = signal.find_peaks(
                    reach, prominence=0.20, distance=separation, wlen=window
                )
                if not len(peaks):
                    continue
                _, _, lefts, rights = signal.peak_widths(
                    reach,
                    peaks,
                    rel_height=0.8,
                    prominence_data=(
                        properties["prominences"],
                        properties["left_bases"],
                        properties["right_bases"],
                    ),
                )
                for n, offset in enumerate(peaks):
                    peak = lo + int(offset)
                    start = max(lo, lo + int(math.floor(lefts[n])) - 1)
                    end = min(hi - 1, lo + int(math.ceil(rights[n])) + 1)
                    if not 180 <= times[end] - times[start] <= 1500:
                        continue
                    probabilities = self.family_probabilities(features[start : end + 1])
                    shape = geometry[peak, h]
                    if (
                        probabilities.argmax() != 0
                        or probabilities[0] < 0.60
                        or shape[4] < 0.55
                        or shape[5] < 100
                        or shape[1] > 0.65
                    ):
                        continue
                    proposals.append(
                        {
                            "hand": hand,
                            "family": "straight",
                            "startMs": float(times[start]),
                            "peakMs": float(times[peak]),
                            "endMs": float(times[end]),
                            "score": float(probabilities[0]),
                            "_rank": float(
                                probabilities[0] * properties["prominences"][n]
                            ),
                        }
                    )
        selected = []
        for event in sorted(proposals, key=lambda event: event["_rank"], reverse=True):
            if not any(
                event["hand"] == other["hand"]
                and abs(event["peakMs"] - other["peakMs"]) < 220
                for other in selected
            ):
                selected.append(event)
        return selected


class RecognizerSession:
    def __init__(self, bundle):
        self.bundle = bundle
        self.epoch = 0
        self.reset()

    def reset(self):
        self.epoch += 1
        self.raw = []
        self.grid = []
        self.next_tick = None
        self.previous_t = None
        self.dimensions = None
        self.curves = [None, None]
        self.pending = []
        self.accepted = []
        self.seen_straights = []
        self.counter = 0
        self.last_proposal_t = -1e9
        self.predictions = []

    def dispose(self):
        self.reset()

    def _curves_at(self, probabilities, valid, frame):
        t = frame["t"]
        for h, hand in enumerate(("left", "right")):
            p = probabilities[h]
            label = int(p.argmax())
            active = label != 0 and p[label] >= 0.65 and valid[h]
            state = self.curves[h]
            # Missing current geometry cannot become a new event or recovery.
            # Brief losses retain classification state only, never joint positions.
            if state is not None and not valid[h]:
                if state.get("uncertainSince") is None:
                    state["uncertainSince"] = t
                if t - state["sourceLast"] > 250:
                    self.curves[h] = None
                continue
            if (
                state is not None
                and state.get("uncertainSince") is not None
                and t - state["sourceLast"] > 250
            ):
                self.curves[h] = None
                state = None
            if state is not None:
                state["uncertainSince"] = None
            if state is None:
                if active:
                    self.curves[h] = {
                        "start": t,
                        "last": t,
                        "label": label,
                        "peak": t,
                        "score": float(p[label]),
                        "frames": 1,
                        "gap": None,
                        "sourceStart": frame.get("_sourceT", t),
                        "sourceLast": frame.get("_sourceT", t),
                        "sourceCount": 1,
                    }
                continue
            if active and label == state["label"]:
                state["last"] = t
                if frame.get("_sourceT", t) != state["sourceLast"]:
                    state["sourceCount"] += 1
                state["sourceLast"] = frame.get("_sourceT", t)
                state["frames"] += 1
                state["gap"] = None
                if p[label] > state["score"]:
                    state["score"] = float(p[label])
                    state["peak"] = t
                if t - state["start"] > 1800:
                    self.curves[h] = None
                continue
            if state["gap"] is None:
                state["gap"] = t
            if t - state["gap"] < 100:
                continue
            if (
                state["label"] in (1, 2, 3)
                and state["last"] - state["start"] >= 66
                and state["frames"] >= 3
                and state["sourceCount"] >= 3
                and state["sourceLast"] - state["sourceStart"] >= 66
            ):
                samples = [
                    value
                    for value in self.grid
                    if state["start"] <= value["t"] <= state["last"]
                ]
                _, geometry, peak_scores = self.bundle.classification_features(samples)
                # Peak is an observed projected trajectory extremum, not the
                # time of maximum network confidence or inferred impact.
                points = geometry[:, h]
                finite = self.bundle.np.isfinite(points).all(1) & (
                    peak_scores[:, [2 + h, 4 + h]].min(1) >= 0.55
                )
                if state["label"] == 1:
                    # A learned straight label still needs an observed extension.
                    # Use the same peak shape as geometric straight proposals;
                    # prominence is deliberately not required here.
                    finite &= (
                        (points[:, 4] >= 0.55)
                        & (points[:, 5] >= 100)
                        & (points[:, 1] <= 0.65)
                    )
                if finite.any():
                    ids = self.bundle.np.flatnonzero(finite)
                    if state["label"] == 3:
                        selected = ids[points[ids, 1].argmin()]
                    elif state["label"] == 1:
                        selected = ids[points[ids, 4].argmax()]
                    else:
                        motion = self.bundle.np.linalg.norm(
                            points[ids, :2] - points[ids[0], :2], axis=1
                        )
                        selected = ids[motion.argmax()]
                    peak = samples[int(selected)].get(
                        "_sourceT", samples[int(selected)]["t"]
                    )
                else:
                    self.curves[h] = None
                    continue
                intervals = [(state["sourceStart"], peak, state["sourceLast"])]
                if state["label"] == 1:
                    # Same-family fast repeats need an observed return trough,
                    # not a change in the family's neural label.
                    observed = []
                    previous = None
                    for sample in samples:
                        source_t = sample.get("_sourceT", sample["t"])
                        if source_t != previous:
                            observed.append({**sample, "t": source_t})
                            previous = source_t
                    _, raw_points, raw_scores = self.bundle.classification_features(
                        observed
                    )
                    valid_points = self.bundle.np.isfinite(raw_points[:, h]).all(1) & (
                        raw_scores[:, [h, 2 + h, 4 + h]].min(1) >= 0.55
                    )
                    ids = self.bundle.np.flatnonzero(valid_points)
                    if len(ids) >= 5:
                        reach = raw_points[ids, h, 4]
                        peaks, _ = self.bundle.signal.find_peaks(reach, prominence=0.20)
                        peaks = [
                            int(v)
                            for v in peaks
                            if raw_points[ids[v], h, 5] >= 100
                            and raw_points[ids[v], h, 4] >= 0.55
                            and raw_points[ids[v], h, 1] <= 0.65
                        ]
                        selected = []
                        for v in sorted(peaks, key=lambda v: reach[v], reverse=True):
                            if all(
                                abs(observed[ids[v]]["t"] - observed[ids[q]]["t"])
                                >= 180
                                for q in selected
                            ):
                                selected.append(v)
                        selected.sort()
                        if len(selected) >= 2 and all(
                            observed[ids[k + 1]]["t"] - observed[ids[k]]["t"] <= 100
                            for k in range(selected[0], selected[-1])
                        ):
                            valleys = [
                                int(a + self.bundle.np.argmin(reach[a : b + 1]))
                                for a, b in zip(selected, selected[1:])
                            ]
                            bounds = (
                                [state["sourceStart"]]
                                + [observed[ids[v]]["t"] for v in valleys]
                                + [state["sourceLast"]]
                            )
                            intervals = [
                                (bounds[k], observed[ids[v]]["t"], bounds[k + 1])
                                for k, v in enumerate(selected)
                            ]
                for start, peak, end in intervals:
                    if end - start < 66:
                        continue
                    self.pending.append(
                        {
                            "hand": hand,
                            "family": {1: "straight", 2: "hook", 3: "uppercut"}[
                                state["label"]
                            ],
                            "startMs": start,
                            "peakMs": peak,
                            "endMs": end,
                            "score": state["score"],
                            "_proposal": "temporal",
                            "_ready": max(t, state["last"] + LOOKAHEAD_MS),
                        }
                    )
            self.curves[h] = None
            if active:
                self.curves[h] = {
                    "start": t,
                    "last": t,
                    "label": label,
                    "peak": t,
                    "score": float(p[label]),
                    "frames": 1,
                    "gap": None,
                    "sourceStart": frame.get("_sourceT", t),
                    "sourceLast": frame.get("_sourceT", t),
                    "sourceCount": 1,
                }

    def update(self, frame):
        started = time.perf_counter()
        t = frame.get("t")
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError(
                "Recognizer requires a finite nonnegative source timestamp"
            )
        if frame.get("estimator") != self.bundle.expected_estimator:
            raise ValueError("Recognizer frame estimator metadata mismatch")
        dimensions = (frame.get("width"), frame.get("height"))
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in dimensions
        ):
            raise ValueError("Recognizer frame dimensions are invalid")
        if (
            self.previous_t is not None
            and (t <= self.previous_t or t - self.previous_t > MAXIMUM_GAP_MS)
        ) or (self.dimensions is not None and dimensions != self.dimensions):
            self.reset()
        self.previous_t, self.dimensions = t, dimensions
        self.raw.append(frame)
        self.raw = [value for value in self.raw if t - value["t"] <= HISTORY_MS]
        self.counter += 1
        if self.next_tick is None:
            self.next_tick = t
        added = []
        while self.next_tick <= t + 1e-6:
            source = next(
                (
                    value
                    for value in reversed(self.raw)
                    if value["t"] <= self.next_tick + 1e-6
                ),
                None,
            )
            if source is None:
                self.next_tick += 1000 / 30
                continue
            age = self.next_tick - source["t"]
            previous_source = self.grid[-1].get("_sourceT") if self.grid else None
            item = {
                **source,
                "t": self.next_tick,
                "_sourceT": source["t"],
                "_ageMs": age,
                "_newSample": previous_source != source["t"],
                "landmarks": source.get("landmarks", []) if age <= 100 else [],
            }
            self.grid.append(item)
            added.append(item)
            self.next_tick += 1000 / 30
        self.grid = [value for value in self.grid if t - value["t"] <= HISTORY_MS]
        if added:
            context = self.grid[-(31 + len(added)) :]
            probabilities, valid = self.bundle.temporal_probabilities(context)
            for i, value in enumerate(added, start=len(context) - len(added)):
                self.predictions.append(
                    {
                        "t": value["t"],
                        "sourceT": value.get("_sourceT", value["t"]),
                        "p": probabilities[:, :, i].copy(),
                        "valid": valid[:, i].copy(),
                    }
                )
                self._curves_at(probabilities[:, :, i], valid[:, i], value)
            self.predictions = [p for p in self.predictions if t - p["t"] <= HISTORY_MS]
        if len(self.raw) >= 12 and t - self.last_proposal_t >= 66:
            self.last_proposal_t = t
            for event in self.bundle.straight_proposals(self.raw):
                h = 0 if event["hand"] == "left" else 1
                support = [
                    p["p"][h]
                    for p in self.predictions
                    if event["startMs"] <= p["sourceT"] <= event["endMs"]
                    and p["valid"][h]
                ]
                if not support or self.bundle.np.mean(support, axis=0)[1] < 0.5:
                    continue
                if (
                    t - event["endMs"] < 66
                    or t - event["peakMs"] < 150
                    or t - event["peakMs"] > 1800
                ):
                    continue
                if self.raw[0]["t"] > 0 and event["startMs"] < self.raw[0]["t"] + 550:
                    continue
                if any(
                    event["hand"] == old["hand"]
                    and abs(event["peakMs"] - old["peakMs"]) < 220
                    for old in self.seen_straights
                ):
                    continue
                self.seen_straights.append(event)
                self.pending.append(
                    {
                        **event,
                        "_proposal": "geometry",
                        "_ready": max(t, event["endMs"] + LOOKAHEAD_MS),
                    }
                )
        events = []
        for event in sorted(
            self.pending,
            key=lambda value: (value.get("_proposal") != "geometry", value["startMs"]),
        ):
            if event["_ready"] > t:
                continue
            overlaps = lambda other: event["hand"] == other["hand"] and min(
                event["endMs"], other["endMs"]
            ) > max(event["startMs"], other["startMs"])
            if (
                event.get("_proposal") == "temporal"
                and event["family"] == "straight"
                and any(
                    overlaps(other)
                    for other in self.accepted + self.pending
                    if other is not event and other.get("_proposal") == "geometry"
                )
            ):
                continue
            if any(
                event["hand"] == other["hand"]
                and abs(event["peakMs"] - other["peakMs"]) < 220
                for other in self.accepted
            ):
                continue
            emitted = {
                key: event[key]
                for key in ("hand", "family", "startMs", "peakMs", "endMs", "score")
            }
            emitted.update(
                {
                    "id": f"{self.epoch}-{event['hand']}-{event['family']}-{event['startMs']:.3f}-{event['peakMs']:.3f}",
                    "detectedAtMs": t,
                }
            )
            events.append(emitted)
            self.accepted.append({**emitted, "_proposal": event.get("_proposal")})
        self.pending = [event for event in self.pending if event["_ready"] > t]
        self.accepted = [
            event for event in self.accepted if t - event["endMs"] <= HISTORY_MS
        ]
        self.seen_straights = [
            event for event in self.seen_straights if t - event["peakMs"] <= HISTORY_MS
        ]
        pins = {
            key: self.bundle.model_info[key]
            for key in ("protocolVersion", "recognizerId", "fingerprint")
        }
        return {
            **pins,
            "events": events,
            "state": (
                "warming"
                if len(self.grid) < 31
                else "active" if frame.get("landmarks") else "uncertain"
            ),
            "inferenceMs": (time.perf_counter() - started) * 1000,
        }
