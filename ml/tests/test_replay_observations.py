from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml.replay_observations import read_source, run
from ml.replay_personal import digest


POLICY = {"id": "rtmpose-m", "scoreType": "simcc", "minimumScore": .55}


class State:
    def __init__(self, info, after=None):
        self.info, self.after = info, after
        self.epoch, self.resets = 1, 0
        self.seen, self.predictions, self.pending = [], [], []
        self.curves = [None, None]
        self.disposed = False

    def reset(self):
        self.epoch += 1
        self.resets += 1
        self.predictions = []
        self.curves = [None, None]
        self.pending = []

    def update(self, frame):
        self.seen.append(deepcopy(frame))
        self.predictions.append({"t": frame["t"], "sourceT": frame["t"],
                                 "p": [[.1, .9, 0, 0], [1, 0, 0, 0]]})
        self.curves = [{"sourceCount": len(self.seen)}, None]
        events = []
        if len(self.seen) == 3:
            events = [{"id": "physical-1", "hand": "left", "family": "straight",
                       "startMs": self.seen[0]["t"], "peakMs": self.seen[1]["t"],
                       "endMs": frame["t"], "detectedAtMs": frame["t"], "score": .9}]
        if self.after:
            self.after()
        return {"fingerprint": self.info["fingerprint"], "state": "active",
                "inferenceMs": 2, "events": events}

    def dispose(self):
        self.disposed = True


class Bundle:
    def __init__(self, manifest, after=None):
        root = Path(__file__).resolve().parents[1]
        self.model_info = {"fingerprint": digest(manifest),
                           "runtimeSourceSha256": digest(root / "recognizer.py"),
                           "featureSourceSha256": digest(root / "recognizer_features.py")}
        self.after, self.states = after, []

    def create_session(self):
        state = State(self.model_info, self.after)
        self.states.append(state)
        return state


class ObservationReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_path = self.root / "observations.json"
        self.provenance_path = self.root / "provenance.json"
        self.manifest = self.root / "model.json"
        weights = {}
        for key in ("checkpoint", "externalModel"):
            path = self.root / f"{key}.bin"
            path.write_bytes(key.encode())
            weights[key] = {"path": path.name, "sha256": digest(path)}
        self.manifest.write_text(json.dumps(weights))
        self.source = {
            "sourceId": "external-shot", "sourceSha256": "a" * 64,
            "sourceFrameCount": 4, "stance": None,
            "modelInfo": {"id": "rtmpose-m", "estimator": POLICY, "recognizer": {"old": True}},
            "frames": [{"sourceFrameIndex0Based": i + 7, "t": t,
                        "result": {"t": t, "width": 640, "height": 360,
                                   "estimator": POLICY,
                                   "landmarks": [{"x": .3, "y": .4, "score": 1.17}] * 33,
                                   "_sourceT": 90000, "referenceLabel": "hook",
                                   "recognition": {"events": ["old"]}}}
                       for i, t in enumerate((1000.25, 1041.96, 1092.1, 1142.15))],
            "annotations": [{"label": "uppercut"}],
        }
        self.provenance = {"purpose": "inspection-only", "sourceGroup": "one-original-video",
                           "timestampSemantics": "declared native presentation milliseconds",
                           "videoSha256": "a" * 64}
        self.save()
        self.bundle = Bundle(self.manifest)

    def save(self):
        self.source_path.write_text(json.dumps(self.source))
        self.provenance_path.write_text(json.dumps(self.provenance))

    def replay(self, name="result", **kwargs):
        return run(self.source_path, self.manifest, self.root / name,
                   provenance_path=self.provenance_path,
                   bundle_loader=lambda *_: self.bundle, **kwargs)

    def test_unknown_stance_vfr_clocks_native_scores_and_input_whitelist(self):
        original = self.source_path.read_bytes()
        summary = self.replay(trace=True)
        state = self.bundle.states[0]
        self.assertTrue(state.disposed)
        self.assertEqual([f["t"] for f in state.seen], [1000.25, 1041.96, 1092.1, 1142.15])
        for f in state.seen:
            self.assertEqual(set(f), {"t", "width", "height", "landmarks", "estimator"})
            self.assertEqual(f["landmarks"][0]["score"], 1.17)
        self.assertEqual(summary["events"][0]["family"], "straight")
        self.assertEqual(summary["events"][0]["hand"], "left")
        self.assertNotIn("label", summary["events"][0])
        self.assertNotIn("role", summary["events"][0])
        self.assertNotIn("annotations", summary)
        self.assertEqual(summary["firstSourceTimeMs"], 1000.25)
        self.assertEqual(self.source_path.read_bytes(), original)
        self.assertEqual(len(state.seen), 4)  # No final flush or invented warmup.
        self.assertEqual(summary["unfinalizedAtEnd"]["activeCandidates"][0]["sourceCount"], 4)

    def test_passive_trace_and_captured_parity_preserve_decisions(self):
        first = self.replay(trace=True)
        lines = [json.loads(line) for line in (self.root / "result/decisions.jsonl").read_text().splitlines()]
        for row, line in zip(self.source["frames"], lines):
            row["result"]["recognition"] = {**line["recognition"], "inferenceMs": 999}
        self.save()
        self.bundle = Bundle(self.manifest)
        second = self.replay("paired", require_parity=True)
        self.assertEqual(first["events"], second["events"])
        self.assertTrue(second["requiredCapturedParityPassed"])
        ticks = [json.loads(line)["ticks"] for line in (self.root / "result/trace.jsonl").read_text().splitlines()]
        self.assertEqual([p["sourceT"] for group in ticks for p in group], [r["t"] for r in self.source["frames"]])

    def test_declared_shot_boundaries_and_gap_size_reset_reasons(self):
        self.provenance["resetBeforeFrameIndices"] = [8]
        self.source["frames"][2]["result"]["width"] = 800
        self.source["frames"][3]["result"]["width"] = 800
        self.source["frames"][3]["t"] = self.source["frames"][3]["result"]["t"] = 1500
        self.save()
        result = self.replay(trace=True)
        self.assertEqual(result["resetCounts"], {"initial": 1, "declared_source_boundary": 1,
                                                "dimensions_changed": 1, "observation_gap": 1})
        self.assertEqual(self.bundle.states[0].resets, 1)
        rows = [json.loads(line) for line in (self.root / "result/trace.jsonl").read_text().splitlines()]
        self.assertEqual(rows[1]["epoch"], 2)
        self.assertEqual(rows[1]["ticks"][0]["sourceT"], self.source["frames"][1]["t"])

    def test_parity_failure_preserves_first_mismatch_without_completion(self):
        with self.assertRaisesRegex(ValueError, "parity failed"):
            self.replay(require_parity=True)
        self.assertTrue((self.root / "result/failure.json").exists())
        self.assertEqual(len((self.root / "result/decisions.jsonl").read_text().splitlines()), 1)
        self.assertFalse((self.root / "result/summary.json").exists())
        self.assertTrue(self.bundle.states[0].disposed)

    def test_reject_bad_clock_indices_score_policy_and_missing_native_score(self):
        mutations = [
            lambda s: s["frames"][1].update(t=s["frames"][0]["t"]),
            lambda s: s["frames"][1].update(sourceFrameIndex0Based=7),
            lambda s: s.update(sourceFrameCount=5),
            lambda s: s["frames"][0]["result"].update(t=123),
            lambda s: s["frames"][0]["result"].update(width=True),
            lambda s: s["frames"][0]["result"].update(estimator={**POLICY, "minimumScore": .1}),
            lambda s: s["frames"][0]["result"]["landmarks"][0].pop("score"),
            lambda s: s["frames"][0]["result"]["landmarks"][0].update(x=float("nan")),
            lambda s: s.update(partial=True),
        ]
        original = deepcopy(self.source)
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.source = deepcopy(original)
                mutate(self.source)
                self.save()
                with self.assertRaises(ValueError):
                    self.replay()
                self.assertFalse((self.root / "result").exists())
        self.assertFalse(self.bundle.states)

    def test_reject_mismatched_video_unknown_boundary_and_missing_provenance(self):
        for change in ({"videoSha256": "b" * 64}, {"resetBeforeFrameIndices": [99]},
                       {"resetBeforeFrameIndices": [9, 8]}, {"resetBeforeFrameIndices": [8, 8]},
                       {"sourceGroup": ""}, {"timestampSemantics": ""}):
            original = deepcopy(self.provenance)
            self.provenance.update(change)
            self.save()
            with self.assertRaises(ValueError):
                read_source(self.source_path, self.provenance_path)
            self.provenance = original

    def test_no_overwrite_and_byte_limits(self):
        self.replay()
        with self.assertRaises(FileExistsError):
            self.replay()
        with patch("ml.replay_observations.MAX_INPUT_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                self.replay("too-large")
        with patch("ml.replay_observations.MAX_OUTPUT_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                self.replay("too-much-output")
        self.assertFalse((self.root / "too-much-output/summary.json").exists())
        self.assertTrue(self.bundle.states[-1].disposed)

    def test_nonfinite_metadata_and_duplicate_json_keys_rejected(self):
        original = self.source_path.read_text()
        for prefix in ('"unrelated":1e309,', '"sourceId":"duplicate",'):
            self.source_path.write_text("{" + prefix + original[1:])
            with self.assertRaises(ValueError):
                self.replay()
            self.assertFalse((self.root / "result").exists())

    def test_changed_weight_bytes_during_replay_withhold_completion(self):
        weight = self.root / "checkpoint.bin"
        self.bundle = Bundle(self.manifest, after=lambda: weight.write_bytes(b"changed"))
        with self.assertRaisesRegex(ValueError, "Pinned input changed"):
            self.replay()
        self.assertFalse((self.root / "result/summary.json").exists())
        self.assertTrue(self.bundle.states[0].disposed)


if __name__ == "__main__":
    unittest.main()
