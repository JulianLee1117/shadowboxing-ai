from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ml.replay_personal import digest, match_changes, read_source, replay, run
from ml.evaluate import evaluate_session


POLICY = {"id": "rtmpose-m", "scoreType": "simcc", "minimumScore": .55}


def source():
    return {
        "schemaVersion": "1.0", "id": "source-video", "source": "camera",
        "model": "rtmpose-m", "stance": "orthodox", "durationMs": 500,
        "modelManifest": {"id": "rtmpose-m", "estimator": POLICY,
                          "modelManifest": {"detector": {"sha256": "a" * 64},
                                            "pose": {"sha256": "b" * 64}}},
        "frames": [{"t": t, "width": 640, "height": 480, "inferenceMs": 23,
                    "estimator": POLICY, "landmarks": [], "frameAgeMs": 40}
                   for t in range(0, 501, 100)],
        "events": [], "annotationsComplete": True,
        "annotations": [{"id": "reference-1", "startMs": 0, "endMs": 250,
                         "hand": "left", "label": "jab"}],
    }


class FakeState:
    def __init__(self, model_info, emit=True):
        self.info, self.emit = model_info, emit
        self.epoch = 1
        self.raw, self.grid, self.predictions, self.pending, self.accepted = [], [], [], [], []
        self.curves = [None, None]
        self.seen = []
        self.disposed = False

    def update(self, frame):
        self.seen.append(deepcopy(frame))
        self.raw.append(frame)
        self.grid.append(frame)
        self.predictions.append({"t": frame["t"], "sourceT": frame["t"],
                                 "p": [[1, 0, 0, 0], [1, 0, 0, 0]], "valid": [False, False]})
        events = []
        if frame["t"] == 300 and self.emit:
            events = [{"id": "observed-1", "hand": "left", "family": "straight",
                       "startMs": 0, "peakMs": 100, "endMs": 200,
                       "detectedAtMs": 300, "score": .9}]
        return {**{k: self.info[k] for k in ("protocolVersion", "recognizerId", "fingerprint")},
                "state": "active", "inferenceMs": 2, "events": events}

    def dispose(self):
        self.disposed = True


class FakeBundle:
    def __init__(self, fingerprint="c" * 64, emit=True):
        root = Path(__file__).resolve().parents[1]
        self.model_info = {
            "protocolVersion": "shadowbox-recognition-v1", "recognizerId": "personal-hybrid-v1",
            "fingerprint": fingerprint, "checkpointSha256": "d" * 64,
            "externalModelSha256": "e" * 64, "poseModelSha256": "b" * 64,
            "runtimeSourceSha256": digest(root / "recognizer.py"),
            "featureSourceSha256": digest(root / "recognizer_features.py"),
            "trainingProtocolSha256": "f" * 64,
        }
        self.emit = emit
        self.states = []

    def create_session(self):
        state = FakeState(self.model_info, self.emit)
        self.states.append(state)
        return state


class PersonalReplayTests(unittest.TestCase):
    def test_labels_old_decisions_and_internal_grid_fields_cannot_enter_inference(self):
        s = source()
        for f in s["frames"]:
            f.update({"recognition": {"events": ["old"]}, "_sourceT": 9000,
                      "_ageMs": 9000, "_newSample": False, "referenceLabel": "uppercut"})
        original = deepcopy(s)
        bundle = FakeBundle()
        output, timing = replay(s, bundle, "1" * 64)
        self.assertEqual(s, original)
        self.assertEqual(output["annotations"], original["annotations"])
        for frame in bundle.states[0].seen:
            self.assertEqual(set(frame), {"t", "width", "height", "landmarks", "estimator"})
        self.assertTrue(bundle.states[0].disposed)
        self.assertEqual(output["events"][0]["label"], "jab")
        self.assertEqual(output["frames"][0]["inferenceMs"], 23)
        self.assertEqual(timing["recognitionOnlyMs"]["p50"], 2)
        self.assertEqual(output["benchmark"]["sourceSessionId"], original["id"])

    def test_trace_observes_each_tick_once_and_does_not_change_decisions(self):
        baseline, _ = replay(source(), FakeBundle(), "1" * 64)
        ticks = []
        traced, timing = replay(source(), FakeBundle(), "1" * 64, trace=ticks.append)
        self.assertEqual(baseline["events"], traced["events"])
        self.assertEqual([t["ticks"][0]["t"] for t in ticks], list(range(0, 501, 100)))
        self.assertEqual(timing["resetCounts"], {"initial": 1})
        self.assertEqual(timing["emittedAfterObservedPeakMs"]["p50"], 200)
        self.assertEqual(timing["emittedAfterObservedEndMs"]["p50"], 100)

    def test_physical_hand_is_retained_when_stance_maps_to_cross(self):
        s = source()
        s["stance"] = "southpaw"
        result, _ = replay(s, FakeBundle(), "1" * 64)
        self.assertEqual(result["events"][0]["label"], "cross")
        self.assertEqual(result["events"][0]["hand"], "left")

    def test_video_binding_is_preserved_for_independent_peak_references(self):
        s = source()
        s["benchmark"] = {"sourceVideoSha256": "7" * 64}
        result, _ = replay(s, FakeBundle(), "1" * 64)
        self.assertEqual(result["benchmark"]["sourceVideoSha256"], "7" * 64)
        with self.assertRaisesRegex(ValueError, "video fingerprint"):
            evaluate_session(result, peak_reference={"sourceVideoSha256": "8" * 64,
                "sessionId": s["id"], "peaks": [{"annotationId": "reference-1", "peakMs": 100}]})

    def test_no_extra_flush_or_reference_generated_event_at_end(self):
        s = source()
        s["frames"] = s["frames"][:3]
        bundle = FakeBundle()
        result, _ = replay(s, bundle, "1" * 64)
        self.assertEqual(result["events"], [])
        self.assertEqual(len(bundle.states[0].seen), 3)

    def test_cleanup_runs_when_passive_trace_fails(self):
        bundle = FakeBundle()
        def fail(_):
            raise OSError("full disk")
        with self.assertRaisesRegex(OSError, "full disk"):
            replay(source(), bundle, "1" * 64, trace=fail)
        self.assertTrue(bundle.states[0].disposed)

    def test_duplicate_observations_rejected_not_silently_deduplicated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "source.json"
            s = source()
            s["frames"][1]["t"] = 0
            p.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                read_source(p)

    def test_paired_run_preserves_capture_report_and_writes_immutable_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source()))
            original_bytes = source_path.read_bytes()
            baseline = root / "base.json"
            candidate = root / "candidate.json"
            baseline.write_text("baseline")
            candidate.write_text("candidate")
            def load(path, _):
                return FakeBundle(digest(path), emit=path == candidate.resolve())
            result = run([source_path], candidate, root / "result",
                         baseline_manifest=baseline, trace=True, bundle_loader=load)
            row = result["views"][0]
            self.assertEqual(row["pairedStrictChanges"]["recoveredAnnotationIds"], ["reference-1"])
            self.assertEqual(row["capturedEventMetrics"]["tp"], 0)
            self.assertEqual(row["replays"]["candidate"]["eventMetrics"]["tp"], 1)
            self.assertIsNone(result["independentRecordingCount"])
            self.assertEqual(source_path.read_bytes(), original_bytes)
            self.assertTrue((root / "result/comparison.json").is_file())
            with self.assertRaises(FileExistsError):
                run([source_path], candidate, root / "result", bundle_loader=load)

    def test_unreviewed_source_does_not_gain_accuracy_from_replay(self):
        self.assertIsNone(match_changes(None, {"matches": []}))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            s = source()
            s["annotationsComplete"] = False
            p, manifest = root / "source.json", root / "model.json"
            p.write_text(json.dumps(s))
            manifest.write_text("model")
            result = run([p], manifest, root / "result",
                         bundle_loader=lambda path, _: FakeBundle(digest(path)))
            self.assertIsNone(result["views"][0]["replays"]["candidate"]["eventMetrics"])

    def test_partial_recall_scope_does_not_silently_filter_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p, manifest = root / "source.json", root / "model.json"
            p.write_text(json.dumps(source()))
            manifest.write_text("model")
            result = run([p], manifest, root / "result", label_scopes=[("hook",)],
                         bundle_loader=lambda path, _: FakeBundle(digest(path)))
            row = result["views"][0]
            self.assertEqual(row["recallLabels"], ["hook"])
            self.assertEqual(row["replays"]["candidate"]["eventMetrics"]["fp"], 1)
            self.assertEqual(row["replays"]["candidate"]["eventMetrics"]["fn"], 0)
            self.assertEqual(len(result["executableSourceHashes"]), 2)

    def test_invalid_scope_rejected_before_outputs_or_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for scope in [[()], [("kick",)], [("jab",), ("cross",)]]:
                with self.assertRaisesRegex(ValueError, "label scope"):
                    run([root / "source.json"], root / "model.json", root / "result",
                        label_scopes=scope,
                        bundle_loader=lambda *_: self.fail("Invalid scope loaded a model"))
            self.assertFalse((root / "result").exists())

    def test_input_change_fails_without_completed_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p, manifest = root / "source.json", root / "model.json"
            p.write_text(json.dumps(source()))
            manifest.write_text("model")
            def load(path, _):
                p.write_text(json.dumps({**source(), "id": "changed"}))
                return FakeBundle(digest(path))
            with self.assertRaisesRegex(ValueError, "Input changed"):
                run([p], manifest, root / "result", bundle_loader=load)
            self.assertFalse((root / "result/comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
