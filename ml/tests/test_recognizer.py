import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from ml.recognizer import (
    FEATURE_VERSION,
    PROTOCOL_VERSION,
    RECOGNIZER_ID,
    ModelBundle,
    RecognizerSession,
    _validate_pose,
    _verified_path,
)

ESTIMATOR = {"id": "rtmpose-m", "scoreType": "simcc", "minimumScore": 0.55}


class Vector(list):
    def argmax(self):
        return self.index(max(self))


class Matrix:
    def __init__(self, kind):
        self.kind = kind

    def __getitem__(self, _):
        return (
            [Vector([1, 0, 0, 0]), Vector([1, 0, 0, 0])]
            if self.kind == "probabilities"
            else [True, True]
        )


class FakeBundle:
    expected_estimator = ESTIMATOR
    model_info = {
        "protocolVersion": PROTOCOL_VERSION,
        "recognizerId": RECOGNIZER_ID,
        "fingerprint": "a" * 64,
    }

    def temporal_probabilities(self, frames):
        return Matrix("probabilities"), Matrix("valid")

    def straight_proposals(self, frames):
        return []


def frame(t, width=1280):
    return {
        "t": t,
        "width": width,
        "height": 720,
        "landmarks": [],
        "estimator": dict(ESTIMATOR),
    }


class RecognizerTests(unittest.TestCase):
    def test_source_policy_and_color_are_mandatory(self):
        manifest = {
            "pose": {
                "id": "rtmpose-m",
                "estimator": ESTIMATOR,
                "poseSha256": "a" * 64,
                "detectorSha256": "b" * 64,
                "poseInputColorOrder": "RGB",
                "detectorScoreThreshold": 0.7,
            }
        }
        actual = {
            "id": "rtmpose-m",
            "estimator": ESTIMATOR,
            "modelManifest": {
                "pose": {"sha256": "a" * 64, "inputColorOrder": "RGB"},
                "detector": {"sha256": "b" * 64},
            },
            "detectorPostprocessing": {"effectivePostNmsScoreThreshold": 0.7},
        }
        _validate_pose(manifest, actual)
        for mutate in (
            lambda v: v["estimator"].update(minimumScore=0.65),
            lambda v: v["modelManifest"]["pose"].update(inputColorOrder="BGR"),
            lambda v: v["detectorPostprocessing"].update(
                effectivePostNmsScoreThreshold=0.3
            ),
        ):
            wrong = json.loads(json.dumps(actual))
            mutate(wrong)
            with self.assertRaises(ValueError):
                _validate_pose(manifest, wrong)

    def test_local_checkpoint_hash_and_directory_are_enforced_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "weights.pt"
            weight.write_bytes(b"local-test-weights")
            item = {
                "path": "weights.pt",
                "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            }
            self.assertEqual(_verified_path(root, item), weight.resolve())
            with self.assertRaises(ValueError):
                _verified_path(root, {**item, "sha256": "0" * 64})
            with self.assertRaises(ValueError):
                _verified_path(root, {**item, "path": "../outside.pt"})

    def test_unsupported_feature_contract_fails_without_loading_optional_dependencies(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"featureVersion": "unrecognized"}))
            with self.assertRaises(ValueError):
                ModelBundle.load(path, {})

    def test_grid_uses_only_observed_past_frames_and_records_sample_age(self):
        observer = RecognizerSession(FakeBundle())
        observer.update(frame(0))
        observer.update(frame(70))
        self.assertEqual(len(observer.grid), 3)
        self.assertEqual([value["_sourceT"] for value in observer.grid], [0, 0, 0])
        self.assertAlmostEqual(observer.grid[-1]["_ageMs"], 2000 / 30)
        observer.update(frame(105))
        self.assertEqual(observer.grid[-1]["_sourceT"], 70)
        self.assertLessEqual(observer.grid[-1]["_sourceT"], observer.grid[-1]["t"])

    def test_seek_long_gap_and_dimensions_reset_pending_events(self):
        for next_frame in (frame(50), frame(400), frame(133, 640)):
            observer = RecognizerSession(FakeBundle())
            observer.update(frame(100))
            observer.pending.append(
                {
                    "hand": "right",
                    "family": "uppercut",
                    "startMs": 0,
                    "peakMs": 20,
                    "endMs": 50,
                    "score": 0.9,
                    "_ready": 120,
                }
            )
            epoch = observer.epoch
            result = observer.update(next_frame)
            self.assertEqual(observer.epoch, epoch + 1)
            self.assertEqual(result["events"], [])
            self.assertEqual(observer.pending, [])

    def test_sessions_are_independent_and_history_is_bounded(self):
        first, second = RecognizerSession(FakeBundle()), RecognizerSession(FakeBundle())
        for t in range(0, 10000, 33):
            first.update(frame(t))
        self.assertEqual(second.raw, [])
        self.assertLessEqual(len(first.grid), 128)
        self.assertLessEqual(first.raw[-1]["t"] - first.raw[0]["t"], 4200)
        first.dispose()
        self.assertEqual(first.grid, [])

    def test_pending_event_is_emitted_only_on_current_observed_time_and_once(self):
        observer = RecognizerSession(FakeBundle())
        observer.update(frame(0))
        observer.pending.append(
            {
                "hand": "left",
                "family": "hook",
                "startMs": 0,
                "peakMs": 10,
                "endMs": 20,
                "score": 0.9,
                "_ready": 80,
            }
        )
        self.assertEqual(observer.update(frame(66))["events"], [])
        result = observer.update(frame(100))
        self.assertEqual(result["events"][0]["detectedAtMs"], 100)
        self.assertEqual(observer.update(frame(133))["events"], [])

    def test_fast_distinct_peaks_allow_overlapping_action_intervals(self):
        observer = RecognizerSession(FakeBundle())
        observer.update(frame(0))
        observer.accepted.append(
            {
                "hand": "left",
                "family": "straight",
                "startMs": 0,
                "peakMs": 10,
                "endMs": 90,
                "score": 0.9,
            }
        )
        observer.pending.append(
            {
                "hand": "left",
                "family": "straight",
                "startMs": 80,
                "peakMs": 250,
                "endMs": 280,
                "score": 0.9,
                "_ready": 300,
            }
        )
        for t in (100, 200):
            observer.update(frame(t))
        self.assertEqual(len(observer.update(frame(300))["events"]), 1)

    def test_one_observation_held_on_multiple_grid_ticks_cannot_form_an_action(self):
        observer = RecognizerSession(FakeBundle())
        hook = [Vector([0, 0, 0.99, 0.01]), Vector([1, 0, 0, 0])]
        idle = [Vector([1, 0, 0, 0]), Vector([1, 0, 0, 0])]
        for t in (0, 1000 / 30, 2000 / 30):
            observer._curves_at(hook, [True, True], {"t": t, "_sourceT": 0})
        observer._curves_at(idle, [False, False], {"t": 100, "_sourceT": 100})
        observer._curves_at(idle, [False, False], {"t": 200, "_sourceT": 200})
        observer._curves_at(idle, [False, False], {"t": 300, "_sourceT": 300})
        self.assertEqual(observer.pending, [])
        self.assertIsNone(observer.curves[0])

    def test_no_finite_observed_geometry_abstains_instead_of_inventing_peak(self):
        class Nonfinite:
            def __getitem__(self, _):
                return self

            def all(self, _):
                return self

            def any(self):
                return False

            def min(self, _):
                return self

            def __ge__(self, _):
                return self

            def __and__(self, _):
                return self

        bundle = FakeBundle()
        bundle.np = SimpleNamespace(isfinite=lambda _: Nonfinite())
        bundle.classification_features = lambda _: (None, Nonfinite(), Nonfinite())
        bundle.family_probabilities = lambda _: [0, 0.9, 0.1]
        observer = RecognizerSession(bundle)
        observer.grid = [{"t": t, "_sourceT": t} for t in (0, 33, 66)]
        hook = [Vector([0, 0, 0.99, 0.01]), Vector([1, 0, 0, 0])]
        idle = [Vector([1, 0, 0, 0]), Vector([1, 0, 0, 0])]
        for value in observer.grid:
            observer._curves_at(hook, [True, True], value)
        for t in (100, 200):
            observer._curves_at(idle, [True, True], {"t": t, "_sourceT": t})
        self.assertEqual(observer.pending, [])
        self.assertIsNone(observer.curves[0])

    def test_temporal_compute_uses_receptive_tail_without_truncating_event_history(
        self,
    ):
        bundle = FakeBundle()
        lengths = []

        def compute(frames):
            lengths.append(len(frames))
            return Matrix("probabilities"), Matrix("valid")

        bundle.temporal_probabilities = compute
        observer = RecognizerSession(bundle)
        for t in range(0, 5000, 50):
            observer.update(frame(t))
        self.assertGreater(len(observer.grid), 100)
        self.assertLessEqual(max(lengths), 33)

    def test_short_uncertainty_retains_state_but_cannot_add_observations(self):
        observer = RecognizerSession(FakeBundle())
        hook = [Vector([0, 0, 0.99, 0.01]), Vector([1, 0, 0, 0])]
        for t in (0, 33, 66):
            observer._curves_at(hook, [True, True], {"t": t, "_sourceT": t})
        count = observer.curves[0]["sourceCount"]
        for t in (100, 200):
            observer._curves_at(hook, [False, True], {"t": t, "_sourceT": t})
        self.assertEqual(observer.curves[0]["sourceCount"], count)
        self.assertEqual(observer.curves[0]["sourceLast"], 66)
        observer._curves_at(hook, [True, True], {"t": 250, "_sourceT": 250})
        self.assertEqual(observer.curves[0]["sourceStart"], 0)
        observer._curves_at(hook, [False, True], {"t": 400, "_sourceT": 400})
        observer._curves_at(hook, [True, True], {"t": 550, "_sourceT": 550})
        self.assertEqual(observer.curves[0]["sourceStart"], 550)
        self.assertEqual(observer.pending, [])

    def test_distinct_peaks_with_shared_start_have_distinct_stable_ids(self):
        observer = RecognizerSession(FakeBundle())
        observer.update(frame(0))
        for peak in (20, 260):
            observer.pending.append(
                {
                    "hand": "left",
                    "family": "straight",
                    "startMs": 0,
                    "peakMs": peak,
                    "endMs": 280,
                    "score": 0.9,
                    "_ready": 300,
                }
            )
        for t in (100, 200):
            observer.update(frame(t))
        events = observer.update(frame(300))["events"]
        self.assertEqual(len(events), 2)
        self.assertNotEqual(events[0]["id"], events[1]["id"])

    def test_missing_or_changed_frame_estimator_fails_explicitly(self):
        observer = RecognizerSession(FakeBundle())
        for bad in ({**frame(0), "estimator": None}, {**frame(0), "t": float("nan")}):
            with self.assertRaises(ValueError):
                observer.update(bad)


if __name__ == "__main__":
    unittest.main()
