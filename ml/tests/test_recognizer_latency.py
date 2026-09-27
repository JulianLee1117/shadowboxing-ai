"""Causal delivery and proposal arbitration for the versioned 150 ms hold."""

import unittest

try:
    import numpy as np
except ImportError:
    np = None

from ml.recognizer import LOOKAHEAD_MS, RecognizerSession
from ml.tests.test_recognizer import FakeBundle, frame


def proposal(kind, *, end=100, hand="left"):
    return {
        "hand": hand,
        "family": "straight",
        "startMs": 0,
        "peakMs": 50,
        "endMs": end,
        "score": 0.9,
        "_proposal": kind,
        "_ready": end + LOOKAHEAD_MS,
    }


class ArbitrationDeliveryTests(unittest.TestCase):
    def test_ready_event_waits_for_an_actual_consumed_source_frame(self):
        observer = RecognizerSession(FakeBundle())
        observer.update(frame(100))
        observer.pending.append(proposal("geometry"))
        self.assertEqual(observer.update(frame(200))["events"], [])
        self.assertEqual(observer.update(frame(249))["events"], [])
        emitted = observer.update(frame(267))["events"]
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["detectedAtMs"], 267)
        self.assertEqual(emitted[0]["endMs"], 100)
        self.assertEqual(observer.update(frame(300))["events"], [])

    def test_already_pending_geometry_wins_even_when_temporal_is_ready_first(self):
        observer = RecognizerSession(FakeBundle())
        observer.update(frame(100))
        observer.pending.extend(
            [proposal("temporal", end=100), proposal("geometry", end=150)]
        )
        self.assertEqual(observer.update(frame(200))["events"], [])
        self.assertEqual(observer.update(frame(250))["events"], [])
        emitted = observer.update(frame(300))["events"]
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["endMs"], 150)
        self.assertEqual(observer.accepted[0]["_proposal"], "geometry")

    def test_later_geometry_cannot_rewrite_or_duplicate_an_emitted_temporal_event(self):
        observer = RecognizerSession(FakeBundle())
        observer.update(frame(100))
        observer.pending.append(proposal("temporal", end=100))
        self.assertEqual(observer.update(frame(200))["events"], [])
        emitted = observer.update(frame(250))["events"]
        self.assertEqual(len(emitted), 1)
        original = dict(emitted[0])
        observer.pending.append(proposal("geometry", end=150))
        self.assertEqual(observer.update(frame(300))["events"], [])
        self.assertEqual(emitted[0], original)
        self.assertEqual(len(observer.accepted), 1)
        self.assertEqual(observer.accepted[0]["_proposal"], "temporal")
        self.assertEqual(observer.accepted[0]["endMs"], 100)


@unittest.skipIf(np is None, "optional NumPy ML environment is unavailable")
class TemporalHoldTests(unittest.TestCase):
    def test_observed_completion_and_hold_overlap_without_inventing_early_emission(
        self,
    ):
        bundle = FakeBundle()
        bundle.np = np

        def features(frames):
            geometry = np.zeros((len(frames), 2, 6))
            for index, sample in enumerate(frames):
                geometry[index, 0, 0] = sample["_sourceT"] / 100
            return None, geometry, np.ones((len(frames), 8))

        bundle.classification_features = features
        observer = RecognizerSession(bundle)
        active = np.array([[0.01, 0.01, 0.97, 0.01], [0.97, 0.01, 0.01, 0.01]])
        idle = np.array([[0.97, 0.01, 0.01, 0.01]] * 2)
        for timestamp in (0, 33, 66, 100, 133, 166, 200, 233):
            sample = {"t": timestamp, "_sourceT": timestamp}
            observer.grid.append(sample)
            observer._curves_at(
                active if timestamp <= 100 else idle, [True, True], sample
            )
        self.assertEqual(len(observer.pending), 1)
        self.assertEqual(observer.pending[0]["endMs"], 100)
        # The 100 ms closure evidence overlaps the 150 ms post-end hold.
        self.assertEqual(observer.pending[0]["_ready"], 250)
        self.assertEqual(observer.update(frame(233))["events"], [])
        emitted = observer.update(frame(267))["events"]
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["peakMs"], 100)
        self.assertEqual(emitted[0]["detectedAtMs"], 267)


if __name__ == "__main__":
    unittest.main()
