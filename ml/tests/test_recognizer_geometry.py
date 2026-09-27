"""Numerical observation-contract tests; run with the optional ML environment."""

import unittest
from types import SimpleNamespace

try:
    import numpy as np
    import scipy.signal as signal
except ImportError:
    np = signal = None
from ml.recognizer import RecognizerSession
from ml.tests.test_recognizer import FakeBundle, Vector


@unittest.skipIf(np is None, "optional NumPy/SciPy ML environment is unavailable")
class ObservedGeometryTests(unittest.TestCase):
    def run_action(self, times, reach, label=1, hidden=(), angles=None):
        values = dict(zip(times, reach))
        bundle = FakeBundle()
        bundle.np, bundle.signal = np, signal

        def features(frames):
            geometry = np.zeros((len(frames), 2, 6))
            scores = np.ones((len(frames), 8))
            for i, f in enumerate(frames):
                t = f.get("_sourceT", f["t"])
                r = values[t]
                geometry[i, 0] = [r, 0, 0.3, 0.2, r, angles[t] if angles else 160]
                if t in hidden:
                    scores[i, 4] = 0.1
            return None, geometry, scores

        bundle.classification_features = features
        observer = RecognizerSession(bundle)
        active = [Vector([0, 0, 0, 0]), Vector([1, 0, 0, 0])]
        active[0][label] = 0.99
        idle = [Vector([1, 0, 0, 0]), Vector([1, 0, 0, 0])]
        for t in times:
            f = {"t": t, "_sourceT": t}
            observer.grid.append(f)
            observer._curves_at(active, [t not in hidden, True], f)
        for t in (times[-1] + 33, times[-1] + 133):
            observer._curves_at(idle, [True, True], {"t": t, "_sourceT": t})
        return observer.pending

    def test_low_score_gap_extremum_cannot_be_an_observed_peak(self):
        events = self.run_action(
            [0, 33, 66, 100, 133, 166],
            [0.1, 0.2, 0.3, 99, 0.4, 0.5],
            label=2,
            hidden=(100,),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["peakMs"], 166)
        self.assertNotEqual(events[0]["peakMs"], 100)

    def test_two_observed_cycles_split_even_when_family_remains_straight(self):
        events = self.run_action(
            list(range(0, 401, 50)), [0.2, 0.6, 1, 0.5, 0.2, 0.5, 1, 0.5, 0.2]
        )
        self.assertEqual([e["peakMs"] for e in events], [100, 300])
        self.assertEqual(events[0]["endMs"], events[1]["startMs"])

    def test_missing_trough_samples_cannot_fabricate_two_cycles(self):
        events = self.run_action(
            list(range(0, 401, 50)),
            [0.2, 0.6, 1, 0.5, 0.2, 0.5, 1, 0.5, 0.2],
            hidden=(150, 200),
        )
        self.assertEqual(len(events), 1)

    def test_forehead_cover_is_not_a_straight_despite_learned_family_support(self):
        times = [0, 33, 66, 100]
        events = self.run_action(
            times, [0.49, 0.47, 0.51, 0.54], angles=dict(zip(times, [26, 38, 50, 59]))
        )
        self.assertEqual(events, [])

    def test_low_prominence_observed_extension_remains_a_straight(self):
        times = [0, 33, 66, 100]
        events = self.run_action(
            times,
            [0.58, 0.60, 0.61, 0.60],
            angles=dict(zip(times, [80, 125, 158, 130])),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["peakMs"], 66)

    def test_below_extension_guard_abstains_even_with_learned_straight_label(self):
        events = self.run_action(list(range(0, 401, 50)), [0.5] * 9)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
