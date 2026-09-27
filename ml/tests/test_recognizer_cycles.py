"""Observed-cycle preservation at the causal recognizer's active-run limit."""

import unittest

try:
    import numpy as np
    import scipy.signal as signal
except ImportError:
    np = signal = None

from ml.recognizer import RecognizerSession


class GeometryBundle:
    """Supply observed geometry without loading a trained network or weights."""

    np = np
    signal = signal

    def classification_features(self, frames):
        return (
            np.zeros((len(frames), 16)),
            np.array([frame["geometry"] for frame in frames]),
            np.array([frame["scores"] for frame in frames]),
        )


@unittest.skipIf(np is None, "optional NumPy/SciPy ML environment is unavailable")
class ObservedCycleTests(unittest.TestCase):
    def append_observation(
        self, observer, timestamp, reach, *, label=1, hidden=False, source_t=None
    ):
        """Consume one grid sample and retain its distinct source timestamp."""
        source_t = timestamp if source_t is None else source_t
        scores = np.ones(8)
        if hidden:
            # The classifier can remain confident despite an unobserved shoulder.
            # Such a sample cannot qualify the observed split geometry.
            scores[0] = 0.1
        frame = {
            "t": timestamp,
            "_sourceT": source_t,
            "geometry": [
                [reach, 0.2, 0.3, 0.3, reach, 150],
                [0.3, 0.2, 0.1, 0.2, 0.36, 70],
            ],
            "scores": scores.tolist(),
        }
        probabilities = np.full((2, 4), 0.02)
        probabilities[0, label] = 0.94
        probabilities[1, 0] = 0.94
        valid = np.array([True, True])
        observer.grid.append(frame)
        observer.predictions.append(
            {
                "t": timestamp,
                "sourceT": source_t,
                "p": probabilities,
                "valid": valid,
            }
        )
        observer._curves_at(probabilities, valid, frame)

    def run_case(self, points, *, end=1900, hidden=(), label=1):
        observer = RecognizerSession(GeometryBundle())
        times, reaches = zip(*points)
        for timestamp in range(0, end + 1, 50):
            reach = float(np.interp(timestamp, times, reaches))
            self.append_observation(
                observer,
                timestamp,
                reach,
                label=label,
                hidden=timestamp in hidden,
            )
        return observer

    def test_two_observed_peaks_preserve_first_and_retain_second(self):
        observer = self.run_case(
            [(0, 0.4), (300, 1), (700, 0.4), (1500, 1), (1800, 0.4), (1900, 0.4)]
        )
        self.assertEqual(len(observer.pending), 1)
        event = observer.pending[0]
        self.assertEqual(
            (event["startMs"], event["peakMs"], event["endMs"]), (0, 300, 700)
        )
        self.assertGreaterEqual(event["_ready"], 1850)
        self.assertEqual(observer.curves[0]["sourceStart"], 700)

    def test_continuously_held_single_peak_is_not_split(self):
        observer = self.run_case([(0, 0.4), (300, 1), (1900, 1)])
        self.assertEqual(observer.pending, [])

    def test_static_guard_is_not_split(self):
        observer = self.run_case([(0, 0.4), (1900, 0.4)])
        self.assertEqual(observer.pending, [])

    def test_qualified_source_gap_blocks_split(self):
        observer = self.run_case(
            [(0, 0.4), (300, 1), (700, 0.4), (1500, 1), (1900, 0.4)],
            hidden=range(800, 1101, 50),
        )
        self.assertEqual(observer.pending, [])

    def test_three_cycles_keep_two_remaining_without_end_flush(self):
        observer = self.run_case(
            [
                (0, 0.4),
                (300, 1),
                (650, 0.4),
                (1000, 1),
                (1300, 0.4),
                (1600, 1),
                (1900, 0.4),
            ]
        )
        self.assertEqual(len(observer.pending), 1)
        self.assertEqual(observer.pending[0]["peakMs"], 300)
        self.assertEqual(observer.curves[0]["sourceStart"], 650)

    def test_retained_cycles_close_once_after_observed_background(self):
        observer = self.run_case(
            [
                (0, 0.4),
                (300, 1),
                (650, 0.4),
                (1000, 1),
                (1300, 0.4),
                (1600, 1),
                (1900, 0.4),
            ]
        )
        self.assertEqual(len(observer.pending), 1)
        for timestamp in (1950, 2000, 2050, 2100):
            self.append_observation(observer, timestamp, 0.4, label=0)
        self.assertEqual(
            [event["peakMs"] for event in observer.pending], [300, 1000, 1600]
        )
        self.assertTrue(
            all(
                first["endMs"] <= second["startMs"]
                for first, second in zip(observer.pending, observer.pending[1:])
            )
        )
        self.assertIsNone(observer.curves[0])

    def test_hook_family_does_not_use_straight_cycle_preservation(self):
        observer = self.run_case(
            [(0, 0.4), (300, 1), (700, 0.4), (1500, 1), (1900, 0.4)], label=2
        )
        self.assertEqual(observer.pending, [])

    def test_prefix_does_not_use_an_unconsumed_second_return(self):
        observer = self.run_case(
            [(0, 0.4), (300, 1), (700, 0.4), (1900, 1), (2200, 0.4)], end=1800
        )
        self.assertEqual(observer.pending, [])

    def test_peaks_below_straight_shape_gate_do_not_qualify(self):
        observer = self.run_case(
            [(0, 0.1), (300, 0.5), (700, 0.1), (1500, 0.5), (1900, 0.1)]
        )
        self.assertEqual(observer.pending, [])

    def test_repeated_source_holds_are_not_distinct_punch_observations(self):
        observer = RecognizerSession(GeometryBundle())
        # Three causal grid ticks hold one real, unchanged source observation.
        for timestamp in (0, 33, 66):
            self.append_observation(observer, timestamp, 1.0, source_t=0)
        self.assertEqual(observer.curves[0]["sourceCount"], 1)
        for timestamp in (100, 150, 200):
            self.append_observation(observer, timestamp, 0.4, label=0)
        self.assertEqual(observer.pending, [])
        self.assertIsNone(observer.curves[0])


if __name__ == "__main__":
    unittest.main()
