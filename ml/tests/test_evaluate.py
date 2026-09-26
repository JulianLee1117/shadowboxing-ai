import copy
import json
from pathlib import Path
import tempfile
import unittest

from ml.evaluate import build_report, evaluate_session, main, match_events, percentile, temporal_iou


def event(event_id="p1", start=100, end=300, label="jab", hand="left"):
    return {"id": event_id, "startMs": start, "endMs": end, "label": label, "hand": hand}


def session():
    return {"schemaVersion": "1.0", "id": "real-session", "source": "file", "model": "full",
            "durationMs": 1000, "frames": [], "events": [], "annotations": []}


def metrics(s):
    return evaluate_session(s, annotations_complete=True)["eventMetrics"]


class EvaluationTests(unittest.TestCase):
    def test_duplicate_predictions_are_not_both_true_positives(self):
        s = session()
        s["annotations"] = [event("a1")]
        s["events"] = [event("p1"), event("p2")]
        m = metrics(s)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 0))
        self.assertEqual(m["precision"], .5)
        self.assertEqual(m["falseEventsPerMinute"], 60)

    def test_wrong_hand_is_false_positive_and_false_negative(self):
        s = session()
        s["annotations"] = [event("a1")]
        s["events"] = [event(hand="right")]
        m = metrics(s)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))

    def test_wrong_label_is_not_a_match(self):
        s = session()
        s["annotations"] = [event("a1")]
        s["events"] = [event(label="cross")]
        m = metrics(s)
        self.assertEqual(m["perClass"]["jab"]["fn"], 1)
        self.assertEqual(m["perClass"]["cross"]["fp"], 1)

    def test_maximum_cardinality_does_not_lose_greedy_match(self):
        # P1 can match either truth; P2 can only match A1 at this threshold.
        preds = [event("p1", 100, 400), event("p2", 0, 200)]
        truths = [event("a1", 0, 300), event("a2", 200, 500)]
        self.assertEqual(set(match_events(preds, truths, .5)), {(0, 1), (1, 0)})

    def test_tiou_threshold_boundary_is_inclusive(self):
        p, a = event(start=0, end=100), event(start=0, end=200)
        self.assertEqual(temporal_iou(p, a), .5)
        self.assertEqual(match_events([p], [a]), [(0, 0)])

    def test_empty_session_does_not_claim_perfect_accuracy(self):
        m = metrics(session())
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 0, 0))
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["f1"])
        self.assertEqual(m["falseEventsPerMinute"], 0)

    def test_empty_predictions_with_truth_is_missed_event(self):
        s = session()
        s["annotations"] = [event("a1")]
        self.assertEqual(metrics(s)["recall"], 0)

    def test_completeness_is_required_but_export_flag_is_supported(self):
        s = session()
        self.assertIsNone(evaluate_session(s)["eventMetrics"])
        s["annotationsComplete"] = True
        self.assertIsNotNone(evaluate_session(s)["eventMetrics"])
        s["annotationsComplete"] = "true"
        with self.assertRaises(ValueError):
            evaluate_session(s)

    def test_synthetic_rejected_or_entire_report_marked(self):
        s = session()
        s["source"] = "demo"
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            evaluate_session(s)
        report = build_report([s], allow_synthetic=True)
        self.assertEqual(report["status"], "synthetic_software_check")
        s["source"], s["model"] = "camera", "synthetic"
        with self.assertRaises(ValueError):
            evaluate_session(s)

    def test_unobservable_intervals_remove_only_fully_contained_predictions(self):
        s = session()
        s["annotations"] = [event("u1", 0, 400, "unobservable", "unknown")]
        s["events"] = [event("ignored", 100, 300), event("crossing", 300, 600)]
        m = metrics(s)
        self.assertEqual(m["excludedPredictions"], ["ignored"])
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["evaluatedExposureMs"], 600)

    def test_conflicting_observability_labels_require_adjudication(self):
        s = session()
        s["annotations"] = [event("a1"), event("u1", 0, 400, "unobservable", "unknown")]
        with self.assertRaisesRegex(ValueError, "Adjudicate"):
            metrics(s)

    def test_unsupported_punch_does_not_make_wrong_jab_correct(self):
        s = session()
        s["annotations"] = [event("a1", label="hook")]
        s["events"] = [event()]
        m = metrics(s)
        self.assertEqual((m["fp"], m["fn"], m["unsupportedTruthCount"]), (1, 0, 1))

    def test_coverage_missing_confidence_and_missing_joints_are_not_visible(self):
        s = session()
        joints = [{"x": .5, "y": .5, "visibility": .9} for _ in range(33)]
        base = {"t": 0, "width": 640, "height": 480, "landmarks": joints, "inferenceMs": 5}
        missing = copy.deepcopy(base)
        missing["t"] = 40
        missing["landmarks"][15].pop("visibility")
        s["frames"] = [base, missing]
        r = evaluate_session(s)
        self.assertEqual(r["coverage"]["fraction"], .5)
        self.assertEqual(r["timing"]["observedProcessedFps"], 25)

    def test_optional_telemetry_is_measured_not_invented(self):
        s = session()
        s["annotations"] = [event("a1")]
        s["events"] = [{**event(), "detectedAtMs": 450}]
        s["frames"] = [{"t": 0, "width": 640, "height": 480, "landmarks": [],
                        "inferenceMs": 20, "frameAgeMs": 25}]
        r = evaluate_session(s, annotations_complete=True)
        self.assertEqual(r["timing"]["eventFinalizationDelayMs"]["p95"], 150)
        self.assertEqual(r["timing"]["processedFrameAgeMs"]["p95"], 25)

    def test_aggregate_counts_are_weighted_by_events_not_sessions(self):
        a, b = session(), session()
        a["id"], b["id"] = "one", "two"
        a["annotations"] = [event("a1")]
        a["events"] = [event("p1")]
        b["events"] = [event("p2"), event("p3")]
        report = build_report([a, b], annotations_complete=True)
        self.assertAlmostEqual(report["eventMetrics"]["precision"], 1 / 3)
        self.assertEqual(report["eventMetrics"]["falseEventsPerMinute"], 60)

    def test_duplicate_session_id_and_invalid_intervals_rejected(self):
        with self.assertRaises(ValueError):
            build_report([session(), session()])
        s = session()
        s["events"] = [event(start=300, end=100)]
        with self.assertRaises(ValueError):
            metrics(s)

    def test_percentiles(self):
        self.assertIsNone(percentile([], .95))
        self.assertEqual(percentile([0, 100], .95), 95)

    def test_cli_writes_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "session.json", Path(directory) / "report.json"
            source.write_text(json.dumps(session()))
            self.assertEqual(main([str(source), "--output", str(output)]), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "capture_quality_only")
            self.assertIsNone(report["eventMetrics"])


if __name__ == "__main__":
    unittest.main()
