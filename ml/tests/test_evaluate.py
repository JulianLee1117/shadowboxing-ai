import copy
import json
from pathlib import Path
import tempfile
import unittest

from ml.evaluate import build_report, evaluate_session, main, match_events, match_peak_events, percentile, temporal_iou


def event(event_id="p1", start=100, end=300, label="jab", hand="left"):
    return {"id": event_id, "startMs": start, "endMs": end, "label": label, "hand": hand}


def session():
    return {"schemaVersion": "1.0", "id": "real-session", "source": "file", "model": "full",
            "durationMs": 1000, "frames": [], "events": [], "annotations": []}


def metrics(s):
    return evaluate_session(s, annotations_complete=True)["eventMetrics"]


class EvaluationTests(unittest.TestCase):
    def test_native_predictions_match_causal_decisions_and_model_fingerprints(self):
        s = session()
        s.update(model="rtmpose-m", stance="orthodox")
        policy = {"id": "rtmpose-m", "scoreType": "simcc", "minimumScore": .55}
        recognizer = {"protocolVersion": "shadowbox-recognition-v1", "recognizerId": "personal-hybrid-v1",
                      "fingerprint": "c" * 64, "checkpointSha256": "d" * 64,
                      "externalModelSha256": "e" * 64, "poseModelSha256": "b" * 64}
        decision = {"id": "native-1", "hand": "right", "family": "straight", "startMs": 100,
                    "peakMs": 200, "endMs": 400, "detectedAtMs": 850, "score": .9}
        s["modelManifest"] = {"id": "rtmpose-m", "estimator": policy, "recognizer": recognizer,
                              "modelManifest": {"detector": {"sha256": "a" * 64}, "pose": {"sha256": "b" * 64}}}
        s["frames"] = [{"t": 900, "width": 640, "height": 480, "inferenceMs": 30,
                        "estimator": policy, "landmarks": [], "recognition": {
                            **{k: recognizer[k] for k in ("protocolVersion", "recognizerId", "fingerprint")},
                            "state": "active", "events": [decision]}}]
        s["events"] = [{**decision, "label": "cross", "role": "rear", "extension": None}]
        del s["events"][0]["family"]
        evaluate_session(s)
        for target, key, value in [("event", "label", "jab"), ("event", "id", "invented"),
                                   ("decision", "detectedAtMs", 901), ("recognition", "fingerprint", "f" * 64)]:
            broken = copy.deepcopy(s)
            selected = {"event": broken["events"][0],
                        "decision": broken["frames"][0]["recognition"]["events"][0],
                        "recognition": broken["frames"][0]["recognition"]}[target]
            selected[key] = value
            with self.assertRaises(ValueError):
                evaluate_session(broken)

    def test_native_browser_scores_require_matching_provenance_and_own_policy(self):
        s = session()
        s["model"] = "rtmpose-m"
        policy = {"id": "rtmpose-m", "scoreType": "simcc", "minimumScore": .55}
        s["modelManifest"] = {"id": "rtmpose-m", "estimator": policy,
                              "modelManifest": {"detector": {"sha256": "a" * 64},
                                                "pose": {"sha256": "b" * 64}}}
        s["frames"] = [{"t": 0, "width": 640, "height": 480, "inferenceMs": 28,
                        "estimator": copy.deepcopy(policy),
                        "landmarks": [{"x": .5, "y": .5, "score": .6} for _ in range(33)]}]
        result = evaluate_session(s, min_confidence=.9)
        self.assertEqual(result["coverage"]["fraction"], 1)
        self.assertEqual(result["coverage"]["nativeScorePolicies"], [policy])
        self.assertIsNone(result["coverage"]["minimumConfidence"])
        s["frames"][0]["landmarks"][15] = {"x": .5, "y": .5, "score": .3, "visibility": 1}
        self.assertEqual(evaluate_session(s)["coverage"]["fraction"], 0)
        broken = copy.deepcopy(s)
        broken["frames"][0]["estimator"]["minimumScore"] = .2
        with self.assertRaisesRegex(ValueError, "must match"):
            evaluate_session(broken)
        broken = copy.deepcopy(s)
        broken["modelManifest"]["estimator"]["id"] = "rtmw-l"
        broken["frames"][0]["estimator"]["id"] = "rtmw-l"
        with self.assertRaisesRegex(ValueError, "model identity"):
            evaluate_session(broken)
        broken = copy.deepcopy(s)
        broken["model"] = "full"
        with self.assertRaisesRegex(ValueError, "cannot claim native"):
            evaluate_session(broken)
        del s["modelManifest"]["modelManifest"]["pose"]
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            evaluate_session(s)

    def test_research_model_requires_explicit_fingerprinted_benchmark(self):
        s = session()
        s["model"] = "rtmpose-m-offline"
        with self.assertRaisesRegex(ValueError, "Session model"):
            evaluate_session(s)
        s["artifactType"] = "detector-benchmark-session"
        s["modelManifest"] = {"family": "rtmpose-body", "pose": {"sha256": "a" * 64}}
        s["benchmark"] = {"modelId": s["model"], "inputSessionSha256": "b" * 64,
                          "detectorSourceSha256": "c" * 64, "trackingSource": "replacement-pose-series",
                          "sourceVideoSha256": "d" * 64, "poseSeriesSha256": "e" * 64,
                          "timestampMode": "decoded-pts"}
        result = evaluate_session(s)
        self.assertEqual(result["model"], "rtmpose-m-offline")
        self.assertEqual(result["benchmark"], s["benchmark"])
        self.assertIsNone(result["eventMetrics"])
        for missing in ("sourceVideoSha256", "poseSeriesSha256", "detectorSourceSha256"):
            broken = copy.deepcopy(s)
            del broken["benchmark"][missing]
            with self.assertRaises(ValueError):
                evaluate_session(broken)
        s["benchmark"]["trackingSource"] = "saved-frames"
        with self.assertRaisesRegex(ValueError, "retain the browser model"):
            evaluate_session(s)

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


class PeakOccurrenceTests(unittest.TestCase):
    def fixture(self):
        s = session()
        s["annotationsComplete"] = True
        s["annotations"] = [{**event("a1", 100, 800), "peakMs": 400}]
        s["events"] = [{**event("p1", 350, 500), "peakMs": 420, "detectedAtMs": 900}]
        return s

    def diagnostic(self, s, **kwargs):
        return evaluate_session(s, labels=("jab", "cross", "hook", "uppercut"), **kwargs)["peakOccurrenceDiagnostics"]

    def test_short_interval_can_match_peak_without_changing_strict_score(self):
        s = self.fixture()
        result = evaluate_session(s)
        self.assertEqual((result["eventMetrics"]["tp"], result["eventMetrics"]["fp"], result["eventMetrics"]["fn"]), (0, 1, 1))
        d = result["peakOccurrenceDiagnostics"]
        self.assertEqual((d["metrics"]["tp"], d["metrics"]["fp"], d["metrics"]["fn"]), (1, 0, 0))
        self.assertFalse(d["matches"][0]["alsoStrictMatchedPair"])
        self.assertEqual(d["timingMs"]["peakErrorMs"]["p50"], 20)
        self.assertEqual(d["timingMs"]["sourcePeakToEmissionMs"]["p50"], 500)
        self.assertEqual(d["timingMs"]["predictedPeakToEmissionMs"]["p50"], 480)
        self.assertEqual(d["timingMs"]["referenceEndToEmissionMs"]["p50"], 100)

    def test_duplicate_predictions_only_match_one_occurrence(self):
        s = self.fixture()
        s["events"].append({**s["events"][0], "id": "duplicate"})
        d = self.diagnostic(s)
        self.assertEqual((d["metrics"]["tp"], d["metrics"]["fp"], d["metrics"]["fn"]), (1, 1, 0))
        self.assertEqual(len(d["matches"]), 1)
        self.assertEqual(d["confusionAssociations"], [])

    def test_wrong_hand_and_family_remain_errors_with_separate_confusion(self):
        for hand, label, wrong_hand, wrong_family in [
            ("right", "jab", True, False), ("left", "hook", False, True), ("right", "uppercut", True, True)
        ]:
            with self.subTest(hand=hand, label=label):
                s = self.fixture()
                s["events"][0].update(hand=hand, label=label)
                d = self.diagnostic(s)
                self.assertEqual((d["metrics"]["tp"], d["metrics"]["fp"], d["metrics"]["fn"]), (0, 1, 1))
                self.assertEqual(len(d["confusionAssociations"]), 1)
                confusion = d["confusionAssociations"][0]
                self.assertEqual((confusion["wrongHand"], confusion["wrongFamily"]), (wrong_hand, wrong_family))

    def test_jab_cross_share_family_but_not_strict_label(self):
        s = self.fixture()
        s["events"][0].update(label="cross", startMs=100, endMs=800)
        r = evaluate_session(s)
        self.assertEqual(r["eventMetrics"]["tp"], 0)
        self.assertEqual(r["peakOccurrenceDiagnostics"]["metrics"]["tp"], 1)

    def test_fixed_250ms_tolerance_inclusive_both_directions(self):
        for error, expected in [(-250, 1), (250, 1), (-250.001, 0), (250.001, 0)]:
            s = self.fixture()
            s["events"][0].update(startMs=100, endMs=800, peakMs=400 + error)
            self.assertEqual(self.diagnostic(s)["metrics"]["tp"], expected)

    def test_peak_assignment_maximizes_cardinality_not_greedy_nearest(self):
        predictions = [{**event("p1", 0, 900), "peakMs": 450}, {**event("p2", 0, 900), "peakMs": 200}]
        truths = [{**event("a1", 0, 900), "peakMs": 400}, {**event("a2", 0, 900), "peakMs": 650}]
        self.assertEqual(set(match_peak_events(predictions, truths)), {(0, 1), (1, 0)})

    def test_missing_peaks_withhold_metrics_without_fabricating_midpoint(self):
        for field, expected_key in [("annotations", "missingReferencePeakIds"), ("events", "missingPredictionPeakIds")]:
            s = self.fixture()
            del s[field][0]["peakMs"]
            d = self.diagnostic(s)
            self.assertIsNone(d["metrics"])
            self.assertEqual(d["status"], "withheld_missing_explicit_peaks")
            self.assertEqual(d[expected_key], [s[field][0]["id"]])
            self.assertEqual(d["matches"], [])
            self.assertIsNone(d["timingMs"]["sourcePeakToEmissionMs"]["p50"])

    def test_invalid_or_conflicting_peaks_error(self):
        for field in ("events", "annotations"):
            for value in (None, True, float("nan"), float("inf"), -1, 999):
                with self.subTest(field=field, value=value):
                    s = self.fixture()
                    s[field][0]["peakMs"] = value
                    with self.assertRaises(ValueError):
                        self.diagnostic(s)
        s = self.fixture()
        s["annotations"][0]["peakTMs"] = 401
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.diagnostic(s)

    def test_delivery_clock_never_substitutes_for_missing_source_peak(self):
        s = self.fixture()
        s["events"][0]["detectedAtMs"] = 1000
        self.assertEqual(self.diagnostic(s)["metrics"]["tp"], 1)
        self.assertEqual(self.diagnostic(s)["timingMs"]["sourcePeakToEmissionMs"]["p50"], 600)
        del s["events"][0]["peakMs"]
        s["events"][0]["detectedAtMs"] = 400
        self.assertIsNone(self.diagnostic(s)["metrics"])

    def test_missing_emission_telemetry_is_not_invented_and_early_emission_errors(self):
        s = self.fixture()
        del s["events"][0]["detectedAtMs"]
        self.assertEqual(self.diagnostic(s)["timingMs"]["sourcePeakToEmissionMs"]["count"], 0)
        s["events"][0]["detectedAtMs"] = 499
        with self.assertRaisesRegex(ValueError, "emission precedes"):
            self.diagnostic(s)

    def test_external_reference_aliases_preserve_source_and_strict_metric(self):
        for collection, peak_key, id_key in [("actionObservations", "peakMs", "id"),
                                             ("approximatePeaks", "approximatePeakMs", "annotationId"),
                                             ("peaks", "peakTMs", "annotationId")]:
            s = self.fixture()
            del s["annotations"][0]["peakMs"]
            original = copy.deepcopy(s)
            reference = {"sessionId": s["id"], collection: [{id_key: "a1", peak_key: 400}]}
            r = evaluate_session(s, peak_reference=reference)
            self.assertEqual(r["peakOccurrenceDiagnostics"]["metrics"]["tp"], 1)
            self.assertEqual(s, original)
            self.assertEqual(r["eventMetrics"], evaluate_session(s)["eventMetrics"])
            self.assertEqual(len(r["peakOccurrenceDiagnostics"]["reference"]["referenceContentSha256"]), 64)

    def test_reference_identity_boundaries_and_conflicts_checked(self):
        s = self.fixture()
        for ref in [{"sessionId": "wrong", "peaks": []},
                    {"peaks": [{"id": "unknown", "peakMs": 200}]},
                    {"peaks": [{"id": "a1", "peakMs": 400, "hand": "right"}]},
                    {"peaks": [{"id": "a1", "peakMs": 400, "startMs": 101}]},
                    {"peaks": [{"id": "a1", "peakMs": 401}]}]:
            with self.assertRaises(ValueError):
                self.diagnostic(s, peak_reference=ref)
        s["benchmark"] = {"sourceSessionId": "original-id", "sourceVideoSha256": "a" * 64}
        self.assertEqual(self.diagnostic(s, peak_reference={"sourceSessionId": "original-id", "peaks": []})["metrics"]["tp"], 1)
        with self.assertRaisesRegex(ValueError, "video fingerprint"):
            self.diagnostic(s, peak_reference={"sourceVideoSha256": "b" * 64, "peaks": []})

    def test_raw_uncertain_notes_do_not_add_truth_or_change_exclusions(self):
        s = self.fixture()
        s["annotations"].append(event("unknown", 800, 1000, "unobservable", "unknown"))
        s["events"].append({**event("ignored", 850, 950), "peakMs": 900})
        reference = {"actionObservations": [{"id": "ambiguous-not-scored", "peakMs": 900, "label": "hook"}]}
        r = evaluate_session(s, peak_reference=reference)
        d = r["peakOccurrenceDiagnostics"]
        self.assertEqual(d["metrics"]["tp"], 1)
        self.assertEqual(d["metrics"]["fp"], 0)
        self.assertEqual(d["metrics"]["evaluatedExposureMs"], 800)
        self.assertEqual(d["reference"]["unscoredReferenceObservationIds"], ["ambiguous-not-scored"])

    def test_empty_and_incomplete_reference_aggregate(self):
        empty = session()
        empty["annotationsComplete"] = True
        r = build_report([empty])
        self.assertIsNone(r["peakOccurrenceDiagnostics"]["metrics"]["precision"])
        complete, missing = self.fixture(), self.fixture()
        missing["id"] = "missing"
        del missing["annotations"][0]["peakMs"]
        r = build_report([complete, missing])
        self.assertIsNone(r["peakOccurrenceDiagnostics"]["metrics"])
        self.assertEqual(r["peakOccurrenceDiagnostics"]["scoredSessionCount"], 1)
        self.assertEqual(r["peakOccurrenceDiagnostics"]["timingMs"]["sourcePeakToEmissionMs"]["count"], 1)
        with self.assertRaisesRegex(ValueError, "input session IDs"):
            build_report([complete], peak_references={"different": {"peaks": []}})

    def test_cli_reference_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            source, reference, output = [Path(directory) / name for name in ("session.json", "peaks.json", "report.json")]
            s = self.fixture()
            del s["annotations"][0]["peakMs"]
            source.write_text(json.dumps(s))
            reference.write_text(json.dumps({"peaks": [{"annotationId": "a1", "peakTMs": 400}]}))
            self.assertEqual(main([str(source), "--peak-reference", str(reference), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["peakOccurrenceDiagnostics"]["metrics"]["tp"], 1)
            original = reference.read_bytes()
            self.assertEqual(main([str(source), "--peak-reference", str(reference), "--output", str(reference)]), 2)
            self.assertEqual(reference.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
