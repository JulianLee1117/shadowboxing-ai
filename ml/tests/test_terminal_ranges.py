from copy import deepcopy
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest

from ml.terminal_ranges import INPUT_VERSION, MAX_ITEMS, diagnose, main, read_document


def prediction(id="p", peak=500, start=0, end=1000, hand="left", family="straight"):
    return {"id": id, "hand": hand, "family": family, "startMs": start,
            "peakMs": peak, "endMs": end, "peakObserved": True}


def reference(id="r", start=0, end=1000, lo=400, hi=600, hand="left", family="straight"):
    return {"id": id, "hand": hand, "family": family, "startMs": start, "endMs": end,
            "peakMs": None, "terminalRange": {"startMs": lo, "endMs": hi,
                "firstFrameIndex0Based": int(lo), "lastFrameIndex0BasedInclusive": int(hi)}}


def document(predictions=None, references=None, unknown=None, complete=True):
    return {"schemaVersion": INPUT_VERSION,
            "source": {"id": "fixture", "videoSha256": "a" * 64, "clock": "source-presentation-ms"},
            "referencesComplete": complete,
            "predictions": [prediction()] if predictions is None else predictions,
            "references": [reference()] if references is None else references,
            "unknownIntervals": [] if unknown is None else unknown}


class TerminalRangeTests(unittest.TestCase):
    def test_held_plateau_preserves_null_scalar_without_midpoint(self):
        data = document([prediction(peak=590)], [reference(lo=400, hi=600)])
        original = deepcopy(data)
        result = diagnose(data)
        self.assertEqual(result["hands"]["left"]["matches"][0]["distanceToTerminalRangeMs"], 0)
        self.assertEqual(result["referenceEvidence"][0]["terminalWidthMs"], 200)
        self.assertIsNone(result["referenceEvidence"][0]["scalarPeakMs"])
        self.assertEqual(data, original)
        self.assertNotIn("accuracy", result)
        self.assertNotIn("eventMetrics", result)

    def test_adjacent_double_is_order_preserving_and_one_to_one(self):
        result = diagnose(document(
            [prediction("p2", 760), prediction("p1", 420)],
            [reference("r2", 700, 950, 740, 790), reference("r1", 300, 650, 400, 450)]))
        self.assertEqual([(m["predictionId"], m["referenceId"]) for m in result["hands"]["left"]["matches"]],
                         [("p1", "r1"), ("p2", "r2")])

    def test_one_prediction_two_overlapping_refs_reports_both_optima(self):
        result = diagnose(document(references=[reference("z"), reference("a")]))["hands"]["left"]
        self.assertEqual(result["matches"][0]["referenceId"], "a")
        self.assertTrue(result["matches"][0]["assignmentAmbiguous"])
        self.assertEqual(result["predictionAlternatives"][0]["referenceIds"], ["a", "z"])
        self.assertFalse(result["predictionAlternatives"][0]["canBeUnmatched"])
        self.assertTrue(all(r["canBeUnmatched"] and r["ambiguous"] for r in result["referenceAlternatives"]))
        self.assertEqual(result["orderUncertainty"]["overlappingTerminalRangePairs"], [["a", "z"]])

    def test_duplicate_predictions_are_not_both_matches(self):
        hand = diagnose(document([prediction("z"), prediction("a")]))["hands"]["left"]
        self.assertEqual(len(hand["matches"]), 1)
        self.assertEqual(hand["matches"][0]["predictionId"], "a")
        self.assertEqual(hand["unmatchedPredictionIds"], ["z"])
        self.assertTrue(all(p["canBeUnmatched"] for p in hand["predictionAlternatives"]))

    def test_hand_and_family_cannot_be_reassigned(self):
        result = diagnose(document([prediction(hand="right"), prediction("wrong-family", family="hook")]))
        self.assertEqual(result["hands"]["left"]["matches"], [])
        self.assertEqual(result["hands"]["right"]["matches"], [])
        self.assertEqual(result["hands"]["right"]["unmatchedPredictionIds"], ["p"])

    def test_fixed_tolerance_inclusive_but_action_bound_also_required(self):
        ref = reference(start=100, end=1000, lo=400, hi=500)
        for peak, matches in [(150, 1), (149.999, 0), (750, 1), (750.001, 0), (99, 0)]:
            with self.subTest(peak=peak):
                result = diagnose(document([prediction(peak=peak)], [ref]))
                self.assertEqual(len(result["hands"]["left"]["matches"]), matches)
        # Close to the range but outside its original action is not eligible.
        self.assertEqual(diagnose(document([prediction(peak=390)], [reference(start=400, lo=400)]))["hands"]["left"]["matches"], [])

    def test_minimum_distance_wins_before_id_tie_break(self):
        result = diagnose(document([prediction("a", 200), prediction("z", 480)]))["hands"]["left"]
        self.assertEqual(result["matches"][0]["predictionId"], "z")
        self.assertFalse(result["matches"][0]["assignmentAmbiguous"])

    def test_order_constraint_can_forbid_crossed_families(self):
        ps = [prediction("early", 450, family="hook"), prediction("late", 550)]
        rs = [reference("early-ref", lo=400, hi=500), reference("late-ref", lo=500, hi=600, family="hook")]
        # A free bipartite assignment could return two crossed pairs. This one cannot.
        self.assertEqual(len(diagnose(document(ps, rs))["hands"]["left"]["matches"]), 1)

    def test_unknown_straddle_and_touch_excluded_prominently(self):
        mask = {"id": "u", "hand": "left", "startMs": 900, "endMs": 1200}
        result = diagnose(document(unknown=[mask]))
        self.assertEqual(result["hands"]["left"]["matches"], [])
        self.assertEqual(result["coverage"]["excludedPredictions"], 1)
        self.assertEqual(result["coverage"]["excludedReferences"], 1)
        self.assertEqual(result["excluded"]["predictions"][0]["unknownIntervalIds"], ["u"])
        self.assertEqual(result["hands"]["left"]["unmatchedPredictionIds"], [])
        # Touching closed endpoint also excludes; other hand does not.
        mask["startMs"] = 1000
        self.assertEqual(diagnose(document(unknown=[mask]))["coverage"]["excludedPredictions"], 1)
        mask["hand"] = "right"
        self.assertEqual(diagnose(document(unknown=[mask]))["coverage"]["excludedPredictions"], 0)
        mask["hand"] = "both"
        self.assertEqual(diagnose(document(unknown=[mask]))["coverage"]["excludedPredictions"], 1)

    def test_incomplete_and_empty_inputs_do_not_claim_metrics(self):
        result = diagnose(document([prediction()], [], complete=False))
        self.assertFalse(result["referencesComplete"])
        self.assertEqual(result["hands"]["left"]["unmatchedPredictionIds"], ["p"])
        empty = diagnose(document([], []))
        self.assertEqual(empty["coverage"]["eligibleReferences"], 0)

    def test_delivery_time_does_not_replace_peak_or_affect_pairing(self):
        data = document()
        baseline = diagnose(data)["hands"]
        data["predictions"][0]["detectedAtMs"] = 9000
        self.assertEqual(diagnose(data)["hands"], baseline)
        data["predictions"][0]["peakMs"] = None
        with self.assertRaises(ValueError):
            diagnose(data)

    def test_invalid_duplicate_and_nonfinite_inputs_rejected(self):
        changes = [
            lambda d: d["predictions"].append(deepcopy(d["predictions"][0])),
            lambda d: d["references"].append(deepcopy(d["references"][0])),
            lambda d: d["predictions"][0].update(peakMs=float("nan")),
            lambda d: d["references"][0]["terminalRange"].update(endMs=float("inf")),
            lambda d: d["predictions"][0].update(peakMs=True),
            lambda d: d["predictions"][0].update(peakObserved=False),
            lambda d: d["predictions"][0].update(peakMs=1001),
            lambda d: d["predictions"][0].update(startMs=10**1000),
            lambda d: d["references"][0].update(family="jab"),
            lambda d: d["references"][0].update(hand="lead"),
            lambda d: d["references"][0]["terminalRange"].update(firstFrameIndex0Based=-1),
            lambda d: d["references"][0]["terminalRange"].update(startMs=-1),
            lambda d: d["source"].update(clock="wall-time"),
            lambda d: d.update(referencesComplete=1),
            lambda d: d["predictions"][0].update(startMs=500, endMs=500),
            lambda d: d.update(predictions=[prediction(str(i)) for i in range(MAX_ITEMS + 1)]),
        ]
        for change in changes:
            with self.subTest(change=change):
                data = document()
                change(data)
                with self.assertRaises(ValueError):
                    diagnose(data)

    def test_conflicting_native_frame_clock_evidence_rejected(self):
        a, b = reference("a"), reference("b", lo=500, hi=700)
        b["terminalRange"]["firstFrameIndex0Based"] = 400
        with self.assertRaisesRegex(ValueError, "conflicting"):
            diagnose(document(references=[a, b]))
        b["terminalRange"]["firstFrameIndex0Based"] = 800
        b["terminalRange"]["lastFrameIndex0BasedInclusive"] = 900
        with self.assertRaisesRegex(ValueError, "frame order"):
            diagnose(document(references=[a, b]))

    def test_against_exhaustive_small_assignments_and_input_permutations(self):
        rng = random.Random(20260927)
        for _ in range(30):
            ps = [prediction(f"p{i}", rng.randrange(100, 900, 100), family=rng.choice(["straight", "hook"])) for i in range(3)]
            rs = [reference(f"r{i}", lo=t, hi=t+100, family=rng.choice(["straight", "hook"])) for i, t in enumerate([200, 400, 600])]
            p = sorted(ps, key=lambda x: (x["peakMs"], x["id"]))
            r = sorted(rs, key=lambda x: (x["terminalRange"]["startMs"], x["terminalRange"]["endMs"], x["startMs"], x["id"]))
            assignments = []
            for k in range(4):
                for pi in itertools.combinations(range(3), k):
                    for ri in itertools.combinations(range(3), k):
                        pairs = list(zip(pi, ri));costs=[]
                        for a,b in pairs:
                            d=max(r[b]["terminalRange"]["startMs"]-p[a]["peakMs"], p[a]["peakMs"]-r[b]["terminalRange"]["endMs"],0)
                            if p[a]["family"] != r[b]["family"] or d>250:
                                break
                            costs.append(Fraction(d))
                        else:
                            assignments.append((k,sum(costs),tuple((p[a]["id"],r[b]["id"]) for a,b in pairs)))
            best_count=max(x[0] for x in assignments);best_cost=min(x[1] for x in assignments if x[0]==best_count)
            optimal=[x[2] for x in assignments if x[:2]==(best_count,best_cost)]
            result=diagnose(document(ps,rs))["hands"]["left"]
            self.assertEqual(tuple((m["predictionId"],m["referenceId"]) for m in result["matches"]),min(optimal))
            for options in result["predictionAlternatives"]:
                expected={b for matching in optimal for a,b in matching if a==options["predictionId"]}
                self.assertEqual(set(options["referenceIds"]),expected)
                self.assertEqual(options["canBeUnmatched"],any(not any(a==options["predictionId"] for a,_ in matching) for matching in optimal))
            self.assertEqual(diagnose(document(list(reversed(ps)),list(reversed(rs))))["hands"]["left"],result)

    def test_cli_strict_json_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"in.json", Path(directory)/"out.json"
            source.write_text(json.dumps(document()))
            self.assertEqual(main([str(source),"--output",str(output)]),0)
            report=json.loads(output.read_text())
            self.assertEqual(len(report["inputSha256"]),64)
            self.assertEqual(main([str(source),"--output",str(output)]),1)
            for value in ['{"x":1,"x":2}', '{"x":1e309}', '{"x":NaN}']:
                source.write_text(value)
                with self.assertRaises(ValueError):
                    read_document(source)


if __name__ == "__main__":
    unittest.main()
