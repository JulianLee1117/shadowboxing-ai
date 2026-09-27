import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from ml.coaching import (CRITERIA, build_packet, compare_reviews, main, packet_id,
                         prepare, review_template, validate_packet, validate_review)


def session():
    return {"schemaVersion": "1.0", "id": "local-round", "source": "file", "model": "full",
            "stance": "orthodox", "durationMs": 5000, "videoOffsetMs": 120,
            "frames": [], "events": [{"id": "prediction", "label": "cross", "hand": "right",
                                       "startMs": 800, "endMs": 900}],
            "annotations": [{"id": "a", "label": "jab", "hand": "left", "startMs": 1000, "endMs": 1300},
                            {"id": "b", "label": "cross", "hand": "right", "startMs": 2000, "endMs": 2400}]}


def packet(s=None):
    return build_packet(s or session(), session_sha256="a" * 64, video_sha256="b" * 64,
                        video_name="round.webm", video_bytes=10)


def review(p, reviewer="coach-a"):
    value = review_template(p)
    value["reviewer"] = {"id": reviewer, "expertise": "coach_self_reported"}
    return value


def rate(row, judgment="pass", **changes):
    row.update(judgment=judgment, context="eligible", view="adequate", outcomeObserved=True,
               feedbackWarranted="no", evidence={"startMs": 900, "endMs": 2600},
               rationale="Independent example evidence for software verification only.")
    row.update(changes)


class CoachingTests(unittest.TestCase):
    def test_prediction_blind_packet_and_no_prefilled_judgments(self):
        s = session(); original = copy.deepcopy(s)
        p = packet(s); validate_packet(p)
        self.assertEqual(s, original)
        self.assertEqual([a["id"] for a in p["items"]], ["a", "b"])
        self.assertNotIn("prediction", [a["id"] for a in p["items"]])
        self.assertNotIn("events", p); self.assertNotIn("frames", p)
        self.assertEqual(p["source"]["videoOffsetMs"], 120)
        r = review(p)
        self.assertEqual(validate_review(p, r)["counts"]["not_reviewed"], 4)
        self.assertTrue(all(row["evidence"] is None for row in r["ratings"]))

    def test_hooks_retained_without_form_rubric_unknown_hands_accounted_for(self):
        s = session()
        s["annotations"] += [{"id": "h", "label": "hook", "hand": "left", "startMs": 3000, "endMs": 3300},
                             {"id": "u", "label": "uppercut", "hand": "unknown", "startMs": 4000, "endMs": 4400}]
        p = packet(s); validate_packet(p)
        self.assertEqual(p["items"][-1]["criteria"], [])
        self.assertEqual(p["excludedActions"], [{"id": "u", "reason": "physical_hand_unknown"}])
        self.assertEqual(len(review_template(p)["ratings"]), 4)
        r = review(p); r["ratings"][0]["actionId"] = "h"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_review(p, r)

    def test_no_synthetic_truth_or_predictions_as_review_targets(self):
        s = session(); s["source"] = "demo"
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            packet(s)
        s = session(); s["annotations"] = []
        with self.assertRaisesRegex(ValueError, "Human action"):
            packet(s)

    def test_stance_mapping_and_conflicts(self):
        s = session(); s["stance"] = "southpaw"
        for action in s["annotations"]:
            action["hand"] = "right" if action["hand"] == "left" else "left"
        p = packet(s); validate_packet(p)
        self.assertEqual(p["items"][0]["role"], "lead")
        self.assertEqual(p["items"][0]["hand"], "right")
        s["annotations"][0]["hand"] = "left"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            packet(s)

    def test_packet_binding_and_structural_validation(self):
        p = packet(); r = review(p)
        altered_rubric = copy.deepcopy(p)
        altered_rubric["criteria"]["guard_recovery"]["question"] = "Changed question"
        altered_rubric["packetId"] = packet_id(altered_rubric)
        with self.assertRaisesRegex(ValueError, "criterion"):
            validate_packet(altered_rubric)
        direct = packet(); direct["criteria"]["guard_recovery"]["question"] = "Caller mutation"
        self.assertNotEqual(CRITERIA["guard_recovery"]["question"], "Caller mutation")
        changed = copy.deepcopy(p); changed["source"]["videoSha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_packet(changed)
        changed["packetId"] = packet_id(changed)
        with self.assertRaisesRegex(ValueError, "exact packet"):
            validate_review(changed, r)
        for key, value in (("stance", "unknown"), ("videoOffsetMs", True), ("videoOffsetMs", -1),
                           ("videoBytes", 0), ("videoBytes", True), ("videoName", "")):
            changed = copy.deepcopy(p); changed["source"][key] = value; changed["packetId"] = packet_id(changed)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_packet(changed)
        changed = copy.deepcopy(p); changed["items"][0]["role"] = "rear"; changed["packetId"] = packet_id(changed)
        with self.assertRaisesRegex(ValueError, "role"):
            validate_packet(changed)

    def test_pass_fail_require_context_view_and_observed_outcome(self):
        p = packet()
        for judgment in ("pass", "fail"):
            for changes in ({"context": "unknown"}, {"context": "not_applicable"},
                            {"view": "inadequate"}, {"view": "unknown"}, {"outcomeObserved": False}):
                r = review(p); rate(r["ratings"][0], judgment, **changes)
                with self.subTest(judgment=judgment, changes=changes), self.assertRaisesRegex(ValueError, "Pass/fail"):
                    validate_review(p, r)

    def test_uncertainty_does_not_become_correction_and_fail_need_not_prompt(self):
        p = packet(); r = review(p)
        rate(r["ratings"][0], "fail", feedbackWarranted="no")
        validate_review(p, r)
        rate(r["ratings"][0], "unobservable", outcomeObserved=False, view="inadequate", feedbackWarranted="yes")
        with self.assertRaisesRegex(ValueError, "correction"):
            validate_review(p, r)
        r["ratings"][0]["feedbackWarranted"] = "uncertain"
        self.assertEqual(validate_review(p, r)["counts"]["unobservable"], 1)
        r["ratings"][0]["outcomeObserved"] = True
        with self.assertRaisesRegex(ValueError, "unobservable"):
            validate_review(p, r)

    def test_evidence_can_exceed_navigation_window_but_not_source(self):
        p = packet(); r = review(p)
        rate(r["ratings"][0], evidence={"startMs": 1000, "endMs": 4000})
        self.assertLess(p["items"][0]["windowEndMs"], 4000)
        validate_review(p, r)
        for evidence in (None, {"startMs": True, "endMs": 3000}, {"startMs": -1, "endMs": 3000},
                         {"startMs": 1000, "endMs": 5001}, {"startMs": 2000, "endMs": 2000},
                         {"startMs": float("nan"), "endMs": 3000}):
            r["ratings"][0]["evidence"] = evidence
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, "Evidence"):
                validate_review(p, r)

    def test_no_missing_duplicate_or_finding_bearing_unreviewed_rows(self):
        p = packet()
        for mutation in (lambda r: r["ratings"].pop(), lambda r: r["ratings"].append(r["ratings"][0]),
                         lambda r: r["ratings"][0].update(feedbackWarranted="no")):
            r = review(p); mutation(r)
            with self.assertRaises(ValueError):
                validate_review(p, r)
        r = review(p); rate(r["ratings"][0], "not_applicable", context="eligible")
        with self.assertRaisesRegex(ValueError, "inapplicable"):
            validate_review(p, r)

    def test_agreement_has_abstention_coverage_and_disagreement_fields(self):
        p = packet(); a = review(p); b = review(p, "coach-b")
        # Guard: pass/pass, fail/pass -> agreement .5, kappa 0.
        for r, judgments in ((a, ("pass", "fail")), (b, ("pass", "pass"))):
            for i, judgment in zip((0, 2), judgments):
                rate(r["ratings"][i], judgment)
        rate(a["ratings"][1], "ambiguous", context="eligible", outcomeObserved=False, feedbackWarranted="uncertain")
        rate(b["ratings"][1], "ambiguous", context="unknown", outcomeObserved=False, feedbackWarranted="uncertain")
        result = compare_reviews(p, a, b)
        guard = result["criteria"]["guard_recovery"]
        self.assertEqual(guard["passFailAgreement"], .5)
        self.assertEqual(guard["cohensKappa"], 0)
        self.assertEqual(guard["decisiveCoverage"], 1)
        other = result["criteria"]["non_punching_hand_guard"]
        self.assertEqual(other["bothReviewed"], 1)
        self.assertEqual(other["decisiveCoverage"], 0)
        self.assertIsNone(other["cohensKappa"])
        self.assertEqual(other["fieldDisagreements"]["context"], 1)
        self.assertEqual(other["disagreements"], ["a"])
        self.assertEqual(other["judgmentCounts"][0]["not_reviewed"], 1)
        with self.assertRaisesRegex(ValueError, "distinct"):
            compare_reviews(p, a, a)

    def test_prepare_fingerprints_preserves_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source.json"; video = root / "original.webm"
            source.write_text(json.dumps(session())); video.write_bytes(b"opaque video bytes; no decode performed")
            originals = (source.read_bytes(), video.read_bytes())
            out = root / "packet"
            result = prepare(source, video, out)
            p = json.loads((out / "packet.json").read_text())
            self.assertEqual(p["source"]["videoSha256"], hashlib.sha256(originals[1]).hexdigest())
            self.assertEqual(result["ratingSlots"], 4)
            self.assertEqual((source.read_bytes(), video.read_bytes()), originals)
            with self.assertRaises(FileExistsError):
                prepare(source, video, out)
            r = review(p); review_path = out / "review.json"; review_path.write_text(json.dumps(r))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["check", str(out / "packet.json"), str(review_path)]), 0)
                self.assertEqual(main(["prepare", str(source), "--video", str(video), "--output", str(out)]), 2)
                self.assertEqual(main(["compare", str(out / "packet.json"), str(review_path), str(review_path), "--output", str(out / "agreement.json")]), 2)
            self.assertFalse((out / "agreement.json").exists())


if __name__ == "__main__":
    unittest.main()
