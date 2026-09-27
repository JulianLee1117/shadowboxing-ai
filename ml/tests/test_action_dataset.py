import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ml.action_dataset import CLASSES, audit_manifest, build_targets, main


def session(identifier="round-a", perturb=0):
    return {"schemaVersion": "1.0", "id": identifier, "source": "file", "model": "full",
            "stance": "orthodox", "durationMs": 1000, "events": [], "annotationsComplete": True,
            "frames": [{"t": t, "width": 1280, "height": 720, "inferenceMs": 8,
                        "landmarks": [{"x": .1 + perturb, "y": .2, "visibility": .123}]}
                       for t in range(0, 1000, 100)],
            "annotations": [{"id": "jab", "label": "jab", "hand": "left", "startMs": 100, "endMs": 200},
                            {"id": "cross", "label": "cross", "hand": "right", "startMs": 300, "endMs": 400}]}


def entry(identifier="a", **overrides):
    result = {"id": identifier, "session": f"{identifier}.json", "split": "train", "participantId": "p1",
              "captureDay": "2026-09-26", "sourceGroup": f"original-{identifier}", "domain": "shadowboxing",
              "view": "three-quarter", "rights": {"status": "owned", "reference": "operator declaration"},
              "labelScope": {"labels": ["jab", "cross"], "complete": True}}
    result.update(overrides)
    return result


def write_manifest(root, entries, policy="recording"):
    path = root / "manifest.json"
    path.write_text(json.dumps({"schemaVersion": "action-dataset-1", "splitPolicy": policy,
                                "expectedDomains": ["shadowboxing", "bag"], "entries": entries}))
    return path


class ActionDatasetTests(unittest.TestCase):
    def test_straight_only_complete_data_never_fabricates_multiclass_background(self):
        s = session()
        rows, events = build_targets(s, {"labels": ["jab", "cross"], "complete": True})
        self.assertFalse(rows[0]["left"]["mask"])
        self.assertIsNone(rows[0]["left"]["familyIndex"])
        self.assertTrue(rows[1]["left"]["mask"])
        self.assertEqual(rows[1]["left"]["familyIndex"], 1)
        self.assertFalse(rows[2]["left"]["mask"])  # half-open event end
        self.assertEqual([e["sixClass"] for e in events], ["jab", "cross"])

    def test_six_class_semantics_preserve_southpaw_physical_hands(self):
        s = session(); s["stance"] = "southpaw"; s["annotations"] = []
        for i, (label, hand) in enumerate((("jab", "right"), ("cross", "left"), ("hook", "right"),
                                          ("hook", "left"), ("uppercut", "right"), ("uppercut", "left"))):
            s["annotations"].append({"id": str(i), "label": label, "hand": hand, "startMs": i*100, "endMs": (i+1)*100})
        rows, events = build_targets(s, {"labels": ["jab", "cross", "hook", "uppercut"], "complete": True})
        self.assertEqual([e["sixClass"] for e in events], list(CLASSES))
        self.assertEqual(events[0]["hand"], "right")
        self.assertEqual(events[0]["role"], "lead")
        self.assertEqual(rows[-1]["left"]["familyIndex"], 0)

    def test_unknown_masks_without_changing_pose_or_other_visible_arm(self):
        s = session(); original = copy.deepcopy(s["frames"])
        s["annotations"] += [{"id": "hidden", "label": "unobservable", "hand": "left", "startMs": 100, "endMs": 200},
                             {"id": "uncertain", "label": "hook", "hand": "unknown", "startMs": 500, "endMs": 600}]
        rows, _ = build_targets(s, {"labels": ["jab", "cross", "hook", "uppercut"], "complete": True})
        self.assertFalse(rows[1]["left"]["mask"])
        self.assertTrue(rows[1]["right"]["mask"])
        self.assertFalse(rows[5]["left"]["mask"])
        self.assertFalse(rows[5]["right"]["mask"])
        self.assertEqual(s["frames"], original)

    def test_conflicting_role_and_same_arm_boundaries_are_rejected(self):
        s = session(); s["annotations"][0]["hand"] = "right"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            build_targets(s, entry()["labelScope"])
        s = session(); s["annotations"].append({"id": "hook", "label": "hook", "hand": "left", "startMs": 150, "endMs": 250})
        with self.assertRaisesRegex(ValueError, "boundary adjudication"):
            build_targets(s, entry()["labelScope"])
        s["annotations"][-1]["hand"] = "right"
        build_targets(s, entry()["labelScope"])  # different arms may overlap

    def test_incomplete_annotation_assertion_cannot_be_upgraded_by_manifest(self):
        s = session(); s["annotationsComplete"] = False
        with self.assertRaisesRegex(ValueError, "annotationsComplete"):
            build_targets(s, entry()["labelScope"])
        rows, _ = build_targets(s, {"labels": ["jab", "cross", "hook", "uppercut"], "complete": False})
        self.assertFalse(rows[-1]["left"]["mask"])

    def test_duplicate_capture_ignores_ids_and_inference_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = session(); b = copy.deepcopy(a); b["id"] = "renamed"
            b["frames"][0]["inferenceMs"] = 900
            for name, value in (("a", a), ("b", b)):
                (root / f"{name}.json").write_text(json.dumps(value))
            report, _ = audit_manifest(write_manifest(root, [entry(), entry("b", split="test")]))
            self.assertFalse(report["valid"])
            self.assertIn("Duplicate pose capture", str(report["errors"]))

    def test_group_leakage_obeys_declared_policy_and_always_blocks_source_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, name in enumerate(("a", "b")):
                (root / f"{name}.json").write_text(json.dumps(session(name, i*.01)))
            entries = [entry(), entry("b", split="test")]
            report, _ = audit_manifest(write_manifest(root, entries))
            self.assertTrue(report["valid"])
            self.assertTrue(report["crossSplitGroups"]["participantDay"])
            report, _ = audit_manifest(write_manifest(root, entries, "participant-day"))
            self.assertFalse(report["valid"])
            entries[1]["captureDay"] = "2026-09-27"
            report, _ = audit_manifest(write_manifest(root, entries, "participant-day"))
            self.assertTrue(report["valid"])
            report, _ = audit_manifest(write_manifest(root, entries, "participant"))
            self.assertFalse(report["valid"])
            entries[1]["sourceGroup"] = entries[0]["sourceGroup"]
            report, _ = audit_manifest(write_manifest(root, entries))
            self.assertFalse(report["valid"])
            self.assertIn("sourceGroup", str(report["errors"]))

    def test_video_hash_is_bytes_provenance_not_proof_of_session_association(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "a.json").write_text(json.dumps(session()))
            (root / "clip.webm").write_bytes(b"nonvideo byte fixture")
            e = entry(video="clip.webm", videoSha256=hashlib.sha256(b"nonvideo byte fixture").hexdigest())
            report, prepared = audit_manifest(write_manifest(root, [e]))
            self.assertTrue(report["valid"])
            self.assertEqual(prepared[0]["source"]["videoAssociation"], "operator_declared_not_cryptographically_established")
            e["videoSha256"] = "incorrect"
            report, _ = audit_manifest(write_manifest(root, [e]))
            self.assertFalse(report["valid"])

    def test_coverage_exposes_missing_punches_domains_and_heldout_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "a.json").write_text(json.dumps(session()))
            report, _ = audit_manifest(write_manifest(root, [entry()]))
            self.assertTrue(report["valid"])
            self.assertEqual(report["missingClasses"], list(CLASSES[2:]))
            self.assertEqual(report["splits"]["test"]["sessions"], 0)
            self.assertEqual(report["splits"]["train"]["missingDomainClasses"]["bag"], list(CLASSES))
            self.assertNotIn("known-background", report["armFrameTargets"])

    def test_unreviewed_rights_block_prepare_but_audit_records_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "a.json").write_text(json.dumps(session()))
            manifest = write_manifest(root, [entry(rights={"status": "unreviewed", "reference": "external source"})])
            report = root / "report.json"
            self.assertEqual(main(["audit", str(manifest), "--output", str(report)]), 2)
            self.assertIn("unreviewed", str(json.loads(report.read_text())["errors"]))
            target = root / "prepared"
            self.assertEqual(main(["prepare", str(manifest), "--output", str(target)]), 2)
            self.assertFalse(target.exists())

    def test_prepare_preserves_sources_masks_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "a.json"; original = json.dumps(session()); source.write_text(original)
            manifest = write_manifest(root, [entry()]); target = root / "prepared"
            self.assertEqual(main(["prepare", str(manifest), "--output", str(target)]), 0)
            prepared = json.loads((target / "session-a.json").read_text())
            self.assertEqual(prepared["frames"], session()["frames"])
            self.assertEqual(source.read_text(), original)
            self.assertFalse(prepared["targets"][0]["left"]["mask"])
            self.assertEqual(main(["prepare", str(manifest), "--output", str(target)]), 2)


if __name__ == "__main__":
    unittest.main()
