import copy
import json
from pathlib import Path
import tempfile
import unittest

from ml.diagnose import diagnose, frame_observation, main


def frame(t=0):
    points = [{"x": .5, "y": .5, "visibility": .99} for _ in range(33)]
    for index, x, y in [(0, .5, .15), (11, .4, .3), (12, .6, .3),
                        (13, .35, .45), (14, .65, .45),
                        (15, .4, .25), (16, .6, .25),
                        (23, .45, .7), (24, .55, .7)]:
        points[index].update(x=x, y=y)
    return {"t": t, "width": 1280, "height": 720,
            "landmarks": points, "inferenceMs": 10}


def session(frames=None):
    return {"schemaVersion": "1.0", "id": "software-fixture", "source": "file",
            "model": "full", "durationMs": 1000, "frames": frames or [],
            "events": [], "annotations": []}


class DiagnosticTests(unittest.TestCase):
    def test_opposite_wrist_failure_is_separated_from_observed_arm(self):
        f = frame()
        f["landmarks"][15]["visibility"] = .64
        r = diagnose(session([f]))["summary"]
        self.assertEqual(r["legacyAllNineGateCoverage"], 0)
        self.assertEqual(r["arms"]["right"]["trackingGateCoverage"], 1)
        self.assertEqual(r["arms"]["right"]["oppositeArmOnlyBlockFrames"], 1)
        self.assertEqual(r["arms"]["left"]["trackingGateCoverage"], 0)
        self.assertEqual(r["joints"]["leftWrist"]["failureCounts"], {"low_visibility": 1})

    def test_clipped_hip_disables_normalized_geometry_but_not_arm_joint_coverage(self):
        f = frame()
        f["landmarks"][24]["y"] = 1.001
        r = diagnose(session([f]))["summary"]
        for hand in ("left", "right"):
            self.assertEqual(r["arms"][hand]["jointCoverage"], 1)
            self.assertEqual(r["arms"][hand]["trackingGateCoverage"], 0)
            self.assertEqual(r["arms"][hand]["oppositeArmOnlyBlockFrames"], 0)
        self.assertEqual(r["joints"]["rightHip"]["failureCounts"], {"outside_image": 1})

    def test_hip_clipping_is_not_confused_with_low_confidence(self):
        f = frame()
        f["landmarks"][24].update(y=1.001, visibility=.98)
        observed = frame_observation(f, .65, .5)
        self.assertEqual(observed["jointIssues"]["rightHip"], ["outside_image"])
        self.assertIn("rightHip", observed["sharedIssues"])

    def test_geometry_is_aspect_corrected(self):
        f = frame()
        padded = copy.deepcopy(f)
        padded["width"] *= 2
        for point in padded["landmarks"]:
            point["x"] /= 2
        a = frame_observation(f, .65, .5)["arms"]["right"]
        b = frame_observation(padded, .65, .5)["arms"]["right"]
        self.assertAlmostEqual(a["elbowAngle2d"], b["elbowAngle2d"])
        self.assertAlmostEqual(a["reachTorsoUnits"], b["reachTorsoUnits"])

    def test_speed_does_not_bridge_gap_or_failed_tracking(self):
        frames = [frame(t) for t in (0, 33, 66, 100, 333)]
        frames[2]["landmarks"][16]["visibility"] = .2
        for i, f in enumerate(frames):
            f["landmarks"][16]["x"] += i * .01
        r = diagnose(session(frames), include_timeline=True)
        speeds = [f["arms"]["right"]["wristSpeedTorsoPerSecond"] for f in r["timeline"]]
        self.assertIsNone(speeds[0])
        self.assertGreater(speeds[1], 0)
        self.assertEqual(speeds[2:], [None, None, None])
        self.assertEqual(r["timing"]["gapsAboveLimit"], 1)

    def test_missing_landmarks_do_not_fabricate_angles(self):
        f = frame()
        f["landmarks"] = []
        r = diagnose(session([f]), include_timeline=True)
        for arm in r["timeline"][0]["arms"].values():
            self.assertFalse(arm["passesTrackingGate"])
            self.assertIsNone(arm["elbowAngle2d"])
            self.assertIsNone(arm["estimatedWorldElbowAngle"])
        self.assertEqual(r["summary"]["legacyAllNineGateCoverage"], 0)
        self.assertNotIn("eventMetrics", r)

    def test_synthetic_requires_explicit_software_check(self):
        s = session([frame()])
        s["source"] = "demo"
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            diagnose(s)
        self.assertTrue(diagnose(s, allow_synthetic=True)["synthetic"])

    def test_empty_capture_reports_null_coverage(self):
        r = diagnose(session())
        self.assertIsNone(r["summary"]["legacyAllNineGateCoverage"])
        self.assertEqual(r["windows"], [])
        self.assertEqual(r["highElbowAngleRuns"]["right"], [])

    def test_cli_writes_local_report_and_refuses_input_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "round.json"
            output = Path(directory) / "diagnosis.json"
            source.write_text(json.dumps(session([frame()])))
            original = source.read_bytes()
            self.assertEqual(main([str(source), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["reportType"], "pose_observation_diagnostics")
            self.assertEqual(main([str(source), "--output", str(source)]), 2)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
