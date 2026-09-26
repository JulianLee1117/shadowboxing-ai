import tempfile
from pathlib import Path
import json
import unittest
from unittest.mock import Mock

from ml.extract import detector_postprocessing, infer_single_person, load_frame_manifest, load_manifest, native_keypoints, sha256
from ml.landmarks import map_body_landmarks


class LandmarkTests(unittest.TestCase):
    def test_body_mapping_preserves_anatomical_sides_and_normalization(self):
        points = [[i * 10, i * 20] for i in range(17)]
        mapped = map_body_landmarks(points, [.8] * 17, 200, 400)
        self.assertEqual(mapped[15]["x"], .45)  # COCO left wrist 9 → MP 15
        self.assertEqual(mapped[16]["x"], .5)   # COCO right wrist 10 → MP 16
        self.assertEqual(mapped[23]["y"], .55)  # COCO left hip 11 → MP 23
        self.assertNotIn("z", mapped[15])

    def test_unsupported_fingers_and_eyes_stay_unavailable(self):
        mapped = map_body_landmarks([[10, 10]] * 133, [.9] * 133, 100, 100)
        self.assertEqual(mapped[19]["visibility"], 0)
        self.assertEqual(mapped[2]["presence"], 0)
        self.assertEqual(mapped[15]["visibility"], .9)

    def test_openpose_or_halpe_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            map_body_landmarks([[0, 0]] * 18, [1] * 18, 100, 100)
        with self.assertRaises(ValueError):
            map_body_landmarks([[0, 0]] * 26, [1] * 26, 100, 100)

    def test_provenance_records_hash_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"test fixture, not a real ONNX model")
            manifest = {"family": "rtmpose-body", **{key: {"path": "model.onnx",
                       "sourceUrl": "https://example.com/fixture", "inputSize": [192, 256]}
                       for key in ("detector", "pose")}}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(load_manifest(path)["pose"]["sha256"], sha256(model))
            manifest["pose"]["expectedSha256"] = "wrong"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_manifest(path)

    def test_zero_or_multiple_people_do_not_run_full_image_pose_fallback(self):
        for boxes in ([], [[0, 0, 20, 20], [30, 0, 50, 20]]):
            model = Mock()
            model.det_model.return_value = boxes
            count, points, scores = infer_single_person(model, "pixels")
            self.assertEqual(count, len(boxes))
            self.assertIsNone(points)
            self.assertIsNone(scores)
            model.pose_model.assert_not_called()

    def test_one_detected_person_preserves_detector_box(self):
        model = Mock()
        boxes = [[0, 0, 20, 20]]
        model.det_model.return_value = boxes
        points, scores = [[10, 12]] * 17, [.8] * 17
        model.pose_model.return_value = [points], [scores]
        self.assertEqual(infer_single_person(model, "pixels"), (1, points, scores))
        model.pose_model.assert_called_once_with("pixels", bboxes=boxes)

    def test_explicit_rgb_pose_contract_does_not_reorder_detector_input(self):
        class Pixels:
            def __getitem__(self, key):
                self.key = key
                return "RGB pixels"
        pixels = Pixels()
        model = Mock()
        boxes = [[0, 0, 20, 20]]
        model.det_model.return_value = boxes
        model.pose_model.return_value = [[[10, 12]] * 17], [[.8] * 17]
        infer_single_person(model, pixels, "RGB")
        model.det_model.assert_called_once_with(pixels)
        model.pose_model.assert_called_once_with("RGB pixels", bboxes=boxes)
        self.assertEqual(pixels.key, (slice(None), slice(None), slice(None, None, -1)))

    def test_native_observations_preserve_raw_score_and_missing_values_separately(self):
        self.assertEqual(native_keypoints([[10, 20], [float("nan"), 1]], [1.2, .9], 100, 200),
                         [{"x": .1, "y": .1, "score": 1.2}, None])

    def test_embedded_nms_telemetry_reports_upstream_effective_filter(self):
        model = Mock()
        model.det_model.score_thr = .7
        model.det_model.nms_thr = .45
        output = Mock()
        output.name, output.shape = "dets", [1, "num_dets", 5]
        model.det_model.session.get_outputs.return_value = [output]
        report = detector_postprocessing(model)
        self.assertEqual(report["configuredScoreThreshold"], .7)
        self.assertEqual(report["effectivePostNmsScoreThreshold"], .3)
        self.assertTrue(report["embeddedNms"])
        output.shape = [1, 8400, 6]
        self.assertEqual(detector_postprocessing(model)["effectivePostNmsScoreThreshold"], .7)

    def test_decoded_image_manifest_preserves_variable_source_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.jpg", "b.jpg"):
                (root / name).write_bytes(b"file existence fixture")
            path = root / "frames.json"
            data = {"frames": [{"file": "a.jpg", "ptsMs": 21}, {"file": "b.jpg", "ptsMs": 58.2}]}
            path.write_text(json.dumps(data))
            self.assertEqual([t for _, t in load_frame_manifest(path)[1]], [21, 58.2])
            for invalid in (21, -1, float("nan"), True):
                data["frames"][1]["ptsMs"] = invalid
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "ptsMs"):
                    load_frame_manifest(path)

    def test_decoded_image_manifest_rejects_missing_or_outside_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "frames.json"
            for name in ("missing.jpg", "../outside.jpg", "https://example.com/image.jpg"):
                path.write_text(json.dumps({"frames": [{"file": name, "ptsMs": 0}]}))
                with self.assertRaisesRegex(ValueError, "[Ff]rame file"):
                    load_frame_manifest(path)


if __name__ == "__main__":
    unittest.main()
