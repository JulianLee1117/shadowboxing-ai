import tempfile
from pathlib import Path
import json
import unittest

from ml.extract import load_manifest, sha256
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


if __name__ == "__main__":
    unittest.main()
