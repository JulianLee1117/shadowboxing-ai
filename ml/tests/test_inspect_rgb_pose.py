"""Focused schema/geometry tests, no dataset labels or human recognition claims."""

import unittest
from types import SimpleNamespace

try:
    import numpy as np
    import cv2
except ImportError:
    np = cv2 = None
from ml.inspect_rgb_pose import (
    bounded_box,
    person_observations,
    rgb_crop,
    main,
    validate_resource_budget,
    pin_inputs,
    verify_pinned_inputs,
    validate_video_declaration,
    decode_samples,
    MAXIMUM_INSTANCES_PER_FRAME,
)
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch
from contextlib import closing


@unittest.skipIf(
    np is None or cv2 is None,
    "optional NumPy/OpenCV extraction environment unavailable",
)
class InspectionFeaturesTests(unittest.TestCase):
    def test_crop_is_clamped_expanded_actual_rgb(self):
        image = np.zeros((20, 30, 3), np.uint8)
        image[:, :, 0] = 7
        image[:, :, 1] = 13
        image[:, :, 2] = 91
        rgb, bounds = rgb_crop(image, [-4, -3, 25, 18])
        self.assertEqual(bounds, (0, 0, 30, 20))
        self.assertEqual(rgb.shape, (112, 112, 3))
        np.testing.assert_array_equal(rgb[0, 0], [91, 13, 7])

    def test_nonfinite_and_disjoint_boxes_fail(self):
        for box in ([0, 0, float("nan"), 1], [10, 10, 1, 1], [50, 50, 80, 80]):
            with self.assertRaises(ValueError):
                bounded_box(box, 30, 20)

    def test_all_person_instances_and_native_scores_are_preserved(self):
        boxes = np.array([[0, 0, 5, 8], [10, 0, 15, 8]], np.float32)
        xy = np.arange(68, dtype=np.float32).reshape(2, 17, 2)
        scores = np.full((2, 17), 1.17, np.float32)
        calls = []

        def pose(pixels, bboxes):
            calls.append((pixels.copy(), bboxes.copy()))
            return xy, scores

        model = SimpleNamespace(det_model=lambda pixels: boxes, pose_model=pose)
        pixels = np.array([[[1, 2, 3]]], np.uint8)
        actual_boxes, actual_xy, actual_scores = person_observations(
            model, pixels, "RGB"
        )
        np.testing.assert_array_equal(actual_boxes, boxes)
        np.testing.assert_array_equal(actual_xy, xy)
        np.testing.assert_array_equal(actual_scores, scores)
        self.assertGreater(float(actual_scores[0, 0]), 1)
        np.testing.assert_array_equal(calls[0][0][0, 0], [3, 2, 1])

    def test_zero_person_never_uses_pose_fallback(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("Pose called without detected box")

        model = SimpleNamespace(
            det_model=lambda pixels: np.empty((0, 4)), pose_model=forbidden
        )
        boxes, xy, scores = person_observations(model, np.zeros((2, 2, 3)), "RGB")
        self.assertEqual(xy.shape, (0, 17, 2))
        self.assertEqual(scores.shape, (0, 17))
        self.assertEqual(len(boxes), 0)

    def test_pose_box_count_mismatch_fails(self):
        model = SimpleNamespace(
            det_model=lambda pixels: np.zeros((2, 4)),
            pose_model=lambda *args, **kwargs: (
                np.zeros((1, 17, 2)),
                np.zeros((1, 17)),
            ),
        )
        with self.assertRaises(ValueError):
            person_observations(model, np.zeros((2, 2, 3)), "RGB")

    def test_noninspection_purpose_fails_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video.mp4").write_bytes(b"not-decoded")
            (root / "provenance.json").write_text(
                json.dumps({"purpose": "training", "sourceGroup": "sample"})
            )
            with self.assertRaisesRegex(ValueError, "inspection-only"):
                main(
                    [
                        str(root / "video.mp4"),
                        "--manifest",
                        str(root / "missing.json"),
                        "--provenance",
                        str(root / "provenance.json"),
                        "--output",
                        str(root / "output"),
                    ]
                )

    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video.mp4").write_bytes(b"not-decoded")
            (root / "provenance.json").write_text(
                json.dumps({"purpose": "inspection-only", "sourceGroup": "sample"})
            )
            (root / "output").mkdir()
            with self.assertRaisesRegex(ValueError, "Output exists"):
                main(
                    [
                        str(root / "video.mp4"),
                        "--manifest",
                        str(root / "missing.json"),
                        "--provenance",
                        str(root / "provenance.json"),
                        "--output",
                        str(root / "output"),
                    ]
                )

    def test_nonfinite_pose_cannot_be_written_as_success(self):
        for value in (float("nan"), float("inf")):
            points = np.zeros((1, 17, 2))
            points[0, 0, 0] = value
            model = SimpleNamespace(
                det_model=lambda pixels: np.array([[0, 0, 1, 1]]),
                pose_model=lambda *args, **kwargs: (points, np.ones((1, 17))),
            )
            with self.assertRaisesRegex(ValueError, "finite"):
                person_observations(model, np.zeros((2, 2, 3)), "RGB")
        model = SimpleNamespace(
            det_model=lambda pixels: np.array([[0, 0, 1, 1]]),
            pose_model=lambda *args, **kwargs: (
                np.zeros((1, 17, 2)),
                np.full((1, 17), float("nan")),
            ),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            person_observations(model, np.zeros((2, 2, 3)), "RGB")

    def test_crowd_limit_fails_before_pose_inference(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("Excessive crowd sent to pose")

        model = SimpleNamespace(
            det_model=lambda pixels: np.zeros((MAXIMUM_INSTANCES_PER_FRAME + 1, 4)),
            pose_model=forbidden,
        )
        with self.assertRaisesRegex(ValueError, "Too many people"):
            person_observations(model, np.zeros((2, 2, 3)), "RGB")

    def test_decode_coverage_and_release_are_explicit(self):
        for limit, expected in ((1, "frame_limit"), (30, "interval_boundary_observed")):
            cap = Mock()
            cap.isOpened.return_value = True
            cap.read.return_value = (True, np.zeros((2, 2, 3), np.uint8))
            cap.get.side_effect = [0, 50, 100]
            coverage = {}
            with patch("cv2.VideoCapture", return_value=cap):
                list(
                    decode_samples(
                        "unused", duration_ms=100, max_frames=limit, coverage=coverage
                    )
                )
            self.assertEqual(coverage["terminationReason"], expected)
            self.assertEqual(coverage["intervalComplete"], limit == 30)
            cap.release.assert_called_once()

    def test_inference_failure_closes_decode_generator(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((2, 2, 3), np.uint8))
        cap.get.return_value = 0
        with patch("cv2.VideoCapture", return_value=cap):
            with self.assertRaisesRegex(RuntimeError, "inference"):
                with closing(decode_samples("unused")) as samples:
                    next(samples)
                    raise RuntimeError("inference failed")
        cap.release.assert_called_once()


class ResourceTests(unittest.TestCase):
    def test_hashes_reject_false_video_declaration_and_changed_inputs(self):
        with self.assertRaisesRegex(ValueError, "videoSha256"):
            validate_video_declaration({"videoSha256": "a" * 64}, "b" * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"original")
            pins = pin_inputs([path])
            verify_pinned_inputs(pins)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_pinned_inputs(pins)

    def test_cumulative_feature_limits_prevent_large_crop_accumulation(self):
        self.assertEqual(
            validate_resource_budget(1, 1),
            112 * 112 * 3 * 4 + 8 + 17 * 3 * 4 + 8 * 4 + 4,
        )
        for frames, instances in ((301, 0), (30, 1001), (300, 1000)):
            with self.assertRaisesRegex(ValueError, "limit exceeded"):
                validate_resource_budget(frames, instances)


if __name__ == "__main__":
    unittest.main()
