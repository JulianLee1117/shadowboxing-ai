"""Public trainer software checks; synthetic geometry is not accuracy evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ml import recognizer
from ml.action_dataset import build_targets, file_digest
from ml.train_personal import (
    VERSION,
    TARGET_POLICY,
    audit_inputs,
    contiguous_frames,
    main,
    pose_info,
    supervised_sequences,
    train,
    train_network,
)

NUMERICAL = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "scipy"))
ESTIMATOR = {"id": "rtmpose-m", "scoreType": "simcc", "minimumScore": 0.55}
ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value))
    return {"path": path.name, "sha256": file_digest(path)}


def model_info():
    return {
        "id": "rtmpose-m",
        "estimator": ESTIMATOR,
        "modelManifest": {
            "pose": {"sha256": "a" * 64, "inputColorOrder": "RGB"},
            "detector": {"sha256": "b" * 64},
        },
        "detectorPostprocessing": {"effectivePostNmsScoreThreshold": 0.7},
    }


def sample(t, shift=0):
    points = [{"x": 0.5 + shift, "y": 0.3, "score": 0.9} for _ in range(33)]
    for j, x, y in (
        (11, 0.4, 0.3),
        (12, 0.6, 0.3),
        (13, 0.3, 0.4),
        (14, 0.7, 0.4),
        (15, 0.4, 0.4),
        (16, 0.6, 0.4),
        (23, 0.4, 0.65),
        (24, 0.6, 0.65),
    ):
        points[j].update(x=x + shift, y=y)
    return {
        "t": t,
        "width": 640,
        "height": 480,
        "inferenceMs": 1,
        "estimator": ESTIMATOR,
        "landmarks": points,
    }


def session(shift=0):
    return {
        "schemaVersion": "1.0",
        "id": "software-fixture",
        "source": "file",
        "model": "rtmpose-m",
        "modelManifest": model_info(),
        "stance": "orthodox",
        "durationMs": 1000,
        "events": [],
        "annotationsComplete": True,
        "frames": [sample(i * 1000 / 30, shift) for i in range(31)],
        "annotations": [
            {
                "id": label,
                "label": label,
                "hand": "left",
                "startMs": start,
                "endMs": end,
            }
            for label, start, end in (
                ("jab", 100, 250),
                ("hook", 300, 450),
                ("uppercut", 500, 650),
            )
        ],
    }


def fixture(directory: Path, numerical=False):
    if numerical:
        import torch

        torch.manual_seed(123)
        temporal, family = recognizer._networks(torch)
        torch.save(temporal.state_dict(), directory / "base.pt")
        torch.save(family.state_dict(), directory / "family.pt")
    else:
        (directory / "base.pt").write_bytes(b"audit-only base fingerprint")
        (directory / "family.pt").write_bytes(b"audit-only external fingerprint")
    (directory / "license.txt").write_text(
        "Synthetic software-check license fixture; not real model weights"
    )
    base = {
        "protocolVersion": recognizer.PROTOCOL_VERSION,
        "recognizerId": recognizer.RECOGNIZER_ID,
        "featureVersion": recognizer.FEATURE_VERSION,
        "architecture": "causal-arm-tcn-40-48-d1248-v1",
        "decoderVersion": recognizer.DECODER_VERSION,
        "runtimeSourceSha256": file_digest(ROOT / "recognizer.py"),
        "featureSourceSha256": file_digest(ROOT / "recognizer_features.py"),
        "personalTraining": "Synthetic software fixture",
        "trainingProtocolSha256": "c" * 64,
        "pose": {
            "id": "rtmpose-m",
            "estimator": ESTIMATOR,
            "poseSha256": "a" * 64,
            "detectorSha256": "b" * 64,
            "poseInputColorOrder": "RGB",
            "detectorInputColorOrder": None,
            "detectorScoreThreshold": 0.7,
        },
        "checkpoint": {"path": "base.pt", "sha256": file_digest(directory / "base.pt")},
        "externalModel": {
            "path": "family.pt",
            "sha256": file_digest(directory / "family.pt"),
            "source": "local synthetic software fixture",
            "commit": "fixture",
            "license": "test-only",
            "licensePath": "license.txt",
            "licenseSha256": file_digest(directory / "license.txt"),
        },
    }
    base_pin = write(directory / "base.json", base)
    source = session()
    source_pin = write(directory / "session.json", source)
    (directory / "video.webm").write_bytes(
        b"local original-video byte fingerprint fixture"
    )
    entry = {
        "id": "capture",
        "session": "session.json",
        "split": "train",
        "participantId": "fixture-person",
        "captureDay": "2026-09-26",
        "sourceGroup": "whole-original-recording",
        "domain": "shadowboxing",
        "view": "three-quarter",
        "rights": {"status": "owned", "reference": "synthetic software test"},
        "video": "video.webm",
        "videoSha256": file_digest(directory / "video.webm"),
        "labelScope": {
            "labels": ["jab", "cross", "hook", "uppercut"],
            "complete": True,
        },
    }
    dataset = {
        "schemaVersion": "action-dataset-1",
        "splitPolicy": "recording",
        "entries": [entry],
    }
    config = {
        "schemaVersion": VERSION,
        "dataset": write(directory / "dataset.json", dataset),
        "baseRecognizer": base_pin,
        "seed": 41729,
        "steps": 2,
        "entries": [
            {
                "id": "capture",
                "sessionSha256": source_pin["sha256"],
                "cadenceFps": [15, 20, 25, 30],
            }
        ],
    }
    write(directory / "train.json", config)
    return directory / "train.json", config, dataset, source


def repin(directory, config, dataset=None, source=None):
    if source is not None:
        config["entries"][0]["sessionSha256"] = write(
            directory / "session.json", source
        )["sha256"]
    if dataset is not None:
        config["dataset"] = write(directory / "dataset.json", dataset)
    write(directory / "train.json", config)


class PersonalTrainingAuditTests(unittest.TestCase):
    def test_uncalibrated_native_scores_above_one_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, config, dataset, source = fixture(root)
            source["frames"][0]["landmarks"][15]["score"] = 1.02
            repin(root, config, dataset, source)
            audited = audit_inputs(path)
            self.assertEqual(
                audited["sessions"][0]["frames"][0]["landmarks"][15]["score"], 1.02
            )

    def test_audit_requires_no_optional_training_imports_and_changes_no_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, config, _, _ = fixture(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = audit_inputs(path)
            self.assertEqual(result["report"]["entriesPrepared"], 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            self.assertEqual(main(["--manifest", str(path), "--audit-only"]), 0)

    def test_input_hashes_and_duplicate_json_keys_fail_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, config, _, _ = fixture(root)
            (root / "session.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                audit_inputs(path)
            path.write_text(
                '{"schemaVersion":"personal-training-1","schemaVersion":"personal-training-1"}'
            )
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                audit_inputs(path)

    def test_whole_video_variants_cannot_cross_splits_or_use_different_groups(self):
        for same_group in (True, False):
            with self.subTest(
                same_group=same_group
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, config, dataset, source = fixture(root)
                alternate = session(0.01)
                pin = write(root / "alternate.json", alternate)
                second = copy.deepcopy(dataset["entries"][0])
                second.update(id="alternate", session="alternate.json", split="test")
                if not same_group:
                    second["sourceGroup"] = "disguised-recording"
                dataset["entries"].append(second)
                config["entries"].append(
                    {
                        "id": "alternate",
                        "sessionSha256": pin["sha256"],
                        "cadenceFps": [30],
                    }
                )
                repin(root, config, dataset)
                with self.assertRaisesRegex(ValueError, "split-leakage"):
                    audit_inputs(path)

    def test_variants_must_keep_frozen_labels_and_explicit_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, config, dataset, _ = fixture(root)
            alternate = session(0.01)
            alternate["annotations"][0]["endMs"] = 240
            pin = write(root / "alternate.json", alternate)
            second = copy.deepcopy(dataset["entries"][0])
            second.update(id="alternate", session="alternate.json")
            dataset["entries"].append(second)
            config["entries"].append(
                {"id": "alternate", "sessionSha256": pin["sha256"], "cadenceFps": [30]}
            )
            repin(root, config, dataset)
            with self.assertRaisesRegex(ValueError, "frozen label"):
                audit_inputs(path)
            config["entries"].pop()
            repin(root, config)
            with self.assertRaisesRegex(ValueError, "every dataset entry"):
                audit_inputs(path)

    def test_unknown_and_incomplete_scope_never_become_unlabelled_background(self):
        source = session()
        source["annotations"] += [
            {
                "id": "occluded",
                "label": "unobservable",
                "hand": "left",
                "startMs": 100,
                "endMs": 200,
            },
            {
                "id": "unknown-hand",
                "label": "hook",
                "hand": "unknown",
                "startMs": 700,
                "endMs": 800,
            },
        ]
        rows, _ = build_targets(
            source, {"labels": ["jab", "cross", "hook", "uppercut"], "complete": False}
        )
        self.assertFalse(rows[0]["left"]["mask"])
        self.assertFalse(rows[3]["left"]["mask"])
        self.assertFalse(rows[21]["left"]["mask"])
        self.assertFalse(rows[21]["right"]["mask"])
        rows, _ = build_targets(
            source, {"labels": ["jab", "cross", "hook", "uppercut"], "complete": True}
        )
        self.assertTrue(rows[3]["right"]["mask"])
        self.assertFalse(rows[3]["left"]["mask"])
        self.assertEqual(rows[15]["left"]["familyIndex"], 3)
        self.assertEqual(TARGET_POLICY, "half-open-uncertainty-first-v1")

    def test_score_and_pose_provenance_mismatch_and_bad_cadence_are_rejected(self):
        for mutate, message in (
            (
                lambda c, d, s: s["modelManifest"]["estimator"].update(
                    minimumScore=0.65
                ),
                "policy",
            ),
            (lambda c, d, s: s["frames"][0]["landmarks"][15].pop("score"), "scores"),
            (lambda c, d, s: c["entries"][0].update(cadenceFps=[14]), "Cadences"),
            (
                lambda c, d, s: s.update(benchmark={"sourceVideoSha256": "e" * 64}),
                "provenance",
            ),
        ):
            with self.subTest(
                message=message
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, config, dataset, source = fixture(root)
                # Isolate nested native metadata from fixture constants.
                source = copy.deepcopy(source)
                mutate(config, dataset, source)
                repin(root, config, dataset, source)
                with self.assertRaisesRegex(ValueError, message):
                    audit_inputs(path)

    def test_context_does_not_cross_runtime_gap_or_resolution_reset(self):
        frames = [
            sample(0),
            sample(50),
            sample(201),
            sample(230),
            {**sample(260), "width": 1280},
        ]
        self.assertEqual(
            [[f["t"] for f in part] for part in contiguous_frames(frames)],
            [[0, 50], [201, 230], [260]],
        )

    def test_existing_output_is_never_overwritten_even_before_optional_dependencies(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "already"
            output.mkdir()
            (output / "keep").write_text("untouched")
            self.assertEqual(
                main(
                    ["--manifest", str(root / "missing.json"), "--output", str(output)]
                ),
                2,
            )
            self.assertEqual((output / "keep").read_text(), "untouched")


@unittest.skipUnless(
    NUMERICAL, "Optional local Torch/numpy/scipy dependencies are not installed"
)
class PersonalTrainingNumericalTests(unittest.TestCase):
    def test_only_train_groups_contribute_features_and_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _, _ = fixture(root)
            prepared = audit_inputs(path)["sessions"]
            baseline, counts = supervised_sequences(prepared)
            heldout = copy.deepcopy(prepared[0])
            heldout.update(id="heldout", split="test")
            heldout["frames"] = None
            actual, again = supervised_sequences([*prepared, heldout])
            self.assertEqual(len(actual), len(baseline))
            self.assertEqual(counts.tolist(), again.tolist())
            self.assertEqual(baseline[0][0].shape[1], 40)
            incomplete = copy.deepcopy(prepared)
            incomplete[0]["labelScope"]["complete"] = False
            with self.assertRaisesRegex(ValueError, "background"):
                supervised_sequences(incomplete)

    def test_two_step_cli_outputs_runtime_loadable_weights_and_deterministic_causal_predictions(
        self,
    ):
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _, _ = fixture(root, numerical=True)
            before = (root / "session.json").read_bytes()
            output = root / "result"
            result = train(path, output)
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual(
                result["trainingProtocolSha256"], file_digest(output / "protocol.json")
            )
            self.assertEqual(protocol["trainingEntries"], ["capture"])
            self.assertEqual(protocol["classCounts"][1] > 0, True)
            self.assertEqual((root / "session.json").read_bytes(), before)
            self.assertEqual(
                (output / "EXTERNAL-LICENSE.txt").read_bytes(),
                (root / "license.txt").read_bytes(),
            )
            bundle = recognizer.ModelBundle.load(
                output / "manifest.json", pose_info(result)
            )
            sequences, counts = supervised_sequences(audit_inputs(path)["sessions"])
            repeat, _, _ = train_network(sequences, counts, seed=41729, steps=2)
            self.assertTrue(
                all(
                    torch.equal(v, repeat.state_dict()[k])
                    for k, v in bundle.temporal.state_dict().items()
                )
            )
            x = torch.tensor(sequences[0][0])
            full = bundle.temporal(x).detach().numpy()
            prefix = bundle.temporal(x[:, :, :12]).detach().numpy()
            np.testing.assert_allclose(prefix, full[:, :, :12], rtol=1e-5, atol=1e-5)
            state = bundle.create_session()
            for observed in session()["frames"]:
                result = state.update(observed)
                self.assertEqual(result["protocolVersion"], recognizer.PROTOCOL_VERSION)
                self.assertTrue(
                    all(e["detectedAtMs"] == observed["t"] for e in result["events"])
                )
            state.dispose()
            with self.assertRaisesRegex(ValueError, "already exists"):
                train(path, output)


if __name__ == "__main__":
    unittest.main()
