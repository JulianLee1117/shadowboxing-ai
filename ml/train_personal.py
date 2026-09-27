"""Train the opt-in causal40 personal model from explicitly pinned local evidence.

No downloads, inferred labels, production activation, or source-file mutations.
Optional numerical dependencies load only after the complete input audit passes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time

from .action_dataset import (
    audit_manifest,
    build_targets,
    canonical,
    file_digest,
    require,
)
from .recognizer import (
    DECODER_VERSION,
    FEATURE_VERSION,
    MAXIMUM_GAP_MS,
    PROTOCOL_VERSION,
    RECOGNIZER_ID,
    ModelBundle,
    _networks,
    _validate_pose,
    _verified_path,
)

VERSION = "personal-training-1"
TARGET_POLICY = "half-open-uncertainty-first-v1"
ARCHITECTURE = "causal-arm-tcn-40-48-d1248-v1"
CADENCES = (15, 20, 25, 30)
SOURCE = Path(__file__).resolve()
MAX_ENTRIES = 128
MAX_FRAMES = 6000
MAX_TOTAL_FRAMES = 200_000


def source_hashes():
    return {
        name: file_digest(SOURCE.with_name(name))
        for name in (
            "train_personal.py",
            "recognizer.py",
            "recognizer_features.py",
            "action_dataset.py",
            "evaluate.py",
        )
    }


def read_json(path: Path, limit: int = 2_000_000):
    require(
        path.is_file() and path.stat().st_size <= limit,
        f"Missing or oversized JSON: {path}",
    )

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON: {value}")
        ),
    )


def pinned_path(root: Path, item, field: str, limit: int = 2_000_000) -> Path:
    require(
        isinstance(item, dict) and set(item) == {"path", "sha256"},
        f"{field} requires path and sha256",
    )
    value = item["path"]
    require(
        isinstance(value, str) and value.strip() and "://" not in value,
        f"{field} must be local",
    )
    path = (root / value).resolve()
    require(
        path.is_file() and path.stat().st_size <= limit, f"Missing or oversized {field}"
    )
    require(file_digest(path) == item["sha256"], f"{field} SHA-256 mismatch")
    return path


def pose_info(manifest: dict) -> dict:
    """Construct only the compatibility contract, never actual provider claims."""
    pose = manifest["pose"]
    return {
        "id": pose["id"],
        "estimator": pose["estimator"],
        "modelManifest": {
            name: {
                "sha256": pose[name + "Sha256"],
                "inputColorOrder": pose.get(name + "InputColorOrder"),
            }
            for name in ("pose", "detector")
        },
        "detectorPostprocessing": {
            "effectivePostNmsScoreThreshold": pose["detectorScoreThreshold"]
        },
    }


def audit_inputs(path: Path):
    """Dependency-light audit, before any Torch import or output creation."""
    path = path.resolve()
    config = read_json(path)
    require(
        isinstance(config, dict) and config.get("schemaVersion") == VERSION,
        f"Expected {VERSION}",
    )
    require(
        set(config)
        == {"schemaVersion", "dataset", "baseRecognizer", "seed", "steps", "entries"},
        "Training manifest has missing or unsupported fields",
    )
    for name, low, high in (("seed", 0, 2**32 - 1), ("steps", 1, 10_000)):
        require(
            type(config.get(name)) is int and low <= config[name] <= high,
            f"Invalid {name}",
        )
    dataset_path = pinned_path(path.parent, config["dataset"], "dataset")
    base_path = pinned_path(
        path.parent, config["baseRecognizer"], "baseRecognizer", 100_000
    )
    base = read_json(base_path, 100_000)
    require(isinstance(base, dict), "Base recognizer must be an object")
    expected = {
        "protocolVersion": PROTOCOL_VERSION,
        "recognizerId": RECOGNIZER_ID,
        "featureVersion": FEATURE_VERSION,
        "architecture": ARCHITECTURE,
        "decoderVersion": DECODER_VERSION,
        "runtimeSourceSha256": file_digest(SOURCE.with_name("recognizer.py")),
        "featureSourceSha256": file_digest(SOURCE.with_name("recognizer_features.py")),
    }
    require(
        all(base.get(k) == value for k, value in expected.items()),
        "Base recognizer source/protocol mismatch",
    )
    required_pose = base.get("pose")
    require(
        isinstance(required_pose, dict)
        and required_pose.get("id") == "rtmpose-m"
        and required_pose.get("estimator")
        == {"id": "rtmpose-m", "scoreType": "simcc", "minimumScore": 0.55},
        "This feature protocol requires RTMPose-M native scores with the explicit .55 policy",
    )
    _validate_pose(base, pose_info(base))
    _verified_path(base_path.parent, base.get("checkpoint"))
    external = base.get("externalModel")
    external_path = _verified_path(base_path.parent, external)
    require(
        all(
            isinstance(external.get(k), str) and external[k].strip()
            for k in ("source", "commit", "license", "licensePath", "licenseSha256")
        ),
        "External model must preserve source, commit and local license provenance",
    )
    license_path = _verified_path(
        base_path.parent,
        {"path": external["licensePath"], "sha256": external["licenseSha256"]},
    )
    dataset = read_json(dataset_path)
    require(
        isinstance(dataset, dict)
        and isinstance(dataset.get("entries"), list)
        and 1 <= len(dataset["entries"]) <= MAX_ENTRIES,
        "Dataset entry count is out of bounds",
    )
    selections = config.get("entries")
    require(
        isinstance(selections, list) and len(selections) == len(dataset["entries"]),
        "Explicit hashes/cadences must cover every dataset entry, including held-out entries",
    )
    by_id = {}
    for selection in selections:
        require(
            isinstance(selection, dict)
            and set(selection) == {"id", "sessionSha256", "cadenceFps"},
            "Each training entry requires id, sessionSha256 and cadenceFps",
        )
        identifier = selection["id"]
        require(
            isinstance(identifier, str) and identifier not in by_id,
            "Duplicate/invalid training entry id",
        )
        cadences = selection["cadenceFps"]
        require(
            isinstance(cadences, list)
            and cadences
            and all(type(c) is int and c in CADENCES for c in cadences)
            and len(cadences) == len(set(cadences)),
            "Cadences must be unique choices from 15,20,25,30",
        )
        by_id[identifier] = selection
    # Size/hash/JSON checks happen before the shared audit reads source sessions.
    total_frames = 0
    for entry in dataset["entries"]:
        require(
            isinstance(entry, dict) and entry.get("id") in by_id,
            "Training/dataset entry IDs must match exactly",
        )
        source = pinned_path(
            dataset_path.parent,
            {
                "path": entry.get("session"),
                "sha256": by_id[entry["id"]]["sessionSha256"],
            },
            "session",
            100_000_000,
        )
        value = read_json(source, 100_000_000)
        require(
            isinstance(value, dict)
            and isinstance(value.get("frames"), list)
            and 2 <= len(value["frames"]) <= MAX_FRAMES,
            "Session frame count is out of bounds",
        )
        total_frames += len(value["frames"])
        require(total_frames <= MAX_TOTAL_FRAMES, "Dataset exceeds total frame budget")
        require(
            entry.get("video") and entry.get("videoSha256"),
            "Every entry requires original local video and videoSha256",
        )
    report, prepared = audit_manifest(dataset_path)
    require(report["valid"], "Dataset audit failed: " + json.dumps(report["errors"]))
    groups, videos = {}, {}
    for item in prepared:
        selection = by_id[item["id"]]
        require(
            item["source"]["sessionSha256"] == selection["sessionSha256"],
            "Session changed during audit",
        )
        _validate_pose(base, item.get("poseModelManifest") or {})
        original = read_json(Path(item["source"]["sessionPath"]), 100_000_000)
        require(
            isinstance(original.get("benchmark", {}), dict),
            "Malformed session benchmark provenance",
        )
        claimed = original.get("benchmark", {}).get("sourceVideoSha256")
        video_sha = item["source"]["videoSha256"]
        require(
            claimed is None or claimed == video_sha,
            "Session source video provenance contradicts dataset",
        )
        group = item["sourceGroup"]
        identity = (
            video_sha,
            item["participantId"],
            item["captureDay"],
            item["stance"],
            original.get("videoOffsetMs", 0),
            item["source"]["annotationSha256"],
            item["labelScope"],
        )
        require(
            group not in groups or groups[group] == identity,
            "Pose versions of one recording must share original video, clock, participant and frozen label scope",
        )
        require(
            video_sha not in videos or videos[video_sha] == group,
            "The same original video must use one recording sourceGroup",
        )
        groups[group], videos[video_sha] = identity, group
        for frame in item["frames"]:
            require(
                frame.get("estimator") == base["pose"]["estimator"],
                "Every frame must retain the pinned native estimator",
            )
            require(
                all(
                    type(frame.get(k)) in (int, float)
                    and math.isfinite(frame[k])
                    and frame[k] > 0
                    for k in ("width", "height")
                ),
                "Frames require positive finite dimensions",
            )
            require(
                len(frame["landmarks"]) in (0, 33),
                "Native common geometry must contain zero or 33 landmarks",
            )
            for point in frame["landmarks"]:
                require(
                    isinstance(point, dict)
                    and all(
                        type(point.get(k)) in (int, float) and math.isfinite(point[k])
                        for k in ("x", "y", "score")
                    )
                    and 0 <= point["score"],
                    "Native coordinates and scores must be finite; missing observations use empty frames",
                )
        item["cadenceFps"] = selection["cadenceFps"]
    require(
        any(s["split"] == "train" for s in prepared), "No explicit training recordings"
    )
    return {
        "config": config,
        "configPath": path,
        "datasetPath": dataset_path,
        "basePath": base_path,
        "base": base,
        "externalPath": external_path,
        "licensePath": license_path,
        "report": report,
        "sessions": prepared,
    }


def contiguous_frames(frames):
    """Match runtime resets; a gap or resolution change cannot share TCN context."""
    segment = []
    for frame in frames:
        if segment and (
            frame["t"] - segment[-1]["t"] > MAXIMUM_GAP_MS
            or (frame["width"], frame["height"])
            != (segment[-1]["width"], segment[-1]["height"])
        ):
            yield segment
            segment = []
        segment.append(frame)
    if segment:
        yield segment


def supervised_sequences(sessions):
    import numpy as np
    from .recognizer_features import grid, temporal_features

    output, counts = [], np.zeros(4, dtype=np.int64)
    for session in sessions:
        if session["split"] != "train":
            continue
        for fps in session["cadenceFps"]:
            for segment in contiguous_frames(session["frames"]):
                frames = grid(segment, fps)
                require(
                    len(frames) <= MAX_FRAMES,
                    "Resampled sequence exceeds the frame budget; split long recordings explicitly",
                )
                features, valid = temporal_features(frames)
                targets, _ = build_targets(
                    {
                        "frames": frames,
                        "annotations": session["annotations"],
                        "stance": session["stance"],
                        "annotationsComplete": session["labelScope"]["complete"],
                    },
                    session["labelScope"],
                )
                labels = np.array(
                    [
                        [row[hand]["familyIndex"] or 0 for row in targets]
                        for hand in ("left", "right")
                    ],
                    dtype=np.int64,
                )
                mask = (
                    np.array(
                        [
                            [row[hand]["mask"] for row in targets]
                            for hand in ("left", "right")
                        ],
                        dtype=bool,
                    )
                    & valid
                )
                if not mask.any():
                    continue
                require(
                    np.isfinite(features).all(),
                    "Feature generation produced nonfinite values",
                )
                counts += np.bincount(labels[mask], minlength=4)
                output.append((features, labels, mask))
    require(
        output and (counts > 0).all(),
        "Training requires observed supervised background, straight, hook and uppercut targets",
    )
    return output, counts


def train_network(sequences, counts, *, seed: int, steps: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, _ = _networks(torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    weights = torch.tensor(
        np.sqrt(counts.sum() / np.maximum(counts, 1)), dtype=torch.float32
    )
    weights /= weights.mean()
    tensors = [
        (torch.tensor(x), torch.tensor(y), torch.tensor(mask))
        for x, y, mask in sequences
    ]
    losses = []
    for step in range(steps):
        x, y, mask = tensors[step % len(tensors)]
        model.train()
        optimizer.zero_grad()
        noisy = x + torch.randn_like(x) * 0.015
        if step % 2:
            noisy = noisy.clone()
            noisy[:, list(range(0, 14, 2)) + list(range(19, 33, 2)), :] *= -1
        loss = torch.nn.functional.cross_entropy(
            model(noisy), y, weight=weights, reduction="none"
        )[mask].mean()
        require(bool(torch.isfinite(loss)), "Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return (
        model,
        {"first": losses[0], "last": losses[-1]},
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": 1,
        },
    )


def write_json(path, value):
    path.write_bytes(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )


def train(manifest_path: Path, output: Path):
    output = output.resolve()
    require(
        not output.exists(), "Output already exists; every run needs a new directory"
    )
    audited = audit_inputs(manifest_path)
    executable_hashes = source_hashes()
    # Verify the complete local base bundle including tensor shapes. It supplies
    # only the fixed external family model and pose contract, never initial TCN weights.
    ModelBundle.load(audited["basePath"], pose_info(audited["base"]))
    sequences, counts = supervised_sequences(audited["sessions"])
    start = time.monotonic()
    model, losses, packages = train_network(
        sequences,
        counts,
        seed=audited["config"]["seed"],
        steps=audited["config"]["steps"],
    )
    import torch

    # Do not copy an input that changed while fitting into a supposedly pinned bundle.
    final_audit = audit_inputs(manifest_path)
    require(
        canonical(final_audit["report"]) == canonical(audited["report"])
        and canonical(final_audit["config"]) == canonical(audited["config"]),
        "Inputs changed during training",
    )
    require(
        source_hashes() == executable_hashes,
        "Executable source changed during training",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".personal-training-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        torch.save(model.state_dict(), staging / "model.pt")
        shutil.copyfile(audited["externalPath"], staging / "external-family.pth")
        shutil.copyfile(audited["licensePath"], staging / "EXTERNAL-LICENSE.txt")
        protocol = {
            "schemaVersion": VERSION,
            "architecture": ARCHITECTURE,
            "featureVersion": FEATURE_VERSION,
            "targetPolicy": TARGET_POLICY,
            "seed": audited["config"]["seed"],
            "steps": audited["config"]["steps"],
            "optimizer": {
                "name": "AdamW",
                "learningRate": 0.002,
                "weightDecay": 0.01,
                "gradientClipNorm": 1,
                "classWeights": "sqrt(total/observed-class-count), mean normalized",
            },
            "featureNoiseStd": 0.015,
            "horizontalReflection": "odd steps: relative x and x-velocity channels; physical family unchanged",
            "timeGridHz": 30,
            "maximumGapMs": MAXIMUM_GAP_MS,
            "interpolation": False,
            "trainingEntries": [
                s["id"] for s in audited["sessions"] if s["split"] == "train"
            ],
            "trainingRecordingGroups": sorted(
                {s["sourceGroup"] for s in audited["sessions"] if s["split"] == "train"}
            ),
            "entryCadences": {s["id"]: s["cadenceFps"] for s in audited["sessions"]},
            "entrySplits": {s["id"]: s["split"] for s in audited["sessions"]},
            "inputFingerprints": audited["report"]["fingerprints"],
            "classCounts": counts.tolist(),
            "trainingManifestSha256": file_digest(audited["configPath"]),
            "trainingManifestReference": str(audited["configPath"]),
            "datasetManifestSha256": file_digest(audited["datasetPath"]),
            "datasetManifestReference": str(audited["datasetPath"]),
            "baseRecognizerSha256": file_digest(audited["basePath"]),
            "trainerSourceSha256": file_digest(SOURCE),
            "targetSourceSha256": file_digest(SOURCE.with_name("action_dataset.py")),
            "executableSourceHashes": executable_hashes,
            "runtimeSourceSha256": file_digest(SOURCE.with_name("recognizer.py")),
            "featureSourceSha256": file_digest(
                SOURCE.with_name("recognizer_features.py")
            ),
            "checkpointSha256": file_digest(staging / "model.pt"),
            "packages": packages,
            "loss": losses,
            "processingSeconds": time.monotonic() - start,
            "status": "development-only; no accuracy or prospective validation claim; not activated",
        }
        write_json(staging / "protocol.json", protocol)
        write_json(staging / "audit.json", audited["report"])
        shutil.copyfile(audited["configPath"], staging / "training-inputs.json")
        shutil.copyfile(audited["datasetPath"], staging / "dataset-inputs.json")
        base = audited["base"]
        result = {
            key: copy.deepcopy(base[key])
            for key in (
                "protocolVersion",
                "recognizerId",
                "featureVersion",
                "architecture",
                "decoderVersion",
                "pose",
                "runtimeSourceSha256",
                "featureSourceSha256",
            )
        }
        result.update(
            checkpoint={"path": "model.pt", "sha256": protocol["checkpointSha256"]},
            externalModel={
                **base["externalModel"],
                "path": "external-family.pth",
                "licensePath": "EXTERNAL-LICENSE.txt",
            },
            parentBundleSha256=protocol["baseRecognizerSha256"],
            trainingProtocolSha256=file_digest(staging / "protocol.json"),
            personalTraining=f"Local personal development fit: {len(protocol['trainingRecordingGroups'])} original recordings; explicit recording-group splits. No independent accuracy or technique validation.",
        )
        write_json(staging / "manifest.json", result)
        loaded = ModelBundle.load(staging / "manifest.json", pose_info(result))
        require(
            all(
                torch.equal(model.state_dict()[k], loaded.temporal.state_dict()[k])
                for k in model.state_dict()
            ),
            "Runtime checkpoint load parity failed",
        )
        # mkdir is atomic and refuses collisions; manifest is copied last so a
        # partially interrupted publication cannot look like a complete bundle.
        output.mkdir()
        for file in staging.iterdir():
            if file.name != "manifest.json":
                shutil.copyfile(file, output / file.name)
        shutil.copyfile(staging / "manifest.json", output / "manifest.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, help="New output directory; never overwritten"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Check pinned inputs without numerical dependencies or output",
    )
    args = parser.parse_args(argv)
    try:
        if args.audit_only:
            audit = audit_inputs(args.manifest)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "entries": len(audit["sessions"]),
                        "warnings": audit["report"]["warnings"],
                    }
                )
            )
        else:
            require(args.output is not None, "--output is required for training")
            manifest = train(args.manifest, args.output)
            print(
                json.dumps(
                    {
                        "manifest": str(args.output / "manifest.json"),
                        "checkpointSha256": manifest["checkpoint"]["sha256"],
                        "activated": False,
                    }
                )
            )
        return 0
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        ImportError,
    ) as error:
        print(f"Personal training failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
