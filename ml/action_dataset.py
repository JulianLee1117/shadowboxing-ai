"""Audit and prepare local six-punch research data; never infer missing labels.

No downloads, training, pose correction, or technique grading. Rights fields are
operator declarations, not legal verification. Original evidence is read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys

from .evaluate import PUNCH_LABELS, validate_session

VERSION = "action-dataset-1"
CLASSES = ("jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut")
FAMILIES = ("background", "straight", "hook", "uppercut")
SPLITS = ("train", "validation", "test")
DOMAINS = ("shadowboxing", "bag", "padwork", "sparring", "unknown")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(root: Path, value, name: str) -> Path:
    require(isinstance(value, str) and value.strip() and "://" not in value,
            f"{name} must be a local path")
    path = (root / value).resolve()
    require(path.is_file(), f"Missing {name}: {path}")
    return path


def identity(annotation: dict, stance: str) -> dict | None:
    """Keep physical hand distinct from stance-dependent role and family."""
    if annotation["label"] not in PUNCH_LABELS or annotation["hand"] == "unknown":
        return None
    lead = "left" if stance == "orthodox" else "right"
    role = "lead" if annotation["hand"] == lead else "rear"
    label = annotation["label"]
    if label in ("jab", "cross"):
        require(label == ("jab" if role == "lead" else "cross"),
                f"Annotation {annotation['id']} label conflicts with its physical hand and stance")
        family, six_class = "straight", label
    else:
        family, six_class = label, f"{role}_{label}"
    return {"hand": annotation["hand"], "role": role,
            "family": family, "sixClass": six_class}


def build_targets(session: dict, scope: dict) -> tuple[list[dict], list[dict]]:
    """Half-open labels at real observation times, with unknown taking priority.

    A complete straight-only source cannot establish multiclass background.
    Labels are independent of pose confidence; no landmark is substituted.
    """
    labels = scope.get("labels")
    require(isinstance(labels, list) and labels and len(set(labels)) == len(labels)
            and all(label in PUNCH_LABELS for label in labels), "Invalid labelScope.labels")
    require(type(scope.get("complete")) is bool, "labelScope.complete must be boolean")
    if scope["complete"]:
        require(session.get("annotationsComplete") is True,
                "Complete label scope requires session.annotationsComplete=true")
    background_known = scope["complete"] and set(labels) == set(PUNCH_LABELS)
    events = []
    for annotation in session["annotations"]:
        semantic = identity(annotation, session["stance"])
        if semantic:
            events.append({**semantic, "id": annotation["id"],
                           "startMs": annotation["startMs"], "endMs": annotation["endMs"]})
    for hand in ("left", "right"):
        ordered = sorted((e for e in events if e["hand"] == hand), key=lambda e: e["startMs"])
        require(all(a["endMs"] <= b["startMs"] for a, b in zip(ordered, ordered[1:])),
                f"Overlapping {hand} punch intervals require boundary adjudication")
    output = []
    for frame in session["frames"]:
        row = {"t": frame["t"]}
        for hand in ("left", "right"):
            inside = lambda a: a["startMs"] <= frame["t"] < a["endMs"]
            unknown = any(inside(a) and (
                (a["label"] == "unobservable" and a["hand"] in (hand, "unknown"))
                or (a["label"] in PUNCH_LABELS and a["hand"] == "unknown"))
                for a in session["annotations"])
            event = next((e for e in events if e["hand"] == hand and inside(e)), None)
            known = not unknown and (event is not None or background_known)
            family = event["family"] if event and known else "background" if known else None
            row[hand] = {"familyIndex": FAMILIES.index(family) if family else None,
                         "mask": known, "eventId": event["id"] if event and known else None}
        output.append(row)
    return output, events


def audit_manifest(path: Path) -> tuple[dict, list[dict]]:
    path = path.resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    require(isinstance(manifest, dict) and manifest.get("schemaVersion") == VERSION,
            f"Expected schemaVersion {VERSION}")
    policy = manifest.get("splitPolicy")
    require(policy in ("recording", "participant-day", "participant"), "Explicit splitPolicy is required")
    expected_domains = manifest.get("expectedDomains", ["shadowboxing"])
    require(isinstance(expected_domains, list) and expected_domains
            and all(d in DOMAINS for d in expected_domains), "Invalid expectedDomains")
    entries = manifest.get("entries")
    require(isinstance(entries, list) and entries, "Manifest entries must be nonempty")
    errors, warnings, prepared = [], [], []
    seen_ids, capture_hashes = set(), {}
    groups = {name: defaultdict(set) for name in ("sourceGroup", "videoSha256", "participantDay", "participantId")}
    for index, entry in enumerate(entries):
        name = entry.get("id", f"entry-{index}") if isinstance(entry, dict) else f"entry-{index}"
        try:
            require(isinstance(entry, dict), "Entry must be an object")
            require(isinstance(name, str) and len(name) <= 120
                    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name), "Invalid entry id")
            require(name.casefold() not in seen_ids, "Duplicate entry id")
            seen_ids.add(name.casefold())
            require(entry.get("split") in SPLITS, "Invalid split")
            for field in ("participantId", "sourceGroup", "view"):
                require(isinstance(entry.get(field), str) and entry[field].strip(), f"Explicit {field} required")
            require(isinstance(entry.get("captureDay"), str), "captureDay must be YYYY-MM-DD")
            require(date.fromisoformat(entry["captureDay"]).isoformat() == entry["captureDay"],
                    "captureDay must be YYYY-MM-DD")
            require(entry.get("domain") in DOMAINS, "Invalid domain")
            rights = entry.get("rights", {})
            require(isinstance(rights, dict) and rights.get("status") in ("owned", "licensed", "unreviewed")
                    and isinstance(rights.get("reference"), str) and rights["reference"].strip(),
                    "Rights declaration needs status and reference")
            require(rights["status"] != "unreviewed", "Rights declaration is unreviewed; preparation is blocked")
            source = local_path(path.parent, entry.get("session"), "session")
            session_bytes = source.read_bytes()
            session = json.loads(session_bytes)
            validate_session(session)
            require(session.get("stance") in ("orthodox", "southpaw"), "Explicit stance required")
            require(session["frames"], "Session must have observed pose frames")
            require(all(a["t"] < b["t"] for a, b in zip(session["frames"], session["frames"][1:])),
                    "Source frame times must be strictly increasing")
            scope = entry.get("labelScope", {})
            require(isinstance(scope, dict), "labelScope must be an object")
            targets, events = build_targets(session, scope)
            pose_hash = digest(canonical([{key: f[key] for key in ("t", "width", "height", "landmarks")}
                                          for f in session["frames"]]))
            require(pose_hash not in capture_hashes,
                    f"Duplicate pose capture also appears as {capture_hashes.get(pose_hash)}")
            capture_hashes[pose_hash] = name
            video_path = local_path(path.parent, entry["video"], "video") if entry.get("video") is not None else None
            video_hash = file_digest(video_path) if video_path else None
            if entry.get("videoSha256") is not None:
                require(video_hash is not None and entry["videoSha256"] == video_hash,
                        "Declared videoSha256 does not match supplied local video")
            split = entry["split"]
            participant_day = f"{entry['participantId']}@{entry['captureDay']}"
            for field, value in (("sourceGroup", entry["sourceGroup"]), ("videoSha256", video_hash),
                                 ("participantDay", participant_day), ("participantId", entry["participantId"])):
                if value is not None:
                    groups[field][value].add(split)
            if not scope["complete"] or set(scope["labels"]) != set(PUNCH_LABELS):
                warnings.append({"entryId": name, "code": "unknown-multiclass-background",
                                 "message": "Unlabeled intervals remain masked; label scope does not establish all-family background."})
            if any(a["hand"] == "unknown" or a["label"] == "unobservable" for a in session["annotations"]):
                warnings.append({"entryId": name, "code": "unknown-label-intervals-masked",
                                 "message": "Unknown-hand punches mask both arms; hand-specific unobservable intervals mask that arm."})
            if video_path:
                warnings.append({"entryId": name, "code": "video-association-operator-declared",
                                 "message": "The video hash fingerprints supplied bytes; it does not prove this legacy session came from that video."})
            if entry["domain"] == "unknown" or entry["view"] == "unknown":
                warnings.append({"entryId": name, "code": "unknown-domain-or-view"})
            prepared.append({"artifactType": "prepared-action-session", "schemaVersion": VERSION, "id": name,
                             "split": split, "participantId": entry["participantId"], "captureDay": entry["captureDay"],
                             "sourceGroup": entry["sourceGroup"], "domain": entry["domain"], "view": entry["view"],
                             "rightsDeclaration": rights, "labelScope": scope, "stance": session["stance"],
                             "durationMs": session["durationMs"], "poseModel": session["model"],
                             "poseModelManifest": session.get("modelManifest"),
                             "source": {"sessionPath": str(source), "sessionSha256": digest(session_bytes),
                                        "poseCaptureSha256": pose_hash, "annotationSha256": digest(canonical(session["annotations"])),
                                        "videoPath": str(video_path) if video_path else None, "videoSha256": video_hash,
                                        "videoAssociation": "operator_declared_not_cryptographically_established" if video_path else "not_supplied"},
                             "frames": session["frames"], "annotations": session["annotations"],
                             "events": events, "targets": targets})
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append({"entryId": name, "message": str(error)})
    enforced_groups = ["sourceGroup", "videoSha256"] + (
        ["participantDay"] if policy == "participant-day" else ["participantId"] if policy == "participant" else [])
    overlaps = {field: {value: sorted(splits) for value, splits in values.items() if len(splits) > 1}
                for field, values in groups.items()}
    for field in enforced_groups:
        for value, splits in overlaps[field].items():
            errors.append({"code": "split-leakage", "group": field, "value": value, "splits": splits})
    all_classes = Counter(e["sixClass"] for s in prepared for e in s["events"])
    split_coverage = {}
    for split in SPLITS:
        selected = [s for s in prepared if s["split"] == split]
        classes = Counter(e["sixClass"] for s in selected for e in s["events"])
        domains = {domain: Counter(e["sixClass"] for s in selected if s["domain"] == domain for e in s["events"])
                   for domain in expected_domains}
        split_coverage[split] = {"sessions": len(selected), "classEvents": dict(classes),
                                 "missingClasses": [c for c in CLASSES if not classes[c]],
                                 "participants": sorted({s["participantId"] for s in selected}),
                                 "participantDays": sorted({f"{s['participantId']}@{s['captureDay']}" for s in selected}),
                                 "views": sorted({s["view"] for s in selected}),
                                 "domainClassEvents": {k: dict(v) for k, v in domains.items()},
                                 "missingDomainClasses": {d: [c for c in CLASSES if not counts[c]] for d, counts in domains.items()}}
    target_counts = Counter("known-background" if hand["mask"] and hand["familyIndex"] == 0 else
                            "known-punch" if hand["mask"] else "unknown-masked"
                            for s in prepared for row in s["targets"] for hand in (row["left"], row["right"]))
    class_targets = Counter()
    for session in prepared:
        events_by_id = {e["id"]: e for e in session["events"]}
        for row in session["targets"]:
            for hand in (row["left"], row["right"]):
                if hand["eventId"] is not None:
                    class_targets[events_by_id[hand["eventId"]]["sixClass"]] += 1
    report = {"schemaVersion": VERSION, "manifestSha256": digest(raw), "valid": not errors,
              "splitPolicy": policy, "classes": list(CLASSES), "families": list(FAMILIES),
              "entriesRequested": len(entries), "entriesPrepared": len(prepared), "errors": errors, "warnings": warnings,
              "classEvents": dict(all_classes), "missingClasses": [c for c in CLASSES if not all_classes[c]],
              "classKnownArmFrames": dict(class_targets),
              "classesWithoutKnownTargets": [c for c in CLASSES if not class_targets[c]],
              "splits": split_coverage, "crossSplitGroups": overlaps, "armFrameTargets": dict(target_counts),
              "fingerprints": [{"id": s["id"], **s["source"]} for s in prepared],
              "interpretation": ["Coverage counts and software validation do not establish sufficient training data, accuracy, rights, or technique-quality labels.",
                                 "Rights are operator declarations; this tool performs no legal verification.",
                                 "Recording-level separation does not establish unseen-day or unseen-participant generalization.",
                                 "Source groups and participant/day identities are operator metadata; altered or re-encoded duplicates require truthful grouping.",
                                 "Labels retain physical hand and role. Pose coordinates/confidences are preserved, never repaired or substituted.",
                                 "Prepared targets describe observed-time action labels, not boxing form or joint correctness."]}
    return report, prepared


def write_new(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON file for audit, or new directory for prepare")
    args = parser.parse_args(argv)
    try:
        report, sessions = audit_manifest(args.manifest)
        if args.command == "audit":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_new(args.output, report)
        else:
            require(report["valid"], "Dataset audit failed; run audit to inspect errors before preparation")
            args.output.mkdir(parents=True, exist_ok=False)
            write_new(args.output / "audit.json", report)
            for session in sessions:
                write_new(args.output / f"session-{session['id']}.json", session)
            write_new(args.output / "index.json", {"schemaVersion": VERSION, "manifestSha256": report["manifestSha256"],
                                                  "families": list(FAMILIES), "classes": list(CLASSES),
                                                  "sessions": [{"id": s["id"], "split": s["split"], "file": f"session-{s['id']}.json"} for s in sessions]})
        print(f"{'Valid' if report['valid'] else 'Invalid'} dataset: {len(sessions)} prepared source(s); "
              f"missing classes: {', '.join(report['missingClasses']) or 'none'}. No model trained.")
        return 0 if report["valid"] else 2
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"Action dataset failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
