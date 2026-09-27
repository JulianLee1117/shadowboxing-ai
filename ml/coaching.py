"""Prepare local, prediction-blind coaching review packets and compare human labels.

This collects criterion evidence. It never judges technique or issues a cue.
No downloads, uploads, pose-based grades, or external dependencies are involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from ml.evaluate import PUNCH_LABELS, finite, validate_session

PACKET_VERSION = "coaching-packet-1"
REVIEW_VERSION = "coaching-review-1"
RUBRIC_VERSION = "isolated-high-guard-draft-1"
JUDGMENTS = ("pass", "fail", "ambiguous", "unobservable", "not_applicable", "not_reviewed")
CRITERIA = {
    "guard_recovery": {
        "name": "Return to the intended high guard",
        "question": "After this isolated straight, did the punching hand visibly return to the intended high-guard region before the next purposeful action?",
        "requiredEvidence": "Identify the physical hand and head reference in the original video. Observe the post-punch outcome; the detector's event end or guardReturn field is not evidence of guard recovery.",
        "exceptions": "A subsequent purposeful action, occlusion, or clip ending before the outcome can be judged is unobservable, not a failed return.",
    },
    "non_punching_hand_guard": {
        "name": "Non-punching hand during the straight",
        "question": "Did the identifiable non-punching hand remain in the intended high-guard region throughout this isolated straight?",
        "requiredEvidence": "The non-striking hand and head reference must remain identifiable across the stroke; a high pose score does not establish that the actual hand is visible.",
        "exceptions": "A deliberate parry, block, feint, other punch or alternative guard makes this rubric inapplicable. Hidden or uncertain evidence cannot establish pass or fail.",
    },
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def packet_id(packet):
    return hashlib.sha256(canonical({k: v for k, v in packet.items() if k != "packetId"})).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build_packet(session, *, session_sha256, video_sha256, video_name, video_bytes):
    validate_session(session)
    if session.get("stance") not in ("orthodox", "southpaw"):
        raise ValueError("An explicit anatomical stance is required")
    offset = session.get("videoOffsetMs", 0)
    if not finite(offset) or offset < 0:
        raise ValueError("videoOffsetMs must be finite and nonnegative")
    actions = [a for a in session["annotations"] if a["label"] in PUNCH_LABELS and a["hand"] in ("left", "right")]
    if not actions:
        raise ValueError("Human action annotations are required; predictions are never review targets")
    items = []
    for action in sorted(actions, key=lambda a: (a["startMs"], a["id"])):
        role = "lead" if action["hand"] == ("left" if session["stance"] == "orthodox" else "right") else "rear"
        if (action["label"] == "jab" and role != "lead") or (action["label"] == "cross" and role != "rear"):
            raise ValueError("Straight action label conflicts with physical hand and declared stance")
        items.append({
            "id": action["id"], "action": action["label"], "hand": action["hand"], "role": role,
            "startMs": action["startMs"], "endMs": action["endMs"],
            "windowStartMs": max(0, action["startMs"] - 800),
            "windowEndMs": min(session["durationMs"], action["endMs"] + 800),
            "criteria": list(CRITERIA) if action["label"] in ("jab", "cross") else [],
        })
    packet = {
        "format": PACKET_VERSION, "createdAt": datetime.now(timezone.utc).isoformat(),
        "rubricVersion": RUBRIC_VERSION, "rubricStatus": "draft_requires_coach_review_not_validated",
        "criteria": deepcopy(CRITERIA),
        "source": {"sessionId": session["id"], "sessionSha256": session_sha256,
                   "videoSha256": video_sha256, "videoName": video_name, "videoBytes": video_bytes,
                   "durationMs": session["durationMs"], "videoOffsetMs": offset, "stance": session["stance"],
                   "association": "operator_supplied_video; capture_export_has_no_verified_embedded_video_hash"},
        "timing": "Evidence uses session milliseconds; video time equals session time plus videoOffsetMs. Windows are navigation aids, not fixed recovery deadlines.",
        "annotationScope": "Supplied human action annotations only; neither action-label correctness nor complete coverage is certified.",
        "excludedActions": [{"id": a["id"], "reason": "physical_hand_unknown"}
                            for a in session["annotations"] if a["label"] in PUNCH_LABELS and a["hand"] == "unknown"],
        "items": items,
        "limitations": ["No detector predictions, pose coordinates or model scores are included.",
                        "The same video must be independently reviewed; no correctness labels are prefilled.",
                        "Current draft criteria apply only to eligible isolated high-guard straights; hooks and uppercuts have no form rubric here.",
                        "Role mapping assumes the declared stance throughout the recording; stance switches are unsupported.",
                        "Reviewer expertise is self-reported. Agreement is not accuracy, coaching benefit, or permission to issue live cues."],
    }
    packet["packetId"] = packet_id(packet)
    return packet


def validate_packet(packet):
    if not isinstance(packet, dict) or packet.get("format") != PACKET_VERSION or packet.get("rubricVersion") != RUBRIC_VERSION:
        raise ValueError("Unsupported packet or rubric version")
    if packet.get("packetId") != packet_id(packet) or packet.get("criteria") != CRITERIA:
        raise ValueError("Packet fingerprint or criterion definitions changed")
    source = packet.get("source")
    if not isinstance(source, dict) or not nonempty(source.get("sessionId")) or not nonempty(source.get("videoName")):
        raise ValueError("Invalid packet source")
    if source.get("stance") not in ("orthodox", "southpaw") or not finite(source.get("videoOffsetMs")) or source["videoOffsetMs"] < 0:
        raise ValueError("Invalid source stance or video offset")
    if type(source.get("videoBytes")) is not int or source["videoBytes"] <= 0:
        raise ValueError("Source video must have positive byte length")
    for name in ("sessionSha256", "videoSha256"):
        value = source.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid source {name}")
    duration = source.get("durationMs")
    if not finite(duration) or duration <= 0:
        raise ValueError("Invalid packet duration")
    seen = set()
    if not isinstance(packet.get("items"), list) or not packet["items"]:
        raise ValueError("Packet must contain annotated actions")
    for item in packet["items"]:
        if not isinstance(item, dict) or not nonempty(item.get("id")) or item["id"] in seen:
            raise ValueError("Packet action IDs must be unique")
        seen.add(item["id"])
        times = [item.get(k) for k in ("windowStartMs", "startMs", "endMs", "windowEndMs")]
        if not all(finite(v) for v in times) or not 0 <= times[0] <= times[1] < times[2] <= times[3] <= duration:
            raise ValueError("Invalid action/context timing")
        expected = list(CRITERIA) if item.get("action") in ("jab", "cross") else []
        if item.get("action") not in PUNCH_LABELS or item.get("hand") not in ("left", "right") or item.get("criteria") != expected:
            raise ValueError("Invalid action or unsupported criteria")
        role = "lead" if item["hand"] == ("left" if source["stance"] == "orthodox" else "right") else "rear"
        if item.get("role") != role or (item["action"] == "jab" and role != "lead") or (item["action"] == "cross" and role != "rear"):
            raise ValueError("Action role conflicts with physical hand and stance")
    return packet


def review_template(packet):
    validate_packet(packet)
    return {"format": REVIEW_VERSION, "packetId": packet["packetId"], "rubricVersion": RUBRIC_VERSION,
            "reviewer": {"id": "", "expertise": "not_reported"},
            "ratings": [{"actionId": item["id"], "criterion": criterion,
                         "judgment": "not_reviewed", "context": "unknown", "view": "unknown",
                         "outcomeObserved": False, "feedbackWarranted": "not_reviewed",
                         "evidence": None, "rationale": ""}
                        for item in packet["items"] for criterion in item["criteria"]]}


def validate_review(packet, review):
    validate_packet(packet)
    if not isinstance(review, dict) or review.get("format") != REVIEW_VERSION or review.get("packetId") != packet["packetId"] or review.get("rubricVersion") != RUBRIC_VERSION:
        raise ValueError("Review must match the exact packet and rubric")
    reviewer = review.get("reviewer", {})
    if not isinstance(reviewer, dict) or not nonempty(reviewer.get("id")) or reviewer.get("expertise") not in ("coach_self_reported", "participant", "researcher", "not_reported"):
        raise ValueError("A reviewer ID and explicit expertise declaration are required")
    items = {item["id"]: item for item in packet["items"]}
    expected = {(item["id"], c) for item in items.values() for c in item["criteria"]}
    seen = set()
    if not isinstance(review.get("ratings"), list):
        raise ValueError("Ratings must be an array")
    for rating in review["ratings"]:
        if not isinstance(rating, dict):
            raise ValueError("Each rating must be an object")
        if not nonempty(rating.get("actionId")) or not nonempty(rating.get("criterion")):
            raise ValueError("Action and criterion IDs must be nonempty strings")
        key = (rating.get("actionId"), rating.get("criterion"))
        if key not in expected or key in seen:
            raise ValueError("Duplicate, unknown or unsupported action/criterion rating")
        seen.add(key)
        judgment = rating.get("judgment")
        if judgment not in JUDGMENTS or rating.get("context") not in ("eligible", "not_applicable", "unknown") or rating.get("view") not in ("adequate", "inadequate", "unknown"):
            raise ValueError("Invalid judgment, context or view")
        if not isinstance(rating.get("outcomeObserved"), bool) or rating.get("feedbackWarranted") not in ("yes", "no", "uncertain", "not_reviewed"):
            raise ValueError("Outcome and feedback fields must be explicit")
        if judgment in ("pass", "fail") and (rating["context"] != "eligible" or rating["view"] != "adequate" or not rating["outcomeObserved"]):
            raise ValueError("Pass/fail requires eligible context, adequate view and an observed outcome")
        if judgment == "not_applicable" and rating["context"] != "not_applicable":
            raise ValueError("An inapplicable rating requires explicit inapplicable context")
        if judgment in ("ambiguous", "unobservable") and rating["context"] == "not_applicable":
            raise ValueError("Use not_applicable for a known inapplicable context")
        if judgment == "unobservable" and rating["outcomeObserved"]:
            raise ValueError("An unobservable outcome cannot be declared observed")
        if rating["feedbackWarranted"] == "yes" and judgment != "fail":
            raise ValueError("A correction requires an independently supported fail judgment")
        if judgment == "not_reviewed":
            if rating.get("evidence") is not None or rating["feedbackWarranted"] != "not_reviewed" or rating["outcomeObserved"]:
                raise ValueError("Unreviewed entries cannot carry findings")
            continue
        if not nonempty(rating.get("rationale")):
            raise ValueError("Every reviewed judgment needs a rationale")
        evidence = rating.get("evidence")
        if not isinstance(evidence, dict) or not all(finite(evidence.get(k)) for k in ("startMs", "endMs")) or not 0 <= evidence["startMs"] < evidence["endMs"] <= packet["source"]["durationMs"]:
            raise ValueError("Evidence must identify an interval inside the source recording")
    if seen != expected:
        raise ValueError("Keep every rating row; use not_reviewed rather than dropping missing labels")
    counts = Counter(r["judgment"] for r in review["ratings"])
    return {"reviewerId": reviewer["id"], "expertise": reviewer["expertise"],
            "counts": {state: counts[state] for state in JUDGMENTS}, "totalRatings": len(expected),
            "reviewedRatings": len(expected) - counts["not_reviewed"],
            "limitation": "Schema validity does not verify expertise or establish the correctness of any judgment."}


def compare_reviews(packet, first, second):
    first_summary = validate_review(packet, first)
    second_summary = validate_review(packet, second)
    if first_summary["reviewerId"] == second_summary["reviewerId"]:
        raise ValueError("Agreement needs distinct reviewer IDs; independence is still operator-declared")
    second_ratings = {(r["actionId"], r["criterion"]): r for r in second["ratings"]}
    report = {}
    fields = ("judgment", "context", "view", "outcomeObserved", "feedbackWarranted")
    for criterion in CRITERIA:
        pairs = [(r, second_ratings[(r["actionId"], criterion)]) for r in first["ratings"] if r["criterion"] == criterion]
        reviewed = [(a, b) for a, b in pairs if a["judgment"] != "not_reviewed" and b["judgment"] != "not_reviewed"]
        matrix = {a: {b: 0 for b in JUDGMENTS[:-1]} for a in JUDGMENTS[:-1]}
        for a, b in reviewed:
            matrix[a["judgment"]][b["judgment"]] += 1
        decisive = [(a, b) for a, b in reviewed if a["judgment"] in ("pass", "fail") and b["judgment"] in ("pass", "fail")]
        n = len(decisive)
        agree = sum(a["judgment"] == b["judgment"] for a, b in decisive) / n if n else None
        expected = sum(sum(a["judgment"] == c for a, _ in decisive) * sum(b["judgment"] == c for _, b in decisive) for c in ("pass", "fail")) / (n * n) if n else None
        report[criterion] = {"totalPairs": len(pairs), "bothReviewed": len(reviewed), "confusion": matrix,
                             "judgmentCounts": [{state: sum(pair[i]["judgment"] == state for pair in pairs) for state in JUDGMENTS} for i in (0, 1)],
                             "fieldDisagreements": {field: sum(a[field] != b[field] for a, b in reviewed) for field in fields},
                             "decisivePairs": n, "decisiveCoverage": n / len(pairs) if pairs else None,
                             "passFailAgreement": agree,
                             "cohensKappa": (agree - expected) / (1 - expected) if expected is not None and expected < 1 else None,
                             "disagreements": [a["actionId"] for a, b in reviewed if any(a[field] != b[field] for field in fields)]}
    return {"format": "coaching-agreement-1", "packetId": packet["packetId"],
            "reviewers": [first_summary, second_summary], "criteria": report,
            "limitations": ["Agreement is not accuracy or coaching efficacy; consensus may still be wrong.",
                            "Kappa uses only mutually decisive pass/fail pairs; exclusions and coverage are reported.",
                            "No source video, source annotation or review is modified, adjudicated, or uploaded."]}


def prepare(session_path, video_path, output):
    session_path, video_path, output = map(Path, (session_path, video_path, output))
    if not video_path.is_file():
        raise ValueError("A local source video is required")
    packet = build_packet(json.loads(session_path.read_text()), session_sha256=file_hash(session_path),
                          video_sha256=file_hash(video_path), video_name=video_path.name,
                          video_bytes=video_path.stat().st_size)
    validate_packet(packet)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "packet.json", packet)
    write_new(output / "review-template.json", review_template(packet))
    (output / "README.md").write_text(
        "# Independent local coaching review\n\n"
        "Use the original video, not its tracking overlay or predicted detections. "
        "Verify the video SHA-256 in packet.json. The source video is not copied or uploaded.\n\n"
        "Copy review-template.json for each reviewer; keep independent reviews separate. "
        "Before labeling, agree with a qualified coach on the intended high-guard reference "
        "for this drill; this draft is not a universal guard standard. "
        "Set a pseudonymous reviewer ID and self-reported expertise. Confirm context and view "
        "before pass/fail, and cite session-time evidence plus rationale. Record whether a "
        "correction is warranted separately. Never turn hidden hands or a truncated return into fail.\n\n"
        "An 800 ms context margin is a navigation aid, not a required return time. If the "
        "window cannot establish the outcome, inspect more of the original recording and cite "
        "that interval; evidence can extend anywhere within the source duration. If the outcome "
        "remains hidden or truncated, mark unobservable. Hooks/uppercuts have no criterion ratings in this draft.\n\n"
        "Validate: `python3 -m ml.coaching check packet.json reviewer-a.json`\n\n"
        "Compare: `python3 -m ml.coaching compare packet.json reviewer-a.json reviewer-b.json "
        "--output agreement.json`\n\n"
        "The rubric is a draft for coach review. This packet contains no automated form labels "
        "and does not enable coaching. Keep packet, reviews and footage outside Git.\n")
    return {"packetId": packet["packetId"], "actions": len(packet["items"]),
            "ratingSlots": len(review_template(packet)["ratings"]), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("session", type=Path)
    create.add_argument("--video", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("packet", type=Path)
    check.add_argument("review", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("packet", type=Path)
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.session, args.video, args.output)
        elif args.command == "check":
            result = validate_review(json.loads(args.packet.read_text()), json.loads(args.review.read_text()))
        else:
            result = compare_reviews(json.loads(args.packet.read_text()), json.loads(args.first.read_text()), json.loads(args.second.read_text()))
            write_new(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"Coaching review failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
