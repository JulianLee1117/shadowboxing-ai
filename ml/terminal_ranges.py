"""Compatibility with observed terminal ranges, separate from accuracy evaluation.

Consumes one explicit physical-hand/family JSON document. No Session conversion,
stance inference, interpolation, reference midpoint, fitting or model execution.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

INPUT_VERSION = "terminal-range-diagnostic-input-1"
VERSION = "terminal-range-compatibility-1"
TOLERANCE_MS = 250
MAX_ITEMS = 256
MAX_INPUT_BYTES = 4 * 1024 * 1024
HANDS = ("left", "right")
FAMILIES = ("straight", "hook", "uppercut")
ZERO = (0, Fraction(0))


def number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and abs(value) <= 1e12 and math.isfinite(value))


def identifier(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 200


def span(row, name):
    if (not isinstance(row, dict) or not number(row.get("startMs"))
            or not number(row.get("endMs")) or row["startMs"] > row["endMs"]):
        raise ValueError(f"Invalid {name} interval")


def validate(document):
    try:
        json.dumps(document, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Input must contain finite JSON values") from error
    if not isinstance(document, dict) or document.get("schemaVersion") != INPUT_VERSION:
        raise ValueError(f"Expected {INPUT_VERSION}")
    source = document.get("source")
    if (not isinstance(source, dict) or not identifier(source.get("id"))
            or source.get("clock") != "source-presentation-ms"
            or not isinstance(source.get("videoSha256"), str)
            or len(source["videoSha256"]) != 64
            or any(c not in "0123456789abcdef" for c in source["videoSha256"])):
        raise ValueError("Explicit source identity, SHA-256 and presentation clock required")
    if type(document.get("referencesComplete")) is not bool:
        raise ValueError("referencesComplete must be an explicit boolean")
    frame_times = {}
    for key in ("predictions", "references", "unknownIntervals"):
        rows = document.get(key)
        if not isinstance(rows, list) or len(rows) > MAX_ITEMS:
            raise ValueError(f"{key} must be a list with at most {MAX_ITEMS} items")
        ids = set()
        for row in rows:
            span(row, key)
            if not identifier(row.get("id")) or row["id"] in ids:
                raise ValueError(f"Invalid or duplicate {key} ID")
            ids.add(row["id"])
            hands = (*HANDS, "both") if key == "unknownIntervals" else HANDS
            if row.get("hand") not in hands:
                raise ValueError(f"Invalid physical hand in {key}")
            if key == "unknownIntervals":
                continue
            if row["startMs"] == row["endMs"]:
                raise ValueError("Action intervals must have positive duration")
            if row.get("family") not in FAMILIES:
                raise ValueError("Explicit straight/hook/uppercut family required; no jab/cross inference")
            if key == "predictions":
                if (row.get("peakObserved") is not True or not number(row.get("peakMs"))
                        or not row["startMs"] <= row["peakMs"] <= row["endMs"]):
                    raise ValueError("Prediction requires a declared observed peak inside its interval")
                continue
            terminal = row.get("terminalRange")
            span(terminal, "terminalRange")
            if not row["startMs"] <= terminal["startMs"] <= terminal["endMs"] <= row["endMs"]:
                raise ValueError("Terminal range must lie within its reference action")
            first, last = terminal.get("firstFrameIndex0Based"), terminal.get("lastFrameIndex0BasedInclusive")
            if (type(first) is not int or type(last) is not int or first < 0 or last < first
                    or (first == last) != (terminal["startMs"] == terminal["endMs"])):
                raise ValueError("Terminal endpoints require consistent observed frame indices")
            for index, t in ((first, terminal["startMs"]), (last, terminal["endMs"])):
                if index in frame_times and frame_times[index] != t:
                    raise ValueError("One source frame cannot have conflicting terminal timestamps")
                frame_times[index] = t
            scalar = row.get("peakMs")
            if scalar is not None and (not number(scalar) or not terminal["startMs"] <= scalar <= terminal["endMs"]):
                raise ValueError("An explicit reference scalar must be inside its terminal range")
    ordered_times = [frame_times[i] for i in sorted(frame_times)]
    if any(a >= b for a, b in zip(ordered_times, ordered_times[1:])):
        raise ValueError("Source frame order must agree with increasing terminal timestamps")
    return document


def distance(prediction, reference):
    """Exact decimal-input arithmetic avoids floating addition breaking ties."""
    t = Fraction(str(prediction["peakMs"]))
    r = reference["terminalRange"]
    return max(Fraction(str(r["startMs"])) - t, t - Fraction(str(r["endMs"])), Fraction(0))


def better(a, b):
    return a if a[0] > b[0] or (a[0] == b[0] and a[1] <= b[1]) else b


def add(a, b, edge=None):
    return a[0] + b[0] + (edge is not None), a[1] + b[1] + (edge if edge is not None else 0)


def pair_hand(predictions, references):
    # These total orders are declared policy, not inferred physical causality.
    predictions = sorted(predictions, key=lambda p: (p["peakMs"], p["id"]))
    references = sorted(references, key=lambda r: (r["terminalRange"]["startMs"],
                        r["terminalRange"]["endMs"], r["startMs"], r["id"]))
    n, m = len(predictions), len(references)
    edges = {}
    for i, p in enumerate(predictions):
        for j, r in enumerate(references):
            if p["family"] == r["family"] and r["startMs"] <= p["peakMs"] <= r["endMs"]:
                d = distance(p, r)
                if d <= TOLERANCE_MS:
                    edges[i, j] = d
    prefix = [[ZERO for _ in range(m + 1)] for _ in range(n + 1)]
    suffix = [[ZERO for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            q = better(prefix[i][j + 1], prefix[i + 1][j])
            if (i, j) in edges:
                q = better(q, add(prefix[i][j], ZERO, edges[i, j]))
            prefix[i + 1][j + 1] = q
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            q = better(suffix[i + 1][j], suffix[i][j + 1])
            if (i, j) in edges:
                q = better(q, add(suffix[i + 1][j + 1], ZERO, edges[i, j]))
            suffix[i][j] = q
    optimum = suffix[0][0]
    # A stable display pairing: lexicographically smallest sequence of ID pairs
    # among all count/distance-optimal order-preserving assignments.
    chosen, i, j = [], 0, 0
    while suffix[i][j][0]:
        options = [(predictions[a]["id"], references[b]["id"], a, b)
                   for (a, b), d in edges.items() if a >= i and b >= j
                   and add(suffix[a + 1][b + 1], ZERO, d) == suffix[i][j]]
        _, _, a, b = min(options)
        chosen.append((a, b))
        i, j = a + 1, b + 1
    # Union of optimal edges and skip states, without enumerating matchings.
    optimal_edges = [(i, j) for (i, j), d in edges.items()
                     if add(prefix[i][j], suffix[i + 1][j + 1], d) == optimum]
    p_options = []
    for i, p in enumerate(predictions):
        partners = sorted(references[j]["id"] for a, j in optimal_edges if a == i)
        unmatched = any(add(prefix[i][j], suffix[i + 1][j]) == optimum for j in range(m + 1))
        p_options.append({"predictionId": p["id"], "referenceIds": partners,
                          "canBeUnmatched": unmatched, "ambiguous": len(partners) + unmatched > 1})
    r_options = []
    for j, r in enumerate(references):
        partners = sorted(predictions[i]["id"] for i, b in optimal_edges if b == j)
        unmatched = any(add(prefix[i][j], suffix[i][j + 1]) == optimum for i in range(n + 1))
        r_options.append({"referenceId": r["id"], "predictionIds": partners,
                          "canBeUnmatched": unmatched, "ambiguous": len(partners) + unmatched > 1})
    chosen_set = set(chosen)
    return {
        "matches": [{"predictionId": predictions[i]["id"], "referenceId": references[j]["id"],
                     "hand": predictions[i]["hand"], "family": predictions[i]["family"],
                     "predictedObservedPeakMs": predictions[i]["peakMs"],
                     "distanceToTerminalRangeMs": float(edges[i, j]),
                     "terminalRange": deepcopy(references[j]["terminalRange"]),
                     "terminalWidthMs": references[j]["terminalRange"]["endMs"] - references[j]["terminalRange"]["startMs"],
                     "assignmentAmbiguous": p_options[i]["ambiguous"] or r_options[j]["ambiguous"]}
                    for i, j in chosen],
        "unmatchedPredictionIds": [p["id"] for i, p in enumerate(predictions) if not any(a == i for a, _ in chosen_set)],
        "unmatchedReferenceIds": [r["id"] for j, r in enumerate(references) if not any(b == j for _, b in chosen_set)],
        "predictionAlternatives": p_options, "referenceAlternatives": r_options,
        "orderUncertainty": {
            "equalPeakPredictionPairs": [[p["id"], q["id"]] for i, p in enumerate(predictions)
                                         for q in predictions[i + 1:] if p["peakMs"] == q["peakMs"]],
            "overlappingTerminalRangePairs": [[r["id"], q["id"]] for i, r in enumerate(references)
                                              for q in references[i + 1:]
                                              if q["terminalRange"]["startMs"] <= r["terminalRange"]["endMs"]],
        },
    }


def diagnose(document):
    validate(document)
    eligible, excluded = {}, {}
    for key in ("predictions", "references"):
        eligible[key], excluded[key] = [], []
        for row in document[key]:
            masks = sorted(u["id"] for u in document["unknownIntervals"]
                           if u["hand"] in (row["hand"], "both")
                           and max(u["startMs"], row["startMs"]) <= min(u["endMs"], row["endMs"]))
            if masks:
                excluded[key].append({"id": row["id"], "unknownIntervalIds": masks,
                                      "reason": "action_span_intersects_unknown_closed_interval"})
            else:
                eligible[key].append(row)
        excluded[key].sort(key=lambda r: r["id"])
    results = {hand: pair_hand([p for p in eligible["predictions"] if p["hand"] == hand],
                              [r for r in eligible["references"] if r["hand"] == hand]) for hand in HANDS}
    return {
        "diagnosticVersion": VERSION, "source": deepcopy(document["source"]),
        "protocol": {"toleranceMs": TOLERANCE_MS, "intervalEndpoints": "inclusive",
                     "matching": "per_hand_order_preserving_max_count_min_total_range_distance_then_lexicographic_ID_pair_sequence",
                     "predictionOrder": ["peakMs", "id"],
                     "referenceOrder": ["terminalRange.startMs", "terminalRange.endMs", "startMs", "id"],
                     "unknownPolicy": "exclude_any_closed_action_span_intersection_for_same_hand_or_both",
                     "ambiguity": "all_partners_and_unmatched_possibility_over_count_distance_optima_before_ID_tiebreak",
                     "maximumItemsPerInputList": MAX_ITEMS},
        "referencesComplete": document["referencesComplete"],
        "coverage": {"inputPredictions": len(document["predictions"]), "inputReferences": len(document["references"]),
                     "eligiblePredictions": len(eligible["predictions"]), "eligibleReferences": len(eligible["references"]),
                     "excludedPredictions": len(excluded["predictions"]), "excludedReferences": len(excluded["references"]),
                     "referenceScalarPeaksProvided": sum(r.get("peakMs") is not None for r in document["references"])},
        "excluded": excluded, "unknownIntervals": deepcopy(document["unknownIntervals"]),
        "referenceEvidence": [{"id": r["id"], "terminalRange": deepcopy(r["terminalRange"]),
                               "terminalWidthMs": r["terminalRange"]["endMs"] - r["terminalRange"]["startMs"],
                               "scalarPeakMs": r.get("peakMs"), "scalarUsedForMatching": False}
                              for r in sorted(document["references"], key=lambda r: r["id"])],
        "hands": results,
        "limitations": [
            "Development evidence compatibility only; no aggregate accuracy, true/false-positive classification or promotion decision.",
            "Observed peaks, terminal endpoints and common source clock are caller declarations; this tool does not verify source pixels or synchronization.",
            "Incomplete references leave unmatched predictions unexplained, not established background errors.",
            "Unknown scalar reference peaks stay unknown; ranges are not replaced with midpoints or favorable endpoints.",
            "ID tie-breaking is reproducibility, not occurrence certainty. Alternatives are conditional on the declared total orders; overlapping ranges/equal peaks also leave physical order uncertain.",
            "Mask policy differs from evaluate.py fully-contained exclusions: totals cannot be compared across matchers; unexplained exclusion transitions block a clean candidate acceptance claim.",
        ],
    }


def read_document(path):
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Input byte limit exceeded")
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate JSON key")
            result[k] = v
        return result
    def decimal(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)
    document = json.loads(raw, parse_constant=invalid, parse_float=decimal, object_pairs_hook=pairs)
    return document, hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Output exists; choose a new artifact path")
        document, fingerprint = read_document(args.input)
        result = diagnose(document)
        result["inputSha256"] = fingerprint
        result["toolSourceSha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    except (ValueError, OSError) as error:
        print(f"Terminal range diagnostic failed: {error}", file=sys.stderr)
        return 1
    print("Wrote terminal-range compatibility diagnostic; no accuracy metric computed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
