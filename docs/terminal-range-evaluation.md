# Terminal-range compatibility diagnostic

`ml.terminal_ranges` checks whether physical-hand/family predictions are compatible with reviewed, interval-valued terminal evidence. It is a separate **development diagnostic**, not recognition accuracy, a technique assessment or a model promotion rule. It does not change `ml.evaluate`, strict temporal-IoU matching, scalar-peak evaluation or the recognizer.

A held extension can have several plausible terminal frames. Keep that range and a null scalar peak. Do not manufacture a midpoint, select an endpoint after seeing predictions, or call a compatible association a confirmed action occurrence.

## Frozen matching policy

The policy version is `terminal-range-compatibility-1`. Freeze the source, reference, tool and test hashes before comparing candidates.

1. Separate physical `left` and `right` hands. Accepted families are `straight`, `hook` and `uppercut`; the tool never converts stance, jab/cross labels or screen-side coordinates.
2. Within each hand, sort predictions by `(peakMs, id)` and references by `(terminalRange.startMs, terminalRange.endMs, startMs, id)`.
3. A pair is eligible only when the families match, the predicted observed peak lies inside the reference's original action interval, and its distance to the terminal range is **at most 250 ms**, inclusive. Distance is zero inside the range. This tolerance is fixed, not a CLI option.
4. Find an order-preserving one-to-one assignment maximizing pair count, then minimizing total range distance. Exact decimal-input arithmetic avoids floating-point addition changing ties.
5. For reproducible display, choose the lexicographically smallest sequence of `(predictionId, referenceId)` pairs among those optima. IDs do not make an uncertain association certain.

The report includes every partner and unmatched possibility that belongs to any count/distance-optimal assignment, before the ID tie-break. It computes this union with forward/backward dynamic programming rather than enumerating exponentially many assignments. An item's `ambiguous` flag means its partner or matched status can differ between optimal assignments. Alternative edges need not all coexist in one assignment.

The total-order policy itself is an assumption. Overlapping terminal ranges and equal predicted peak times are separately reported as `orderUncertainty`, even if the chosen ordered solution is unique. An order-constrained result must not be presented as proof of physical action order.

## Unknowns and completeness

`referencesComplete` must be explicitly `true` or `false`. It records a caller assertion about the supplied reference scope; it never unlocks accuracy metrics. Incomplete references leave unmatched predictions unexplained, not established background errors.

`unknownIntervals` must be supplied explicitly, including an empty list when none were declared. Masks use the same source clock and physical `left`, `right` or `both` hands. **Any intersection, including endpoint contact, between an item's closed action interval and an applicable mask excludes that item from pairing.** Both predictions and references are handled this way. Excluded IDs, mask IDs, reasons and coverage counts remain prominent in the output. A straddling prediction does not become either a match or a background error.

This conservative policy deliberately differs from `ml.evaluate`'s fully-contained prediction exclusion policy. **Do not compare totals across the two matchers.** A candidate experiment must separately compare exclusion IDs and resolve changed exclusions; an apparent gain produced by mask exclusions cannot establish improvement. Unresolved exclusion transitions block a clean acceptance claim.

## Explicit input contract

Use one JSON object; there is no browser Session, CSV or legacy-reference autodetection. This small synthetic example declares one source observation and one reviewed range:

```json
{
  "schemaVersion": "terminal-range-diagnostic-input-1",
  "source": {
    "id": "example-source",
    "videoSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "clock": "source-presentation-ms"
  },
  "referencesComplete": false,
  "predictions": [
    {
      "id": "prediction-1",
      "hand": "left",
      "family": "straight",
      "startMs": 1000,
      "peakMs": 1300,
      "endMs": 1600,
      "peakObserved": true
    }
  ],
  "references": [
    {
      "id": "reference-1",
      "hand": "left",
      "family": "straight",
      "startMs": 950,
      "endMs": 1650,
      "peakMs": null,
      "terminalRange": {
        "startMs": 1270,
        "endMs": 1370,
        "firstFrameIndex0Based": 38,
        "lastFrameIndex0BasedInclusive": 41
      }
    }
  ],
  "unknownIntervals": []
}
```

The original source clock is preserved; negative presentation origins are permitted. All times must be finite numbers within ±10¹² ms. Action intervals must have positive duration. Prediction peaks must be present, declared observed and inside their own action interval. Missing/invalid peaks cause an error; delivery/emission time cannot replace them. A reference terminal range must lie within the original action bounds and have ordered, nonnegative source-frame endpoints; a single observed frame has equal start/end times. Endpoint evidence across references must agree on the timestamp of each native frame and preserve increasing source-frame order. Optional scalar reference peaks must be finite and inside the range. They remain unused by this matcher, and missing/null values remain null in the report.

The caller must independently bind frame indices/times to hashed native source evidence and prove that predictions and references use the same clock. `peakObserved: true` is a declaration, not verification of pixels, joints, synchronization or model causality. Keep that preparation provenance alongside the diagnostic. Do not repair clock alignment by searching prediction offsets.

IDs must be unique within each of the three input lists. Duplicate JSON keys, nonfinite values, invalid hands/families/intervals and excessive input are rejected. Limits are 4 MiB of CLI input and 256 items in each list. The bound limits DP memory, tie analysis and report size. Input list order does not affect pairings.

## Use and output

```sh
python3 -m ml.terminal_ranges prepared-input.json --output terminal-diagnostic.json
python3 -m unittest ml.tests.test_terminal_ranges
```

The CLI refuses to overwrite an existing output and records `inputSha256` and `toolSourceSha256`. It reads no media or model files and performs no inference. For an already prepared object, call `diagnose(document)`; programmatic callers must pin their input/tool bytes themselves.

Output keys include:

- `coverage`, `excluded`, `unknownIntervals` and `referencesComplete` for the evidence scope and exclusions.
- `referenceEvidence` for every range, its width and the original scalar/null declaration.
- `hands.left` and `hands.right`, each containing `matches`, `unmatchedPredictionIds`, `unmatchedReferenceIds`, `predictionAlternatives`, `referenceAlternatives` and `orderUncertainty`.
- Each display match contains the observed prediction peak, range, range width, distance and `assignmentAmbiguous` flag.

There are no TP/FP/FN, precision, recall, aggregate accuracy or acceptance fields. Unmatched IDs require source review under a separately frozen protocol. The initial tests cover held plateaus, adjacent doubles, overlaps, wrong hand/family, duplicate predictions, conflicting order, exact tolerance bounds, straddling unknown masks, missing/nonfinite peaks, delivery-time independence and exhaustive small assignment comparisons. No real candidate was scored while developing this policy.
