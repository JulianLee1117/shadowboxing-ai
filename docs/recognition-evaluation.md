# Recognition evaluation

The canonical score remains one-to-one matching with temporal intersection-over-union (tIoU) at least 0.5, exact punch label and physical hand. `eventMetrics` and the existing strict matching algorithm are unchanged. A short detection inside a long punch can still be a false positive and a missed reference under this score.

The separate `peakOccurrenceDiagnostics` field asks whether an event identifies the same punch occurrence near its visible peak. This is a **development diagnostic pending evaluation on fresh future recordings**, not a replacement accuracy claim or evidence of correct technique. Its fixed protocol is `peak-occurrence-development-v1`:

- Match source-time peaks within **±250 ms**, inclusive, with the same physical hand and family. Jab and cross both map to `straight`; hook and uppercut retain their families. Consequently, a jab/cross naming error can match here while remaining wrong under the strict score.
- Use maximum-cardinality bipartite matching. One detection cannot match two punches, and duplicate detections remain unmatched. Edges are visited by smallest absolute peak difference, then reference order; this does not guarantee minimum total timing error among equally large matchings.
- Preserve the selected recall labels, complete-session annotation requirement, full continuous background, and strict protocol's exclusion intervals. Only predictions fully contained in excluded intervals are ignored. No event windows are selected from model outputs.
- Withhold occurrence precision, recall and counts if **any relevant prediction or reference lacks an explicit peak**. List those IDs and the available/required counts. Available matches and their timing distributions are labeled as a subset. An aggregate is withheld if any input session is incomplete. Empty complete sessions produce zero counts and undefined precision/recall, not perfect accuracy.
- Reject nonfinite, negative, conflicting or out-of-interval explicit peaks. Do not substitute interval midpoints, emission times, model confidence maxima or interpolated coordinates for reference peaks.

Predicted peaks describe what the recognizer selected; reference peaks must come from independent video annotation. A ±250 ms match can be ambiguous during rapid combinations. Both metrics, unmatched event IDs, timing distributions and the original footage remain necessary for interpretation. Do not tune this tolerance on the recordings being compared.

When video review supports several plausible terminal frames, keep those ranges and leave scalar peaks unknown. The separate [terminal-range compatibility tool](terminal-range-evaluation.md) reports deterministic associations and ambiguity without changing either score or inventing a peak. Its conservative unknown-mask policy differs from this evaluator; totals must not be compared across them.

## Running the evaluator

```bash
python3 -m ml.evaluate data/session.json \
  --labels jab,cross,hook,uppercut \
  --peak-reference data/video-reference.json \
  --output data/evaluation.json
```

The source session must assert `annotationsComplete: true`, or the caller must deliberately use `--annotations-complete`. For multiple session paths, repeat `--peak-reference` once per input, in the same order. Output cannot overwrite a source session or peak-reference file. The existing default recall scope remains `jab,cross`; specify all four label families to include all six hand/role punches.

In Python:

```python
report = build_report(
    [session],
    labels=("jab", "cross", "hook", "uppercut"),
    peak_references={session["id"]: video_reference},
)
# For one session: evaluate_session(session, peak_reference=video_reference, ...)
```

Without an external reference, explicit peaks already present on `session.annotations` are used. References are joined to existing annotations by `id` or `annotationId`; they **never replace or add scored annotations**. Supported fields are `peakMs`, `peakTMs`, and the explicitly approximate legacy field `approximatePeakMs`. If more than one is present, their values must agree exactly. Supported reference arrays are `annotations`, `actionObservations`, `approximatePeaks`, and `peaks`, for example:

```json
{
  "sessionId": "the-original-session-id",
  "peaks": [{ "annotationId": "punch-01", "peakTMs": 1234 }]
}
```

Any supplied label, physical hand, or interval boundary must match the stored annotation. Supplied session identity must match the session or its explicit `benchmark.sourceSessionId`; source-video hashes must agree where both are available. The report records a SHA-256 of the canonical JSON reference content. This content hash is distinct from a file-byte hash.

Raw `actionObservations` may include ambiguous video notes that were deliberately excluded from scored annotations. Unjoined notes are listed in `unscoredReferenceObservationIds` and are not converted to truth. An unjoined ID in a dedicated peak list is an error. Existing unknown intervals and provisional family decisions remain frozen; predictions must never be used to revise them.

## Timing and error diagnosis

Every occurrence match reports signed source-peak error, absolute peak error, start/end boundary error, tIoU and whether the same pair also passed strict matching. With recorded `detectedAtMs`, it additionally reports:

| Field                       | Definition                                                         |
| --------------------------- | ------------------------------------------------------------------ |
| `sourcePeakToEmissionMs`    | Recorded emission source time minus independent reference peak     |
| `predictedPeakToEmissionMs` | Recorded emission source time minus predicted peak                 |
| `referenceEndToEmissionMs`  | Recorded emission source time minus reference end; may be negative |

Each distribution includes sample count, median, p95 and maximum. Missing emission telemetry contributes no latency sample. Emission before a prediction's own end is rejected when that prediction is eligible for peak diagnostics. No latency is inferred from a video frame's position in an array.

These source-clock delays are **not sensor-to-display latency**. `processedFrameAgeMs` separately measures the browser frame callback through completed inference; `inferenceMs` is another, narrower measurement. Neither establishes camera acquisition or display latency.

After correct identity matches, a second one-to-one time-only association among unmatched predictions/references describes possible wrong-hand and wrong-family confusions. These are possible co-occurrences, not verified causes. They never become true positives. A nearby unmatched event may instead be a duplicate, boundary fragment, simultaneous other-arm motion or provisional-label disagreement; inspect video before assigning a cause.

## Comparing models without losing the evidence

Preserve original videos, source exports, frozen video-only references, source/model fingerprints and the untouched prospective report. Save replays as derived artifacts. Cold replay can differ from the actual live result when pre-round model history is unavailable; report those as separate experiments. A paired live/full-frame version of one video remains one recording, not two independent participants or trials.

If a recording later enters training or guides model selection, mark subsequent results as development. Preserve its earlier prospective result separately. Evaluate the final frozen model on newly recorded mixed actions and nonpunch movement before making an improvement claim; neither development metric is a form grade or population-level accuracy estimate.
