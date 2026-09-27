# Observed recognition events

These modules describe observed movement evidence. They do not validate technique, prove anatomical identity, infer missing punches, or use the requested drill to make a sequence appear. They do not speak or schedule cues. The source `PunchEvent` remains unchanged.

## Straight-action boundaries and recovery

`projected-straight-v6-observed-repeats` separates action counting from return to the original guard position. A supported observed extension still needs a flexed elbow and at least two recovery observations spanning 30 ms. Recovery can reach the original reach allowance or retract at least half the measured excursion. This allows a partial return between fast repetitions; it does not certify a complete or correct guard return. An arm held extended, an unsupported spike, or a missing active arm cannot satisfy this evidence.

An isolated stroke still requires at least .45 torso lengths of observed reach increase and a supported projected elbow angle of at least 145 degrees. An **already accepted full stroke** can establish a short-lived reference for the same hand's next extension. A repeat then needs at least .225 torso lengths of its own travel, the same supported elbow straightening, and a reach at least .45 beyond the preceding full stroke's origin. Its own onset, path and recovery must still qualify. This avoids demanding a full isolated stroke's travel again after a partial return.

The reference expires 650 ms after the preceding full stroke's observed peak. Repeat-only events cannot move its origin or refresh its deadline; only another independently full-qualified accepted event can create a new reference. Rejected candidates and tracking/timing/framing resets clear repeat context. Without a preceding accepted full stroke, a short motion remains uncounted. If the first jab in a double is missed, this rule cannot infer it or rescue the second using intended sequence labels.

`startMs` is the observed onset estimate, `peakMs` is the observed maximum reach, and `endMs` is the first qualifying recovery observation. `detectedAtMs` is the later observation that confirms recovery. These are detector boundaries, not the full duration of a coach-labeled action. Earlier recovery can shorten an event enough to fail strict temporal-overlap matching even when it corresponds to a real punch. Keep those mismatches in reported evaluation results.

`guardReturn` describes spatial return to that repetition's origin **by detection time**. `not-observed` does not mean that the hand failed to return later. This field is neither a defensive-skill grade nor a requirement for recognizing the next repetition. The confirmed flexed observation can rearm counting independently.

A ready static reference can follow coherent inward motion over at least three observations and 60 ms; it cannot follow an outgoing hand. After a small candidate fails to establish a supported extension, only the last 200 ms of valid acquisition observations survive. The rejected candidate, peak evidence and ready reference do not survive. Tracking loss, timing gaps, invalid geometry and framing changes still discard the relevant history.

Straight-looking relaxed arm lowering can still satisfy the projected-motion rules. A confidently misplaced wrist can also suggest a recovery that did not happen physically. Neither event detection nor the diagnostic confidence gate resolves these ambiguities. Do not convert those signals into technique corrections or hide false events by adjusting the reference labels.

## Combinations

`src/lib/combinations.ts` exports `groupCombinations(events, options)` for a complete round and `CombinationRecognizer` for a future causal consumer. The supported sequences are jab–cross (`1-2`), double jab (`1-1`), double jab–cross (`1-1-2`), and jab–cross–jab (`1-2-1`). These are experimental temporal groupings, not coach-validated intent labels.

Events are ordered by source `peakMs`, so one punch can start before the previous punch finishes returning. Each member must have a label, role, and anatomical hand consistent with the selected stance. Guard return is **not** required by default: a spatial return-to-origin measure is not a sequence or technique grade. `requireObservedReturn: true` is available as an explicit additional caller policy.

The grouper selects the longest supported pattern at the earliest available starting event and consumes its members once. It never skips an intervening event to connect a sequence. A four-event `1-2-1-2` becomes one `1-2-1`, with the final cross left ungrouped. This deliberate nonoverlap policy is not every possible interpretation of a boxing combination.

Default timing constraints are 60–650 ms between adjacent peaks and at most 1,300 ms from first to last peak. They are configurable engineering rules, not learned boxing norms. Uncertainty intervals intersecting the full combination span, including gaps, block grouping for either participating hand. Opposite-arm uncertainty alone does not block a same-hand double jab. Intersections include boundaries conservatively. No uncertainty input means **no supplied uncertainty evidence**, not independently verified tracking.

Each `ComboEvent` contains stable source-event IDs, a session-scoped deterministic ID, notation/name, source start/last-peak/end times, `finalizedAtMs`, algorithm/timing/stance provenance, source hands, and timestamp provenance. `confidence` is the minimum constituent heuristic punch score; `confidenceSemantics` explicitly says it is uncalibrated. It is not the probability the combination happened or was performed well.

```ts
const combinations = groupCombinations(session.events, {
  stance: session.stance,
  sessionId: session.id,
  uncertaintyIntervals: report.uncertaintyIntervals,
});

const recognizer = new CombinationRecognizer({
  stance: "orthodox",
  sessionId: "current-round",
});
recognizer.consume(newlyFinalizedPunches, sourceTimeMs, newlyKnownUncertainty);
recognizer.flush(roundEndSourceTimeMs); // asserts all round evidence has arrived
```

Streaming uses an explicit maximum arrival delay, 1,800 ms by default, to accommodate different punch recovery/finalization delays. A complete three-punch pattern waits until the ordering watermark passes its last peak; a shorter pair waits for the possible longer pattern to expire. Therefore a pair can finalize about **3.1 seconds after its first peak, plus the next consumer update**, not instantly. Offline end-of-round flushing can finalize earlier because the caller asserts the round is complete. This distinction is recorded in `finalizationReason`.

The source clock must be nondecreasing. Repeated identical punch IDs do not emit duplicates. Conflicting IDs, evidence outside the declared arrival bound, and late uncertainty intersecting an already finalized affected combination produce errors; replay complete evidence rather than silently changing an earlier voice/event log. Repeated `flush` is idempotent. `reset` clears observations and uncertainty (new intervals can be supplied); use a new recognizer/session ID for a different round. Missing `detectedAtMs` is explicitly recorded as consumer time in streaming and an event-end fallback in offline grouping. No sensor exposure or speech latency is synthesized.

## Tracking trust diagnostics

`src/lib/trackingTrust.ts` exports `summarizeTrackingTrust(frames, durationMs)` for uncertainty intervals and `assessTrackingTrust(frames, options)` for a detailed report. The round-analysis service stores these flags separately. They do not change motion detection, source coordinates, labels, or confidence scores.

`trusted` means **none of these checks found an inconsistency**, not correct pose or verified identity. Every report/interval carries `identityVerified: false`. A high-confidence sustained tracking error can remain undetected. The current observability check uses the current MediaPipe-oriented motion thresholds; these are not calibrated for another pose model's native scores.

The checks flag missing/low-confidence or severely foreshortened joints, frame gaps over 200 ms, framing changes, a short isolated wrist excursion that immediately returns near its prior position, and a strong bilateral assignment discontinuity with stable shoulders. The latter is ambiguous evidence; hands are never automatically swapped. An isolated detour can also be genuine fast motion, so it is flagged for inspection rather than called an error or corrected. Fast monotonic extension alone does not trigger that check.

After an uncertain observation, an arm needs at least three consistent samples spanning 120 ms to reenter the diagnostic trusted state. Arms are independent except where torso normalization or evidence for both arms is required. Intervals partition source time per arm; unsampled beginning/end time and large gaps remain uncertain. These sample-supported intervals do not establish continuous visibility between camera exposures. Nonmonotonic timestamps are rejected rather than silently reordered.

The isolated-detour check uses the following observed frame. Its `assessedAtMs` therefore records when the evidence becomes available; retrospective interval flags are not instantaneous live warnings. A future voice system must hold evidence until the required observations arrive and use the causal combination interface, not play offline timestamps as if they were live decisions.

Late in-flight pose results beyond a round's declared deadline are excluded from this report and counted in `outsideRoundFrames`; the original observations and duration remain unchanged. Interval reason codes distinguish missing samples, ordinary initial/reacquisition time, low-confidence joints, and trajectory anomalies. A UI must not present the number of all uncertain intervals as the number of proven tracking errors.

Unit tests cover longest/nonoverlapping patterns, overlapping punch intervals, stance and hand consistency, uncertainty gaps, delayed arrivals, stable IDs, reset/late-evidence behavior, arm independence, suspicious high-confidence motion, legitimate monotonic extension, timing gaps, nonmutation, and complete per-arm interval coverage. Private A/B/C review is a diagnostic exercise on one participant, not a calibrated tracking or combination-accuracy benchmark.
