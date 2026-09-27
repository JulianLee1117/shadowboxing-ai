import { punchName, punchNotation } from "../lib/punches";
import type { PunchEvent } from "../lib/types";
import "./LiveFeedback.css";

function emissionTime(event: PunchEvent): number {
  return event.detectedAtMs ?? event.endMs;
}

/** Keep the first valid receipt for each ID. Invalid later duplicates cannot
 * replace a confirmed identity or inflate its total/announcement.
 */
export function acceptedDetections(
  events: readonly PunchEvent[],
  elapsedMs: number,
): PunchEvent[] {
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return [];
  const seen = new Set<string>();
  return events.filter((event) => {
    const emittedAt = emissionTime(event);
    if (
      typeof event.id !== "string" ||
      !event.id ||
      seen.has(event.id) ||
      !Number.isFinite(event.startMs) ||
      !Number.isFinite(event.peakMs) ||
      !Number.isFinite(event.endMs) ||
      !Number.isFinite(emittedAt) ||
      event.startMs < 0 ||
      event.peakMs < event.startMs ||
      event.endMs < event.peakMs ||
      emittedAt < event.endMs ||
      emittedAt > elapsedMs + 150
    )
      return false;
    seen.add(event.id);
    return true;
  });
}

/** Source-clock recency remains available to callers; it is not receipt age.
 * The UI's keyed CSS arrival pulse and persistent history do not depend on it.
 */
export function recentDetection(
  events: readonly PunchEvent[],
  elapsedMs: number,
): PunchEvent | null {
  const event = acceptedDetections(events, elapsedMs).at(-1);
  if (!event) return null;
  const emittedAt = emissionTime(event);
  const age = elapsedMs - emittedAt;
  return Number.isFinite(emittedAt) &&
    emittedAt >= 0 &&
    age >= -150 &&
    age <= 900
    ? event
    : null;
}

/** Receipt order is intentional: later confirmation must not insert a row
 * halfway down the log. Source-peak times remain visible beside each identity.
 * The round owns this array and clears it on every start. Never mutate it here.
 */
export function detectionHistory(
  events: readonly PunchEvent[],
  elapsedMs: number,
): { event: PunchEvent; ordinal: number }[] {
  return historyFromAccepted(acceptedDetections(events, elapsedMs));
}

function historyFromAccepted(events: readonly PunchEvent[]) {
  return events
    .map((event, index) => ({ event, ordinal: index + 1 }))
    .slice(-4)
    .reverse();
}

function actionTime(ms: number): string {
  const tenths = Math.floor(ms / 100);
  return `${Math.floor(tenths / 600)}:${((tenths % 600) / 10)
    .toFixed(1)
    .padStart(4, "0")}`;
}

export function LiveFeedback({
  events,
  elapsedMs,
  trackingUnclear,
}: {
  events: readonly PunchEvent[];
  elapsedMs: number;
  trackingUnclear: boolean;
}) {
  const accepted = acceptedDetections(events, elapsedMs);
  const history = historyFromAccepted(accepted);
  const latest = history[0]?.event;
  const batch = latest
    ? accepted.filter((event) => emissionTime(event) === emissionTime(latest))
    : [];
  const batchIds = new Set(batch.map((event) => event.id));
  const hiddenBatchCount = batch.filter(
    (event) => !history.some((row) => row.event.id === event.id),
  ).length;
  const accent = latest ? `hand-${latest.hand}` : "";
  return (
    <div className={`live-feedback ${accent}`} aria-label="Live punch feedback">
      <section className="live-history" aria-label="Punch history">
        <div className="live-history-heading">
          <span>Detected</span>
          <span className="live-history-order">Newest first</span>
          {latest && (
            <span className="detection-tick" key={latest.id} aria-hidden="true">
              +{batch.length}
            </span>
          )}
        </div>
        {history.length ? (
          <ol className="punch-history" aria-label="Recent confirmed punches">
            {history.map(({ event, ordinal }, index) => (
              <li
                className={`punch-history-item hand-${event.hand} ${index === 0 ? "latest" : ""}${batchIds.has(event.id) ? " newest-batch" : ""}`}
                key={event.id}
                data-event-id={event.id}
                data-sequence={ordinal}
              >
                <span className="punch-history-number" aria-hidden="true">
                  {punchNotation(event)}
                </span>
                <span className="punch-history-identity">
                  <strong>{punchName(event)}</strong>
                  <span>{event.hand === "left" ? "Left" : "Right"} hand</span>
                </span>
                <span className="punch-history-time">
                  <span aria-label={`Detection ${ordinal}`}>#{ordinal}</span>
                  <time aria-label={`Punch peak ${actionTime(event.peakMs)}`}>
                    {actionTime(event.peakMs)}
                  </time>
                </span>
              </li>
            ))}
          </ol>
        ) : (
          <div className="live-history-empty">
            <strong>Ready when you are</strong>
            <span>Your punches will stay here</span>
          </div>
        )}
        {hiddenBatchCount > 0 && (
          <p className="live-history-overflow">
            +{hiddenBatchCount} more in this batch · saved in Review
          </p>
        )}
        <div
          className={`live-tracking-note ${trackingUnclear ? "needs-framing" : ""}`}
          role="status"
        >
          {trackingUnclear
            ? "Tracking unclear · keep your arms in view"
            : "Live detections · not a form score"}
        </div>
      </section>
      <div
        className="live-total"
        aria-label={`${accepted.length} punches detected`}
      >
        <strong data-testid="live-punch-count">{accepted.length}</strong>
        <span>detected</span>
      </div>
      <span
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {latest
          ? `${batch.map((event) => `${punchName(event)}, ${event.hand} hand`).join(". ")}. ${accepted.length} detected.`
          : "Round started. Waiting for detections."}
      </span>
    </div>
  );
}
