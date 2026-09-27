import { punchName } from "../lib/punches";
import type { PunchEvent } from "../lib/types";
import "./LiveFeedback.css";

/** A finalized detection is recent at emission time, not at its earlier peak.
 * A clock reset must never bring a previous round's label back onto the stage.
 */
export function recentDetection(
  events: readonly PunchEvent[],
  elapsedMs: number,
): PunchEvent | null {
  const event = events.at(-1);
  if (!event || !Number.isFinite(elapsedMs) || elapsedMs < 0) return null;
  const detectedAt = event.detectedAtMs ?? event.endMs;
  const age = elapsedMs - detectedAt;
  // The displayed clock updates every 100 ms and can trail a fresh result.
  return Number.isFinite(detectedAt) &&
    detectedAt >= 0 &&
    age >= -150 &&
    age <= 3500
    ? event
    : null;
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
  const event = recentDetection(events, elapsedMs);
  const emittedAt = event?.detectedAtMs ?? event?.endMs ?? NaN;
  const justDetected = event && elapsedMs - emittedAt <= 900;
  const batchSize = justDetected
    ? events.filter((item) => (item.detectedAtMs ?? item.endMs) === emittedAt)
        .length
    : 0;
  return (
    <div
      className={`live-feedback ${event ? `hand-${event.hand}` : ""}`}
      aria-label="Live punch feedback"
    >
      <div
        className="live-punch"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        <span className="live-eyebrow">
          {event
            ? "● Detected"
            : trackingUnclear
              ? "Round continues"
              : "Your round"}
          {justDetected && (
            <span className="detection-tick" key={event.id} aria-hidden="true">
              +{batchSize}
            </span>
          )}
        </span>
        <strong
          className={
            event
              ? "live-punch-name detected"
              : `live-punch-name waiting ${trackingUnclear ? "needs-framing" : ""}`
          }
          key={event?.id ?? "waiting"}
        >
          {event
            ? punchName(event)
            : trackingUnclear
              ? "Tracking unclear"
              : events.length
                ? "Keep moving"
                : "Find your rhythm"}
        </strong>
        <span className="live-punch-identity">
          {event
            ? `${event.hand === "left" ? "Left" : "Right"} hand · ${event.role === "lead" ? "Lead" : "Rear"}`
            : trackingUnclear
              ? "Keep your arms and torso in view"
              : "Punch detections will appear here"}
        </span>
      </div>
      <div
        className="live-total"
        aria-label={`${events.length} punches detected`}
      >
        <strong data-testid="live-punch-count">{events.length}</strong>
        <span>punches detected</span>
      </div>
      <span className="live-feedback-note">
        Experimental detection · not a form assessment
      </span>
    </div>
  );
}
