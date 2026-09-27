import type { PunchEvent, Stance } from "./types";
import type { TrackingUncertaintyInterval } from "./trackingTrust";

export const COMBINATION_VERSION = "observed-combinations-v1";
export type ComboName =
  | "jab-cross"
  | "double-jab"
  | "double-jab-cross"
  | "jab-cross-jab";
export interface CombinationOptions {
  stance: Stance;
  sessionId?: string;
  minimumPeakGapMs?: number;
  maximumPeakGapMs?: number;
  maximumComboSpanMs?: number;
  maximumArrivalDelayMs?: number;
  requireObservedReturn?: boolean;
  uncertaintyIntervals?: readonly TrackingUncertaintyInterval[];
}
export interface ComboEvent {
  id: string;
  name: ComboName;
  notation: "1-2" | "1-1" | "1-1-2" | "1-2-1";
  eventIds: string[];
  startMs: number;
  peakMs: number;
  endMs: number;
  finalizedAtMs: number;
  confidence: number;
  confidenceSemantics: "minimum-uncalibrated-punch-score";
  experimental: true;
  provenance: {
    algorithmVersion: string;
    stance: Stance;
    sessionId: string | null;
    ordering: "source-peak-time";
    uncertaintyPolicy: "block-intersecting-participating-arm-intervals";
    finalizationSources: ("detector" | "consumer" | "event-end-offline")[];
    sourceHands: PunchEvent["hand"][];
    finalizationReason: "round-complete" | "ordering-watermark";
    timing: {
      minimumPeakGapMs: number;
      maximumPeakGapMs: number;
      maximumComboSpanMs: number;
      maximumArrivalDelayMs: number;
    };
  };
}
type Pending = {
  event: PunchEvent;
  finalizationSource: "detector" | "consumer" | "event-end-offline";
};
const patterns: {
  labels: string[];
  name: ComboName;
  notation: ComboEvent["notation"];
}[] = [
  {
    labels: ["jab", "jab", "cross"],
    name: "double-jab-cross",
    notation: "1-1-2",
  },
  { labels: ["jab", "cross", "jab"], name: "jab-cross-jab", notation: "1-2-1" },
  { labels: ["jab", "cross"], name: "jab-cross", notation: "1-2" },
  { labels: ["jab", "jab"], name: "double-jab", notation: "1-1" },
];
const overlaps = (a: number, b: number, c: number, d: number) =>
  a <= d && c <= b;
const identity = (e: PunchEvent) =>
  JSON.stringify([
    e.id,
    e.hand,
    e.role,
    e.label,
    e.startMs,
    e.peakMs,
    e.endMs,
    e.detectedAtMs ?? null,
    e.score,
    e.extension,
    e.guardReturn,
  ]);

/** Finalized punch inputs only. The bounded holdback prevents arrival order from
 * pretending to be movement order. flush closes a round; reset begins another. */
export class CombinationRecognizer {
  private readonly timing;
  private readonly options: CombinationOptions;
  private pending: Pending[] = [];
  private seen = new Map<string, string>();
  private intervals: TrackingUncertaintyInterval[] = [];
  private emitted: ComboEvent[] = [];
  private clock = -1;
  private closed = false;

  constructor(options: CombinationOptions) {
    if (!["orthodox", "southpaw"].includes(options.stance))
      throw new Error("Invalid stance");
    this.options = { ...options };
    this.timing = {
      minimumPeakGapMs: options.minimumPeakGapMs ?? 60,
      maximumPeakGapMs: options.maximumPeakGapMs ?? 650,
      maximumComboSpanMs: options.maximumComboSpanMs ?? 1300,
      maximumArrivalDelayMs: options.maximumArrivalDelayMs ?? 1800,
    };
    if (
      Object.values(this.timing).some((n) => !Number.isFinite(n) || n < 0) ||
      this.timing.minimumPeakGapMs <= 0 ||
      this.timing.maximumPeakGapMs < this.timing.minimumPeakGapMs ||
      this.timing.maximumComboSpanMs < this.timing.maximumPeakGapMs
    )
      throw new Error("Invalid combination timing constraints");
    this.addIntervals(options.uncertaintyIntervals ?? []);
  }

  reset(
    uncertaintyIntervals: readonly TrackingUncertaintyInterval[] = [],
  ): void {
    this.pending = [];
    this.seen.clear();
    this.emitted = [];
    this.clock = -1;
    this.closed = false;
    this.intervals = [];
    this.addIntervals(uncertaintyIntervals);
  }

  consume(
    events: readonly PunchEvent[],
    nowMs: number,
    uncertaintyIntervals: readonly TrackingUncertaintyInterval[] = [],
  ): ComboEvent[] {
    if (this.closed)
      throw new Error("Round is closed; reset before consuming more events");
    this.checkClock(nowMs);
    this.addIntervals(uncertaintyIntervals);
    for (const input of events) {
      const key = identity(input),
        prior = this.seen.get(input.id);
      if (prior !== undefined) {
        if (prior !== key)
          throw new Error(
            "Conflicting punch evidence for an existing event ID",
          );
        continue;
      }
      if (
        !input.id ||
        ![input.startMs, input.peakMs, input.endMs, input.score].every(
          Number.isFinite,
        ) ||
        input.startMs < 0 ||
        input.startMs > input.peakMs ||
        input.peakMs > input.endMs ||
        input.endMs > nowMs ||
        input.score < 0 ||
        input.score > 1 ||
        (input.detectedAtMs !== undefined &&
          (!Number.isFinite(input.detectedAtMs) ||
            input.detectedAtMs < input.endMs ||
            input.detectedAtMs > nowMs))
      )
        throw new Error("Invalid finalized punch evidence");
      if (nowMs - input.peakMs > this.timing.maximumArrivalDelayMs)
        throw new Error(
          "Punch arrived after the declared ordering window; reset or replay complete evidence",
        );
      this.seen.set(input.id, key);
      this.pending.push({
        event: { ...input },
        finalizationSource:
          input.detectedAtMs === undefined ? "consumer" : "detector",
      });
    }
    this.pending.sort(
      (a, b) =>
        a.event.peakMs - b.event.peakMs || a.event.id.localeCompare(b.event.id),
    );
    this.clock = nowMs;
    return this.drain(nowMs, false);
  }

  /** Caller asserts that all finalized punches and uncertainty for the round arrived. */
  flush(nowMs: number): ComboEvent[] {
    this.checkClock(nowMs);
    if (this.closed) return [];
    this.closed = true;
    this.clock = nowMs;
    return this.drain(nowMs, true);
  }

  private checkClock(nowMs: number): void {
    if (!Number.isFinite(nowMs) || nowMs < 0 || nowMs < this.clock)
      throw new Error("Source clock must be finite and nondecreasing");
  }

  private addIntervals(
    intervals: readonly TrackingUncertaintyInterval[],
  ): void {
    for (const interval of intervals) {
      if (
        !["left", "right", "both"].includes(interval.hand) ||
        !Number.isFinite(interval.startMs) ||
        !Number.isFinite(interval.endMs) ||
        interval.startMs < 0 ||
        interval.endMs < interval.startMs
      )
        throw new Error("Invalid uncertainty interval");
      const key = JSON.stringify(interval);
      if (this.intervals.some((old) => JSON.stringify(old) === key)) continue;
      if (
        this.emitted.some(
          (event) =>
            (interval.hand === "both" ||
              event.provenance.sourceHands.includes(interval.hand)) &&
            overlaps(
              event.startMs,
              event.endMs,
              interval.startMs,
              interval.endMs,
            ),
        )
      )
        throw new Error(
          "Late uncertainty overlaps a finalized combination; replay the round",
        );
      this.intervals.push({ ...interval, reasons: [...interval.reasons] });
    }
  }

  private eligible(event: PunchEvent): boolean {
    const lead = this.options.stance === "orthodox" ? "left" : "right";
    return (
      ((event.label === "jab" &&
        event.role === "lead" &&
        event.hand === lead) ||
        (event.label === "cross" &&
          event.role === "rear" &&
          event.hand !== lead &&
          ["left", "right"].includes(event.hand))) &&
      (this.options.requireObservedReturn !== true ||
        event.guardReturn === "returned")
    );
  }

  private matches(labels: string[]): boolean {
    const events = this.pending.slice(0, labels.length).map((p) => p.event);
    if (
      events.length !== labels.length ||
      events.some(
        (event, i) => !this.eligible(event) || event.label !== labels[i],
      )
    )
      return false;
    for (let i = 1; i < events.length; i++) {
      const gap = events[i].peakMs - events[i - 1].peakMs;
      if (
        gap < this.timing.minimumPeakGapMs ||
        gap > this.timing.maximumPeakGapMs
      )
        return false;
    }
    if (
      events.at(-1)!.peakMs - events[0].peakMs >
      this.timing.maximumComboSpanMs
    )
      return false;
    const start = Math.min(...events.map((event) => event.startMs)),
      end = Math.max(...events.map((event) => event.endMs));
    return !this.intervals.some(
      (interval) =>
        (interval.hand === "both" ||
          events.some((event) => event.hand === interval.hand)) &&
        overlaps(start, end, interval.startMs, interval.endMs),
    );
  }

  private drain(nowMs: number, force: boolean): ComboEvent[] {
    const result: ComboEvent[] = [],
      watermark = nowMs - this.timing.maximumArrivalDelayMs;
    while (this.pending.length) {
      const head = this.pending[0].event;
      if (!force && head.peakMs > watermark) break;
      const match = patterns.find((pattern) => this.matches(pattern.labels));
      const mature =
        force ||
        watermark > head.peakMs + this.timing.maximumComboSpanMs ||
        (match?.labels.length === 3 &&
          watermark >= this.pending[2].event.peakMs);
      if (!mature) break;
      if (!match) {
        this.pending.shift();
        continue;
      }
      const members = this.pending.splice(0, match.labels.length),
        events = members.map((p) => p.event),
        ids = events.map((e) => e.id);
      const event: ComboEvent = {
        id: `${COMBINATION_VERSION}:${encodeURIComponent(this.options.sessionId ?? "unscoped")}:${match.notation}:${ids.map(encodeURIComponent).join("+")}`,
        name: match.name,
        notation: match.notation,
        eventIds: ids,
        startMs: Math.min(...events.map((e) => e.startMs)),
        peakMs: events.at(-1)!.peakMs,
        endMs: Math.max(...events.map((e) => e.endMs)),
        finalizedAtMs: nowMs,
        confidence: Math.min(...events.map((e) => e.score)),
        confidenceSemantics: "minimum-uncalibrated-punch-score",
        experimental: true,
        provenance: {
          algorithmVersion: COMBINATION_VERSION,
          stance: this.options.stance,
          sessionId: this.options.sessionId ?? null,
          ordering: "source-peak-time",
          uncertaintyPolicy: "block-intersecting-participating-arm-intervals",
          finalizationSources: members.map((p) => p.finalizationSource),
          sourceHands: events.map((event) => event.hand),
          finalizationReason: force ? "round-complete" : "ordering-watermark",
          timing: { ...this.timing },
        },
      };
      this.emitted.push(structuredClone(event));
      result.push(event);
    }
    return result;
  }
}

/** Complete-round grouping; no requested drill or labels are used to invent punches. */
export function groupCombinations(
  events: readonly PunchEvent[],
  options: CombinationOptions,
): ComboEvent[] {
  // Complete evidence can declare its actual finalization delay. Keep the
  // streaming API's bound strict, while a late curved event still interrupts
  // an apparent jab–cross sequence rather than crashing review or being omitted.
  const maximumArrivalDelayMs = events.reduce(
    (delay, event) =>
      Math.max(delay, (event.detectedAtMs ?? event.endMs) - event.peakMs),
    options.maximumArrivalDelayMs ?? 1800,
  );
  const recognizer = new CombinationRecognizer({
      ...options,
      maximumArrivalDelayMs,
    }),
    result: ComboEvent[] = [];
  const sorted = [...events].sort(
    (a, b) =>
      (a.detectedAtMs ?? a.endMs) - (b.detectedAtMs ?? b.endMs) ||
      a.peakMs - b.peakMs,
  );
  let now = 0;
  for (const event of sorted) {
    now = event.detectedAtMs ?? event.endMs;
    result.push(...recognizer.consume([event], now));
  }
  result.push(...recognizer.flush(now));
  const missing = new Set(
    events.filter((e) => e.detectedAtMs === undefined).map((e) => e.id),
  );
  for (const combo of result)
    combo.provenance.finalizationSources = combo.eventIds.map((id) =>
      missing.has(id) ? "event-end-offline" : "detector",
    );
  return result;
}
