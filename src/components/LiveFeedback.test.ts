import { describe, expect, it } from "vitest";
import {
  LiveFeedback,
  acceptedDetections,
  detectionHistory,
  recentDetection,
} from "./LiveFeedback";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type { PunchEvent } from "../lib/types";

const event: PunchEvent = {
  id: "round-a-punch-1",
  hand: "left",
  role: "lead",
  label: "jab",
  startMs: 300,
  peakMs: 600,
  endMs: 900,
  detectedAtMs: 1200,
  score: 0.8,
  extension: 0.8,
  guardReturn: "returned",
  experimental: true,
};
const render = (
  events: PunchEvent[],
  elapsedMs: number,
  trackingUnclear = false,
) =>
  renderToStaticMarkup(
    createElement(LiveFeedback, { events, elapsedMs, trackingUnclear }),
  );

describe("live detection history", () => {
  it("shows each delivered punch immediately, using receipt order rather than inventing combo order", () => {
    const laterArrival: PunchEvent = {
      ...event,
      id: "delayed-cross",
      hand: "right",
      role: "rear",
      label: "cross",
      peakMs: 500,
      detectedAtMs: 1300,
    };
    const events = [event, laterArrival];
    expect(
      detectionHistory(events, 1250).map((row) => [row.event.id, row.ordinal]),
    ).toEqual([
      ["delayed-cross", 2],
      [event.id, 1],
    ]);
    expect(events).toEqual([event, laterArrival]);
    const html = render(events, 1250);
    expect(html.indexOf('data-event-id="delayed-cross"')).toBeLessThan(
      html.indexOf('data-event-id="round-a-punch-1"'),
    );
    expect(html).toContain("Right hand");
    expect(html).toContain("Left hand");
    expect(html).toContain("0:00.5");
  });
  it("keeps history readable during a pause, independently of source-time recency", () => {
    expect(recentDetection([event], 1100)).toBe(event);
    expect(recentDetection([event], 2200)).toBeNull();
    expect(detectionHistory([event], 29000)[0].event).toBe(event);
    const html = render([event], 29000);
    expect(html).toContain("Jab");
    // The keyed receipt badge has a CSS lifetime. Old source evidence must
    // still signal arrival when it is first delivered to this component.
    expect(html).toContain("detection-tick");
    expect(html).toContain('data-testid="live-punch-count">1');
  });
  it("does not revive a previous round's results after a clock reset", () => {
    expect(recentDetection([event], 0)).toBeNull();
    expect(detectionHistory([event], 0)).toEqual([]);
    expect(detectionHistory([], 0)).toEqual([]);
    expect(render([], 0)).toContain("Ready when you are");
  });
  it("bounds the visible log, keeps repeated jabs distinct and preserves the full count", () => {
    const events = Array.from({ length: 8 }, (_, i) => ({
      ...event,
      id: `jab-${i}`,
      detectedAtMs: 1200 + i * 200,
    }));
    expect(detectionHistory(events, 2700).map((row) => row.ordinal)).toEqual([
      8, 7, 6, 5,
    ]);
    const html = render(events, 2700);
    expect((html.match(/data-event-id=/g) ?? []).length).toBe(4);
    expect(html).toContain('data-testid="live-punch-count">8');
    expect(html).toContain("+1");
  });
  it("renders both names from a same-frame batch, with one +2 receipt indicator", () => {
    const html = render(
      [
        event,
        {
          ...event,
          id: "same-batch-cross",
          hand: "right",
          role: "rear",
          label: "cross",
          endMs: 1100,
        },
      ],
      1250,
    );
    expect((html.match(/data-event-id=/g) ?? []).length).toBe(2);
    expect(html).toContain("Jab");
    expect(html).toContain("Cross");
    expect(html).toContain(">+2</span>");
    expect(html).toContain('data-testid="live-punch-count">2');
  });
  it("does not hide recorded detections when tracking is lost", () => {
    const html = render([event], 10000, true);
    expect(html).toContain("Jab");
    expect(html).toContain("Tracking unclear · keep your arms in view");
    expect(html).not.toContain("Keep moving");
  });
  it("omits invalid and future timestamps and does not duplicate an event", () => {
    expect(
      detectionHistory(
        [event, { ...event, id: "bad", detectedAtMs: NaN }],
        1250,
      ).map((row) => row.event.id),
    ).toEqual([event.id]);
    expect(detectionHistory([event], NaN)).toEqual([]);
    expect(detectionHistory([event], -1)).toEqual([]);
    expect(detectionHistory([event, event], 1300)).toHaveLength(1);
    expect(recentDetection([{ ...event, detectedAtMs: NaN }], 1300)).toBeNull();
  });
  it("uses the same accepted receipts for the log, total, batch and announcement", () => {
    const invalid: PunchEvent = {
      ...event,
      id: "invalid-uppercut",
      label: "uppercut",
      peakMs: NaN,
    };
    const duplicate = { ...event, label: "cross" as const };
    const html = render([event, invalid, duplicate], 1250);
    expect(acceptedDetections([event, invalid, duplicate], 1250)).toEqual([
      event,
    ]);
    expect((html.match(/data-event-id=/g) ?? []).length).toBe(1);
    expect(html).toContain('data-testid="live-punch-count">1');
    expect(html).toContain(">+1</span>");
    expect(html).not.toContain("uppercut");
    expect(html).not.toContain("Cross");
    expect(html).toContain("Jab, left hand. 1 detected.");
  });
  it("rejects noncausal or unordered times and lets the first valid receipt own an ID", () => {
    const invalids = [
      { ...event, startMs: -1 },
      { ...event, startMs: 700 },
      { ...event, endMs: 500 },
      { ...event, endMs: 1400 },
      { ...event, detectedAtMs: Infinity },
      { ...event, id: "" },
    ];
    expect(acceptedDetections(invalids, 1250)).toEqual([]);
    expect(acceptedDetections([...invalids, event], 1250)).toEqual([event]);
    expect(render(invalids, 1250)).toContain(
      'data-testid="live-punch-count">0',
    );
    expect(render(invalids, 1250)).not.toContain("detection-tick");
  });
  it("marks the full newest batch for visibility and explicitly reports overflow beyond four rows", () => {
    const batch = Array.from({ length: 6 }, (_, index) => ({
      ...event,
      id: `batch-${index}`,
    }));
    const html = render(batch, 1250);
    expect((html.match(/data-event-id=/g) ?? []).length).toBe(4);
    expect((html.match(/newest-batch/g) ?? []).length).toBe(4);
    expect(html).toContain("+2 more in this batch · saved in Review");
    expect(html).toContain(">+6</span>");
    expect(html).toContain('data-testid="live-punch-count">6');
    const triple = render(batch.slice(0, 3), 1250);
    expect((triple.match(/newest-batch/g) ?? []).length).toBe(3);
    expect(triple).not.toContain("live-history-overflow");
  });
  it("supports legacy events with no emission timestamp and never renders 0:60.0", () => {
    expect(
      detectionHistory([{ ...event, detectedAtMs: undefined }], 1000),
    ).toHaveLength(1);
    expect(
      render(
        [{ ...event, peakMs: 59999, endMs: 60100, detectedAtMs: 60500 }],
        60500,
      ),
    ).toContain("0:59.9");
  });
  it("keeps role and physical hand separate for a southpaw lead hook", () => {
    const html = render(
      [{ ...event, hand: "right", role: "lead", label: "hook" }],
      1250,
    );
    expect(html).toContain("Lead hook");
    expect(html).toContain("Right hand");
    expect(html).toContain(
      'class="punch-history-item hand-right latest newest-batch"',
    );
    expect(html).toContain('class="punch-history-number" aria-hidden="true">3');
    expect(html).not.toContain("correct form");
  });
});
