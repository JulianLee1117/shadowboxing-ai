import { describe, expect, it } from "vitest";
import { LiveFeedback, recentDetection } from "./LiveFeedback";
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

describe("recent live feedback", () => {
  it("shows a finalized observation using emission time even when its peak is older", () => {
    expect(recentDetection([{ ...event, detectedAtMs: 2900 }], 3000)?.id).toBe(
      event.id,
    );
  });
  it("expires feedback while preserving a new observation that leads the coarse display clock", () => {
    expect(recentDetection([event], 1100)).toBe(event);
    expect(recentDetection([event], 4701)).toBeNull();
  });
  it("does not replay a prior-round label after the clock resets", () => {
    expect(recentDetection([event], 0)).toBeNull();
    expect(recentDetection([], 0)).toBeNull();
  });
  it("does not fall back to an older event if the latest timestamp is invalid", () => {
    expect(
      recentDetection(
        [event, { ...event, id: "bad", detectedAtMs: NaN }],
        1250,
      ),
    ).toBeNull();
    expect(recentDetection([event], NaN)).toBeNull();
  });
  it("supports an older event without an emission timestamp", () => {
    expect(
      recentDetection([{ ...event, detectedAtMs: undefined }], 1000)?.id,
    ).toBe(event.id);
  });
  it("gives each new same-hand detection a short +1 indicator while the last label remains readable", () => {
    const render = (events: PunchEvent[], elapsedMs: number) =>
      renderToStaticMarkup(
        createElement(LiveFeedback, {
          events,
          elapsedMs,
          trackingUnclear: false,
        }),
      );
    const settled = render([event], 2300);
    expect(settled).toContain("Left hand · Lead");
    expect(settled).not.toContain("detection-tick");
    const repeated = render(
      [event, { ...event, id: "second-jab", detectedAtMs: 2350 }],
      2400,
    );
    expect(repeated).toContain("detection-tick");
    expect(repeated).toContain("+1");
    expect(repeated).toContain("live-feedback hand-left");
    expect(repeated).toContain('data-testid="live-punch-count">2');
    expect(render([event], 0)).not.toContain("detection-tick");
  });
  it("shows +2 when different hands finalize in one emission batch", () => {
    const html = renderToStaticMarkup(
      createElement(LiveFeedback, {
        events: [
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
        elapsedMs: 1250,
        trackingUnclear: false,
      }),
    );
    expect(html).toContain('class="detection-tick"');
    expect(html).toContain(">+2</span>");
    expect(html).toContain("Right hand · Rear");
    expect(html).toContain('data-testid="live-punch-count">2');
  });
  it("explains unavailable tracking instead of prompting an invisible boxer to keep punching", () => {
    const html = renderToStaticMarkup(
      createElement(LiveFeedback, {
        events: [],
        elapsedMs: 1000,
        trackingUnclear: true,
      }),
    );
    expect(html).toContain("Tracking unclear");
    expect(html).toContain("Keep your arms and torso in view");
    expect(html).toContain("Round continues");
    expect(html).not.toContain("Find your rhythm");
  });
  it("keeps an emitted detection briefly visible, then explains subsequent tracking loss", () => {
    const recent = renderToStaticMarkup(
      createElement(LiveFeedback, {
        events: [event],
        elapsedMs: 1250,
        trackingUnclear: true,
      }),
    );
    const expired = renderToStaticMarkup(
      createElement(LiveFeedback, {
        events: [event],
        elapsedMs: 4800,
        trackingUnclear: true,
      }),
    );
    expect(recent).toContain("● Detected");
    expect(recent).toContain("Left hand · Lead");
    expect(expired).toContain("Tracking unclear");
    expect(expired).not.toContain("Keep moving");
    expect(expired).toContain("1 punches detected");
  });
});
