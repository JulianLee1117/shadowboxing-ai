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
    expect(recentDetection([event], 3201)).toBeNull();
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
        elapsedMs: 3300,
        trackingUnclear: true,
      }),
    );
    expect(recent).toContain("Last detected");
    expect(recent).toContain("Left hand · Lead");
    expect(expired).toContain("Tracking unclear");
    expect(expired).not.toContain("Keep moving");
    expect(expired).toContain("1 punches detected");
  });
});
