import { describe, expect, it } from "vitest";
import { demoFrame } from "./demo";
import { DETECTOR_VERSION } from "./motion";
import { recheckDetections } from "./recheck";
import type { PunchEvent, Session } from "./types";

function session(): Session {
  return {
    id: "original-round",
    createdAt: "2026-01-01T00:00:00.000Z",
    source: "demo",
    stance: "orthodox",
    model: "synthetic",
    drill: "open",
    durationMs: 6400,
    frames: Array.from({ length: 161 }, (_, index) => demoFrame(index * 40)),
    events: [],
    annotations: [],
    measuredFps: 25,
    inferenceP95: 0,
    skippedFrames: 0,
    schemaVersion: "1.0",
    detectorVersion: "original-test-detector",
  };
}

function freezeDeep(value: unknown) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return;
  Object.freeze(value);
  for (const child of Object.values(value)) freezeDeep(child);
}

describe("saved-tracking detection recheck", () => {
  it("derives fresh events without changing frozen original evidence", () => {
    const original = session();
    const snapshot = structuredClone(original);
    freezeDeep(original);
    const report = recheckDetections(original);
    expect(report).toMatchObject({
      reportType: "detector-recheck",
      sessionId: original.id,
      detectorVersion: DETECTOR_VERSION,
      sourceDetectorVersion: "original-test-detector",
      trackingSource: "saved-frames",
      stance: "orthodox",
      frameCount: original.frames.length,
    });
    expect(report.events.length).toBeGreaterThan(0);
    expect(report.events.every((event) => event.experimental)).toBe(true);
    expect(report).not.toHaveProperty("frames");
    expect(report).not.toHaveProperty("video");
    expect(original).toEqual(snapshot);
    expect(original.events).toEqual([]);
  });

  it("uses the saved stance without changing anatomical hand identity", () => {
    const original = session();
    const orthodox = recheckDetections(original);
    const southpaw = recheckDetections({ ...original, stance: "southpaw" });
    expect(southpaw.events.map((event) => event.hand)).toEqual(
      orthodox.events.map((event) => event.hand),
    );
    for (const [index, event] of southpaw.events.entries()) {
      expect(event.label).not.toBe(orthodox.events[index].label);
      expect(event.role).not.toBe(orthodox.events[index].role);
    }
  });

  it("starts a fresh engine per report and never copies saved detections", () => {
    const original = session();
    expect(recheckDetections(original).events.length).toBeGreaterThan(0);
    original.frames = [];
    original.events = [{ id: "saved-only" } as PunchEvent];
    original.detectorVersion = undefined;
    const report = recheckDetections(original);
    expect(report.events).toEqual([]);
    expect(report.sourceDetectorVersion).toBeNull();
    expect(original.events[0].id).toBe("saved-only");
  });
});
