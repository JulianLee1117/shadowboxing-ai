import { describe, expect, it } from "vitest";
import {
  createAnalysisPlan,
  ANALYSIS_LIMITS,
  analyzeRound,
} from "./roundAnalysis";
import { demoFrame } from "./demo";
import type { Session } from "./types";
const session = (overrides: Partial<Session> = {}): Session => ({
  id: "test-source",
  createdAt: "2026-01-01T00:00:00Z",
  source: "camera",
  stance: "orthodox",
  model: "full",
  drill: "Free practice",
  durationMs: 1000,
  frames: [],
  events: [],
  annotations: [],
  video: new Blob(["fixture"], { type: "video/webm" }),
  measuredFps: 0,
  inferenceP95: 0,
  skippedFrames: 0,
  schemaVersion: "1.0",
  ...overrides,
});
describe("saved video analysis plan", () => {
  it("prefers source camera cadence over slow live inference and caps planning at60fps", () => {
    const s = session({
      frames: [demoFrame(0), demoFrame(100)],
      capture: {
        width: 1280,
        height: 720,
        deliveredFps: 60,
        timingSource: "source",
        delegate: "GPU",
      },
    });
    expect(createAnalysisPlan(s)).toMatchObject({
      fps: 60,
      frameCount: 60,
      cadenceSource: "recorded camera cadence",
      truncated: false,
    });
    expect(s.frames.map((f) => f.t)).toEqual([0, 100]);
  });
  it("uses observed frame cadence or an explicit fallback when source cadence is unavailable", () => {
    expect(
      createAnalysisPlan(
        session({ frames: [demoFrame(0), demoFrame(50), demoFrame(100)] }),
      ),
    ).toMatchObject({ fps: 20, cadenceSource: "saved frame median cadence" });
    expect(createAnalysisPlan(session())).toMatchObject({
      fps: 30,
      cadenceSource: "30 fps planning estimate; source cadence not recorded",
    });
  });
  it("caps work and preserves a valid source-video offset", () => {
    expect(
      createAnalysisPlan(session({ durationMs: 240000, videoOffsetMs: 5000 })),
    ).toMatchObject({
      offsetMs: 5000,
      durationMs: 185000,
      frameCount: 5550,
      truncated: true,
    });
    expect(ANALYSIS_LIMITS.maximumWallMs).toBe(240000);
  });
  it("rejects unsupported source, missing video, invalid duration and invalid offset", () => {
    for (const overrides of [
      { source: "demo" as const },
      { video: undefined },
      { durationMs: NaN },
      { videoOffsetMs: -1 },
    ])
      expect(() => createAnalysisPlan(session(overrides))).toThrow();
  });
  it("rejects a canceled job before starting browser resources", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      analyzeRound(session(), { signal: controller.signal }),
    ).rejects.toMatchObject({ name: "AbortError" });
  });
});
