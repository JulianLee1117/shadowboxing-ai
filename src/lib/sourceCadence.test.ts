import { describe, expect, it } from "vitest";
import { SourceCadence } from "./sourceCadence";

describe("source callback cadence", () => {
  it("separates source media cadence from playback delivery and pose processing", () => {
    const cadence = new SourceCadence();
    for (let index = 0; index < 31; index++)
      cadence.observe(100 + (index * 1000) / 30, 200 + (index * 1000) / 15);
    const summary = cadence.summary();
    expect(summary.observations).toBe(31);
    expect(summary.mediaFps).toBeCloseTo(30);
    expect(summary.callbackFps).toBeCloseTo(15);
    expect(summary.mediaSpanMs).toBeCloseTo(1000);
    expect(summary.callbackSpanMs).toBeCloseTo(2000);
    expect(summary.gapMs.p95).toBeCloseTo(1000 / 30);
    expect(summary.gapMs.complete).toBe(true);
  });

  it("keeps missing samples unknown and does not divide count by round duration", () => {
    const cadence = new SourceCadence();
    expect(cadence.summary()).toMatchObject({
      observations: 0,
      mediaFps: null,
      callbackFps: null,
      mediaSpanMs: null,
      gapMs: { p50: null, p95: null, maximum: null },
    });
    cadence.observe(600, 900);
    expect(cadence.summary()).toMatchObject({
      observations: 1,
      mediaFps: null,
      callbackFps: null,
      mediaSpanMs: 0,
    });
    cadence.observe(650, 950);
    expect(cadence.summary()).toMatchObject({
      observations: 2,
      mediaFps: 20,
      callbackFps: 20,
    });
  });

  it("does not turn repeated or invalid callbacks into extra frames", () => {
    const cadence = new SourceCadence();
    cadence.observe(10, 10);
    cadence.observe(10, 20);
    cadence.observe(NaN, 30);
    cadence.observe(30, Infinity);
    cadence.observe(-1, 40);
    cadence.observe(60, 60);
    expect(cadence.summary()).toMatchObject({
      observations: 2,
      duplicates: 1,
      invalid: 3,
      mediaFps: 20,
      callbackFps: 20,
    });
  });

  it("reports clock regression and withholds aggregate rates instead of joining discontinuities", () => {
    const cadence = new SourceCadence();
    cadence.observe(100, 100);
    cadence.observe(90, 110);
    cadence.observe(110, 90);
    cadence.observe(150, 150);
    expect(cadence.summary()).toMatchObject({
      observations: 2,
      regressions: 2,
      mediaFps: null,
      callbackFps: null,
    });
  });

  it("bounds retained gap samples and labels their partial distribution", () => {
    const cadence = new SourceCadence(2);
    [0, 30, 60, 150].forEach((t) => cadence.observe(t, t));
    expect(cadence.summary()).toMatchObject({
      observations: 4,
      mediaFps: 20,
      gapMs: { samples: 2, complete: false, p50: 30, p95: 30, maximum: 90 },
    });
    expect(() => new SourceCadence(0)).toThrow();
    expect(() => new SourceCadence(20_001)).toThrow();
    expect(() => new SourceCadence(1.5)).toThrow();
  });

  it("returns a detached summary and does not carry observations into another round", () => {
    const cadence = new SourceCadence();
    cadence.observe(0, 0);
    cadence.observe(100, 100);
    const summary = cadence.summary();
    summary.gapMs.p95 = 999;
    cadence.observe(200, 200);
    expect(summary.observations).toBe(2);
    expect(cadence.summary().gapMs.p95).toBe(100);
    expect(new SourceCadence().summary().observations).toBe(0);
  });
});
