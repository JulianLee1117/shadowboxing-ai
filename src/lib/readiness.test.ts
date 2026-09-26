import { describe, expect, it } from "vitest";
import { ReadinessGate } from "./readiness";

describe("hands-free recording countdown", () => {
  it("allows eight full seconds to step back, then starts without another click", () => {
    const gate = new ReadinessGate();
    expect(gate.arm(500)).toMatchObject({
      phase: "countdown",
      countdownSeconds: 8,
      pending: true,
      start: false,
    });
    expect(gate.update({ nowMs: 1500 })).toMatchObject({
      countdownSeconds: 7,
      start: false,
    });
    expect(gate.update({ nowMs: 8499 })).toMatchObject({
      countdownSeconds: 1,
      start: false,
    });
    expect(gate.update({ nowMs: 8500 })).toMatchObject({
      phase: "ready",
      countdownSeconds: 0,
      pending: false,
      start: true,
    });
  });
  it.each(["missing", "occluded", "frozen", "backward", "invalid"] as const)(
    "starts at eight seconds with %s pose data",
    (quality) => {
      const gate = new ReadinessGate();
      gate.arm(0);
      for (let nowMs = 0; nowMs < 8000; nowMs += 100) {
        const frameTimeMs =
          quality === "missing"
            ? null
            : quality === "frozen"
              ? 42
              : quality === "backward"
                ? 9000 - nowMs
                : quality === "invalid"
                  ? Number.NaN
                  : nowMs;
        expect(
          gate.update({ nowMs, frameTimeMs, assessable: false }).start,
        ).toBe(false);
      }
      expect(
        gate.update({ nowMs: 8000, frameTimeMs: null, assessable: false })
          .start,
      ).toBe(true);
    },
  );
  it("emits exactly one start and ignores later visibility changes", () => {
    const gate = new ReadinessGate();
    gate.arm(0);
    expect(gate.update({ nowMs: 8000, assessable: false }).start).toBe(true);
    for (let nowMs = 8100; nowMs <= 9000; nowMs += 100)
      expect(
        gate.update({ nowMs, assessable: nowMs % 200 === 0 }),
      ).toMatchObject({ phase: "ready", start: false });
    expect(gate.update({ nowMs: 50_000, assessable: false }).start).toBe(false);
  });
  it("does not extend the countdown after a repeated click", () => {
    const gate = new ReadinessGate();
    gate.arm(0);
    gate.update({ nowMs: 5000 });
    gate.arm(5000);
    expect(gate.update({ nowMs: 8000 }).start).toBe(true);
  });
  it("cancel prevents a delayed recording and retry receives a new countdown", () => {
    const gate = new ReadinessGate();
    gate.arm(0);
    gate.update({ nowMs: 7900 });
    expect(gate.cancel()).toMatchObject({
      phase: "idle",
      pending: false,
      start: false,
    });
    expect(gate.update({ nowMs: 9000 })).toMatchObject({
      phase: "idle",
      start: false,
    });
    gate.arm(10_000);
    expect(gate.update({ nowMs: 17_999 }).start).toBe(false);
    expect(gate.update({ nowMs: 18_000 }).start).toBe(true);
  });
  it("starts on the first tick after its deadline without waiting for tracking", () => {
    const gate = new ReadinessGate();
    gate.arm(0);
    expect(gate.update({ nowMs: 8500, assessable: false }).start).toBe(true);
    expect(gate.update({ nowMs: 8600, assessable: false }).start).toBe(false);
  });
  it("fails safely on a backward or invalid wall clock and permits explicit retry", () => {
    const gate = new ReadinessGate();
    gate.arm(0);
    gate.update({ nowMs: 2000 });
    expect(gate.update({ nowMs: 1000 })).toMatchObject({
      phase: "failed",
      reason: "clock-reset",
      pending: false,
      start: false,
    });
    expect(gate.update({ nowMs: 9000 }).start).toBe(false);
    expect(gate.arm(10_000)).toMatchObject({
      phase: "countdown",
      reason: null,
    });
    expect(gate.update({ nowMs: Number.NaN })).toMatchObject({
      phase: "failed",
      reason: "clock-reset",
    });
    expect(new ReadinessGate().arm(-1)).toMatchObject({
      phase: "failed",
      reason: "clock-reset",
    });
  });
});
