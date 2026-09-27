import { describe, expect, it } from "vitest";
import { assessTrackingTrust, summarizeTrackingTrust } from "./trackingTrust";
import { JOINT, type PoseFrame } from "./types";
function frame(t: number): PoseFrame {
  const landmarks = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.2,
    visibility: 0.99,
  }));
  for (const [i, x, y] of [
    [11, 0.4, 0.3],
    [12, 0.6, 0.3],
    [13, 0.35, 0.48],
    [14, 0.65, 0.48],
    [15, 0.42, 0.4],
    [16, 0.58, 0.4],
    [23, 0.42, 0.7],
    [24, 0.58, 0.7],
  ])
    landmarks[i] = { x, y, visibility: 0.99 };
  return { t, width: 1280, height: 720, landmarks, inferenceMs: 8 };
}
const frames = (count = 20) =>
  Array.from({ length: count }, (_, i) => frame(i * 33));
describe("tracking consistency diagnostics", () => {
  it("requires reacquisition, never verifies identity, and never extrapolates beyond sampled footage", () => {
    const result = assessTrackingTrust(frames(), { durationMs: 700 });
    expect(result.samples[0].arms.left.status).toBe("uncertain");
    expect(result.samples[5].arms.left.status).toBe("trusted");
    expect(result.identityVerified).toBe(false);
    expect(
      result.uncertaintyIntervals.some(
        (i) =>
          i.startMs === 627 &&
          i.endMs === 700 &&
          i.reasons.includes("no-samples"),
      ),
    ).toBe(true);
  });
  it("flags high-confidence isolated wrist detours without changing source coordinates", () => {
    const input = frames(),
      before = JSON.stringify(input);
    input[8].landmarks[JOINT.leftWrist].x = 0.72;
    const snapshot = JSON.stringify(input),
      result = assessTrackingTrust(input, { durationMs: 627 });
    expect(JSON.stringify(input)).toBe(snapshot);
    expect(snapshot).not.toBe(before);
    expect(result.samples[8].arms.left.reasons).toContain(
      "isolated-wrist-detour",
    );
    expect(result.samples[8].assessedAtMs).toBe(input[9].t);
    expect(result.samples[8].arms.right.status).toBe("trusted");
    expect(result.samples[10].arms.left.status).toBe("uncertain");
  });
  it("does not equate fast monotonic extension with a tracking error", () => {
    const input = frames();
    for (let i = 5; i < 12; i++)
      input[i].landmarks[JOINT.leftWrist].x = 0.42 + (i - 5) * 0.045;
    for (let i = 12; i < input.length; i++)
      input[i].landmarks[JOINT.leftWrist].x = 0.69;
    expect(
      assessTrackingTrust(input, { durationMs: 627 }).samples.some((s) =>
        s.arms.left.reasons.includes("isolated-wrist-detour"),
      ),
    ).toBe(false);
  });
  it("flags bilateral assignment discontinuity without swapping anatomical labels", () => {
    const input = frames();
    for (let i = 8; i < input.length; i++)
      [input[i].landmarks[15], input[i].landmarks[16]] = [
        input[i].landmarks[16],
        input[i].landmarks[15],
      ];
    const snapshot = JSON.stringify(input),
      result = assessTrackingTrust(input, { durationMs: 627 });
    expect(result.samples[8].arms.left.reasons).toContain(
      "bilateral-assignment-discontinuity",
    );
    expect(result.samples[8].arms.right.reasons).toContain(
      "bilateral-assignment-discontinuity",
    );
    expect(JSON.stringify(input)).toBe(snapshot);
  });
  it("keeps low-confidence arms independent", () => {
    const input = frames();
    input[8].landmarks[16].visibility = 0.2;
    const result = assessTrackingTrust(input, { durationMs: 627 });
    expect(result.samples[8].arms.right.status).toBe("uncertain");
    expect(result.samples[8].arms.left.status).toBe("trusted");
  });
  it("marks a frame gap uncertain and requires fresh observations", () => {
    const input = frames();
    for (let i = 8; i < input.length; i++) input[i].t += 300;
    const result = assessTrackingTrust(input, { durationMs: 927 });
    expect(
      result.uncertaintyIntervals.some(
        (i) =>
          i.startMs <= 231 && i.endMs >= 564 && i.reasons.includes("frame-gap"),
      ),
    ).toBe(true);
    expect(result.samples[9].arms.left.status).toBe("uncertain");
    expect(result.samples[14].arms.left.status).toBe("trusted");
  });
  it("partitions the round per arm without overlapping or missing time", () => {
    for (const input of [[], [frame(100)], frames()]) {
      const result = assessTrackingTrust(input, { durationMs: 700 });
      for (const hand of ["left", "right"]) {
        const list = result.intervals.filter((i) => i.hand === hand);
        expect(list[0].startMs).toBe(0);
        expect(list.at(-1)!.endMs).toBe(700);
        list
          .slice(1)
          .forEach((interval, i) =>
            expect(interval.startMs).toBe(list[i].endMs),
          );
      }
      expect(summarizeTrackingTrust(input, 700)).toEqual(
        result.uncertaintyIntervals,
      );
    }
  });
  it("rejects nonmonotonic timestamps rather than silently reordering observations", () => {
    expect(() =>
      assessTrackingTrust([frame(30), frame(20)], { durationMs: 100 }),
    ).toThrow(/timestamps/);
    expect(() =>
      assessTrackingTrust([frame(30), frame(30)], { durationMs: 100 }),
    ).toThrow(/timestamps/);
  });
  it("reports and excludes a late in-flight pose after the round deadline", () => {
    const input = frames();
    const result = assessTrackingTrust(input, { durationMs: 620 });
    expect(result.outsideRoundFrames).toBe(1);
    expect(result.samples.at(-1)!.t).toBe(594);
    expect(Math.max(...result.intervals.map((i) => i.endMs))).toBe(620);
    expect(input.at(-1)!.t).toBe(627);
  });
});
