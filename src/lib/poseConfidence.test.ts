import { describe, expect, it } from "vitest";
import { demoFrame } from "./demo";
import { assessArmTracking } from "./motion";
import { observedPoint, observationThreshold } from "./poseConfidence";
import { JOINT, type PoseFrame } from "./types";

const native = (): PoseFrame => {
  const frame = demoFrame(0);
  frame.estimator = { id: "rtmpose-m", scoreType: "simcc", minimumScore: 0.4 };
  frame.landmarks = frame.landmarks.map(({ x, y }) => ({ x, y, score: 0.55 }));
  return frame;
};

describe("estimator-specific observation evidence", () => {
  it("does not treat native scores as MediaPipe visibility", () => {
    const frame = native();
    expect(assessArmTracking(frame).left.assessable).toBe(true);
    const untagged = { ...frame, estimator: undefined };
    expect(assessArmTracking(untagged).left.assessable).toBe(false);
    frame.landmarks[JOINT.leftWrist] = {
      x: 0.5,
      y: 0.4,
      score: 0.2,
      visibility: 1,
    };
    expect(assessArmTracking(frame).left.assessable).toBe(false);
    expect(assessArmTracking(frame).right.assessable).toBe(true);
  });

  it("requires a valid explicit native-score policy and real in-frame points", () => {
    for (const minimumScore of [0, -1, NaN, Infinity, 1]) {
      const frame = native();
      frame.estimator!.minimumScore = minimumScore;
      expect(observationThreshold(frame)).toBe(Infinity);
      expect(assessArmTracking(frame).left.assessable).toBe(false);
    }
    const frame = native();
    for (const point of [
      { x: NaN, y: 0.4, score: 1 },
      { x: 1.1, y: 0.4, score: 1 },
      { x: 0.4, y: 0.4, score: NaN },
    ])
      expect(observedPoint(frame, point)).toBe(false);
  });

  it("keeps MediaPipe visibility and presence gates unchanged", () => {
    const frame = demoFrame(0);
    expect(
      observedPoint(frame, { x: 0.5, y: 0.5, visibility: 0.64, score: 1 }),
    ).toBe(false);
    expect(
      observedPoint(frame, { x: 0.5, y: 0.5, visibility: 0.9, presence: 0.1 }),
    ).toBe(false);
    expect(observedPoint(frame, { x: 0.5, y: 0.5, visibility: 0.9 })).toBe(
      true,
    );
  });
});
