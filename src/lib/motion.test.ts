import { describe, expect, it } from "vitest";
import {
  anatomicalRole,
  armGeometry,
  assessArmTracking,
  assessQuality,
  imageDistance,
  MotionEngine,
} from "./motion";
import { JOINT, type PoseFrame, type PunchEvent } from "./types";
import { demoFrame } from "./demo";

const lerp = (a: number, b: number, fraction: number) => a + (b - a) * fraction;

function frame(
  t: number,
  extension = 0,
  hand: "left" | "right" = "left",
): PoseFrame {
  const landmarks = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    visibility: 0.99,
  }));
  const point = (index: number, x: number, y: number) => {
    landmarks[index] = { x, y, visibility: 0.99 };
  };
  point(JOINT.nose, 0.5, 0.17);
  point(JOINT.leftShoulder, 0.42, 0.3);
  point(JOINT.rightShoulder, 0.58, 0.3);
  point(JOINT.leftHip, 0.44, 0.67);
  point(JOINT.rightHip, 0.56, 0.67);
  for (const side of ["left", "right"] as const) {
    const amount = side === hand ? extension : 0;
    const mirror = side === "left" ? (x: number) => x : (x: number) => 1 - x;
    point(
      side === "left" ? JOINT.leftElbow : JOINT.rightElbow,
      mirror(lerp(0.35, 0.3, amount)),
      lerp(0.44, 0.31, amount),
    );
    point(
      side === "left" ? JOINT.leftWrist : JOINT.rightWrist,
      mirror(lerp(0.4, 0.14, amount)),
      lerp(0.28, 0.31, amount),
    );
  }
  return { t, width: 1280, height: 720, landmarks, inferenceMs: 8 };
}

function cycle(start = 0, hand: "left" | "right" = "left"): PoseFrame[] {
  return [
    0, 0, 0, 0, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1, 1, 0.9, 0.75, 0.6, 0.45,
    0.3, 0.15, 0, 0, 0, 0,
  ].map((value, index) => frame(start + index * 40, value, hand));
}

const collect = (engine: MotionEngine, frames: PoseFrame[]): PunchEvent[] =>
  frames.flatMap((value) => engine.update(value).events);
const engine = () => new MotionEngine({ stance: "orthodox", calibrated: true });

describe("upper-body acquisition quality", () => {
  it("reports each arm independently without substituting for a hidden wrist", () => {
    expect(assessQuality(frame(0)).assessable).toBe(true);
    const hidden = frame(0);
    hidden.landmarks[JOINT.rightWrist].visibility = 0.3;
    expect(assessQuality(hidden)).toMatchObject({
      assessable: true,
      label: "Left arm visible",
    });
    expect(assessArmTracking(hidden)).toMatchObject({
      left: { assessable: true, status: "tracking", reason: null },
      right: {
        assessable: false,
        status: "hidden",
        projectedAngle: null,
        projectedReach: null,
        minimumVisibility: 0.3,
      },
    });
    hidden.landmarks[JOINT.leftWrist].visibility = 0.3;
    expect(assessQuality(hidden).assessable).toBe(false);
    expect(assessQuality({ ...hidden, landmarks: [] }).visibleJoints).toBe(0);
    expect(assessArmTracking({ ...hidden, landmarks: [] }).left).toMatchObject({
      assessable: false,
      status: "invalid",
      minimumVisibility: null,
    });
  });

  it("rejects a clipped or uncertain arm and invalid shared frame geometry", () => {
    const clipped = frame(0);
    clipped.landmarks[JOINT.leftWrist].x = -0.1;
    expect(assessArmTracking(clipped).left.assessable).toBe(false);
    expect(assessArmTracking(clipped).right.assessable).toBe(true);
    const unknown = frame(0);
    delete unknown.landmarks[JOINT.leftElbow].visibility;
    expect(assessArmTracking(unknown).left.assessable).toBe(false);
    expect(assessQuality({ ...frame(0), t: NaN }).assessable).toBe(false);
    expect(assessQuality({ ...frame(0), width: 0 }).assessable).toBe(false);
    const clippedTorso = frame(0);
    clippedTorso.landmarks[JOINT.rightHip].y = 1.001;
    expect(assessArmTracking(clippedTorso)).toMatchObject({
      left: { assessable: false, status: "invalid" },
      right: { assessable: false, status: "invalid" },
    });
  });

  it("does not require the nose for arm geometry, but requires torso anchors", () => {
    const hiddenFace = frame(0);
    hiddenFace.landmarks[JOINT.nose].visibility = 0.1;
    expect(assessQuality(hiddenFace)).toMatchObject({
      assessable: true,
      visibleJoints: 8,
      totalJoints: 8,
    });
    hiddenFace.landmarks[JOINT.leftShoulder].visibility = 0.1;
    expect(assessQuality(hiddenFace).assessable).toBe(false);
  });

  it("withholds projected metrics when an arm segment collapses in the image", () => {
    const collapsed = frame(0);
    collapsed.landmarks[JOINT.rightElbow] = {
      ...collapsed.landmarks[JOINT.rightShoulder],
    };
    // Predicted world-space positions cannot rescue absent projected evidence.
    collapsed.worldLandmarks = frame(0, 1, "right").landmarks;
    expect(assessArmTracking(collapsed)).toMatchObject({
      left: { assessable: true, status: "tracking" },
      right: {
        assessable: false,
        status: "foreshortened",
        projectedAngle: null,
        projectedReach: null,
      },
    });
    const normal = frame(0, 0.7);
    expect(assessArmTracking(normal).left.projectedAngle).toBeCloseTo(
      armGeometry(normal, "left").angle,
    );
    expect(assessArmTracking(normal).left.projectedReach).toBeCloseTo(
      armGeometry(normal, "left").reach,
    );
  });
});

describe("anatomical identity and projected geometry", () => {
  it("maps physical hand to stance without relying on horizontal ordering", () => {
    expect(anatomicalRole("left", "orthodox")).toBe("lead");
    expect(anatomicalRole("right", "orthodox")).toBe("rear");
    expect(anatomicalRole("left", "southpaw")).toBe("rear");
    expect(anatomicalRole("right", "southpaw")).toBe("lead");
    const orthodox = collect(engine(), cycle());
    const southpaw = collect(
      new MotionEngine({ stance: "southpaw", calibrated: true }),
      cycle(),
    );
    expect(orthodox[0]).toMatchObject({
      hand: "left",
      role: "lead",
      label: "jab",
    });
    expect(southpaw[0]).toMatchObject({
      hand: "left",
      role: "rear",
      label: "cross",
    });
    // Mirroring the UI does not change the physical, unmirrored input frames.
    expect(collect(engine(), cycle())).toEqual(orthodox);
    // Even a reflected input keeps the anatomical landmark IDs, not x sorting.
    const reflected = cycle().map((value) => ({
      ...value,
      landmarks: value.landmarks.map((point) => ({ ...point, x: 1 - point.x })),
    }));
    expect(collect(engine(), reflected)[0]).toMatchObject({
      hand: "left",
      role: "lead",
      label: "jab",
    });
  });

  it("aspect-corrects distances and preserves angles across equivalent canvas widths", () => {
    expect(
      imageDistance({ x: 0, y: 0 }, { x: 0.1, y: 0 }, 200, 100),
    ).toBeCloseTo(0.2);
    expect(
      imageDistance({ x: 0, y: 0 }, { x: 0, y: 0.1 }, 200, 100),
    ).toBeCloseTo(0.1);
    const original = frame(0, 0.7);
    const padded = {
      ...original,
      width: 2560,
      landmarks: original.landmarks.map((point) => ({
        ...point,
        x: point.x / 2,
      })),
    };
    expect(armGeometry(padded, "left").angle).toBeCloseTo(
      armGeometry(original, "left").angle,
    );
    expect(armGeometry(padded, "left").reach).toBeCloseTo(
      armGeometry(original, "left").reach,
    );
    const paddedCycle = cycle().map((value) => ({
      ...value,
      width: 2560,
      landmarks: value.landmarks.map((point) => ({ ...point, x: point.x / 2 })),
    }));
    expect(collect(engine(), paddedCycle)).toEqual(collect(engine(), cycle()));
  });
});

describe("conservative causal event detection", () => {
  it.each([30, 60])(
    "recognizes both physical hands in the UI demo at %i fps",
    (fps) => {
      const frames = Array.from(
        { length: Math.floor(9600 / (1000 / fps)) },
        (_, index) => demoFrame((index * 1000) / fps),
      );
      const events = collect(engine(), frames);
      expect(events.map((event) => event.hand)).toEqual([
        "left",
        "right",
        "left",
        "right",
      ]);
      expect(events.map((event) => event.label)).toEqual([
        "jab",
        "cross",
        "jab",
        "cross",
      ]);
      expect(events.every((event) => event.experimental)).toBe(true);
    },
  );

  it("emits exactly once after extension and recovery, with descriptive return evidence", () => {
    const detector = engine();
    const events = collect(detector, cycle());
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      hand: "left",
      label: "jab",
      guardReturn: "returned",
      experimental: true,
    });
    expect(events[0].startMs).toBeLessThan(events[0].peakMs);
    expect(events[0].peakMs).toBeLessThan(events[0].endMs);
    expect(events[0].detectedAtMs).toBeGreaterThan(events[0].endMs);
    expect(events[0].extension).toBeGreaterThan(0.45);
    expect(events[0].score).toBeGreaterThan(0);
    expect(events[0].score).toBeLessThan(1);
    expect(
      collect(
        detector,
        Array.from({ length: 20 }, (_, index) => frame(920 + index * 40)),
      ),
    ).toEqual([]);
  });

  it("does not count a held extension, a partial gesture, or idle movement", () => {
    expect(collect(engine(), cycle().slice(0, 13))).toEqual([]);
    expect(
      collect(
        engine(),
        cycle().map((value) => frame(value.t, 0.25)),
      ),
    ).toEqual([]);
    expect(
      collect(
        engine(),
        cycle().map((value) => frame(value.t)),
      ),
    ).toEqual([]);
    const held = [
      ...cycle().slice(0, 13),
      ...Array.from({ length: 50 }, (_, index) => frame(520 + index * 40, 1)),
    ];
    expect(collect(engine(), held)).toEqual([]);
  });

  it("can rearm for a second repetition and independently detect the rear hand", () => {
    const detector = engine();
    const events = collect(detector, [
      ...cycle(),
      ...cycle(920),
      ...cycle(1840, "right"),
    ]);
    expect(events.map((event) => event.label)).toEqual(["jab", "jab", "cross"]);
    expect(new Set(events.map((event) => event.id)).size).toBe(3);
  });

  it("preserves the resting reference with smaller per-frame movement at higher cadence", () => {
    const original = cycle();
    const dense = original.flatMap((value, index) => {
      const next = original[index + 1];
      if (!next) return [value];
      return [
        value,
        {
          ...value,
          t: value.t + 20,
          landmarks: value.landmarks.map((point, joint) => ({
            ...point,
            x: (point.x + next.landmarks[joint].x) / 2,
            y: (point.y + next.landmarks[joint].y) / 2,
          })),
        },
      ];
    });
    expect(collect(engine(), dense)).toHaveLength(1);
  });

  it("never emits uncalibrated events, and explicit reset discards a partial movement", () => {
    expect(
      collect(
        new MotionEngine({ stance: "orthodox", calibrated: false }),
        cycle(),
      ),
    ).toEqual([]);
    const detector = engine();
    collect(detector, cycle().slice(0, 13));
    detector.reset();
    expect(collect(detector, cycle().slice(13))).toEqual([]);
    detector.reset({ stance: "southpaw", calibrated: true });
    expect(collect(detector, cycle())[0]).toMatchObject({
      hand: "left",
      label: "cross",
    });
  });

  it("drops the entire candidate after a timestamp gap, then recovers on a fresh cycle", () => {
    const detector = engine();
    collect(detector, cycle().slice(0, 13));
    const interrupted = detector.update(frame(1000, 0.8));
    expect(interrupted.quality.assessable).toBe(false);
    expect(interrupted.activeHand).toBe(null);
    expect(interrupted.events).toEqual([]);
    expect(
      collect(
        detector,
        cycle()
          .slice(13)
          .map((value) => ({ ...value, t: value.t + 520 })),
      ),
    ).toEqual([]);
    expect(collect(detector, cycle(1480))).toHaveLength(1);
  });

  it("rejects duplicate/backward timestamps and clears pending activity", () => {
    const detector = engine();
    collect(detector, cycle().slice(0, 12));
    expect(detector.update(frame(440, 1))).toMatchObject({
      events: [],
      activeHand: null,
    });
    expect(collect(detector, cycle().slice(12))).toEqual([]);
    detector.reset();
    collect(detector, cycle().slice(0, 12));
    expect(detector.update(frame(100, 1)).quality.assessable).toBe(false);
  });

  it("rejects an occluded required joint during the critical phase without later stale counts", () => {
    const detector = engine();
    const frames = cycle();
    frames[11].landmarks[JOINT.leftWrist].visibility = 0.1;
    expect(collect(detector, frames)).toEqual([]);
    expect(collect(detector, cycle(920))).toHaveLength(1);
  });

  it.each(["left", "right"] as const)(
    "keeps a visible %s-arm movement when only the guarding arm disappears",
    (hand) => {
      const oppositeWrist =
        hand === "left" ? JOINT.rightWrist : JOINT.leftWrist;
      const oppositeElbow =
        hand === "left" ? JOINT.rightElbow : JOINT.leftElbow;
      const frames = cycle(0, hand);
      // This fails the former all-joints gate during the outgoing and peak phase.
      for (const value of frames.slice(7, 15)) {
        value.landmarks[oppositeWrist].visibility = 0.61;
        value.landmarks[oppositeElbow].visibility = 0.63;
      }
      const events = collect(engine(), frames);
      expect(events).toHaveLength(1);
      expect(events[0].hand).toBe(hand);
      expect(events[0].label).toBe(hand === "left" ? "jab" : "cross");
      // Opposite-arm uncertainty must not alter the visible movement's evidence.
      expect(events).toEqual(collect(engine(), cycle(0, hand)));
    },
  );

  it.each(["left", "right"] as const)(
    "discards the %s-arm candidate if its own wrist is hidden and requires a fresh baseline",
    (hand) => {
      const detector = engine();
      const frames = cycle(0, hand);
      const wrist = hand === "left" ? JOINT.leftWrist : JOINT.rightWrist;
      frames[11].landmarks[wrist].visibility = 0.1;
      expect(assessQuality(frames[11]).assessable).toBe(true);
      expect(collect(detector, frames)).toEqual([]);
      expect(collect(detector, cycle(920, hand))).toHaveLength(1);
    },
  );

  it("does not reinterpret collapsed active-arm projection as extension", () => {
    const frames = cycle(0, "right");
    for (const value of frames.slice(7, 15)) {
      value.worldLandmarks = value.landmarks.map((point) => ({ ...point }));
      value.landmarks[JOINT.rightWrist] = {
        ...value.landmarks[JOINT.rightElbow],
      };
    }
    expect(collect(engine(), frames)).toEqual([]);
  });

  it("requires a new resting baseline after image aspect changes", () => {
    const detector = engine();
    collect(detector, cycle().slice(0, 12));
    const changed = detector.update({ ...frame(480, 1), width: 720 });
    expect(changed).toMatchObject({ events: [], activeHand: null });
    expect(changed.quality.label).toBe("Camera framing changed");
    expect(collect(detector, cycle().slice(13))).toEqual([]);
  });
});
