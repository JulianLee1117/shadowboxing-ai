import { describe, expect, it } from "vitest";
import {
  anatomicalRole,
  armGeometry,
  assessArmTracking,
  assessQuality,
  imageDistance,
  MOTION_LIMITS,
  robustOutboundPath,
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

// One strict extension sample, with near-peak evidence immediately before/after.
// This is a state-machine fixture, not simulated training or an accuracy claim.
const supportedPeak = [
  0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.63, 0.7, 0.63, 0.6, 0.5, 0.4,
  0.3, 0.2, 0.1, 0, 0, 0,
];

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
  it.each([15, 30, 60])(
    "accepts one strict peak with observed outbound/inbound support at %i fps",
    (fps) => {
      for (const hand of ["left", "right"] as const) {
        const frames = supportedPeak.map((p, i) =>
          frame((i * 1000) / fps, p, hand),
        );
        const origin = armGeometry(frames[0], hand);
        expect(
          frames.filter((value) => {
            const sample = armGeometry(value, hand);
            return (
              sample.reach - origin.reach >= MOTION_LIMITS.minimumExtension &&
              sample.angle >= MOTION_LIMITS.minimumPeakAngle
            );
          }),
        ).toHaveLength(1);
        const events = collect(engine(), frames);
        expect(events).toHaveLength(1);
        expect(events[0]).toMatchObject({ hand, guardReturn: "returned" });
        expect(events[0].peakMs).toBeLessThan(events[0].endMs);
      }
    },
  );

  it("requires support on both sides of a lone peak and never bridges an occluded neighbor", () => {
    for (const unsupportedIndex of [11, 13]) {
      const unsupported = supportedPeak.map((p, i) =>
        frame(i * 40, i === unsupportedIndex ? 0.2 : p),
      );
      expect(collect(engine(), unsupported)).toEqual([]);
      const hidden = supportedPeak.map((p, i) => frame(i * 40, p));
      hidden[unsupportedIndex].landmarks[JOINT.leftWrist].visibility = 0.1;
      expect(collect(engine(), hidden)).toEqual([]);
    }
  });

  it.each([15, 24, 30, 60])(
    "rejects short isolated spikes and subthreshold pulses across sample phases at %i fps",
    (fps) => {
      for (const phase of [0, 0.25, 0.5, 0.75]) {
        for (const kind of ["spike", "partial"] as const) {
          const frames = Array.from(
            { length: Math.ceil(fps * 1.5) },
            (_, i) => {
              const t = ((i + phase) * 1000) / fps;
              const p =
                kind === "spike"
                  ? t >= 700 && t <= 710
                    ? 1
                    : 0
                  : Math.max(0, 0.63 * (1 - Math.abs(t - 750) / 250));
              return frame(t, p);
            },
          );
          expect(collect(engine(), frames)).toEqual([]);
        }
      }
    },
  );

  it("measures straightness on the outward path even if elbow flexion lags retraction", () => {
    const frames = cycle();
    for (const value of frames.slice(13, 17)) {
      const shoulder = value.landmarks[JOINT.leftShoulder];
      const wrist = value.landmarks[JOINT.leftWrist];
      value.landmarks[JOINT.leftElbow] = {
        x: (shoulder.x + wrist.x) / 2,
        y: (shoulder.y + wrist.y) / 2,
        visibility: 0.99,
      };
    }
    expect(collect(engine(), frames)).toHaveLength(1);
  });

  it("still rejects a visibly circuitous outward path with extended peak samples", () => {
    const frames = Array.from({ length: 5 }, (_, i) => frame(i * 40));
    for (const [x, y] of [
      [0.2, 0.3],
      [-0.2, 0.4],
      [0.3, 0.6],
      [0.8, 0],
      [0.9, 0],
      [0.9, 0],
      [0.7, 0],
      [0.4, 0],
    ]) {
      const value = frame(frames.length * 40);
      const shoulder = value.landmarks[JOINT.leftShoulder];
      const wrist = {
        x: shoulder.x + (x * 0.37) / (1280 / 720),
        y: shoulder.y + y * 0.37,
        visibility: 0.99,
      };
      value.landmarks[JOINT.leftWrist] = wrist;
      value.landmarks[JOINT.leftElbow] = {
        x: (shoulder.x + wrist.x) / 2,
        y: (shoulder.y + wrist.y) / 2,
        visibility: 0.99,
      };
      frames.push(value);
    }
    for (let i = 0; i < 5; i++) frames.push(frame(frames.length * 40));
    expect(collect(engine(), frames)).toEqual([]);
  });

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
      // The supported retraction is detected before the later spatial return.
      guardReturn: "not-observed",
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

  it.each([15, 30, 60])(
    "uses observed retraction to separate a quick jab-jab-cross at %i fps",
    (fps) => {
      const frames = [
        ...cycle().slice(0, 20),
        ...cycle().slice(5, 20),
        ...cycle(0, "right").slice(5),
      ].map((value, i) => ({ ...value, t: (i * 1000) / fps }));
      const events = collect(engine(), frames);
      expect(events.map((event) => event.label)).toEqual([
        "jab",
        "jab",
        "cross",
      ]);
      expect(events[1].startMs).toBeGreaterThanOrEqual(events[0].endMs);
    },
  );

  it("clears recovery readiness after active-arm occlusion before a quick repeat", () => {
    const frames = [...cycle().slice(0, 20), ...cycle().slice(5)].map(
      (value, i) => ({ ...value, t: i * 40 }),
    );
    frames[20].landmarks[JOINT.leftWrist].visibility = 0.1;
    expect(collect(engine(), frames)).toHaveLength(1);
  });

  it("follows a confirmed hand's continued inward return without following its next extension", () => {
    const frames = [
      ...cycle().slice(0, 20),
      ...supportedPeak.slice(5).map((p) => frame(0, p)),
    ].map((value, i) => {
      // Preserve angles but use a smaller projected arm excursion. Freezing a
      // partially returned reference would wrongly erase the next strict peak.
      const shoulder = value.landmarks[JOINT.leftShoulder];
      for (const joint of [JOINT.leftElbow, JOINT.leftWrist]) {
        const point = value.landmarks[joint];
        point.x = shoulder.x + (point.x - shoulder.x) * 0.55;
        point.y = shoulder.y + (point.y - shoulder.y) * 0.55;
      }
      return { ...value, t: i * 40 };
    });
    const events = collect(engine(), frames);
    expect(events).toHaveLength(2);
    expect(events[1].extension).toBeCloseTo(
      (armGeometry(frame(0, 0.7), "left").reach -
        armGeometry(frame(0), "left").reach) *
        0.55,
    );
    expect(events[1].extension).toBeGreaterThanOrEqual(
      MOTION_LIMITS.minimumExtension,
    );
    expect(events[1].extension).toBeLessThan(0.5);
  });

  it("does not turn return jitter or a held extension into repeated events", () => {
    for (const kind of ["jitter", "held"] as const) {
      const frames = [
        ...cycle().slice(0, 20),
        ...Array.from({ length: 60 }, (_, i) =>
          frame(800 + i * 40, kind === "held" ? 1 : i % 3 === 0 ? 0.15 : 0),
        ),
      ];
      expect(collect(engine(), frames)).toHaveLength(1);
    }
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

describe("time-based observed trough acquisition", () => {
  function movingFrames(
    fps: number,
    points: [number, number][] = [
      [0, 0.4],
      [100, 0.1],
      [400, 1],
      [500, 1],
      [800, 0],
      [1000, 0],
    ],
  ): PoseFrame[] {
    return Array.from(
      { length: Math.floor((points.at(-1)![0] * fps) / 1000) + 1 },
      (_, i) => {
        const t = (i * 1000) / fps;
        const next = points.findIndex(([time]) => time > t);
        if (next < 0) return frame(t, points.at(-1)![1]);
        const [fromT, from] = points[next - 1];
        const [toT, to] = points[next];
        return frame(t, lerp(from, to, (t - fromT) / (toT - fromT)));
      },
    );
  }

  it.each([15, 30, 60])(
    "recognizes one supported moving-start cycle without a stationary guard at %i fps",
    (fps) => {
      const frames = movingFrames(fps);
      const events = collect(engine(), frames);
      expect(events).toHaveLength(1);
      const originReach = Math.min(
        ...frames
          .filter((f) => f.t <= 150)
          .map((f) => armGeometry(f, "left").reach),
      );
      const peakReach = Math.max(
        ...frames.map((f) => armGeometry(f, "left").reach),
      );
      expect(events[0].extension).toBeCloseTo(peakReach - originReach);
      expect(events[0].hand).toBe("left");
    },
  );

  it.each([15, 30, 60])(
    "refreshes an acquired reference only to a deeper observed flexed trough at %i fps",
    (fps) => {
      const frames = movingFrames(fps, [
        [0, 0.3],
        [200, 0.3],
        [800 / 3, 0.25],
        [400, 0.1],
        [2000 / 3, 1],
        [800, 1],
        [1100, 0],
        [1200, 0],
      ]);
      const events = collect(engine(), frames);
      expect(events).toHaveLength(1);
      expect(events[0].extension).toBeCloseTo(
        armGeometry(frame(0, 1), "left").reach -
          armGeometry(frame(0, 0.1), "left").reach,
      );
    },
  );

  it.each([15, 30, 60])(
    "does not turn trough jitter, a held reach, or a brief spike into cycles at %i fps",
    (fps) => {
      for (const points of [
        [
          [0, 0.4],
          [100, 0.1],
          [200, 0.3],
          [300, 0.1],
          [400, 0.3],
          [600, 0],
        ],
        [
          [0, 0.4],
          [100, 0.1],
          [400, 1],
          [1500, 1],
        ],
        [
          [0, 0.4],
          [100, 0.1],
          [350, 0.3],
          [400, 1],
          [410, 0.1],
          [1000, 0],
        ],
      ] as [number, number][][]) {
        expect(collect(engine(), movingFrames(fps, points))).toEqual([]);
      }
    },
  );

  it.each([15, 30, 60])(
    "never joins trough evidence across active-arm occlusion or a timing reset at %i fps",
    (fps) => {
      for (const interruption of [
        "hidden",
        "gap",
        "duplicate-time",
        "reset",
      ] as const) {
        const frames = movingFrames(fps);
        const troughIndex = frames.findIndex((f) => f.t >= 100);
        if (interruption === "hidden")
          frames[troughIndex].landmarks[JOINT.leftWrist].visibility = 0.2;
        if (interruption === "gap")
          for (let i = troughIndex; i < frames.length; i++) frames[i].t += 250;
        if (interruption === "duplicate-time")
          frames[troughIndex].t = frames[troughIndex - 1].t;
        const detector = engine();
        const events = frames.flatMap((f, i) => {
          if (interruption === "reset" && i === troughIndex) detector.reset();
          return detector.update(f).events;
        });
        expect(events, interruption).toEqual([]);
      }
    },
  );

  it("expires old moving-acquisition samples even when individual frame gaps remain valid", () => {
    const values: [number, number][] = [
      [0, 0.4],
      [180, 0.1],
      [360, 0.3],
      [400, 0.45],
      [440, 0.6],
      [480, 0.75],
      [520, 0.9],
      [560, 1],
      [600, 1],
      [640, 0.8],
      [680, 0.6],
      [720, 0.4],
      [760, 0.2],
      [800, 0],
      [840, 0],
      [880, 0],
    ];
    expect(
      collect(
        engine(),
        values.map(([t, x]) => frame(t, x)),
      ),
    ).toEqual([]);
  });

  it("requires elapsed observation time even when three points have arrived", () => {
    const frames = [frame(0, 0.4), frame(10, 0.1), frame(20, 0.3)];
    const detector = engine();
    expect(collect(detector, frames)).toEqual([]);
    // A spike immediately afterward must not turn that short window into a cycle.
    expect(
      collect(detector, [
        frame(30, 1),
        frame(40, 0),
        frame(50, 0),
        frame(60, 0),
      ]),
    ).toEqual([]);
  });
});

const sample = (t: number, x: number, y = 0, angle = 170) => ({
  t,
  wrist: { x, y },
  reach: Math.hypot(x, y),
  angle,
});
const origin = sample(0, 0, 0, 45);
function measure(points: ReturnType<typeof sample>[]) {
  return robustOutboundPath(points, origin, points.at(-1)!.t);
}
describe("outbound path measurement with an isolated detour", () => {
  it("preserves coherent observed straight paths", () => {
    const r = measure([
      origin,
      sample(33, 0.2),
      sample(66, 0.4),
      sample(99, 0.6),
      sample(132, 0.8),
    ]);
    expect(r.removed).toEqual([]);
    expect(r.path).toBeCloseTo(0.8);
  });
  it("removes one reverse detour with multiple independent strict peak observations", () => {
    const r = measure([
      origin,
      sample(33, 0.5),
      sample(66, 0.52, 0.8),
      sample(99, 0.55),
      sample(132, 0.8),
    ]);
    expect(r.removed).toEqual([66]);
    expect(r.path).toBeCloseTo(0.8);
  });
  it("does not remove a lone spike to manufacture supported peak evidence", () => {
    const r = measure([
      origin,
      sample(33, 0.1),
      sample(66, 0.12, 0.8),
      sample(99, 0.15),
      sample(132, 0.2),
    ]);
    expect(r.removed).toEqual([]);
  });
  it("does not alter a selected observed peak", () => {
    const r = measure([
      origin,
      sample(33, 0.5),
      sample(66, 0.6),
      sample(99, 0.62, 0.8),
    ]);
    expect(r.removed).toEqual([]);
  });
  it("does not smooth a sustained curved path", () => {
    const r = measure([
      origin,
      sample(33, 0.2, 0.2),
      sample(66, 0.4, 0.4),
      sample(99, 0.6, 0.3),
      sample(132, 0.8),
    ]);
    expect(r.removed).toEqual([]);
    expect(r.path).toBeGreaterThan(0.8);
  });
  it("does not join a confidence/time gap", () => {
    const r = measure([
      origin,
      sample(33, 0.5),
      sample(166, 0.52, 0.8),
      sample(199, 0.55),
      sample(232, 0.8),
    ]);
    expect(r.removed).toEqual([]);
  });
  it("rejects repeated zigzags rather than cleaning all evidence", () => {
    const r = measure([
      origin,
      sample(33, 0.5),
      sample(66, 0.52, 0.8),
      sample(99, 0.55),
      sample(132, 0.6),
      sample(165, 0.62, 0.8),
      sample(198, 0.65),
      sample(231, 0.8),
    ]);
    expect(r.removed).toEqual([]);
  });
  it("leaves sharp but non-reversing physical direction changes measured", () => {
    const r = measure([
      origin,
      sample(33, 0.5),
      sample(66, 0.52, 0.4),
      sample(99, 0.9, 0.8),
      sample(132, 1.3, 0.8),
    ]);
    expect(r.removed).toEqual([]);
  });
});

it("does not remove a sustained two-sample detour or mutate the observed path", () => {
  const points = [
    origin,
    sample(33, 0.5),
    sample(66, 0.52, 0.8),
    sample(99, 0.54, 0.8),
    sample(132, 0.55),
    sample(165, 0.8),
  ];
  const original = structuredClone(points);
  expect(measure(points).removed).toEqual([]);
  expect(points).toEqual(original);
});

describe("observed recovery independent of exact prior guard", () => {
  function trace(
    fps: number,
    points: [number, number][],
    hand: "left" | "right" = "left",
  ) {
    const end = points.at(-1)![0];
    return Array.from(
      { length: Math.floor((end * fps) / 1000) + 1 },
      (_, i) => {
        const t = (i * 1000) / fps;
        let j = 0;
        while (j + 1 < points.length - 1 && points[j + 1][0] < t) j += 1;
        const [at, a] = points[j],
          [bt, b] = points[j + 1];
        return frame(t, lerp(a, b, (t - at) / (bt - at)), hand);
      },
    );
  }

  it.each([15, 30, 60])(
    "separates two supported cycles with a partial flexed return at %i fps",
    (fps) => {
      const points: [number, number][] = [
        [0, 0],
        [200, 0],
        [450, 1],
        [550, 1],
        [750, 0.35],
        [850, 0.35],
        [1100, 1],
        [1200, 1],
        [1450, 0.35],
        [1550, 0.35],
      ];
      for (const hand of ["left", "right"] as const) {
        const events = collect(engine(), trace(fps, points, hand));
        expect(events.map((e) => e.hand)).toEqual([hand, hand]);
        expect(events[0].endMs).toBeLessThan(events[1].startMs);
        expect(events[0].guardReturn).toBe("not-observed");
      }
    },
  );

  it.each([15, 30, 60])(
    "requires retraction and flexion, not a held or slightly shortened extension at %i fps",
    (fps) => {
      for (const returned of [1, 0.7, 0.6]) {
        const points: [number, number][] = [
          [0, 0],
          [200, 0],
          [450, 1],
          [550, 1],
          [750, returned],
          [2400, returned],
        ];
        expect(collect(engine(), trace(fps, points)), String(returned)).toEqual(
          [],
        );
      }
    },
  );

  it.each([15, 30, 60])(
    "does not turn a subthreshold partial extension into an event at %i fps",
    (fps) => {
      const points: [number, number][] = [
        [0, 0],
        [200, 0],
        [450, 0.5],
        [550, 0.5],
        [750, 0.2],
        [850, 0.2],
        [1100, 0.5],
        [1200, 0.5],
        [1450, 0.2],
        [1600, 0.2],
      ];
      expect(collect(engine(), trace(fps, points))).toEqual([]);
    },
  );

  it("does not bridge a hidden arm between observed extension and partial recovery", () => {
    const frames = trace(30, [
      [0, 0],
      [200, 0],
      [450, 1],
      [550, 1],
      [750, 0.35],
      [850, 0.35],
      [1200, 0.35],
    ]);
    const hidden = frames.find((f) => f.t >= 600)!;
    hidden.landmarks[JOINT.leftWrist].visibility = 0.2;
    expect(collect(engine(), frames)).toEqual([]);
  });
  it.each([15, 30, 60])(
    "retains a coherently flexing static guard before a fast departure at %i fps",
    (fps) => {
      for (const hand of ["left", "right"] as const) {
        const points: [number, number][] = [
          [0, 0.2],
          [200, 0.2],
          [700, 0.15],
          [900, 1],
          [1000, 1],
          [1250, 0.15],
          [1400, 0.15],
        ];
        const frames = trace(fps, points, hand).map((f) => {
          if (f.t < 300) return f;
          const shoulder =
            f.landmarks[
              hand === "left" ? JOINT.leftShoulder : JOINT.rightShoulder
            ];
          const theta = hand === "left" ? -0.7 : 0.7;
          for (const joint of hand === "left"
            ? [JOINT.leftElbow, JOINT.leftWrist]
            : [JOINT.rightElbow, JOINT.rightWrist]) {
            const p = f.landmarks[joint];
            const x = ((p.x - shoulder.x) * f.width) / f.height,
              y = p.y - shoulder.y;
            p.x =
              shoulder.x +
              ((x * Math.cos(theta) - y * Math.sin(theta)) * f.height) /
                f.width;
            p.y = shoulder.y + x * Math.sin(theta) + y * Math.cos(theta);
          }
          return f;
        });
        const events = collect(engine(), frames);
        expect(events.map((e) => e.hand)).toEqual([hand]);
      }
    },
  );

  it.each([15, 30, 60])(
    "recognizes a real extension immediately after rejecting a smaller guard gesture at %i fps",
    (fps) => {
      const points: [number, number][] = [
        [0, 0.2],
        [200, 0.2],
        [360, 0.5],
        [460, 0.5],
        [500, 0.1],
        [750, 1],
        [850, 1],
        [1050, 0.1],
        [1200, 0.1],
      ];
      for (const hand of ["left", "right"] as const) {
        const frames = trace(fps, points, hand);
        const events = collect(engine(), frames);
        expect(events.map((e) => e.hand)).toEqual([hand]);
        expect(events[0].startMs).toBeGreaterThanOrEqual(500);
        expect(events[0].peakMs).toBeGreaterThanOrEqual(750);
      }
    },
  );

  it.each([15, 30, 60])(
    "does not combine repeated rejected guard gestures into an observed straight peak at %i fps",
    (fps) => {
      const points: [number, number][] = [
        [0, 0.2],
        [200, 0.2],
        [360, 0.5],
        [460, 0.5],
        [600, 0.1],
        [800, 0.5],
        [900, 0.5],
        [1100, 0.1],
        [1300, 0.5],
        [1400, 0.5],
        [1600, 0.1],
        [1700, 0.1],
      ];
      expect(collect(engine(), trace(fps, points))).toEqual([]);
    },
  );

  it("does not preserve rejected acquisition history through active-arm occlusion", () => {
    const frames = trace(30, [
      [0, 0.2],
      [200, 0.2],
      [360, 0.5],
      [460, 0.5],
      [600, 0.1],
      [800, 1],
      [900, 1],
      [1100, 0.1],
      [1200, 0.1],
    ]);
    frames.find((f) => f.t >= 600)!.landmarks[JOINT.leftWrist].visibility = 0.1;
    expect(collect(engine(), frames)).toEqual([]);
  });

  it("does not preserve rejected acquisition history through a source-time gap", () => {
    const frames = trace(30, [
      [0, 0.2],
      [200, 0.2],
      [360, 0.5],
      [460, 0.5],
      [600, 0.1],
      [800, 1],
      [900, 1],
      [1100, 0.1],
      [1200, 0.1],
    ]);
    for (const f of frames) if (f.t >= 600) f.t += 250;
    expect(collect(engine(), frames)).toEqual([]);
  });

  it("documents the unresolved arm-lowering ambiguity without treating it as a verified punch", () => {
    const frames = trace(30, [
      [0, 0],
      [300, 0],
      [650, 1],
      [750, 1],
      [1100, 0],
      [1250, 0],
    ]);
    for (const f of frames) {
      const shoulder = f.landmarks[JOINT.leftShoulder];
      const theta = -Math.PI / 2;
      for (const joint of [JOINT.leftElbow, JOINT.leftWrist]) {
        const p = f.landmarks[joint];
        const x = ((p.x - shoulder.x) * f.width) / f.height,
          y = p.y - shoulder.y;
        p.x =
          shoulder.x +
          ((x * Math.cos(theta) - y * Math.sin(theta)) * f.height) / f.width;
        p.y = shoulder.y + x * Math.sin(theta) + y * Math.cos(theta);
      }
    }
    // Known limitation: these image-space thresholds also admit a lowered arm.
    // This is a negative characterization, not validation of punch classification.
    expect(collect(engine(), frames)).toHaveLength(1);
  });
  it.each([15, 30, 60])(
    "counts a shorter same-arm repeat only after an accepted full extension at %i fps",
    (fps) => {
      for (const hand of ["left", "right"] as const) {
        const points: [number, number][] = [
          [0, 0],
          [200, 0],
          [450, 1],
          [550, 1],
          [700, 0.45],
          [800, 0.45],
          [1000, 0.75],
          [1067, 0.75],
          [1200, 0.45],
          [1300, 0.45],
        ];
        const events = collect(engine(), trace(fps, points, hand));
        expect(events.map((e) => e.hand)).toEqual([hand, hand]);
        expect(events[1].extension).toBeLessThan(
          MOTION_LIMITS.minimumExtension,
        );
        expect(events[1].startMs).toBeGreaterThan(events[0].endMs);
        const isolated: [number, number][] = [
          [0, 0.45],
          [200, 0.45],
          [400, 0.75],
          [467, 0.75],
          [600, 0.45],
          [700, 0.45],
        ];
        expect(collect(engine(), trace(fps, isolated, hand))).toEqual([]);
      }
    },
  );

  it.each([15, 30, 60])(
    "does not count shallow straight-arm bobbing after an accepted stroke at %i fps",
    (fps) => {
      const points: [number, number][] = [
        [0, 0.2],
        [200, 0.2],
        [450, 1],
        [550, 1],
        [700, 0.55],
        [800, 0.55],
        [1000, 0.7],
        [1067, 0.7],
        [1200, 0.55],
        [1300, 0.55],
      ];
      expect(collect(engine(), trace(fps, points))).toHaveLength(1);
    },
  );

  it.each([15, 30, 60])(
    "does not let unsupported prior gestures authorize a short repeat at %i fps",
    (fps) => {
      const points: [number, number][] = [
        [0, 0.45],
        [200, 0.45],
        [450, 0.6],
        [550, 0.6],
        [700, 0.45],
        [800, 0.45],
        [1000, 0.75],
        [1067, 0.75],
        [1200, 0.45],
        [1300, 0.45],
      ];
      expect(collect(engine(), trace(fps, points))).toEqual([]);
    },
  );

  it.each([15, 30, 60])(
    "expires the original full-extension context rather than renewing it on a short repeat at %i fps",
    (fps) => {
      const points: [number, number][] = [
        [0, 0],
        [200, 0],
        [450, 1],
        [500, 1],
        [650, 0.45],
        [733, 0.45],
        [933, 0.75],
        [1000, 0.75],
        [1100, 0.45],
        [1167, 0.45],
        [1367, 0.75],
        [1433, 0.75],
        [1600, 0.45],
        [1700, 0.45],
      ];
      const events = collect(engine(), trace(fps, points));
      expect(events).toHaveLength(2);
      expect(events[1].peakMs).toBeLessThan(1100);
    },
  );

  it("clears accepted-repeat context across arm loss or duplicated timestamps", () => {
    const points: [number, number][] = [
      [0, 0],
      [200, 0],
      [450, 1],
      [550, 1],
      [700, 0.45],
      [800, 0.45],
      [1000, 0.75],
      [1067, 0.75],
      [1200, 0.45],
      [1300, 0.45],
    ];
    for (const interruption of ["hidden", "duplicate", "reset"] as const) {
      const frames = trace(30, points);
      const index = frames.findIndex((f) => f.t >= 800);
      if (interruption === "hidden")
        frames[index].landmarks[JOINT.leftWrist].visibility = 0.1;
      if (interruption === "duplicate") frames[index].t = frames[index - 1].t;
      const detector = engine();
      const events = frames.flatMap((f, i) => {
        if (interruption === "reset" && i === index) detector.reset();
        return detector.update(f).events;
      });
      expect(events, interruption).toHaveLength(1);
    }
  });

  it("rejects a repeat whose first straight sample is one millisecond beyond the original peak deadline", () => {
    const first = [
      ...cycle().slice(0, 17),
      ...[680, 740, 800, 860].map((t) => frame(t, 0.45)),
    ];
    const firstEvent = collect(engine(), first);
    expect(firstEvent).toHaveLength(1);
    const values = [
      0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,
      0.45,
    ];
    const firstStraightIndex = 4;
    const expiredStart =
      firstEvent[0].peakMs +
      MOTION_LIMITS.maximumRepeatPeakGapMs +
      1 -
      firstStraightIndex * 30;
    const after = values.map((x, i) => frame(expiredStart + i * 30, x));
    const before = values.map((x, i) => frame(expiredStart - 100 + i * 30, x));
    expect(collect(engine(), [...first, ...before])).toHaveLength(2);
    expect(collect(engine(), [...first, ...after])).toHaveLength(1);
  });

  it("does not carry early repeat support into a selected peak beyond the original deadline", () => {
    const first = [
      ...cycle().slice(0, 17),
      ...[680, 740, 800, 860].map((t) => frame(t, 0.45)),
    ];
    const repeat: [number, number][] = [
      [900, 0.5],
      [930, 0.6],
      [960, 0.68],
      [990, 0.7],
      [1020, 0.72],
      [1050, 0.74],
      [1080, 0.75],
      [1110, 0.7],
      [1140, 0.6],
      [1170, 0.5],
      [1200, 0.45],
      [1230, 0.45],
    ];
    expect(
      collect(engine(), [...first, ...repeat.map(([t, x]) => frame(t, x))]),
    ).toHaveLength(2);
    expect(
      collect(engine(), [
        ...first,
        ...repeat.map(([t, x]) => frame(t + 40, x)),
      ]),
    ).toHaveLength(1);
  });
  it("does not renew repeat context from mixed repeat support and one unsupported full sample", () => {
    const first = [
      ...cycle().slice(0, 17),
      ...[680, 740, 800, 860].map((t) => frame(t, 0.45)),
    ];
    const later: [number, number][] = [
      [900, 0.5],
      [930, 0.6],
      [960, 0.7],
      [990, 0.75],
      [1020, 0.84],
      [1050, 0.75],
      [1080, 0.65],
      [1110, 0.5],
      [1140, 0.45],
      [1170, 0.45],
      [1200, 0.45],
      [1230, 0.5],
      [1260, 0.6],
      [1290, 0.7],
      [1320, 0.75],
      [1350, 0.75],
      [1380, 0.65],
      [1410, 0.5],
      [1440, 0.45],
      [1470, 0.45],
    ];
    const events = collect(engine(), [
      ...first,
      ...later.map(([t, x]) => frame(t, x)),
    ]);
    expect(events).toHaveLength(2);
    expect(events[1].peakMs).toBe(1020);
    // The one larger point is not independently supported as a full stroke;
    // later short movement cannot inherit a fresh deadline from it.
  });
});
