import { describe, expect, it } from "vitest";
import { MotionEngine } from "./motion";
import { JOINT, type PoseFrame, type PunchEvent, type Stance } from "./types";

type Hand = PunchEvent["hand"];
type Position = { wrist: [number, number]; elbow: [number, number] };
const guard: Position = { wrist: [0.4, -0.1], elbow: [0.3, 0.3] };
const load: Position = { wrist: [0.25, 0.65], elbow: [0.2, 0.45] };
const hook: Position = { wrist: [0.95, -0.3], elbow: [0.75, 0.15] };
const uppercut: Position = { wrist: [0.65, -0.65], elbow: [0.5, 0.1] };
const mix = (a: Position, b: Position, f: number): Position => ({
  wrist: [
    a.wrist[0] + (b.wrist[0] - a.wrist[0]) * f,
    a.wrist[1] + (b.wrist[1] - a.wrist[1]) * f,
  ],
  elbow: [
    a.elbow[0] + (b.elbow[0] - a.elbow[0]) * f,
    a.elbow[1] + (b.elbow[1] - a.elbow[1]) * f,
  ],
});

/** Exact geometry fixtures exercise rules; they are not training or accuracy data. */
function pose(t: number, position: Position, hand: Hand = "left"): PoseFrame {
  const landmarks = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    visibility: 0.99,
  }));
  landmarks[JOINT.leftShoulder] = { x: 0.42, y: 0.3, visibility: 0.99 };
  landmarks[JOINT.rightShoulder] = { x: 0.58, y: 0.3, visibility: 0.99 };
  landmarks[JOINT.leftHip] = { x: 0.44, y: 0.67, visibility: 0.99 };
  landmarks[JOINT.rightHip] = { x: 0.56, y: 0.67, visibility: 0.99 };
  for (const side of ["left", "right"] as const) {
    const p = side === hand ? position : guard;
    const x = side === "left" ? 0.42 : 0.58;
    const sign = side === "left" ? 1 : -1;
    const set = (index: number, value: [number, number]) => {
      landmarks[index] = {
        x: x + (sign * value[0] * 0.37) / (1280 / 720),
        y: 0.3 + value[1] * 0.37,
        visibility: 0.99,
      };
    };
    set(side === "left" ? JOINT.leftWrist : JOINT.rightWrist, p.wrist);
    set(side === "left" ? JOINT.leftElbow : JOINT.rightElbow, p.elbow);
  }
  return { t, width: 1280, height: 720, landmarks, inferenceMs: 1 };
}

function trace(
  knots: [number, Position][],
  fps = 30,
  hand: Hand = "left",
  phase = 0,
): PoseFrame[] {
  const result: PoseFrame[] = [];
  for (let t = phase; t <= knots.at(-1)![0]; t += 1000 / fps) {
    let i = 1;
    while (i < knots.length - 1 && knots[i][0] < t) i += 1;
    const [a, b] = [knots[i - 1], knots[i]];
    result.push(pose(t, mix(a[1], b[1], (t - a[0]) / (b[0] - a[0])), hand));
  }
  return result;
}

const cycle = (family: "hook" | "uppercut"): [number, Position][] =>
  family === "hook"
    ? [
        [0, guard],
        [200, guard],
        [450, hook],
        [530, hook],
        [800, guard],
        [1100, guard],
      ]
    : [
        [0, guard],
        [200, guard],
        [350, load],
        [600, uppercut],
        [680, uppercut],
        [950, guard],
        [1200, guard],
      ];
const collect = (frames: PoseFrame[], stance: Stance = "orthodox") => {
  const engine = new MotionEngine({ stance, calibrated: true });
  return frames.flatMap((frame) => engine.update(frame).events);
};

describe("experimental observed curved-punch cycles", () => {
  for (const fps of [15, 30, 60])
    for (const label of ["hook", "uppercut"] as const)
      for (const hand of ["left", "right"] as const) {
        it(`recognizes a supported ${hand} ${label} at ${fps} fps across sampling phases`, () => {
          for (const phase of [0, 250 / fps, 500 / fps]) {
            const frames = trace(cycle(label), fps, hand, phase);
            for (const stance of ["orthodox", "southpaw"] as const) {
              const events = collect(frames, stance);
              expect(events).toHaveLength(1);
              const event = events[0];
              expect(event).toMatchObject({
                hand,
                label,
                role:
                  hand === (stance === "orthodox" ? "left" : "right")
                    ? "lead"
                    : "rear",
                experimental: true,
              });
              expect(event.startMs).toBeLessThan(event.peakMs);
              expect(event.peakMs).toBeLessThan(event.endMs);
              expect(event.endMs).toBeLessThan(event.detectedAtMs!);
              expect(frames.some((f) => f.t === event.peakMs)).toBe(true);
            }
          }
        });
      }

  it.each(["hook", "uppercut"] as const)(
    "does not bridge own-arm confidence loss for a %s",
    (label) => {
      const frames = trace(cycle(label));
      frames.find((frame) => frame.t >= 400)!.landmarks[
        JOINT.leftWrist
      ].visibility = 0.3;
      expect(collect(frames)).toEqual([]);
    },
  );

  it("keeps opposite-arm occlusion independent and never swaps physical hands", () => {
    const frames = trace(cycle("hook"));
    for (const frame of frames)
      frame.landmarks[JOINT.rightWrist].visibility = 0.1;
    expect(collect(frames)).toMatchObject([{ hand: "left", label: "hook" }]);
  });

  it.each([15, 30, 60])(
    "keeps supported rising drives stable under small coordinate jitter at %i fps",
    (fps) => {
      for (const phase of [0, 250 / fps, 500 / fps])
        for (const ratio of [1.45, 1.55, 1.3]) {
          const endpoint: Position = {
            wrist: [load.wrist[0] + 1.05 / ratio, -0.4],
            elbow: [load.wrist[0] + 1.05 / ratio - 0.1, 0.2],
          };
          const frames = trace(
            [
              [0, guard],
              [200, guard],
              [350, load],
              [450, load],
              [700, endpoint],
              [780, endpoint],
              [1050, guard],
              [1250, guard],
            ],
            fps,
            "left",
            phase,
          );
          for (const [i, frame] of frames.entries()) {
            frame.landmarks[JOINT.leftWrist].x += Math.sin(i * 1.7) * 0.00015;
            frame.landmarks[JOINT.leftWrist].y += Math.cos(i * 1.3) * 0.00015;
          }
          expect(collect(frames).map((event) => event.label)).toEqual(
            ratio > 1.4 ? ["uppercut"] : [],
          );
        }
    },
  );

  it.each(["hook", "uppercut"] as const)(
    "discards %s evidence after a frame gap or explicit reset",
    (label) => {
      const original = trace(cycle(label));
      expect(
        collect(original.filter((frame) => frame.t < 300 || frame.t > 650)),
      ).toEqual([]);
      const engine = new MotionEngine({ stance: "orthodox", calibrated: true });
      const events: PunchEvent[] = [];
      original.forEach((frame, i) => {
        if (i === 13) engine.reset();
        events.push(...engine.update(frame).events);
      });
      expect(events).toEqual([]);
    },
  );

  it("rejects one-frame spikes, small guard shifts, and diagonal ambiguity", () => {
    const diagonal = {
      wrist: [0.8, -0.5] as [number, number],
      elbow: [0.7, 0.1] as [number, number],
    };
    for (const target of [mix(guard, hook, 0.4), diagonal]) {
      expect(
        collect(
          trace([
            [0, guard],
            [200, guard],
            [450, target],
            [500, target],
            [800, guard],
            [1100, guard],
          ]),
        ),
      ).toEqual([]);
    }
    const spike = trace([
      [0, guard],
      [1000, guard],
    ]);
    spike[15] = pose(spike[15].t, hook);
    expect(collect(spike)).toEqual([]);
  });

  it("does not relabel the pre-existing downward-straight ambiguity as a curve", () => {
    const lowered: Position = { wrist: [0.4, 0.9], elbow: [0.3, 0.45] };
    const events = collect(
      trace([
        [0, guard],
        [200, guard],
        [450, lowered],
        [500, lowered],
        [800, guard],
        [1100, guard],
      ]),
    );
    // Known straight-detector limitation, independently characterized in motion.test.ts.
    // This increment does not treat its arm lowering as a hook or uppercut.
    expect(events.map((event) => event.label)).toEqual(["jab"]);
  });

  it("requires recovery and does not count a held endpoint or retracting jitter twice", () => {
    expect(
      collect(
        trace([
          [0, guard],
          [200, guard],
          [450, hook],
          [2400, hook],
        ]),
      ),
    ).toEqual([]);
    const frames = trace([
      ...cycle("hook").slice(0, -1),
      [850, mix(guard, hook, 0.12)],
      [900, guard],
      [1000, mix(guard, hook, 0.12)],
      [1300, guard],
    ]);
    expect(collect(frames)).toHaveLength(1);
  });

  it("rearms completed curves without counting either cycle twice", () => {
    const frames = trace([
      [0, guard],
      [200, guard],
      [450, hook],
      [530, hook],
      [800, guard],
      [900, guard],
      [1050, load],
      [1300, uppercut],
      [1380, uppercut],
      [1650, guard],
      [1800, guard],
    ]);
    expect(collect(frames).map((e) => e.label)).toEqual(["hook", "uppercut"]);
  });

  it.each(["straight-first", "hook-first"])(
    "arbitrates %s without duplicate cycles or stale straight repeat anchors",
    (order) => {
      const straight: Position = { wrist: [1.4, -0.1], elbow: [0.7, -0.03] };
      const [first, second] =
        order === "straight-first" ? [straight, hook] : [hook, straight];
      const frames = trace([
        [0, guard],
        [200, guard],
        [450, first],
        [530, first],
        [800, guard],
        [940, guard],
        [1190, second],
        [1270, second],
        [1540, guard],
        [1700, guard],
      ]);
      const events = collect(frames);
      expect(events.map((e) => e.label)).toEqual(
        order === "straight-first" ? ["jab", "hook"] : ["hook", "jab"],
      );
      expect(new Set(events.map((e) => e.id)).size).toBe(2);
      expect(events[0].endMs).toBeLessThan(events[1].startMs);
    },
  );
});
