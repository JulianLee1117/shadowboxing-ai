import type { Landmark, PoseFrame } from "./types";

// A deterministic interface/replay demonstration. This is not a model benchmark.
export function demoFrame(t: number): PoseFrame {
  const bob = Math.sin(t / 450) * 0.004;
  const points: Landmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.2,
    z: 0,
    visibility: 0.99,
    presence: 0.99,
  }));
  const set = (i: number, x: number, y: number) => {
    points[i] = { x, y: y + bob, z: 0, visibility: 0.99, presence: 0.99 };
  };
  set(0, 0.49, 0.18);
  set(7, 0.46, 0.19);
  set(8, 0.52, 0.19);
  set(11, 0.44, 0.31);
  set(12, 0.57, 0.32);
  set(13, 0.39, 0.43);
  set(14, 0.62, 0.44);
  set(15, 0.42, 0.255);
  set(16, 0.585, 0.27);
  set(23, 0.46, 0.57);
  set(24, 0.56, 0.57);
  set(25, 0.4, 0.735);
  set(26, 0.61, 0.72);
  set(27, 0.37, 0.885);
  set(28, 0.65, 0.865);
  set(29, 0.36, 0.9);
  set(30, 0.65, 0.88);
  set(31, 0.32, 0.91);
  set(32, 0.7, 0.885);
  const cycle = t % 4800;
  const punch = (start: number, side: "left" | "right") => {
    const elapsed = cycle - start;
    if (elapsed < 0 || elapsed > 1000) return;
    const p =
      elapsed < 380 ? elapsed / 380 : Math.max(0, 1 - (elapsed - 380) / 620);
    if (side === "left") {
      set(15, 0.42 - 0.26 * p, 0.255 + 0.035 * p);
      set(13, 0.39 - 0.1 * p, 0.43 - 0.135 * p);
    } else {
      set(16, 0.585 + 0.26 * p, 0.27 + 0.055 * p);
      set(14, 0.62 + 0.105 * p, 0.44 - 0.115 * p);
    }
  };
  punch(900, "left");
  punch(2400, "right");
  [17, 19, 21].forEach((i) => {
    points[i] = { ...points[15] };
  });
  [18, 20, 22].forEach((i) => {
    points[i] = { ...points[16] };
  });
  return {
    t,
    width: 1280,
    height: 720,
    landmarks: points,
    inferenceMs: 0,
    frameAgeMs: 0,
  };
}
