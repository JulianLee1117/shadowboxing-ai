import { assessArmTracking, aspectPoint } from "./motion";
import { JOINT, type PoseFrame } from "./types";

export const TRACKING_TRUST_VERSION = "trajectory-consistency-v1";
type Hand = "left" | "right";
type Point = { x: number; y: number };
export interface TrackingUncertaintyInterval {
  hand: Hand | "both";
  startMs: number;
  endMs: number;
  reasons: string[];
}
export interface TrackingTrustInterval extends TrackingUncertaintyInterval {
  status: "trusted" | "uncertain";
  identityVerified: false;
}
export interface TrackingTrustOptions {
  durationMs: number;
  maximumFrameGapMs?: number;
  reacquireMs?: number;
  minimumStableSamples?: number;
}
export interface TrackingTrustSample {
  t: number;
  assessedAtMs: number;
  arms: Record<Hand, { status: "trusted" | "uncertain"; reasons: string[] }>;
}
export interface TrackingTrustReport {
  algorithmVersion: string;
  identityVerified: false;
  meaning: "trusted-means-no-tested-inconsistency-not-correct-pose-or-verified-identity";
  intervals: TrackingTrustInterval[];
  uncertaintyIntervals: TrackingUncertaintyInterval[];
  samples: TrackingTrustSample[];
  options: Required<TrackingTrustOptions>;
  outsideRoundFrames: number;
}
interface Geometry {
  shoulder: Point;
  elbow: Point;
  wrist: Point;
  scale: number;
}
const hands: Hand[] = ["left", "right"];
const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);
const midpoint = (a: Point, b: Point) => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});
function geometry(frame: PoseFrame, hand: Hand): Geometry {
  const p = (i: number) =>
    aspectPoint(frame.landmarks[i], frame.width, frame.height);
  const scale = distance(
    midpoint(p(JOINT.leftShoulder), p(JOINT.rightShoulder)),
    midpoint(p(JOINT.leftHip), p(JOINT.rightHip)),
  );
  return {
    shoulder: p(hand === "left" ? JOINT.leftShoulder : JOINT.rightShoulder),
    elbow: p(hand === "left" ? JOINT.leftElbow : JOINT.rightElbow),
    wrist: p(hand === "left" ? JOINT.leftWrist : JOINT.rightWrist),
    scale,
  };
}
const relative = (value: Geometry) => ({
  x: (value.wrist.x - value.shoulder.x) / value.scale,
  y: (value.wrist.y - value.shoulder.y) / value.scale,
});

/** Diagnostic review only. Flags observations; never swaps anatomical identities,
 * repairs coordinates, changes source scores, or establishes joint correctness. */
export function assessTrackingTrust(
  inputFrames: readonly PoseFrame[],
  options: TrackingTrustOptions,
): TrackingTrustReport {
  const config: Required<TrackingTrustOptions> = {
    durationMs: options.durationMs,
    maximumFrameGapMs: options.maximumFrameGapMs ?? 200,
    reacquireMs: options.reacquireMs ?? 120,
    minimumStableSamples: options.minimumStableSamples ?? 3,
  };
  if (
    Object.values(config).some((v) => !Number.isFinite(v) || v < 0) ||
    config.maximumFrameGapMs <= 0 ||
    !Number.isInteger(config.minimumStableSamples) ||
    config.minimumStableSamples < 2
  )
    throw new Error("Invalid tracking trust configuration");
  for (let i = 0; i < inputFrames.length; i++)
    if (
      !Number.isFinite(inputFrames[i].t) ||
      inputFrames[i].t < 0 ||
      (i > 0 && inputFrames[i].t <= inputFrames[i - 1].t)
    )
      throw new Error(
        "Tracking trust needs finite nonnegative strictly increasing source timestamps",
      );
  // A late in-flight result can fall just beyond a wall-clock round deadline.
  // Exclude and report it; never extend the round or alter its timestamp.
  const frames = inputFrames.filter((frame) => frame.t <= config.durationMs);
  const outsideRoundFrames = inputFrames.length - frames.length;
  const flags = frames.map(() => ({
    left: new Set<string>(),
    right: new Set<string>(),
  }));
  const assessedAt = frames.map((f) => f.t);
  const geometries = frames.map((frame, index) => {
    const tracking = assessArmTracking(frame),
      result: Record<Hand, Geometry | null> = { left: null, right: null };
    for (const hand of hands) {
      if (!tracking[hand].assessable)
        flags[index][hand].add(`unobservable-${tracking[hand].status}`);
      else result[hand] = geometry(frame, hand);
    }
    return result;
  });
  const flag = (
    index: number,
    hand: Hand,
    reason: string,
    knownAt = frames[index].t,
  ) => {
    flags[index][hand].add(reason);
    assessedAt[index] = Math.max(assessedAt[index], knownAt);
  };
  for (let i = 1; i < frames.length; i++) {
    const dt = frames[i].t - frames[i - 1].t;
    const previousAspect = frames[i - 1].width / frames[i - 1].height,
      aspect = frames[i].width / frames[i].height;
    if (
      dt > config.maximumFrameGapMs ||
      !Number.isFinite(aspect) ||
      Math.abs(aspect / previousAspect - 1) > 0.01
    ) {
      for (const hand of hands) {
        flag(
          i - 1,
          hand,
          dt > config.maximumFrameGapMs ? "frame-gap" : "framing-discontinuity",
          frames[i].t,
        );
        flag(
          i,
          hand,
          dt > config.maximumFrameGapMs ? "frame-gap" : "framing-discontinuity",
        );
      }
      continue;
    }
    const before = geometries[i - 1],
      after = geometries[i];
    if (dt <= 80 && hands.every((hand) => before[hand] && after[hand])) {
      const a = before as Record<Hand, Geometry>,
        b = after as Record<Hand, Geometry>,
        scale = (a.left.scale + b.left.scale) / 2;
      const own =
        distance(a.left.wrist, b.left.wrist) +
        distance(a.right.wrist, b.right.wrist);
      const crossed =
        distance(a.left.wrist, b.right.wrist) +
        distance(a.right.wrist, b.left.wrist);
      const shouldersStable = hands.every(
        (hand) => distance(a[hand].shoulder, b[hand].shoulder) / scale < 0.12,
      );
      if (
        shouldersStable &&
        distance(a.left.wrist, a.right.wrist) / scale > 0.35 &&
        own / scale > 0.8 &&
        crossed / scale < 0.2
      ) {
        for (const hand of hands) {
          flag(i - 1, hand, "bilateral-assignment-discontinuity", frames[i].t);
          flag(i, hand, "bilateral-assignment-discontinuity");
        }
      }
    }
  }
  // A single point that leaves and returns to nearby neighbors can be a tracking
  // artifact OR genuine fast motion. Mark ambiguous evidence, never correct it.
  for (let i = 1; i < frames.length - 1; i++) {
    if (
      frames[i].t - frames[i - 1].t > 70 ||
      frames[i + 1].t - frames[i].t > 70
    )
      continue;
    for (const hand of hands) {
      const previous = geometries[i - 1][hand],
        middle = geometries[i][hand],
        next = geometries[i + 1][hand];
      if (!previous || !middle || !next) continue;
      const [a, b, c] = [previous, middle, next].map(relative),
        first = distance(a, b),
        second = distance(b, c),
        bypass = distance(a, c);
      const dot = (b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y);
      if (
        first > 0.3 &&
        second > 0.3 &&
        dot / (first * second) < -0.85 &&
        bypass < Math.min(first, second) * 0.35
      ) {
        for (const j of [i - 1, i, i + 1])
          flag(j, hand, "isolated-wrist-detour", frames[i + 1].t);
      }
    }
  }
  const stable = {
    left: { since: 0, count: 0 },
    right: { since: 0, count: 0 },
  };
  const samples: TrackingTrustSample[] = frames.map((frame, i) => {
    const arms = {} as TrackingTrustSample["arms"];
    for (const hand of hands) {
      const state = stable[hand],
        reasons = [...flags[i][hand]];
      if (reasons.length) state.count = 0;
      else {
        if (state.count === 0) state.since = frame.t;
        state.count++;
        if (
          state.count < config.minimumStableSamples ||
          frame.t - state.since < config.reacquireMs
        )
          reasons.push("acquiring-consistent-observations");
      }
      arms[hand] = {
        status: reasons.length ? "uncertain" : "trusted",
        reasons: reasons.sort(),
      };
    }
    return { t: frame.t, assessedAtMs: assessedAt[i], arms };
  });
  const intervals: TrackingTrustInterval[] = [];
  const append = (
    hand: Hand,
    startMs: number,
    endMs: number,
    reasons: string[],
  ) => {
    if (endMs <= startMs) return;
    reasons = [...new Set(reasons)].sort();
    const status = reasons.length ? "uncertain" : "trusted",
      previous = intervals.at(-1);
    if (
      previous?.hand === hand &&
      previous.endMs === startMs &&
      previous.status === status &&
      JSON.stringify(previous.reasons) === JSON.stringify(reasons)
    )
      previous.endMs = endMs;
    else
      intervals.push({
        hand,
        startMs,
        endMs,
        reasons,
        status,
        identityVerified: false,
      });
  };
  for (const hand of hands) {
    if (frames.length < 2) {
      append(hand, 0, config.durationMs, [
        frames.length ? "insufficient-samples" : "no-samples",
      ]);
      continue;
    }
    append(hand, 0, frames[0].t, ["no-samples"]);
    for (let i = 0; i < samples.length - 1; i++)
      append(hand, samples[i].t, samples[i + 1].t, [
        ...samples[i].arms[hand].reasons,
        ...samples[i + 1].arms[hand].reasons,
      ]);
    append(hand, frames.at(-1)!.t, config.durationMs, ["no-samples"]);
  }
  intervals.sort(
    (a, b) => a.startMs - b.startMs || a.hand.localeCompare(b.hand),
  );
  return {
    algorithmVersion: TRACKING_TRUST_VERSION,
    identityVerified: false,
    meaning:
      "trusted-means-no-tested-inconsistency-not-correct-pose-or-verified-identity",
    options: config,
    outsideRoundFrames,
    samples,
    intervals,
    uncertaintyIntervals: intervals
      .filter((i) => i.status === "uncertain")
      .map(({ hand, startMs, endMs, reasons }) => ({
        hand,
        startMs,
        endMs,
        reasons: [...reasons],
      })),
  };
}

export function summarizeTrackingTrust(
  frames: readonly PoseFrame[],
  durationMs: number,
): TrackingUncertaintyInterval[] {
  return assessTrackingTrust(frames, { durationMs }).uncertaintyIntervals;
}
