import {
  JOINT,
  type EngineOptions,
  type EngineResult,
  type Landmark,
  type PoseFrame,
  type PunchEvent,
  type QualityState,
  type Stance,
} from "./types";

type Hand = PunchEvent["hand"];
type Point = { x: number; y: number };

/** Experimental image-space thresholds, not a validated boxing rubric. */
export const MOTION_LIMITS = {
  minimumVisibility: 0.65,
  maximumFrameGapMs: 200,
  minimumRestMs: 60,
  minimumRestFrames: 3,
  minimumExtension: 0.45,
  minimumPeakAngle: 145,
  minimumEventMs: 180,
  maximumEventMs: 1800,
  minimumRecoveryMs: 30,
} as const;

const REQUIRED_JOINTS = [
  JOINT.nose,
  JOINT.leftShoulder,
  JOINT.rightShoulder,
  JOINT.leftElbow,
  JOINT.rightElbow,
  JOINT.leftWrist,
  JOINT.rightWrist,
  JOINT.leftHip,
  JOINT.rightHip,
];

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);
const midpoint = (a: Point, b: Point): Point => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});
const finitePoint = (point: Landmark | undefined): point is Landmark =>
  !!point && Number.isFinite(point.x) && Number.isFinite(point.y);

/** Coordinates in image-height units: corrects x/y aspect ratio before geometry. */
export function aspectPoint(
  point: Landmark,
  width: number,
  height: number,
): Point {
  return { x: (point.x * width) / height, y: point.y };
}

export function imageDistance(
  a: Landmark,
  b: Landmark,
  width: number,
  height: number,
): number {
  return distance(aspectPoint(a, width, height), aspectPoint(b, width, height));
}

/** Anatomical landmark identity is independent of the mirrored preview. */
export function anatomicalRole(hand: Hand, stance: Stance): PunchEvent["role"] {
  return hand === (stance === "orthodox" ? "left" : "right") ? "lead" : "rear";
}

function angleAt(a: Point, joint: Point, b: Point): number {
  const ax = a.x - joint.x;
  const ay = a.y - joint.y;
  const bx = b.x - joint.x;
  const by = b.y - joint.y;
  const denominator = Math.hypot(ax, ay) * Math.hypot(bx, by);
  if (denominator < 1e-8) return NaN;
  return (
    (Math.acos(Math.max(-1, Math.min(1, (ax * bx + ay * by) / denominator))) *
      180) /
    Math.PI
  );
}

function visible(point: Landmark | undefined): point is Landmark {
  return (
    finitePoint(point) &&
    point.x >= 0 &&
    point.x <= 1 &&
    point.y >= 0 &&
    point.y <= 1 &&
    Number.isFinite(point.visibility) &&
    point.visibility! >= MOTION_LIMITS.minimumVisibility &&
    (point.presence === undefined ||
      (Number.isFinite(point.presence) && point.presence >= 0.5))
  );
}

function torsoScale(frame: PoseFrame): number {
  const p = (index: number) =>
    aspectPoint(frame.landmarks[index], frame.width, frame.height);
  return distance(
    midpoint(p(JOINT.leftShoulder), p(JOINT.rightShoulder)),
    midpoint(p(JOINT.leftHip), p(JOINT.rightHip)),
  );
}

/** Upper-body acquisition gate only; assessable does not mean correct technique. */
export function assessQuality(frame: PoseFrame): QualityState {
  const visibleJoints = REQUIRED_JOINTS.filter((index) =>
    visible(frame.landmarks[index]),
  ).length;
  const reasons: string[] = [];
  if (!Number.isFinite(frame.t) || frame.t < 0)
    reasons.push("Frame time is invalid.");
  if (
    !Number.isFinite(frame.width) ||
    !Number.isFinite(frame.height) ||
    frame.width <= 0 ||
    frame.height <= 0
  ) {
    reasons.push("Image dimensions are invalid.");
  }
  if (visibleJoints < REQUIRED_JOINTS.length) {
    reasons.push(
      "Keep your face, shoulders, elbows, wrists, and hips visible inside the frame.",
    );
  }
  if (reasons.length === 0) {
    const scale = torsoScale(frame);
    if (scale < 0.08)
      reasons.push("Move closer so your upper body is large enough to track.");
    for (const hand of ["left", "right"] as const) {
      const shoulder =
        frame.landmarks[
          hand === "left" ? JOINT.leftShoulder : JOINT.rightShoulder
        ];
      const elbow =
        frame.landmarks[hand === "left" ? JOINT.leftElbow : JOINT.rightElbow];
      const wrist =
        frame.landmarks[hand === "left" ? JOINT.leftWrist : JOINT.rightWrist];
      if (
        imageDistance(shoulder, elbow, frame.width, frame.height) < 0.01 ||
        imageDistance(elbow, wrist, frame.width, frame.height) < 0.01
      ) {
        reasons.push(
          "An arm is too foreshortened or uncertain in this view. Try a slight turn.",
        );
        break;
      }
    }
  }
  return {
    assessable: reasons.length === 0,
    label: reasons.length ? "Tracking unavailable" : "Upper body visible",
    reasons,
    visibleJoints,
    totalJoints: REQUIRED_JOINTS.length,
  };
}

interface ArmSample {
  t: number;
  wrist: Point; // shoulder-relative, in torso-length units
  reach: number;
  angle: number;
}

export function armGeometry(frame: PoseFrame, hand: Hand): ArmSample {
  const p = (index: number) =>
    aspectPoint(frame.landmarks[index], frame.width, frame.height);
  const shoulder = p(
    hand === "left" ? JOINT.leftShoulder : JOINT.rightShoulder,
  );
  const elbow = p(hand === "left" ? JOINT.leftElbow : JOINT.rightElbow);
  const wrist = p(hand === "left" ? JOINT.leftWrist : JOINT.rightWrist);
  const scale = torsoScale(frame);
  const relativeWrist = {
    x: (wrist.x - shoulder.x) / scale,
    y: (wrist.y - shoulder.y) / scale,
  };
  return {
    t: frame.t,
    wrist: relativeWrist,
    reach: Math.hypot(relativeWrist.x, relativeWrist.y),
    angle: angleAt(shoulder, elbow, wrist),
  };
}

interface Candidate {
  origin: ArmSample;
  startMs: number;
  peakMs: number;
  peakReach: number;
  peakAngle: number;
  peakWrist: Point;
  peakFrames: number;
  pathLength: number;
  recovering: boolean;
  recoveryStartMs: number | null;
  recoveryFrames: number;
}

interface ArmState {
  previous: ArmSample | null;
  rest: ArmSample | null;
  restStartMs: number;
  restFrames: number;
  candidate: Candidate | null;
}

const freshArm = (): ArmState => ({
  previous: null,
  rest: null,
  restStartMs: 0,
  restFrames: 0,
  candidate: null,
});

/**
 * Causal extension/recovery heuristic for a supported projected straight-like motion.
 * This cannot distinguish every straight from hooks/uppercuts/other gestures. It is
 * deliberately experimental and has no requested-drill input or learned accuracy.
 */
export class MotionEngine {
  private options: EngineOptions;
  private arms: Record<Hand, ArmState> = {
    left: freshArm(),
    right: freshArm(),
  };
  private lastTime: number | null = null;
  private lastAspect: number | null = null;
  private sequence = 0;

  constructor(options: EngineOptions) {
    this.options = { ...options };
  }

  reset(options?: EngineOptions): void {
    if (options) this.options = { ...options };
    this.clearTracking();
  }

  private clearTracking(): void {
    this.arms = { left: freshArm(), right: freshArm() };
    this.lastTime = null;
    this.lastAspect = null;
  }

  update(frame: PoseFrame): EngineResult {
    let quality = assessQuality(frame);
    if (!this.options.calibrated) {
      quality = {
        ...quality,
        assessable: false,
        label: "Calibration needed",
        reasons: [
          ...quality.reasons,
          "Confirm stance and anatomical sides before counting.",
        ],
      };
    }
    const aspect = frame.width / frame.height;
    const timeGap = this.lastTime === null ? null : frame.t - this.lastTime;
    if (
      timeGap !== null &&
      (timeGap <= 0 || timeGap > MOTION_LIMITS.maximumFrameGapMs)
    ) {
      quality = {
        ...quality,
        assessable: false,
        label: "Timing interrupted",
        reasons: [
          ...quality.reasons,
          "Frame timing changed or frames were missed. Restart from your resting position.",
        ],
      };
    }
    if (
      this.lastAspect !== null &&
      Math.abs(aspect / this.lastAspect - 1) > 0.01
    ) {
      quality = {
        ...quality,
        assessable: false,
        label: "Camera framing changed",
        reasons: [
          ...quality.reasons,
          "Image proportions changed. Restart from your resting position.",
        ],
      };
    }
    if (!quality.assessable) {
      this.clearTracking();
      return { quality, events: [], activeHand: null };
    }
    this.lastTime = frame.t;
    this.lastAspect = aspect;
    const events: PunchEvent[] = [];
    for (const hand of ["left", "right"] as const) {
      const event = this.updateArm(hand, armGeometry(frame, hand));
      if (event) events.push(event);
    }
    const activeHand = this.arms.left.candidate
      ? "left"
      : this.arms.right.candidate
        ? "right"
        : null;
    return { quality, events, activeHand };
  }

  private updateArm(hand: Hand, sample: ArmSample): PunchEvent | null {
    const state = this.arms[hand];
    const previous = state.previous;
    state.previous = sample;
    if (!previous || !Number.isFinite(sample.angle)) return null;

    const candidate = state.candidate;
    if (!candidate) {
      const ready =
        state.rest &&
        state.restFrames >= MOTION_LIMITS.minimumRestFrames &&
        previous.t - state.restStartMs >= MOTION_LIMITS.minimumRestMs;
      if (
        ready &&
        sample.reach - state.rest!.reach > 0.18 &&
        sample.angle - state.rest!.angle > 18 &&
        sample.reach > previous.reach
      ) {
        state.candidate = {
          origin: state.rest!,
          startMs: previous.t,
          peakMs: sample.t,
          peakReach: sample.reach,
          peakAngle: sample.angle,
          peakWrist: sample.wrist,
          peakFrames: 0,
          pathLength: distance(state.rest!.wrist, sample.wrist),
          recovering: false,
          recoveryStartMs: null,
          recoveryFrames: 0,
        };
        this.observePeak(state.candidate, sample);
        return null;
      }
      // Rearm only from a flexed, quiet hand. An extended hand cannot generate
      // repeated events, and every accepted event needs a new resting baseline.
      if (ready) {
        if (
          sample.angle <= 130 &&
          distance(sample.wrist, state.rest!.wrist) <= 0.12
        ) {
          // Refresh freshness without following an outgoing hand: otherwise
          // slow motions or high-fps input can continually move the baseline.
          state.rest = { ...state.rest!, t: sample.t };
        } else if (sample.t - state.rest!.t > 350) {
          state.rest = null;
          state.restFrames = 0;
        }
        return null;
      }
      if (
        sample.angle <= 130 &&
        distance(sample.wrist, previous.wrist) <= 0.06
      ) {
        if (!state.rest || distance(sample.wrist, state.rest.wrist) > 0.18) {
          state.restStartMs = previous.t;
          state.restFrames = 1;
          state.rest = sample;
        } else state.restFrames += 1;
      } else {
        state.rest = null;
        state.restFrames = 0;
      }
      return null;
    }

    if (sample.t - candidate.startMs > MOTION_LIMITS.maximumEventMs) {
      this.arms[hand] = freshArm();
      return null;
    }
    if (!candidate.recovering) {
      candidate.pathLength += distance(sample.wrist, previous.wrist);
      this.observePeak(candidate, sample);
      if (
        candidate.peakReach - sample.reach > 0.14 &&
        sample.angle < candidate.peakAngle - 12
      ) {
        if (candidate.peakFrames < 2) {
          this.arms[hand] = freshArm();
          return null;
        }
        const chord = distance(candidate.origin.wrist, candidate.peakWrist);
        // Reject obvious arcs. Projected linearity is not proof of punch type.
        if (chord < 0.35 || candidate.pathLength / chord > 1.65) {
          this.arms[hand] = freshArm();
          return null;
        }
        candidate.recovering = true;
      }
    }
    if (!candidate.recovering) return null;
    const recovered =
      sample.angle <= 130 && sample.reach <= candidate.origin.reach + 0.2;
    if (!recovered) {
      candidate.recoveryFrames = 0;
      candidate.recoveryStartMs = null;
      return null;
    }
    candidate.recoveryStartMs ??= sample.t;
    candidate.recoveryFrames += 1;
    if (
      candidate.recoveryFrames < 2 ||
      sample.t - candidate.recoveryStartMs < MOTION_LIMITS.minimumRecoveryMs
    )
      return null;
    if (sample.t - candidate.startMs < MOTION_LIMITS.minimumEventMs) {
      this.arms[hand] = freshArm();
      return null;
    }

    const role = anatomicalRole(hand, this.options.stance);
    const extension = candidate.peakReach - candidate.origin.reach;
    const score = Math.min(
      0.99,
      0.55 +
        0.2 * Math.min(extension / 1.4, 1) +
        0.15 *
          Math.min(
            (candidate.peakAngle - MOTION_LIMITS.minimumPeakAngle) / 30,
            1,
          ) +
        0.1 * Math.min(candidate.peakFrames / 3, 1),
    );
    const event: PunchEvent = {
      id: `motion-${++this.sequence}-${hand}-${Math.round(candidate.startMs)}`,
      hand,
      role,
      label: role === "lead" ? "jab" : "cross",
      startMs: candidate.startMs,
      peakMs: candidate.peakMs,
      endMs: candidate.recoveryStartMs,
      detectedAtMs: sample.t,
      score,
      extension,
      // Describes return to this repetition's origin, never a correctness score.
      guardReturn:
        distance(sample.wrist, candidate.origin.wrist) <= 0.28
          ? "returned"
          : "not-observed",
      experimental: true,
    };
    this.arms[hand] = freshArm();
    return event;
  }

  private observePeak(candidate: Candidate, sample: ArmSample): void {
    if (sample.reach > candidate.peakReach) {
      candidate.peakReach = sample.reach;
      candidate.peakWrist = sample.wrist;
      candidate.peakMs = sample.t;
    }
    candidate.peakAngle = Math.max(candidate.peakAngle, sample.angle);
    if (
      sample.reach - candidate.origin.reach >= MOTION_LIMITS.minimumExtension &&
      sample.angle >= MOTION_LIMITS.minimumPeakAngle
    )
      candidate.peakFrames += 1;
  }
}
