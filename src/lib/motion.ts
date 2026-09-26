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

/** Include in saved evidence so replays can identify the counting rules used. */
export const DETECTOR_VERSION = "projected-straight-v3-causal-cycles";

export interface ArmTrackingState {
  hand: Hand;
  assessable: boolean;
  status: "tracking" | "hidden" | "foreshortened" | "invalid";
  reason: string | null;
  /** Image-space estimates only; neither metric establishes physical extension. */
  projectedAngle: number | null;
  projectedReach: number | null;
  minimumVisibility: number | null;
}

/** Experimental image-space thresholds, not a validated boxing rubric. */
export const MOTION_LIMITS = {
  minimumVisibility: 0.65,
  maximumFrameGapMs: 200,
  minimumRestMs: 60,
  minimumRestFrames: 3,
  minimumExtension: 0.45,
  minimumPeakAngle: 145,
  minimumPeakSupportFraction: 0.8,
  minimumEventMs: 180,
  maximumEventMs: 1800,
  minimumRecoveryMs: 30,
} as const;

const TORSO_JOINTS = [
  JOINT.leftShoulder,
  JOINT.rightShoulder,
  JOINT.leftHip,
  JOINT.rightHip,
];
const REQUIRED_JOINTS = [
  ...TORSO_JOINTS,
  JOINT.leftElbow,
  JOINT.rightElbow,
  JOINT.leftWrist,
  JOINT.rightWrist,
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

function sharedTrackingReasons(frame: PoseFrame): string[] {
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
  if (!TORSO_JOINTS.every((index) => visible(frame.landmarks[index]))) {
    reasons.push(
      "Keep both shoulders and hips visible inside the frame to measure arm motion.",
    );
  }
  if (reasons.length === 0) {
    const scale = torsoScale(frame);
    if (scale < 0.08)
      reasons.push("Move closer so your upper body is large enough to track.");
  }
  return reasons;
}

/**
 * Independent arm observability. A hidden guarding hand must not erase the
 * opposite arm's evidence. Both shoulders/hips remain necessary for torso scale.
 * Predicted depth is deliberately not substituted for missing image evidence.
 */
export function assessArmTracking(
  frame: PoseFrame,
): Record<Hand, ArmTrackingState> {
  const sharedReasons = sharedTrackingReasons(frame);
  const assess = (hand: Hand): ArmTrackingState => {
    const indices =
      hand === "left"
        ? [JOINT.leftShoulder, JOINT.leftElbow, JOINT.leftWrist]
        : [JOINT.rightShoulder, JOINT.rightElbow, JOINT.rightWrist];
    const points = indices.map((index) => frame.landmarks[index]);
    const minimumVisibility = points.every((point) =>
      Number.isFinite(point?.visibility),
    )
      ? Math.min(...points.map((point) => point!.visibility!))
      : null;
    const base: ArmTrackingState = {
      hand,
      assessable: false,
      status: "invalid",
      reason: null,
      projectedAngle: null,
      projectedReach: null,
      minimumVisibility,
    };
    if (sharedReasons.length)
      return { ...base, reason: sharedReasons.join(" ") };
    if (!points.every(visible)) {
      return {
        ...base,
        status: "hidden",
        reason: `Your ${hand} elbow or wrist is outside the frame or not reliably visible.`,
      };
    }
    const [shoulder, elbow, wrist] = points;
    if (
      imageDistance(shoulder, elbow, frame.width, frame.height) < 0.01 ||
      imageDistance(elbow, wrist, frame.width, frame.height) < 0.01
    ) {
      return {
        ...base,
        status: "foreshortened",
        reason: `Your ${hand} arm segments overlap or are too small in this view. Try a slight turn.`,
      };
    }
    const geometry = armGeometry(frame, hand);
    return {
      ...base,
      assessable: true,
      status: "tracking",
      projectedAngle: geometry.angle,
      projectedReach: geometry.reach,
    };
  };
  return { left: assess("left"), right: assess("right") };
}

function trackingQuality(
  frame: PoseFrame,
  arms: Record<Hand, ArmTrackingState>,
): QualityState {
  const assessable = arms.left.assessable || arms.right.assessable;
  return {
    assessable,
    label:
      arms.left.assessable && arms.right.assessable
        ? "Both arms visible"
        : assessable
          ? `${arms.left.assessable ? "Left" : "Right"} arm visible`
          : "Tracking unavailable",
    reasons: Array.from(
      new Set(
        [arms.left.reason, arms.right.reason].filter(
          (reason): reason is string => reason !== null,
        ),
      ),
    ),
    visibleJoints: REQUIRED_JOINTS.filter((index) =>
      visible(frame.landmarks[index]),
    ).length,
    totalJoints: REQUIRED_JOINTS.length,
  };
}

/** At least one arm is observable; this does not mean correct technique. */
export function assessQuality(frame: PoseFrame): QualityState {
  return trackingQuality(frame, assessArmTracking(frame));
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
  pathToPeak: number;
  pendingPeak: { before: ArmSample; peak: ArmSample } | null;
  supportedPeak: boolean;
  recovering: boolean;
  recoveryStartMs: number | null;
  recoveryFrames: number;
}

interface ArmState {
  previous: ArmSample | null;
  rest: ArmSample | null;
  restStartMs: number;
  restFrames: number;
  readyFromRecovery: boolean;
  candidate: Candidate | null;
}

const freshArm = (): ArmState => ({
  previous: null,
  rest: null,
  restStartMs: 0,
  restFrames: 0,
  readyFromRecovery: false,
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
    const armTracking = assessArmTracking(frame);
    let quality = trackingQuality(frame, armTracking);
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
      if (!armTracking[hand].assessable) {
        // Never bridge missing active-arm evidence or reuse its stale baseline.
        this.arms[hand] = freshArm();
        continue;
      }
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
        (state.readyFromRecovery ||
          (state.restFrames >= MOTION_LIMITS.minimumRestFrames &&
            previous.t - state.restStartMs >= MOTION_LIMITS.minimumRestMs));
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
          pathToPeak: distance(state.rest!.wrist, sample.wrist),
          pendingPeak: null,
          supportedPeak: false,
          recovering: false,
          recoveryStartMs: null,
          recoveryFrames: 0,
        };
        this.observePeak(state.candidate, sample, previous);
        return null;
      }
      // Maintain the acquired or confirmed-return reference in flexion. An
      // extended hand cannot rearm, and outgoing motion must not drag the origin.
      if (ready) {
        if (
          state.readyFromRecovery &&
          sample.angle <= 130 &&
          sample.reach < state.rest!.reach &&
          sample.angle <= state.rest!.angle
        ) {
          // Confirmation can occur before the hand has finished returning.
          // Follow only observed inward motion with continued elbow flexion;
          // freezing an early return would understate the next excursion.
          state.rest = sample;
          return null;
        }
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
          state.readyFromRecovery = false;
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
        state.readyFromRecovery = false;
      }
      return null;
    }

    if (sample.t - candidate.startMs > MOTION_LIMITS.maximumEventMs) {
      this.arms[hand] = freshArm();
      return null;
    }
    if (!candidate.recovering) {
      candidate.pathLength += distance(sample.wrist, previous.wrist);
      this.observePeak(candidate, sample, previous);
      if (
        candidate.peakReach - sample.reach > 0.14 &&
        sample.angle < candidate.peakAngle - 12
      ) {
        if (candidate.peakFrames < 2 && !candidate.supportedPeak) {
          this.arms[hand] = freshArm();
          return null;
        }
        const chord = distance(candidate.origin.wrist, candidate.peakWrist);
        // Compare the outward path with its outward chord. Including return
        // travel here penalizes a straight punch simply for retracting quickly.
        // Projected linearity is still not proof of punch type.
        if (chord < 0.35 || candidate.pathToPeak / chord > 1.65) {
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
    // Confirmed return already separates two movements. Preserve its real last
    // sample rather than discarding that evidence and forcing a fast double jab
    // to wait for another three quiet frames. No extra frames/time are invented.
    // Uncertain return or later occlusion still requires fresh acquisition.
    this.arms[hand] =
      event.guardReturn === "returned"
        ? {
            ...freshArm(),
            previous: sample,
            rest: sample,
            restStartMs: sample.t,
            restFrames: 1,
            readyFromRecovery: true,
          }
        : freshArm();
    return event;
  }

  private observePeak(
    candidate: Candidate,
    sample: ArmSample,
    previous: ArmSample,
  ): void {
    if (candidate.pendingPeak) {
      const { before, peak } = candidate.pendingPeak;
      const support = MOTION_LIMITS.minimumPeakSupportFraction;
      const reachFloor =
        candidate.origin.reach +
        (peak.reach - candidate.origin.reach) * support;
      const angleFloor =
        candidate.origin.angle +
        (peak.angle - candidate.origin.angle) * support;
      // One observed strict peak can fall between two almost-extended samples.
      // Require immediate observed neighbors on BOTH sides of that peak, each
      // supporting its excursion and elbow opening. Never invent an unseen
      // peak, or accept a lone baseline -> spike -> baseline jump.
      if (
        before.t < peak.t &&
        peak.t < sample.t &&
        before.reach < peak.reach &&
        sample.reach < peak.reach &&
        before.reach >= reachFloor &&
        sample.reach >= reachFloor &&
        before.angle >= angleFloor &&
        sample.angle >= angleFloor
      )
        candidate.supportedPeak = true;
      candidate.pendingPeak = null;
    }
    if (sample.reach > candidate.peakReach) {
      candidate.peakReach = sample.reach;
      candidate.peakWrist = sample.wrist;
      candidate.peakMs = sample.t;
      candidate.pathToPeak = candidate.pathLength;
    }
    candidate.peakAngle = Math.max(candidate.peakAngle, sample.angle);
    if (
      sample.reach - candidate.origin.reach >= MOTION_LIMITS.minimumExtension &&
      sample.angle >= MOTION_LIMITS.minimumPeakAngle
    ) {
      candidate.peakFrames += 1;
      candidate.pendingPeak = { before: previous, peak: sample };
    }
  }
}
