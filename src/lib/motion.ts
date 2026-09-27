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
export const DETECTOR_VERSION = "projected-straight-v6-observed-repeats";

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
  maximumAcquisitionWindowMs: 200,
  maximumAcquisitionSamples: 32,
  minimumTroughReachChange: 0.06,
  minimumTroughAngleChange: 6,
  minimumExtension: 0.45,
  minimumPeakAngle: 145,
  minimumPeakSupportFraction: 0.8,
  minimumEventMs: 180,
  maximumEventMs: 1800,
  minimumRecoveryMs: 30,
  minimumRetractionFraction: 0.5,
  maximumRepeatPeakGapMs: 650,
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

/** Measure the outward path with at most one isolated reversing detour omitted.
 * Does not alter landmarks, peak time, peak reach/angle, or guard evidence.
 * A selected peak can never be removed. At least two OTHER strict observed
 * peaks must remain. This is path robustness, not a pose correction.
 */
export function robustOutboundPath(
  samples: ArmSample[],
  origin: ArmSample,
  peakMs: number,
): { path: number; removed: number[] } {
  const outgoing = samples.filter((s) => s.t <= peakMs);
  const raw = outgoing
    .slice(1)
    .reduce((sum, s, i) => sum + distance(outgoing[i].wrist, s.wrist), 0);
  const suspects: number[] = [];
  for (let i = 1; i < outgoing.length - 1; i++) {
    const [a, b, c] = [outgoing[i - 1], outgoing[i], outgoing[i + 1]];
    if (b.t === peakMs || b.t - a.t > 70 || c.t - b.t > 70) continue;
    const first = distance(a.wrist, b.wrist),
      second = distance(b.wrist, c.wrist);
    const bypass = distance(a.wrist, c.wrist);
    const dot =
      (b.wrist.x - a.wrist.x) * (c.wrist.x - b.wrist.x) +
      (b.wrist.y - a.wrist.y) * (c.wrist.y - b.wrist.y);
    if (
      first > 0.3 &&
      second > 0.3 &&
      dot / (first * second) < -0.85 &&
      bypass < Math.min(first, second) * 0.35
    )
      suspects.push(i);
  }
  if (suspects.length !== 1) return { path: raw, removed: [] };
  const remove = suspects[0];
  const retained = outgoing.filter((_, i) => i !== remove);
  const strict = retained.filter(
    (s) =>
      s.t !== origin.t &&
      s.reach - origin.reach >= MOTION_LIMITS.minimumExtension &&
      s.angle >= MOTION_LIMITS.minimumPeakAngle,
  );
  if (strict.length < 2) return { path: raw, removed: [] };
  return {
    path: retained
      .slice(1)
      .reduce((sum, s, i) => sum + distance(retained[i].wrist, s.wrist), 0),
    removed: [outgoing[remove].t],
  };
}

/** Accepted full-stroke evidence with a fixed lifetime; repeat-only events cannot renew it. */
interface RepeatAnchor {
  origin: ArmSample;
  peakMs: number;
  expiresAtMs: number;
}

interface Candidate {
  repeatAnchor: RepeatAnchor | null;
  fullPeakFrames: number;
  fullSupportedPeak: boolean;
  origin: ArmSample;
  startMs: number;
  peakMs: number;
  peakReach: number;
  peakAngle: number;
  peakWrist: Point;
  peakFrames: number;
  pathLength: number;
  pathToPeak: number;
  pathSamples: ArmSample[];
  pendingPeak: { before: ArmSample; peak: ArmSample; full: boolean } | null;
  supportedPeak: boolean;
  recovering: boolean;
  recoveryStartMs: number | null;
  recoveryFrames: number;
}

interface ArmState {
  repeatAnchor: RepeatAnchor | null;
  previous: ArmSample | null;
  acquisitionSamples: ArmSample[];
  rest: ArmSample | null;
  restStartMs: number;
  restFrames: number;
  readyFromMotion: boolean;
  candidate: Candidate | null;
}

const freshArm = (): ArmState => ({
  repeatAnchor: null,
  previous: null,
  acquisitionSamples: [],
  rest: null,
  restStartMs: 0,
  restFrames: 0,
  readyFromMotion: false,
  candidate: null,
});

/** A flexed reversal observed over time, with no inferred or interpolated samples. */
function observedTrough(samples: ArmSample[]): ArmSample | null {
  const current = samples.at(-1);
  if (!current || samples.length < 3) return null;
  // Search newest first. Both sides must move coherently away from an observed
  // minimum; elapsed-time support avoids a three-frame assumption at high fps.
  for (let troughIndex = samples.length - 2; troughIndex > 0; troughIndex--) {
    const trough = samples[troughIndex];
    if (
      trough.angle > 130 ||
      current.reach - trough.reach < MOTION_LIMITS.minimumTroughReachChange ||
      current.angle - trough.angle < MOTION_LIMITS.minimumTroughAngleChange
    )
      continue;
    let outgoing = true;
    for (let i = troughIndex + 1; i < samples.length; i++) {
      if (
        samples[i].reach < samples[i - 1].reach ||
        samples[i].angle < samples[i - 1].angle
      ) {
        outgoing = false;
        break;
      }
    }
    if (!outgoing) continue;
    for (let i = troughIndex - 1; i >= 0; i--) {
      const before = samples[i];
      const next = samples[i + 1];
      if (before.reach < next.reach || before.angle < next.angle) break;
      if (
        current.t - before.t >= MOTION_LIMITS.minimumRestMs &&
        before.reach - trough.reach >= MOTION_LIMITS.minimumTroughReachChange &&
        before.angle - trough.angle >= MOTION_LIMITS.minimumTroughAngleChange
      )
        return trough;
    }
  }
  return null;
}

/** A guarded reference may follow a coherent observed inward run, never extension. */
function coherentInward(samples: readonly ArmSample[]): boolean {
  if (samples.length < MOTION_LIMITS.minimumRestFrames) return false;
  const latest = samples[samples.length - 1];
  let start = samples.length - 1;
  while (start > 0) {
    const earlier = samples[start - 1];
    const later = samples[start];
    if (
      earlier.angle > 130 ||
      later.reach > earlier.reach ||
      later.angle > earlier.angle
    )
      break;
    start -= 1;
    if (
      samples.length - start >= MOTION_LIMITS.minimumRestFrames &&
      latest.t - earlier.t >= MOTION_LIMITS.minimumRestMs
    )
      return true;
  }
  return false;
}

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
    if (!Number.isFinite(sample.angle)) {
      this.arms[hand] = freshArm();
      return null;
    }
    if (!state.candidate) {
      state.acquisitionSamples = [...state.acquisitionSamples, sample]
        .filter(
          (value) =>
            sample.t - value.t <= MOTION_LIMITS.maximumAcquisitionWindowMs,
        )
        .slice(-MOTION_LIMITS.maximumAcquisitionSamples);
    }
    if (!previous) return null;

    const candidate = state.candidate;
    if (!candidate) {
      const hadReadyReference =
        state.rest &&
        (state.readyFromMotion ||
          (state.restFrames >= MOTION_LIMITS.minimumRestFrames &&
            previous.t - state.restStartMs >= MOTION_LIMITS.minimumRestMs));
      const trough = observedTrough(state.acquisitionSamples);
      if (
        trough &&
        (!hadReadyReference ||
          (trough.reach < state.rest!.reach &&
            trough.angle <= state.rest!.angle))
      ) {
        // A moving arm may acquire an origin at an observed flexed reversal.
        // An existing origin can only move farther inward, never follow extension.
        state.rest = trough;
        state.restStartMs = trough.t;
        state.restFrames = 1;
        state.readyFromMotion = true;
      }
      const ready =
        state.rest &&
        (state.readyFromMotion ||
          (state.restFrames >= MOTION_LIMITS.minimumRestFrames &&
            previous.t - state.restStartMs >= MOTION_LIMITS.minimumRestMs));
      if (
        ready &&
        sample.reach - state.rest!.reach > 0.18 &&
        sample.angle - state.rest!.angle > 18 &&
        sample.reach > previous.reach
      ) {
        state.candidate = {
          repeatAnchor: state.repeatAnchor,
          fullPeakFrames: 0,
          fullSupportedPeak: false,
          origin: state.rest!,
          startMs: previous.t,
          peakMs: sample.t,
          peakReach: sample.reach,
          peakAngle: sample.angle,
          peakWrist: sample.wrist,
          peakFrames: 0,
          pathLength: distance(state.rest!.wrist, sample.wrist),
          pathToPeak: distance(state.rest!.wrist, sample.wrist),
          pathSamples: [state.rest!, sample],
          pendingPeak: null,
          supportedPeak: false,
          recovering: false,
          recoveryStartMs: null,
          recoveryFrames: 0,
        };
        state.acquisitionSamples = [];
        this.observePeak(state.candidate, sample, previous);
        return null;
      }
      // Maintain a reference acquired from rest, reversal, or confirmed return. An
      // extended hand cannot rearm, and outgoing motion must not drag the origin.
      if (ready) {
        if (
          (state.readyFromMotion || coherentInward(state.acquisitionSamples)) &&
          sample.angle <= 130 &&
          sample.reach < state.rest!.reach &&
          sample.angle <= state.rest!.angle
        ) {
          // Follow an already-ready guard only farther inward with elbow flexion.
          // Static acquisition additionally needs a coherent observed run; a
          // moving wrist must not expire solely because its old guard moved.
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
          state.readyFromMotion = false;
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
        state.readyFromMotion = false;
      }
      return null;
    }

    if (sample.t - candidate.startMs > MOTION_LIMITS.maximumEventMs) {
      this.arms[hand] = freshArm();
      return null;
    }
    if (!candidate.recovering) {
      candidate.pathLength += distance(sample.wrist, previous.wrist);
      candidate.pathSamples.push(sample);
      this.observePeak(candidate, sample, previous);
      if (
        candidate.peakReach - sample.reach > 0.14 &&
        sample.angle < candidate.peakAngle - 12
      ) {
        // Earlier qualifying samples must not authorize a later selected peak
        // outside the repeat window. Isolated full qualification is unchanged.
        const fullQualified =
          candidate.fullPeakFrames >= 2 || candidate.fullSupportedPeak;
        const repeatPeakInWindow =
          candidate.repeatAnchor !== null &&
          candidate.peakMs <= candidate.repeatAnchor.expiresAtMs;
        if (
          (candidate.peakFrames < 2 && !candidate.supportedPeak) ||
          (!fullQualified && !repeatPeakInWindow)
        ) {
          // Keep only directly observed recent acquisition evidence, never a
          // ready reference or rejected peak. Tracking/timing resets still erase it.
          this.arms[hand] = {
            ...freshArm(),
            previous: sample,
            acquisitionSamples: candidate.pathSamples
              .filter(
                (value) =>
                  sample.t - value.t <=
                  MOTION_LIMITS.maximumAcquisitionWindowMs,
              )
              .slice(-MOTION_LIMITS.maximumAcquisitionSamples),
          };
          return null;
        }
        const chord = distance(candidate.origin.wrist, candidate.peakWrist);
        // Compare the outward path with its outward chord. Including return
        // travel here penalizes a straight punch simply for retracting quickly.
        // Projected linearity is still not proof of punch type.
        const measuredPath = robustOutboundPath(
          candidate.pathSamples,
          candidate.origin,
          candidate.peakMs,
        );
        if (chord < 0.35 || measuredPath.path / chord > 1.65) {
          this.arms[hand] = freshArm();
          return null;
        }
        candidate.recovering = true;
      }
    }
    if (!candidate.recovering) return null;
    // Confirm a flexed retraction, independently of returning to the old guard.
    // Retain the previous close-origin criterion for shorter observed excursions.
    const recovered =
      sample.angle <= 130 &&
      sample.reach <=
        Math.max(
          candidate.origin.reach + 0.2,
          candidate.origin.reach +
            (candidate.peakReach - candidate.origin.reach) *
              (1 - MOTION_LIMITS.minimumRetractionFraction),
        );
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
      // Spatial return to the origin observed by detectedAtMs, not a form score
      // or a claim about whether the hand returns later.
      guardReturn:
        distance(sample.wrist, candidate.origin.wrist) <= 0.28
          ? "returned"
          : "not-observed",
      experimental: true,
    };
    // Confirmed flexed retraction already separates two movements. Preserve its
    // actual last sample without requiring another quiet guard or spatial return.
    // Only farther inward observations may update it; tracking loss still clears it.
    this.arms[hand] = {
      ...freshArm(),
      // Only independently full-qualified accepted evidence creates a new
      // anchor. A repeat may retain its original origin/deadline, never move them.
      repeatAnchor:
        candidate.fullPeakFrames >= 2 || candidate.fullSupportedPeak
          ? {
              origin: candidate.origin,
              peakMs: candidate.peakMs,
              expiresAtMs:
                candidate.peakMs + MOTION_LIMITS.maximumRepeatPeakGapMs,
            }
          : candidate.repeatAnchor &&
              sample.t <= candidate.repeatAnchor.expiresAtMs
            ? candidate.repeatAnchor
            : null,
      previous: sample,
      rest: sample,
      restStartMs: sample.t,
      restFrames: 1,
      readyFromMotion: true,
    };
    return event;
  }

  private observePeak(
    candidate: Candidate,
    sample: ArmSample,
    previous: ArmSample,
  ): void {
    if (candidate.pendingPeak) {
      const { before, peak, full } = candidate.pendingPeak;
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
      ) {
        candidate.supportedPeak = true;
        if (full) candidate.fullSupportedPeak = true;
      }
      candidate.pendingPeak = null;
    }
    if (sample.reach > candidate.peakReach) {
      candidate.peakReach = sample.reach;
      candidate.peakWrist = sample.wrist;
      candidate.peakMs = sample.t;
      candidate.pathToPeak = candidate.pathLength;
    }
    candidate.peakAngle = Math.max(candidate.peakAngle, sample.angle);
    // Isolated strokes retain the full excursion requirement. A short repeat
    // additionally needs a recent accepted full stroke and its own real motion.
    const full =
      sample.reach - candidate.origin.reach >= MOTION_LIMITS.minimumExtension &&
      sample.angle >= MOTION_LIMITS.minimumPeakAngle;
    const anchor = candidate.repeatAnchor;
    const repeated =
      anchor !== null &&
      sample.t <= anchor.expiresAtMs &&
      sample.t > anchor.peakMs &&
      sample.reach - candidate.origin.reach >=
        MOTION_LIMITS.minimumExtension *
          MOTION_LIMITS.minimumRetractionFraction &&
      sample.reach - anchor.origin.reach >= MOTION_LIMITS.minimumExtension &&
      sample.angle >= MOTION_LIMITS.minimumPeakAngle;
    if (full || repeated) {
      candidate.peakFrames += 1;
      if (full) candidate.fullPeakFrames += 1;
      candidate.pendingPeak = { before: previous, peak: sample, full };
    }
  }
}
