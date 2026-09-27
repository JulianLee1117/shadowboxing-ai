/**
 * Experimental image-plane curved-punch evidence. This is a directional motion
 * heuristic, not a trained six-punch model or a technique-quality assessment.
 * The caller must reset it on missing joints, timing gaps, or framing changes.
 */
export interface CurvedSample {
  t: number;
  wrist: { x: number; y: number };
  elbow: { x: number; y: number };
  angle: number;
}

export interface CurvedEvidence {
  label: "hook" | "uppercut";
  startMs: number;
  peakMs: number;
  endMs: number;
  detectedAtMs: number;
  excursion: number;
  score: number;
  returned: boolean;
}

/** All distances are shoulder-relative torso lengths, after aspect correction. */
export const CURVED_LIMITS = {
  minimumGuardMs: 60,
  minimumGuardFrames: 3,
  minimumExcursion: 0.45,
  minimumRise: 0.4,
  minimumOutboundMs: 60,
  maximumOutboundMs: 900,
  minimumEventMs: 180,
  maximumEventMs: 1800,
  minimumRecoveryMs: 30,
  directionDominance: 1.5,
  uppercutDirectionDominance: 1.4,
  maximumPeakAngle: 140,
} as const;

const distance = (a: CurvedSample["wrist"], b: CurvedSample["wrist"]) =>
  Math.hypot(a.x - b.x, a.y - b.y);
const guarded = (sample: CurvedSample) =>
  sample.angle <= 130 &&
  sample.wrist.y >= -0.65 &&
  sample.wrist.y <= 0.55 &&
  sample.elbow.y > sample.wrist.y + 0.08;

interface CurvedCandidate {
  origin: CurvedSample;
  guardOrigin: CurvedSample;
  loading: boolean;
  loaded: boolean;
  samples: CurvedSample[];
  peakIndex: number;
  peakDistance: number;
  label: CurvedEvidence["label"] | null;
  recoveryStart: number | null;
  recoveryFrames: number;
}

/** Inspect observed samples only. A depth-only arc remains unclassified. */
function family(candidate: CurvedCandidate): CurvedEvidence["label"] | null {
  const { origin, samples, peakIndex, peakDistance } = candidate;
  const peak = samples[peakIndex];
  const before = samples[peakIndex - 1];
  const after = samples[peakIndex + 1];
  if (!before || !after || peakIndex < 3) return null;
  const duration = peak.t - origin.t;
  if (
    duration < CURVED_LIMITS.minimumOutboundMs ||
    duration > CURVED_LIMITS.maximumOutboundMs ||
    peak.angle < 25 ||
    peak.angle > CURVED_LIMITS.maximumPeakAngle ||
    before.angle > CURVED_LIMITS.maximumPeakAngle ||
    after.angle > CURVED_LIMITS.maximumPeakAngle ||
    distance(before.wrist, origin.wrist) < peakDistance * 0.65 ||
    distance(after.wrist, origin.wrist) < peakDistance * 0.65
  )
    return null;

  const dx = peak.wrist.x - origin.wrist.x;
  const dy = peak.wrist.y - origin.wrist.y;
  let path = 0;
  let advances = 0;
  let previousProjection = 0;
  for (let i = 1; i <= peakIndex; i++) {
    const sample = samples[i];
    const step = distance(samples[i - 1].wrist, sample.wrist);
    const projection =
      ((sample.wrist.x - origin.wrist.x) * dx +
        (sample.wrist.y - origin.wrist.y) * dy) /
      peakDistance;
    // Do not let one jumped landmark supply the apparent excursion. Small
    // reversals are tolerated; a loop or repeated direction changes are not.
    if (step > peakDistance * 0.65 || projection < previousProjection - 0.08)
      return null;
    if (projection - previousProjection >= 0.01) advances += 1;
    previousProjection = projection;
    path += step;
  }
  if (advances < 3 || path / peakDistance > 1.8) return null;

  // Curved rising punches can first travel forward from their loaded position.
  // Measure the final observed drive as well as the net displacement; do not
  // confuse that initial forward travel with a horizontal strike.
  const driveEnd = samples
    .slice(1, peakIndex + 1)
    .find((value) => distance(value.wrist, peak.wrist) <= peakDistance * 0.1)!;
  const terminalStart = samples
    .slice(0, peakIndex)
    .find((value) => driveEnd.t > value.t && driveEnd.t - value.t <= 150);
  const terminalDx = terminalStart
    ? driveEnd.wrist.x - terminalStart.wrist.x
    : 0;
  const terminalRise = terminalStart
    ? terminalStart.wrist.y - driveEnd.wrist.y
    : 0;
  const terminalObserved =
    terminalStart !== undefined && driveEnd.t - terminalStart.t >= 60;

  const hook =
    Math.abs(dx) >= CURVED_LIMITS.minimumExcursion &&
    (candidate.loaded
      ? terminalObserved &&
        Math.abs(terminalDx) >= 0.25 &&
        Math.abs(terminalDx) >=
          Math.abs(terminalRise) * CURVED_LIMITS.directionDominance &&
        Math.abs(peak.wrist.y - peak.elbow.y) <= 0.25
      : Math.abs(dx) >= Math.abs(dy) * CURVED_LIMITS.directionDominance) &&
    peak.wrist.y >= -0.65 &&
    peak.wrist.y <= 0.25 &&
    peak.elbow.y <= 0.4 &&
    // A partly extended straight is not a hook: a sweep transports a still-bent
    // arm instead of deriving its displacement chiefly from elbow opening.
    peak.angle - origin.angle <= 45 &&
    Math.sign(dx) * (peak.elbow.x - origin.elbow.x) >= Math.abs(dx) * 0.35 &&
    distance(peak.elbow, origin.elbow) >= 0.15;
  const uppercut =
    -dy >= CURVED_LIMITS.minimumRise &&
    terminalObserved &&
    terminalRise >= 0.25 &&
    terminalRise >=
      Math.abs(terminalDx) * CURVED_LIMITS.uppercutDirectionDominance &&
    origin.wrist.y >= -0.15 &&
    peak.wrist.y <= -0.1 &&
    peak.elbow.y >= -0.1 &&
    peak.elbow.y > peak.wrist.y + 0.12 &&
    peak.elbow.y - peak.wrist.y >=
      Math.abs(peak.elbow.x - peak.wrist.x) * 1.25 &&
    origin.elbow.y - peak.elbow.y >= 0.08;
  if (hook === uppercut) return null;
  return hook ? "hook" : "uppercut";
}

export class CurvedMotionObserver {
  private previous: CurvedSample | null = null;
  private guard: CurvedSample | null = null;
  private guardStart = 0;
  private guardFrames = 0;
  private ready = false;
  private candidate: CurvedCandidate | null = null;
  private returnTarget: CurvedSample | null = null;

  get active(): boolean {
    return this.candidate !== null;
  }

  /** Supported bent-arm evidence must not be relabeled as a straight punch. */
  get blocksStraight(): boolean {
    return this.candidate?.label !== null && this.candidate !== null;
  }

  reset(): void {
    this.previous = null;
    this.guard = null;
    this.guardFrames = 0;
    this.ready = false;
    this.candidate = null;
    this.returnTarget = null;
  }

  update(sample: CurvedSample): CurvedEvidence | null {
    const previous = this.previous;
    this.previous = sample;
    if (!previous) {
      if (guarded(sample)) {
        this.guard = sample;
        this.guardStart = sample.t;
        this.guardFrames = 1;
      }
      return null;
    }
    if (!this.candidate) {
      if (this.guard && this.ready) {
        if (this.returnTarget) {
          // A confirmed return can finish moving inward. Do not mistake the
          // remaining retraction for the beginning of another curved punch.
          const inward =
            distance(sample.wrist, this.returnTarget.wrist) <
            distance(previous.wrist, this.returnTarget.wrist) - 0.001;
          if (inward && sample.angle <= previous.angle + 3 && guarded(sample)) {
            this.guard = sample;
            return null;
          }
          this.returnTarget = null;
        }
        if (distance(sample.wrist, this.guard.wrist) > 0.18) {
          this.candidate = {
            origin: this.guard,
            guardOrigin: this.guard,
            loading: sample.wrist.y - this.guard.wrist.y > 0.18,
            loaded: sample.wrist.y - this.guard.wrist.y > 0.18,
            samples: [this.guard, previous, sample].filter(
              (value, i, all) => i === 0 || value.t > all[i - 1].t,
            ),
            peakIndex: 0,
            peakDistance: 0,
            label: null,
            recoveryStart: null,
            recoveryFrames: 0,
          };
          const candidate = this.candidate;
          if (candidate.loading) {
            candidate.origin = sample;
            candidate.samples = [sample];
          }
          candidate.peakIndex = candidate.samples.length - 1;
          candidate.peakDistance = distance(
            sample.wrist,
            candidate.origin.wrist,
          );
          return null;
        }
        if (guarded(sample)) this.guard = { ...this.guard, t: sample.t };
        else if (sample.t - this.guard.t > 350) this.reset();
        return null;
      }
      if (guarded(sample) && distance(sample.wrist, previous.wrist) <= 0.06) {
        if (!this.guard || distance(sample.wrist, this.guard.wrist) > 0.12) {
          this.guard = sample;
          this.guardStart = previous.t;
          this.guardFrames = 1;
        } else this.guardFrames += 1;
        this.ready =
          this.guardFrames >= CURVED_LIMITS.minimumGuardFrames &&
          sample.t - this.guardStart >= CURVED_LIMITS.minimumGuardMs;
        if (this.ready) this.guard = { ...this.guard!, t: sample.t };
      } else {
        this.guard = null;
        this.guardFrames = 0;
      }
      return null;
    }

    const candidate = this.candidate;
    if (sample.t - candidate.guardOrigin.t > CURVED_LIMITS.maximumEventMs) {
      this.reset();
      return null;
    }
    if (candidate.loading) {
      if (sample.wrist.y >= candidate.origin.wrist.y) {
        candidate.origin = sample;
        candidate.samples = [sample];
        candidate.peakIndex = 0;
        candidate.peakDistance = 0;
        return null;
      }
      if (candidate.origin.wrist.y - sample.wrist.y >= 0.1)
        candidate.loading = false;
    }
    candidate.samples.push(sample);
    const displacement = distance(sample.wrist, candidate.origin.wrist);
    if (displacement > candidate.peakDistance) {
      candidate.peakIndex = candidate.samples.length - 1;
      candidate.peakDistance = displacement;
      // An earlier plausible peak must not authorize a later incompatible one.
      candidate.label = null;
    }
    candidate.label = family(candidate);
    const returnedToGuard = distance(sample.wrist, candidate.guardOrigin.wrist);
    const peak = candidate.samples[candidate.peakIndex];
    const projectedRemaining =
      ((sample.wrist.x - candidate.origin.wrist.x) *
        (peak.wrist.x - candidate.origin.wrist.x) +
        (sample.wrist.y - candidate.origin.wrist.y) *
          (peak.wrist.y - candidate.origin.wrist.y)) /
      Math.max(candidate.peakDistance, 1e-8);
    const hookRetracted =
      candidate.label === "hook" &&
      projectedRemaining <= candidate.peakDistance * 0.5 &&
      distance(sample.wrist, peak.wrist) >= 0.22;
    const returning =
      guarded(sample) &&
      (hookRetracted ||
        (candidate.loaded
          ? returnedToGuard <= 0.35 &&
            (candidate.label === "uppercut"
              ? sample.wrist.y - peak.wrist.y >= 0.18
              : candidate.label === "hook" &&
                Math.sign(peak.wrist.x - candidate.origin.wrist.x) *
                  (peak.wrist.x - sample.wrist.x) >=
                  0.22)
          : displacement <= Math.max(0.25, candidate.peakDistance * 0.4) &&
            candidate.peakDistance - displacement >= 0.22));
    if (!returning) {
      candidate.recoveryFrames = 0;
      candidate.recoveryStart = null;
      return null;
    }
    candidate.recoveryStart ??= sample.t;
    candidate.recoveryFrames += 1;
    if (
      candidate.recoveryFrames < 2 ||
      sample.t - candidate.recoveryStart < CURVED_LIMITS.minimumRecoveryMs
    )
      return null;
    const event =
      candidate.label &&
      sample.t - candidate.origin.t >= CURVED_LIMITS.minimumEventMs
        ? {
            label: candidate.label,
            startMs: candidate.origin.t,
            peakMs: candidate.samples[candidate.peakIndex].t,
            endMs: candidate.recoveryStart,
            detectedAtMs: sample.t,
            excursion: candidate.peakDistance,
            score: Math.min(0.85, 0.55 + candidate.peakDistance * 0.15),
            returned: returnedToGuard <= 0.28,
          }
        : null;
    this.reset();
    this.previous = sample;
    this.guard = sample;
    this.guardStart = sample.t;
    this.guardFrames = 1;
    // Only a completed accepted cycle can immediately rearm another curve.
    this.ready = event !== null;
    this.returnTarget = event ? candidate.guardOrigin : null;
    return event;
  }
}
