/** Hands-free recording delay. Tracking quality must never block capture. */
export const READINESS_DEFAULTS = { countdownMs: 8_000 } as const;

export interface ReadinessOptions {
  countdownMs: number;
}
export interface ReadinessInput {
  /** Monotonic wall time, normally performance.now(). */
  nowMs: number;
  /** Accepted for caller compatibility; source timing does not gate recording. */
  frameTimeMs?: number | null;
  /** Diagnostic only. Recording starts even when no body is visible. */
  assessable?: boolean;
}
export type ReadinessPhase = "idle" | "countdown" | "ready" | "failed";
export interface ReadinessResult {
  phase: ReadinessPhase;
  pending: boolean;
  countdownSeconds: number;
  /** Compatibility field: 0 before the countdown completes, then 1. */
  framingProgress: number;
  /** One-shot action; later updates cannot start another recording. */
  start: boolean;
  reason: "clock-reset" | null;
}

/**
 * Click near the laptop, step back, and record after eight seconds without a
 * second click or visibility test. Call cancel() on navigation/source changes.
 * Camera availability and whether a round is running remain caller concerns.
 */
export class ReadinessGate {
  private readonly countdownMs: number;
  private phase: ReadinessPhase = "idle";
  private reason: ReadinessResult["reason"] = null;
  private armedAt = 0;
  private lastNow = 0;

  constructor(options: Partial<ReadinessOptions> = {}) {
    this.countdownMs = options.countdownMs ?? READINESS_DEFAULTS.countdownMs;
    if (!Number.isFinite(this.countdownMs) || this.countdownMs < 0)
      throw new RangeError("Countdown must be a finite, nonnegative duration.");
  }
  arm(nowMs: number): ReadinessResult {
    // Repeated clicks never restart or extend a pending countdown.
    if (this.phase === "countdown") return this.result();
    this.cancel();
    if (!Number.isFinite(nowMs) || nowMs < 0) return this.fail();
    this.armedAt = nowMs;
    this.lastNow = nowMs;
    this.phase = "countdown";
    return this.result();
  }
  cancel(): ReadinessResult {
    this.phase = "idle";
    this.reason = null;
    this.armedAt = 0;
    this.lastNow = 0;
    return this.result();
  }
  update({ nowMs }: ReadinessInput): ReadinessResult {
    if (this.phase !== "countdown") return this.result();
    if (!Number.isFinite(nowMs) || nowMs < this.lastNow) return this.fail();
    this.lastNow = nowMs;
    if (nowMs - this.armedAt >= this.countdownMs) {
      this.phase = "ready";
      return this.result(true);
    }
    return this.result();
  }
  private fail(): ReadinessResult {
    this.phase = "failed";
    this.reason = "clock-reset";
    return this.result();
  }
  private result(start = false): ReadinessResult {
    return {
      phase: this.phase,
      pending: this.phase === "countdown",
      countdownSeconds:
        this.phase === "countdown"
          ? Math.ceil(
              Math.max(0, this.countdownMs - (this.lastNow - this.armedAt)) /
                1000,
            )
          : 0,
      framingProgress: this.phase === "ready" ? 1 : 0,
      start,
      reason: this.reason,
    };
  }
}
