import type { NativeActionEvent, PoseFrame, PunchEvent, Stance } from "./types";

const FINGERPRINT = /^[a-f0-9]{64}$/;

export function nativeDetectorVersion(modelInfo: unknown): string | undefined {
  if (!modelInfo || typeof modelInfo !== "object") return;
  const info = (modelInfo as { recognizer?: Record<string, unknown> })
    .recognizer;
  if (
    info?.protocolVersion === "shadowbox-recognition-v1" &&
    info.recognizerId === "personal-hybrid-v1" &&
    typeof info.fingerprint === "string" &&
    FINGERPRINT.test(info.fingerprint)
  )
    return `${info.recognizerId}:${info.fingerprint}`;
}

/** Capture uses a continuous source clock; every saved decision uses round time.
 * A movement beginning before the round is not credited to the new round.
 */
export function relativePoseFrame(frame: PoseFrame, offset: number): PoseFrame {
  return {
    ...frame,
    t: frame.t - offset,
    ...(frame.recognition
      ? {
          recognition: {
            ...frame.recognition,
            events: frame.recognition.events
              .filter((event) => event.startMs >= offset)
              .map((event) => ({
                ...event,
                startMs: event.startMs - offset,
                peakMs: event.peakMs - offset,
                endMs: event.endMs - offset,
                detectedAtMs: event.detectedAtMs - offset,
              })),
          },
        }
      : {}),
  };
}

function validEvent(event: NativeActionEvent, t: number): boolean {
  return (
    typeof event.id === "string" &&
    event.id.length > 0 &&
    event.id.length <= 200 &&
    ["left", "right"].includes(event.hand) &&
    ["straight", "hook", "uppercut"].includes(event.family) &&
    [
      event.startMs,
      event.peakMs,
      event.endMs,
      event.detectedAtMs,
      event.score,
    ].every(Number.isFinite) &&
    event.startMs >= 0 &&
    event.startMs <= event.peakMs &&
    event.peakMs <= event.endMs &&
    event.startMs < event.endMs &&
    event.endMs <= event.detectedAtMs &&
    event.detectedAtMs <= t + 0.001 &&
    event.score >= 0 &&
    event.score <= 1
  );
}

/** Accept already finalized, fingerprinted local model decisions. This adapter
 * does not rerun weights, reconstruct landmarks, or grade form/guard return.
 */
export class NativeEventAdapter {
  private fingerprint: string | null = null;
  private lastTime = -1;
  private seen = new Set<string>();

  get configured(): boolean {
    return this.fingerprint !== null;
  }

  reset(): void {
    this.fingerprint = null;
    this.lastTime = -1;
    this.seen.clear();
  }

  update(frame: PoseFrame, stance: Stance, calibrated: boolean): PunchEvent[] {
    const recognition = frame.recognition;
    if (
      !frame.estimator ||
      !recognition ||
      recognition.protocolVersion !== "shadowbox-recognition-v1" ||
      recognition.recognizerId !== "personal-hybrid-v1" ||
      !FINGERPRINT.test(recognition.fingerprint) ||
      !["warming", "active", "uncertain"].includes(recognition.state) ||
      !Array.isArray(recognition.events) ||
      recognition.events.length > 16 ||
      !Number.isFinite(frame.t) ||
      frame.t < 0 ||
      frame.t <= this.lastTime ||
      recognition.events.some((event) => !event || !validEvent(event, frame.t))
    ) {
      throw new Error("Invalid or noncausal local recognition evidence.");
    }
    if (this.fingerprint && recognition.fingerprint !== this.fingerprint)
      throw new Error(
        "Local recognizer changed during this round. Start a new round.",
      );
    this.fingerprint = recognition.fingerprint;
    this.lastTime = frame.t;
    const events: PunchEvent[] = [];
    for (const event of recognition.events) {
      if (this.seen.has(event.id)) continue;
      this.seen.add(event.id);
      // Bounded memory; an ordinary round never approaches this many decisions.
      if (this.seen.size > 4096)
        this.seen.delete(this.seen.values().next().value!);
      if (!calibrated) continue;
      const role =
        event.hand === (stance === "orthodox" ? "left" : "right")
          ? "lead"
          : "rear";
      events.push({
        id: event.id,
        hand: event.hand,
        role,
        label:
          event.family === "straight"
            ? role === "lead"
              ? "jab"
              : "cross"
            : event.family,
        startMs: event.startMs,
        peakMs: event.peakMs,
        endMs: event.endMs,
        detectedAtMs: event.detectedAtMs,
        score: event.score,
        extension: null,
        guardReturn: "unassessable",
        experimental: true,
      });
    }
    return events;
  }
}
