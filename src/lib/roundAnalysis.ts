import modelManifest from "../../model-manifest.json";
import { DETECTOR_VERSION, MotionEngine } from "./motion";
import { nativeDetectorVersion } from "./nativeRecognition";
import { LocalPoseBusyError } from "./localPoseClient";
import {
  summarizeTrackingTrust,
  TRACKING_TRUST_VERSION,
  type TrackingUncertaintyInterval,
} from "./trackingTrust";
export type { TrackingUncertaintyInterval } from "./trackingTrust";
import type {
  ModelVariant,
  PoseDelegate,
  PoseFrame,
  PunchEvent,
  Session,
  Stance,
} from "./types";
import { VisionRunner } from "./vision";
import { fingerprintVideo } from "./mediaFingerprint";
import { openVideoFrames, type VideoFrameSource } from "./videoFrames";

export const REPORT_VERSION = "round-video-analysis-v4-local-recognition";
export const ANALYSIS_VERSION = `${REPORT_VERSION}|${DETECTOR_VERSION}|${TRACKING_TRUST_VERSION}`;
export const ANALYSIS_LIMITS = {
  maximumDurationMs: 185_000,
  maximumFrames: 5550,
  maximumVideoBytes: 250 * 1024 * 1024,
  maximumWallMs: 240_000,
  maximumNativeWallMs: 360_000,
  maximumSamplingFps: 60,
  maximumPixels: 3840 * 2160,
} as const;

export interface AnalysisProgress {
  phase: "loading" | "analyzing";
  completedFrames: number;
  plannedFrames: number;
  fraction: number;
}
export interface RoundAnalysisReport {
  id: string;
  reportType: "round-video-analysis";
  reportVersion: string;
  sourceSessionId: string;
  sourceDetectorVersion: string | null;
  sourceFingerprint: string;
  analysisVersion: string;
  detectorVersion: string;
  createdAt: string;
  stance: Stance;
  frames: PoseFrame[];
  events: PunchEvent[];
  model: ModelVariant;
  delegate: PoseDelegate;
  modelManifest: unknown;
  provenance: {
    source: "saved-video";
    videoBytes: number;
    videoOffsetMs: number;
    sourceDurationMs: number;
    decodedDurationMs: number | null;
    timestampMode: "decoded-media-time";
    decoder: string;
    firstTimestampMs: number;
    rotationDegrees: number;
    interpolation: false;
    mirrored: false;
    sequentialInference: true;
    processingLimitMs?: number;
    trackingTrustVersion: string;
  };
  cadence: {
    requestedFps: number;
    sourceFps: number | null;
    samplingFramesSkipped: number;
    cadenceSource: string;
    observedFps: number;
    maximumGapMs: number;
    decodedTimestampFrames: number;
    estimatedTimestampFrames: number;
    duplicateFramesSkipped: number;
    processingMs: number;
  };
  uncertaintyIntervals: TrackingUncertaintyInterval[];
  warnings: string[];
  completeness: {
    status: "complete" | "partial";
    reason: string | null;
    processedFrames: number;
    plannedFrames: number;
    coveredUntilMs: number;
  };
}
export interface RoundAnalysisOptions {
  model?: ModelVariant;
  signal?: AbortSignal;
  onProgress?: (progress: AnalysisProgress) => void;
}

/** Uses source cadence when recorded; inference cadence is only a fallback. */
export function createAnalysisPlan(session: Session) {
  if (session.source === "demo" || session.model === "synthetic")
    throw new Error("Demo rounds cannot be analyzed from video.");
  if (!(session.video instanceof Blob) || !session.video.size)
    throw new Error("This round has no saved video to analyze.");
  if (session.video.size > ANALYSIS_LIMITS.maximumVideoBytes)
    throw new Error("Saved video exceeds the 250 MB local analysis limit.");
  if (!Number.isFinite(session.durationMs) || session.durationMs <= 0)
    throw new Error("The saved round has no valid duration.");
  const offsetMs = session.videoOffsetMs ?? 0;
  if (!Number.isFinite(offsetMs) || offsetMs < 0)
    throw new Error("The saved video offset is invalid.");
  const gaps = session.frames
    .slice(1)
    .map((f, i) => f.t - session.frames[i].t)
    .filter((gap) => Number.isFinite(gap) && gap > 0)
    .sort((a, b) => a - b);
  const candidates: [number | undefined, string][] = [
    [session.capture?.deliveredFps, "recorded camera cadence"],
    [
      gaps.length ? 1000 / gaps[Math.floor(gaps.length / 2)] : undefined,
      "saved frame median cadence",
    ],
    [session.measuredFps, "saved measured cadence"],
  ];
  const cadence = candidates.find(
    ([fps]) =>
      fps !== undefined && Number.isFinite(fps) && fps >= 1 && fps <= 240,
  );
  const fps = Math.min(ANALYSIS_LIMITS.maximumSamplingFps, cadence?.[0] ?? 30);
  const durationMs = Math.min(
    session.durationMs,
    ANALYSIS_LIMITS.maximumDurationMs,
  );
  const frameCount = Math.min(
    ANALYSIS_LIMITS.maximumFrames,
    Math.ceil((durationMs * fps) / 1000),
  );
  return {
    offsetMs,
    durationMs,
    fps,
    frameCount,
    cadenceSource:
      cadence?.[1] ?? "30 fps planning estimate; source cadence not recorded",
    truncated:
      durationMs < session.durationMs ||
      (frameCount / fps) * 1000 < durationMs - 1,
  };
}

const abortError = () =>
  new DOMException("Local round analysis was canceled.", "AbortError");
function requireActive(signal: AbortSignal) {
  if (signal.aborted) throw abortError();
}

function waitForLocalHandoff(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const cancel = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", cancel);
      reject(abortError());
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", cancel);
      resolve();
    }, 50);
    signal.addEventListener("abort", cancel, { once: true });
    if (signal.aborted) cancel();
  });
}

/** Fresh local pose inference over saved video; never edits or persists the Session. */
export async function analyzeRound(
  session: Session,
  options: RoundAnalysisOptions = {},
): Promise<RoundAnalysisReport> {
  const plan = createAnalysisPlan(session);
  if (options.signal?.aborted) throw abortError();
  const started = performance.now();
  const controller = new AbortController();
  const model =
    options.model ??
    (session.model === "rtmpose-m" || session.model === "rtmw-l"
      ? session.model
      : "full");
  const runner = new VisionRunner(model);
  const processingLimitMs =
    model === "rtmpose-m" || model === "rtmw-l"
      ? ANALYSIS_LIMITS.maximumNativeWallMs
      : ANALYSIS_LIMITS.maximumWallMs;
  const processingLimitMinutes = processingLimitMs / 60_000;
  let decoded: VideoFrameSource | null = null;
  let timedOut = false;
  const cancel = () => controller.abort();
  options.signal?.addEventListener("abort", cancel, { once: true });
  controller.signal.addEventListener("abort", () => runner.dispose(), {
    once: true,
  });
  const deadline = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, processingLimitMs);
  const frames: PoseFrame[] = [],
    events: PunchEvent[] = [];
  const warnings = [
    "A fresh pass is not proof of greater accuracy. Tracking and projected punch labels remain experimental.",
  ];
  const engine = new MotionEngine({ stance: session.stance, calibrated: true });
  let delegate: PoseDelegate | null = null;
  let estimatorManifest: unknown = modelManifest;
  let fingerprint = "";
  let plannedFrames = plan.frameCount;
  let cadenceSource = plan.cadenceSource;
  let requestedFps = plan.fps;
  let samplingFramesSkipped = 0;
  let coveredUntilMs = 0;
  let metadata: VideoFrameSource["metadata"] | null = null;
  let reason: string | null =
    plan.durationMs < session.durationMs
      ? "Saved round exceeds the local analysis duration limit."
      : null;
  const progress = (
    phase: AnalysisProgress["phase"],
    completedFrames: number,
    fraction: number,
  ) =>
    options.onProgress?.({
      phase,
      completedFrames,
      plannedFrames,
      fraction: Math.min(1, fraction),
    });
  try {
    progress("loading", 0, 0);
    fingerprint = await fingerprintVideo(session.video!);
    requireActive(controller.signal);
    decoded = await openVideoFrames(session.video!, {
      signal: controller.signal,
      offsetMs: plan.offsetMs,
      durationMs: plan.durationMs,
      maxPixels: ANALYSIS_LIMITS.maximumPixels,
    });
    metadata = decoded.metadata;
    requireActive(controller.signal);
    const availableMs = Math.min(
      plan.durationMs,
      metadata.durationMs - plan.offsetMs,
    );
    if (availableMs + 100 < plan.durationMs)
      reason = "The saved video ends before the recorded round duration.";
    if (metadata.frameRate) {
      requestedFps = Math.min(
        ANALYSIS_LIMITS.maximumSamplingFps,
        metadata.frameRate,
      );
      cadenceSource = "decoded video packet cadence";
    }
    plannedFrames = Math.min(
      ANALYSIS_LIMITS.maximumFrames,
      Math.ceil((availableMs * requestedFps) / 1000),
    );
    await runner.init();
    estimatorManifest = runner.modelInfo ?? modelManifest;
    delegate = runner.delegate;
    if (!delegate)
      throw new Error("Local pose inference did not report its runtime.");
    // Preserve every native frame at ordinary camera rates. Only higher-rate
    // sources are sampled into 60Hz buckets, avoiding strict 33.333ms thresholds
    // that accidentally discard native 29.97/30/60fps frames due to rounding.
    const sampleHighRate =
      metadata.frameRate !== null &&
      metadata.frameRate > ANALYSIS_LIMITS.maximumSamplingFps + 0.5;
    let previousBucket = -1;
    for await (const image of decoded.frames) {
      requireActive(controller.signal);
      const t = image.timestampMs - plan.offsetMs;
      if (t < 0 || t >= plan.durationMs) continue;
      const bucket = Math.floor(
        (t * ANALYSIS_LIMITS.maximumSamplingFps) / 1000 + 1e-6,
      );
      if (sampleHighRate && bucket === previousBucket) {
        samplingFramesSkipped++;
        continue;
      }
      previousBucket = bucket;
      if (frames.length >= ANALYSIS_LIMITS.maximumFrames) {
        reason = "Local analysis reached its 5550-frame processing limit.";
        break;
      }
      let frame: PoseFrame | undefined;
      // A just-stopped live session can still own the native inference lock.
      // Hold this decoded image/time pair briefly; never retry other failures.
      for (let attempt = 0; frame === undefined; attempt++) {
        requireActive(controller.signal);
        try {
          frame = await runner.detectImage(image.canvas, t);
        } catch (error) {
          if (!(error instanceof LocalPoseBusyError)) throw error;
          if (attempt >= 8)
            throw new Error(
              "The local tracker is busy in another session. Stop that analysis and try again.",
            );
          await waitForLocalHandoff(controller.signal);
        }
      }
      requireActive(controller.signal);
      frames.push(frame);
      events.push(...engine.update(frame).events);
      coveredUntilMs = Math.min(
        session.durationMs,
        t +
          (Number.isFinite(image.durationMs) && image.durationMs > 0
            ? image.durationMs
            : 1000 / requestedFps),
      );
      progress(
        "analyzing",
        frames.length,
        coveredUntilMs / Math.max(1, availableMs),
      );
    }
    // A caller may cancel from the last progress callback, after the final
    // decoded frame. Completion must not turn that cancellation into a report.
    requireActive(controller.signal);
  } catch (error) {
    if (options.signal?.aborted) throw abortError();
    if (timedOut && frames.length)
      reason = `Local analysis reached its ${processingLimitMinutes}-minute processing limit.`;
    else if (timedOut)
      throw new Error(
        `Local analysis exceeded its ${processingLimitMinutes}-minute processing limit before producing frames.`,
      );
    else throw error;
  } finally {
    clearTimeout(deadline);
    options.signal?.removeEventListener("abort", cancel);
    runner.dispose();
    decoded?.dispose();
  }
  if (!delegate || !metadata || !frames.length)
    throw new Error("No decoded video frames could be analyzed.");
  const last = frames.at(-1)!.t;
  if (coveredUntilMs + 100 < plan.durationMs && !reason)
    reason = "Decoded frames did not cover the end of the saved round.";
  if (metadata.frameRate === null)
    warnings.push(
      "Native timestamps were preserved; source frame-rate metadata was unavailable.",
    );
  if (samplingFramesSkipped)
    warnings.push(
      `${samplingFramesSkipped} higher-rate source frames were omitted by the 60fps sampling cap.`,
    );
  return {
    id: crypto.randomUUID(),
    reportType: "round-video-analysis",
    reportVersion: REPORT_VERSION,
    sourceSessionId: session.id,
    sourceDetectorVersion: session.detectorVersion ?? null,
    sourceFingerprint: fingerprint,
    analysisVersion: ANALYSIS_VERSION,
    detectorVersion:
      nativeDetectorVersion(estimatorManifest) ?? DETECTOR_VERSION,
    createdAt: new Date().toISOString(),
    stance: session.stance,
    frames,
    events,
    model,
    delegate,
    modelManifest: structuredClone(estimatorManifest),
    provenance: {
      source: "saved-video",
      videoBytes: session.video!.size,
      videoOffsetMs: plan.offsetMs,
      sourceDurationMs: session.durationMs,
      decodedDurationMs: metadata.durationMs,
      timestampMode: "decoded-media-time",
      decoder: metadata.decoder,
      firstTimestampMs: metadata.firstTimestampMs,
      rotationDegrees: metadata.rotation,
      interpolation: false,
      mirrored: false,
      sequentialInference: true,
      processingLimitMs,
      trackingTrustVersion: TRACKING_TRUST_VERSION,
    },
    cadence: {
      requestedFps,
      sourceFps: metadata.frameRate,
      samplingFramesSkipped,
      cadenceSource,
      observedFps:
        frames.length > 1
          ? ((frames.length - 1) * 1000) / (last - frames[0].t)
          : 0,
      maximumGapMs: Math.max(
        0,
        ...frames.slice(1).map((f, i) => f.t - frames[i].t),
      ),
      decodedTimestampFrames: frames.length,
      estimatedTimestampFrames: 0,
      duplicateFramesSkipped: 0,
      processingMs: performance.now() - started,
    },
    uncertaintyIntervals: summarizeTrackingTrust(frames, session.durationMs),
    warnings,
    completeness: {
      status: reason ? "partial" : "complete",
      reason,
      processedFrames: frames.length,
      plannedFrames,
      coveredUntilMs,
    },
  };
}
