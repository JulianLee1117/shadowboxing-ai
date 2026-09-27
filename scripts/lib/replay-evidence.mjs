import { createHash } from "node:crypto";

export const fingerprint = (bytes) =>
  createHash("sha256").update(bytes).digest("hex");
const requireValue = (condition, message) => {
  if (!condition) throw new Error(message);
};
const finite = (value) => typeof value === "number" && Number.isFinite(value);

/** Offline experiment only. Original evidence and annotations are never changed. */
export function replayEvidence(session, options) {
  const {
    MotionEngine,
    detectorVersion,
    detectorFingerprint,
    sessionFingerprint,
    poses,
    posesFingerprint,
    modelId,
    videoFingerprint,
    poseTimeOffsetMs = 0,
  } = options;
  requireValue(
    session.schemaVersion === "1.0",
    "Expected schemaVersion 1.0 session.",
  );
  requireValue(
    ["orthodox", "southpaw"].includes(session.stance),
    "Select a valid saved stance.",
  );
  requireValue(
    finite(session.durationMs) && session.durationMs > 0,
    "Invalid duration.",
  );
  requireValue(
    Array.isArray(session.annotations),
    "Missing reference annotation array.",
  );
  requireValue(finite(poseTimeOffsetMs), "Invalid pose time offset.");
  if (poses) {
    requireValue(
      poses.artifactType === "research-pose-series" &&
        poses.schemaVersion === "1.0",
      "Expected a research-pose-series artifact.",
    );
    requireValue(
      poses.truncatedByFrameLimit === false,
      "Refusing truncated or unspecified pose coverage.",
    );
    requireValue(
      typeof modelId === "string" && modelId.trim().length > 0,
      "External poses need an explicit --model identifier.",
    );
    requireValue(
      modelId === modelId.trim() &&
        !["full", "heavy", "lite", "synthetic", "rtmpose-m", "rtmw-l"].includes(
          modelId.toLowerCase(),
        ),
      "Use a distinct research model identifier; do not relabel external poses as a browser model.",
    );
    requireValue(
      videoFingerprint && poses.source?.sha256 === videoFingerprint,
      "Pose series does not match the supplied original video SHA-256.",
    );
    requireValue(
      poses.modelManifest &&
        typeof poses.modelManifest === "object" &&
        Object.keys(poses.modelManifest).length > 0,
      "Pose model provenance is required.",
    );
    requireValue(
      typeof poses.timestampMode === "string" && poses.timestampMode.length > 0,
      "Pose timestamp provenance is required.",
    );
    requireValue(
      videoFingerprint && posesFingerprint,
      "External pose/video fingerprints are required.",
    );
  } else
    requireValue(
      poseTimeOffsetMs === 0,
      "Offset applies only to replacement poses.",
    );
  requireValue(
    Array.isArray(poses?.frames ?? session.frames),
    "Missing pose frames.",
  );
  // Validate before windowing: malformed/out-of-order source timestamps must not
  // silently disappear merely because the filter would place them out of range.
  let sourcePrevious = -Infinity;
  for (const frame of poses?.frames ?? session.frames) {
    requireValue(
      !frame?.recognition,
      "Saved learned decisions cannot be recomputed from tracking alone. Rerun video with the fingerprinted local recognizer.",
    );
    requireValue(
      frame && finite(frame.t) && frame.t >= 0 && frame.t > sourcePrevious,
      "Source frame times must be finite, nonnegative and strictly increasing.",
    );
    sourcePrevious = frame.t;
  }
  const frames = structuredClone(poses?.frames ?? session.frames)
    .map((frame) => ({ ...frame, t: frame.t + poseTimeOffsetMs }))
    .filter(
      (frame) => !poses || (frame.t >= 0 && frame.t <= session.durationMs),
    );
  requireValue(frames.length > 0, "No frames within the recorded round.");
  let previous = -1;
  for (const frame of frames) {
    requireValue(
      finite(frame.t) &&
        frame.t >= 0 &&
        frame.t > previous &&
        frame.t <= session.durationMs + 1,
      "Frame times must be strictly increasing and inside the round.",
    );
    requireValue(
      finite(frame.width) &&
        frame.width > 0 &&
        finite(frame.height) &&
        frame.height > 0,
      "Invalid frame dimensions.",
    );
    requireValue(
      finite(frame.inferenceMs) && frame.inferenceMs >= 0,
      "Missing inference timing.",
    );
    requireValue(
      Array.isArray(frame.landmarks) &&
        frame.landmarks.every((p) => p && finite(p.x) && finite(p.y)),
      "Invalid landmark coordinates.",
    );
    previous = frame.t;
  }
  // A complete-series claim must cover the round. Never score a short smoke run
  // as a complete alternative model. Allow one normal terminal frame interval.
  if (poses) {
    const gaps = frames
      .slice(1)
      .map((f, i) => f.t - frames[i].t)
      .sort((a, b) => a - b);
    const tolerance = Math.max(
      100,
      (gaps[Math.floor(gaps.length / 2)] ?? 0) * 2,
    );
    requireValue(
      frames[0].t <= tolerance &&
        session.durationMs - frames.at(-1).t <= tolerance,
      "Replacement poses do not cover the full round; inspect timestamps/offset.",
    );
  }
  const engine = new MotionEngine({ stance: session.stance, calibrated: true });
  const events = frames.flatMap((frame) => engine.update(frame).events);
  const output = structuredClone(session);
  delete output.video;
  output.events = events;
  output.frames = frames;
  output.detectorVersion = detectorVersion;
  output.artifactType = "detector-benchmark-session";
  output.id = `${session.id}-replay-${fingerprint(JSON.stringify([detectorFingerprint, posesFingerprint ?? sessionFingerprint, poseTimeOffsetMs])).slice(0, 12)}`;
  output.benchmark = {
    createdAt: new Date().toISOString(),
    sourceSessionId: session.id,
    inputSessionSha256: sessionFingerprint,
    detectorSourceSha256: detectorFingerprint,
    sourceDetectorVersion: session.detectorVersion ?? null,
    trackingSource: poses ? "replacement-pose-series" : "saved-frames",
    poseSeriesSha256: posesFingerprint ?? null,
    sourceVideoSha256: videoFingerprint ?? null,
    poseTimeOffsetMs,
    timestampMode: poses?.timestampMode ?? "saved-session-timestamps",
    modelId: poses ? modelId : session.model,
    limitations: [
      "Offline development replay, not live-camera or coaching validation.",
      "Reference labels retain their original completeness and review provenance.",
    ],
  };
  if (poses) {
    output.model = modelId;
    output.modelManifest = structuredClone(poses.modelManifest);
    output.source = session.source === "demo" ? "demo" : "file";
    output.capture = {
      width: frames[0].width,
      height: frames[0].height,
      timingSource: poses.timestampMode,
      delegate: "offline: see pose runtime provenance",
    };
    output.benchmark.poseRuntime = structuredClone(poses.runtime ?? {});
    const sorted = frames.map((f) => f.inferenceMs).sort((a, b) => a - b);
    output.inferenceP95 = sorted[Math.ceil(sorted.length * 0.95) - 1];
    output.measuredFps =
      frames.length > 1
        ? ((frames.length - 1) * 1000) / (frames.at(-1).t - frames[0].t)
        : 0;
    // Capture skips are unknown for an external offline extractor, not zero.
    output.skippedFrames = null;
  }
  return output;
}
