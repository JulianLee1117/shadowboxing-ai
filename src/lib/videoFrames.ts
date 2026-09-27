export interface DecodedVideoFrame {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  timestampMs: number;
  durationMs: number;
}
export interface VideoFrameSource {
  frames: AsyncGenerator<DecodedVideoFrame, void, unknown>;
  metadata: {
    decoder: "mediabunny-1.60.0-webcodecs";
    durationMs: number;
    firstTimestampMs: number;
    width: number;
    height: number;
    rotation: number;
    mirrored: boolean;
    frameRate: number | null;
  };
  dispose: () => void;
}

/** Sequential local decoding. Consumers must finish using a canvas before next().
 * One pooled canvas bounds GPU memory. Timestamps are the source frame PTS, never
 * seek targets; no duplicated frames or guessed intermediate poses are created.
 */
export async function openVideoFrames(
  blob: Blob,
  options: {
    signal: AbortSignal;
    offsetMs: number;
    durationMs: number;
    maxPixels: number;
  },
): Promise<VideoFrameSource> {
  const { signal, offsetMs, durationMs, maxPixels } = options;
  const active = () => {
    if (signal.aborted)
      throw new DOMException("Video decoding canceled.", "AbortError");
  };
  active();
  if (
    !(blob instanceof Blob) ||
    !blob.size ||
    !Number.isFinite(offsetMs) ||
    offsetMs < 0 ||
    !Number.isFinite(durationMs) ||
    durationMs <= 0 ||
    !Number.isFinite(maxPixels) ||
    maxPixels <= 0
  )
    throw new Error("Invalid saved-video decode request.");
  if (typeof VideoDecoder === "undefined")
    throw new Error(
      "Local video analysis needs WebCodecs. Open this recording in current desktop Chrome or Edge.",
    );
  // Lazy import keeps media parsing off the initial camera/practice path.
  const { Input, BlobSource, CanvasSink, WEBM, MATROSKA, MP4, QTFF } =
    await import("mediabunny");
  active();
  const input = new Input({
    source: new BlobSource(blob),
    formats: [WEBM, MATROSKA, MP4, QTFF],
  });
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    signal.removeEventListener("abort", dispose);
    input.dispose();
  };
  signal.addEventListener("abort", dispose, { once: true });
  try {
    active();
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error("This recording has no video track.");
    const [width, height, end, first, rotation, mirrored, metrics, canDecode] =
      await Promise.all([
        track.getDisplayWidth(),
        track.getDisplayHeight(),
        track.computeDuration(),
        track.getFirstTimestamp(),
        track.getRotation(),
        track.getFlip(),
        track.computeFrameRateMetrics({ targetPacketCount: 256 }),
        track.canDecode(),
      ]);
    active();
    if (!canDecode)
      throw new Error(
        "This browser cannot decode this recording for local analysis. Try a WebM or MP4 recording in desktop Chrome.",
      );
    if (!width || !height || width * height > maxPixels)
      throw new Error("Saved video exceeds the supported analysis dimensions.");
    if (!Number.isFinite(end) || end * 1000 <= offsetMs)
      throw new Error("The saved round starts beyond the end of its video.");
    // Honor rotation, but reject pre-mirrored container transforms: anatomical
    // hand labels require explicit provenance rather than silently flipping.
    if (mirrored)
      throw new Error(
        "This video declares a mirrored track. Use the original unmirrored recording for anatomical hand analysis.",
      );
    const sink = new CanvasSink(track, { poolSize: 1 });
    const metadata: VideoFrameSource["metadata"] = {
      decoder: "mediabunny-1.60.0-webcodecs",
      durationMs: end * 1000,
      firstTimestampMs: first * 1000,
      width,
      height,
      rotation,
      mirrored,
      frameRate:
        Number.isFinite(metrics.bestGuessFrameRate) &&
        metrics.bestGuessFrameRate > 0
          ? metrics.bestGuessFrameRate
          : null,
    };
    async function* frames(): AsyncGenerator<DecodedVideoFrame, void, unknown> {
      let previous = -Infinity;
      try {
        for await (const item of sink.canvases(
          offsetMs / 1000,
          // Some MediaRecorder WebMs omit the last frame's duration. Their
          // computed track end equals that frame's PTS; clamping an exclusive
          // range to it would silently lose a real final observation.
          (offsetMs + durationMs) / 1000,
        )) {
          active();
          const timestampMs = item.timestamp * 1000;
          if (!Number.isFinite(timestampMs) || timestampMs <= previous)
            throw new Error(
              "Decoded video timestamps are not strictly increasing.",
            );
          previous = timestampMs;
          if (timestampMs < offsetMs || timestampMs >= offsetMs + durationMs)
            continue;
          yield {
            canvas: item.canvas,
            timestampMs,
            durationMs: item.duration * 1000,
          };
        }
      } catch (error) {
        active();
        throw error;
      } finally {
        dispose();
      }
    }
    return { frames: frames(), metadata, dispose };
  } catch (error) {
    dispose();
    active();
    throw error;
  }
}
