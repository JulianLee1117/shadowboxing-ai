import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ModelVariant,
  PoseFrame,
  PunchEvent,
  QualityState,
  Session,
  SourceKind,
  Stance,
} from "../lib/types";
import { VisionRunner, type PreparedVisionFrame } from "../lib/vision";
import { LocalPoseBusyError } from "../lib/localPoseClient";
import { LatestFramePump } from "../lib/latestFramePump";
import {
  nativeDetectorVersion,
  relativePoseFrame,
} from "../lib/nativeRecognition";
import { MotionEngine, assessQuality, DETECTOR_VERSION } from "../lib/motion";
import { demoFrame } from "../lib/demo";
import { percentile } from "../lib/storage";
import modelManifest from "../../model-manifest.json";

const EMPTY_QUALITY: QualityState = {
  assessable: false,
  label: "Camera is off",
  reasons: ["Start your camera to check your setup."],
  visibleJoints: 0,
  totalJoints: 9,
};
interface RoundConfig {
  stance: Stance;
  drill: string;
  durationSeconds: number;
  record: boolean;
  calibrated: boolean;
}
interface RoundData {
  config: RoundConfig;
  start: number;
  startedAt: number; // monotonic wall time; capture deadlines do not depend on pose
  frames: PoseFrame[];
  events: PunchEvent[];
  skipped: number;
  id: string;
  createdAt: string;
  source: SourceKind;
  model: ModelVariant | "synthetic";
  capture: Session["capture"];
  modelManifest: unknown;
  detectorVersion: string;
}

/** Decode an uploaded clip without playing it through model initialization. */
async function firstVideoFrame(video: HTMLVideoElement): Promise<void> {
  video.pause();
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    await new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timeout);
        video.removeEventListener("loadeddata", loaded);
        video.removeEventListener("error", failed);
      };
      const loaded = () => {
        cleanup();
        resolve();
      };
      const failed = () => {
        cleanup();
        reject(
          new Error("Could not decode this video. Try an MP4 or WebM file."),
        );
      };
      const timeout = setTimeout(() => {
        cleanup();
        reject(
          new Error(
            "Timed out decoding the first video frame. Try a shorter MP4 or WebM clip.",
          ),
        );
      }, 15_000);
      video.addEventListener("loadeddata", loaded);
      video.addEventListener("error", failed);
    });
  }
  if (video.currentTime !== 0 || video.seeking) {
    await new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timeout);
        video.removeEventListener("seeked", seeked);
        video.removeEventListener("error", failed);
      };
      const seeked = () => {
        cleanup();
        resolve();
      };
      const failed = () => {
        cleanup();
        reject(new Error("Could not seek to the beginning of this clip."));
      };
      const timeout = setTimeout(() => {
        cleanup();
        reject(new Error("Timed out seeking to the beginning of this clip."));
      }, 10_000);
      video.addEventListener("seeked", seeked);
      video.addEventListener("error", failed);
      video.currentTime = 0;
    });
  }
}

export function useStudio(onComplete: (session: Session) => void) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const runner = useRef<VisionRunner | null>(null);
  const sourceRef = useRef<SourceKind | null>(null);
  const variantRef = useRef<ModelVariant>("full");
  const generation = useRef(0);
  const raf = useRef(0);
  const videoCallback = useRef<number | null>(null);
  const framePump = useRef<LatestFramePump<PreparedVisionFrame> | null>(null);
  const lastMedia = useRef(-1);
  const lastSource = useRef(0);
  const sourceStart = useRef(0);
  const displayAt = useRef(0);
  const statisticsAt = useRef(-Infinity);
  const fileUrl = useRef<string | null>(null);
  const sourceFile = useRef<File | null>(null);
  const finalizing = useRef<Promise<void> | null>(null);
  const closingAt = useRef<number | null>(null);
  const round = useRef<RoundData | null>(null);
  const roundDeadline = useRef<ReturnType<typeof setTimeout> | null>(null);
  const elapsedClock = useRef<ReturnType<typeof setInterval> | null>(null);
  const engine = useRef(
    new MotionEngine({ stance: "orthodox", calibrated: false }),
  );
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const complete = useRef(onComplete);
  complete.current = onComplete;
  const mounted = useRef(true);
  const [source, setSource] = useState<SourceKind | null>(null);
  const [status, setStatus] = useState<"off" | "loading" | "ready">("off");
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState<PoseFrame | null>(null);
  const [quality, setQuality] = useState(EMPTY_QUALITY);
  const [events, setEvents] = useState<PunchEvent[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [running, setRunning] = useState(false);
  const [settings, setSettings] = useState<MediaTrackSettings | null>(null);
  const [fps, setFps] = useState(0);
  const [skipped, setSkipped] = useState(0);
  const recentTimes = useRef<number[]>([]);

  const clearRoundClock = useCallback(() => {
    if (roundDeadline.current !== null) clearTimeout(roundDeadline.current);
    if (elapsedClock.current !== null) clearInterval(elapsedClock.current);
    roundDeadline.current = null;
    elapsedClock.current = null;
  }, []);

  const finishRound = useCallback((): Promise<void> => {
    if (finalizing.current) return finalizing.current;
    const data = round.current;
    if (!data) return Promise.resolve();
    clearRoundClock();
    const endSource =
      data.source === "demo"
        ? performance.now() - sourceStart.current
        : (videoRef.current?.currentTime ?? lastSource.current / 1000) * 1000;
    const durationMs = Math.max(
      0,
      // A captured frame can still be in flight. Its media timestamp may lead
      // the wall clock slightly; freeze both boundaries before draining it.
      // Every accepted late result is bounded by this same endSource below.
      endSource - data.start,
      data.source === "file"
        ? endSource - data.start
        : performance.now() - data.startedAt,
      data.frames.at(-1)?.t ?? 0,
    );
    if (mounted.current) setElapsed(durationMs);
    const originalFile = sourceFile.current;
    closingAt.current = endSource;
    framePump.current?.discardPending();
    const pendingFrame = framePump.current?.whenIdle();
    if (data.source === "file") videoRef.current?.pause();
    const rec = recorder.current;
    recorder.current = null;
    const task = (async () => {
      let video: Blob | undefined =
        data.source === "file" ? (originalFile ?? undefined) : undefined;
      if (rec && rec.state !== "inactive") {
        video = await new Promise<Blob>((resolve) => {
          const savedChunks = chunks.current;
          let settled = false;
          const done = () => {
            if (settled) return;
            settled = true;
            clearTimeout(timeout);
            resolve(
              new Blob(savedChunks, { type: rec.mimeType || "video/webm" }),
            );
          };
          const timeout = setTimeout(done, 3000);
          rec.ondataavailable = (e) => {
            if (e.data.size) savedChunks.push(e.data);
          };
          rec.onstop = done;
          rec.onerror = done;
          rec.stop();
        });
      }
      if (pendingFrame) {
        let timeout: ReturnType<typeof setTimeout> | undefined;
        await Promise.race([
          pendingFrame,
          new Promise<void>((resolve) => {
            timeout = setTimeout(resolve, 3000);
          }),
        ]);
        clearTimeout(timeout);
      }
      if (round.current === data) round.current = null;
      const times = data.frames.map((f) => f.t);
      const span = times.length > 1 ? times.at(-1)! - times[0] : 0;
      const session: Session = {
        id: data.id,
        createdAt: data.createdAt,
        schemaVersion: "1.0",
        source: data.source,
        stance: data.config.stance,
        model: data.model,
        drill: data.config.drill,
        durationMs,
        frames: data.frames,
        events: data.events,
        annotations: [],
        annotationsComplete: false,
        video: video?.size ? video : undefined,
        videoOffsetMs: data.source === "file" ? data.start : 0,
        measuredFps: span ? (times.length - 1) / (span / 1000) : 0,
        inferenceP95: percentile(
          data.frames.map((f) => f.inferenceMs),
          0.95,
        ),
        skippedFrames: data.skipped,
        modelManifest: data.modelManifest,
        detectorVersion: data.detectorVersion,
        capture: data.capture,
      };
      complete.current(session);
    })().finally(() => {
      finalizing.current = null;
      closingAt.current = null;
      if (mounted.current) setRunning(false);
    });
    finalizing.current = task;
    return task;
  }, [clearRoundClock]);
  const finishRef = useRef(finishRound);
  finishRef.current = finishRound;

  const releaseSource = useCallback(() => {
    generation.current++;
    framePump.current?.dispose();
    framePump.current = null;
    clearRoundClock();
    cancelAnimationFrame(raf.current);
    runner.current?.dispose();
    runner.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    const video = videoRef.current;
    if (videoCallback.current !== null)
      video?.cancelVideoFrameCallback?.(videoCallback.current);
    videoCallback.current = null;
    if (video) {
      video.pause();
      video.srcObject = null;
      video.removeAttribute("src");
      video.load();
    }
    if (fileUrl.current) URL.revokeObjectURL(fileUrl.current);
    fileUrl.current = null;
    sourceFile.current = null;
    sourceRef.current = null;
    recentTimes.current = [];
    lastMedia.current = -1;
    lastSource.current = 0;
    displayAt.current = 0;
    statisticsAt.current = -Infinity;
    if (mounted.current) {
      setSource(null);
      setStatus("off");
      setFrame(null);
      setQuality(EMPTY_QUALITY);
      setSettings(null);
      setFps(0);
    }
  }, [clearRoundClock]);

  const stop = useCallback(async () => {
    const pending = finishRound();
    releaseSource();
    await pending;
  }, [finishRound, releaseSource]);
  const receive = useCallback((next: PoseFrame) => {
    lastSource.current = next.t;
    const times = recentTimes.current;
    times.push(next.t);
    while (times.length > 45) times.shift();
    const data = round.current;
    if (
      data &&
      (next.t < data.start ||
        (closingAt.current !== null && next.t > closingAt.current))
    )
      return;
    const analyzed = data ? relativePoseFrame(next, data.start) : next;
    const result = data
      ? engine.current.update(analyzed)
      : { quality: assessQuality(next), events: [], activeHand: null };
    if (data) {
      data.frames.push(analyzed);
      if (result.events.length) {
        data.events.push(...result.events);
        setEvents([...data.events]);
      }
      if (
        data.source === "file" &&
        closingAt.current === null &&
        analyzed.t >= data.config.durationSeconds * 1000
      )
        void finishRef.current();
    }
    const now = performance.now();
    // Every completed observation reaches the overlay. A time cutoff here
    // discards alternating results when inference runs at 20–25 Hz.
    displayAt.current = now;
    setFrame(next);
    setQuality(result.quality);
    // Numeric telemetry can update less often without holding back the pose.
    if (now - statisticsAt.current >= 100) {
      statisticsAt.current = now;
      setFps(
        times.length > 1
          ? (times.length - 1) / ((times.at(-1)! - times[0]) / 1000)
          : 0,
      );
      if (data) {
        if (data.source === "file") setElapsed(analyzed.t);
        setSkipped(data.skipped);
      }
    }
  }, []);

  const loop = useCallback(
    (gen: number) => {
      const vision = runner.current;
      framePump.current?.dispose();
      const pump = new LatestFramePump<PreparedVisionFrame>({
        snapshot: (source, mediaMs) => {
          if (!vision)
            return Promise.reject(new Error("Pose runner is unavailable."));
          return vision.prepareImage(source, mediaMs);
        },
        process: async (bitmap, mediaMs, observedAt) => {
          if (gen !== generation.current || !vision) return;
          const result = await vision.detectPrepared(bitmap, mediaMs);
          if (gen !== generation.current) return;
          receive({ ...result, frameAgeMs: performance.now() - observedAt });
        },
        onSkipped: () => {
          if (gen === generation.current && round.current)
            round.current.skipped++;
        },
        onError: (e) => {
          if (gen !== generation.current) return;
          if (e instanceof LocalPoseBusyError) {
            if (round.current) round.current.skipped++;
            if (performance.now() - displayAt.current > 500) {
              setFrame(null);
              setFps(0);
              setQuality({
                ...EMPTY_QUALITY,
                label: "Tracker busy",
                reasons: ["Another local analysis is using the tracker."],
              });
            }
            return;
          }
          setError(e.message);
          void stop();
        },
      });
      framePump.current = pump;
      const processFrame = (mediaMs: number) => {
        const video = videoRef.current;
        if (
          gen !== generation.current ||
          closingAt.current !== null ||
          !video ||
          !runner.current ||
          video.readyState < 2
        )
          return;
        // Seeking a paused uploaded clip can itself issue a video-frame callback.
        // Its frame is used for warmup, not queued ahead of the next round.
        if (sourceRef.current === "file" && video.paused) return;
        if (mediaMs === lastMedia.current) return;
        lastMedia.current = mediaMs;
        pump.push(video, mediaMs, performance.now());
      };
      const video = videoRef.current;
      if (sourceRef.current !== "demo" && video?.requestVideoFrameCallback) {
        const onFrame = (
          _now: number,
          metadata: VideoFrameCallbackMetadata,
        ) => {
          if (gen !== generation.current) return;
          videoCallback.current = video.requestVideoFrameCallback(onFrame);
          processFrame(metadata.mediaTime * 1000);
        };
        videoCallback.current = video.requestVideoFrameCallback(onFrame);
      } else {
        const tick = () => {
          if (gen !== generation.current) return;
          raf.current = requestAnimationFrame(tick);
          if (sourceRef.current === "demo")
            receive(demoFrame(performance.now() - sourceStart.current));
          else if (video && !video.paused && !video.ended)
            processFrame(video.currentTime * 1000);
        };
        tick();
      }
    },
    [receive, stop],
  );

  const start = useCallback(
    async (kind: SourceKind, variant: ModelVariant, file?: File) => {
      // stop() invalidates older requests synchronously. Capture that generation
      // before awaiting, so a newer start/stop cannot revive this request.
      const pendingStop = stop();
      const gen = generation.current;
      await pendingStop;
      if (gen !== generation.current || !mounted.current) return;
      setError(null);
      setStatus("loading");
      setEvents([]);
      setElapsed(0);
      setSkipped(0);
      sourceRef.current = kind;
      setSource(kind);
      variantRef.current = variant;
      try {
        const video = videoRef.current;
        if (!video) throw new Error("Video surface is not available.");
        if (kind === "demo") {
          sourceStart.current = performance.now();
          setStatus("ready");
          loop(gen);
          return;
        }
        const vision = new VisionRunner(variant);
        runner.current = vision;
        if (kind === "camera") {
          const acquired = await navigator.mediaDevices.getUserMedia({
            video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              frameRate: { ideal: 30 },
              facingMode: "user",
            },
            audio: false,
          });
          if (gen !== generation.current) {
            acquired.getTracks().forEach((t) => t.stop());
            return;
          }
          stream.current = acquired;
          video.srcObject = acquired;
          setSettings(acquired.getVideoTracks()[0].getSettings());
          acquired.getVideoTracks()[0].onended = () => {
            if (gen === generation.current) {
              setError(
                "The camera disconnected. Reconnect it and start again.",
              );
              void stop();
            }
          };
        } else {
          if (!file) throw new Error("Choose a video file.");
          sourceFile.current = file;
          fileUrl.current = URL.createObjectURL(file);
          video.src = fileUrl.current;
          video.preload = "auto";
          video.load();
        }
        if (kind === "file") {
          await Promise.all([vision.init(), firstVideoFrame(video)]);
          if (gen !== generation.current) return;
          // GPU graph/shader setup can be deferred until the first inference.
          // Warm it while paused so a short clip cannot end with every frame busy.
          const warmup = await vision.detect(video, 0);
          if (gen !== generation.current) return;
          lastMedia.current = 0;
          setFrame(warmup);
          setQuality(assessQuality(warmup));
        } else {
          await video.play();
          if (gen !== generation.current) return;
          await vision.init();
          if (gen !== generation.current) return;
        }
        setStatus("ready");
        loop(gen);
      } catch (e) {
        if (gen !== generation.current) return;
        const message =
          e instanceof DOMException && e.name === "NotAllowedError"
            ? "Camera access was denied. Allow camera access in your browser, then try again."
            : e instanceof Error
              ? e.message
              : "Could not start the camera.";
        releaseSource();
        setError(message);
      }
    },
    [loop, releaseSource, stop],
  );

  const beginRound = useCallback(
    async (config: RoundConfig) => {
      if (
        !sourceRef.current ||
        status !== "ready" ||
        round.current ||
        finalizing.current
      )
        return;
      const kind = sourceRef.current;
      if (kind === "file" && (videoRef.current?.currentTime ?? 0) > 0) {
        const gen = ++generation.current;
        framePump.current?.dispose();
        framePump.current = null;
        cancelAnimationFrame(raf.current);
        if (videoCallback.current !== null)
          videoRef.current?.cancelVideoFrameCallback?.(videoCallback.current);
        runner.current?.dispose();
        setStatus("loading");
        const vision = new VisionRunner(variantRef.current);
        runner.current = vision;
        try {
          const video = videoRef.current;
          if (!video) throw new Error("Video surface is not available.");
          await Promise.all([vision.init(), firstVideoFrame(video)]);
          if (gen !== generation.current) return;
          const warmup = await vision.detect(video, 0);
          if (gen !== generation.current) return;
          lastMedia.current = 0;
          setFrame(warmup);
          setQuality(assessQuality(warmup));
          setFps(0);
          setStatus("ready");
          loop(gen);
        } catch (e) {
          if (gen === generation.current) {
            releaseSource();
            setError(
              e instanceof Error ? e.message : "Could not restart analysis.",
            );
          }
          return;
        }
      }
      // Reset imported playback explicitly so ended clips cannot inherit stale timestamps.
      if (kind === "file" && videoRef.current) {
        videoRef.current.currentTime = 0;
        // The first paused frame was already consumed by estimator warmup.
        lastMedia.current = 0;
        recentTimes.current = [];
      }
      const start =
        kind === "demo"
          ? performance.now() - sourceStart.current
          : kind === "file"
            ? 0
            : (videoRef.current?.currentTime ?? 0) * 1000;
      framePump.current?.discardPending();
      engine.current.reset({
        stance: config.stance,
        calibrated: config.calibrated || kind === "demo",
      });
      const trackSettings = stream.current?.getVideoTracks()[0]?.getSettings();
      round.current = {
        config,
        start,
        startedAt: performance.now(),
        frames: [],
        events: [],
        skipped: 0,
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        source: kind,
        model: kind === "demo" ? "synthetic" : variantRef.current,
        modelManifest: runner.current?.modelInfo ?? modelManifest,
        detectorVersion:
          nativeDetectorVersion(runner.current?.modelInfo) ?? DETECTOR_VERSION,
        capture: {
          width: kind === "demo" ? 1280 : (videoRef.current?.videoWidth ?? 0),
          height: kind === "demo" ? 720 : (videoRef.current?.videoHeight ?? 0),
          requestedFps: kind === "camera" ? 30 : undefined,
          deliveredFps: trackSettings?.frameRate,
          timingSource:
            kind === "demo"
              ? "synthetic"
              : videoRef.current?.requestVideoFrameCallback
                ? "requestVideoFrameCallback"
                : "animationFrame-estimate",
          delegate:
            kind === "demo"
              ? "synthetic"
              : (runner.current?.delegate ?? "unknown"),
        },
      };
      chunks.current = [];
      if (config.record && kind === "camera" && stream.current) {
        try {
          const mimeType = [
            "video/webm;codecs=vp9",
            "video/webm;codecs=vp8",
            "video/mp4",
          ].find((m) => MediaRecorder.isTypeSupported(m));
          const rec = new MediaRecorder(
            stream.current,
            mimeType ? { mimeType } : undefined,
          );
          rec.ondataavailable = (e) => {
            if (e.data.size) chunks.current.push(e.data);
          };
          rec.start(1000);
          recorder.current = rec;
        } catch {
          setError(
            "Video recording is unavailable in this browser. This round will save motion data only.",
          );
        }
      }
      const data = round.current;
      const gen = generation.current;
      if (kind === "file") {
        void videoRef.current?.play().catch(() => {
          if (gen !== generation.current || round.current !== data) return;
          setError(
            "Could not play this video format. Try an MP4 or WebM file.",
          );
          void stop();
        });
      } else {
        // Recording duration and its clock remain reliable when pose inference
        // is slow or camera-frame callbacks temporarily stop arriving.
        clearRoundClock();
        const durationMs = config.durationSeconds * 1000;
        elapsedClock.current = setInterval(() => {
          if (
            gen !== generation.current ||
            round.current !== data ||
            closingAt.current !== null
          )
            return;
          setElapsed(
            Math.min(
              durationMs,
              Math.max(0, performance.now() - data.startedAt),
            ),
          );
        }, 100);
        roundDeadline.current = setTimeout(
          () => {
            if (gen === generation.current && round.current === data)
              void finishRef.current();
          },
          Math.max(0, durationMs - (performance.now() - data.startedAt)),
        );
      }
      setEvents([]);
      setElapsed(0);
      setSkipped(0);
      setRunning(true);
    },
    [status, loop, releaseSource, clearRoundClock, stop],
  );

  useEffect(() => {
    mounted.current = true;
    const ended = () => {
      void finishRef.current();
    };
    const video = videoRef.current;
    video?.addEventListener("ended", ended);
    const hidden = () => {
      if (document.hidden && round.current) {
        setError("Round ended because this tab was hidden.");
        void stop();
      }
    };
    document.addEventListener("visibilitychange", hidden);
    return () => {
      mounted.current = false;
      round.current = null;
      if (recorder.current?.state === "recording") recorder.current.stop();
      releaseSource();
      video?.removeEventListener("ended", ended);
      document.removeEventListener("visibilitychange", hidden);
    };
  }, [releaseSource, stop]);

  return {
    videoRef,
    source,
    status,
    error,
    frame,
    quality,
    events,
    elapsed,
    running,
    settings,
    fps,
    skipped,
    start,
    stop,
    beginRound,
    finishRound,
    clearError: () => setError(null),
  };
}
