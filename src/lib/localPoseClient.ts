import type { NativeRecognition, PoseFrame } from "./types";
import { FrameEncoder } from "./frameEncoder";

const PROTOCOL_VERSION = "local-pose-1";
const MAXIMUM_PIXELS = 921_600;
const MAXIMUM_IMAGE_BYTES = 3_000_000;
const RECOGNITION_PROTOCOL = "shadowbox-recognition-v1";

export interface LocalRecognizerInfo {
  protocolVersion: "shadowbox-recognition-v1";
  recognizerId: "personal-hybrid-v1";
  fingerprint: string;
  featureVersion: "arm-offsets-native-v3";
  checkpointSha256: string;
  externalModelSha256: string;
  poseModelSha256: string;
  developmentOnly: true;
  scoreSemantics: "uncalibrated_model_support";
}

export interface LocalPoseModelInfo {
  id: "rtmpose-m" | "rtmw-l";
  family: "rtmpose-body" | "rtmw-wholebody";
  backend: "onnxruntime";
  delegate: "CPU" | "CoreML+CPU";
  estimator: NonNullable<PoseFrame["estimator"]>;
  recognizer?: LocalRecognizerInfo;
  confidenceSemantics: string;
  confidencePolicy: string;
  modelManifest: Record<
    "detector" | "pose",
    {
      sourceUrl: string;
      sha256: string;
      inputSize: [number, number];
      inputColorOrder?: string | null;
    }
  >;
  sessionProviders: Record<"detector" | "pose", string[]>;
  [key: string]: unknown;
}

export class LocalPoseBusyError extends Error {
  constructor(
    message = "Local pose inference is busy. Skip this frame instead of queueing it.",
  ) {
    super(message);
    this.name = "LocalPoseBusyError";
  }
}

interface SessionCredentials {
  sessionId: string;
  token: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

const sha256 = (value: unknown): value is string =>
  typeof value === "string" && /^[a-f0-9]{64}$/.test(value);

function readRecognizerInfo(
  value: unknown,
  poseHash: string,
): LocalRecognizerInfo {
  if (
    !isRecord(value) ||
    value.protocolVersion !== RECOGNITION_PROTOCOL ||
    value.recognizerId !== "personal-hybrid-v1" ||
    value.featureVersion !== "arm-offsets-native-v3" ||
    value.developmentOnly !== true ||
    value.scoreSemantics !== "uncalibrated_model_support" ||
    ![
      value.fingerprint,
      value.checkpointSha256,
      value.externalModelSha256,
      value.poseModelSha256,
    ].every(sha256) ||
    value.poseModelSha256 !== poseHash
  ) {
    throw new Error(
      "Local recognizer checkpoint or feature provenance is incompatible.",
    );
  }
  return value as unknown as LocalRecognizerInfo;
}

function validateRecognition(
  value: unknown,
  expected: LocalRecognizerInfo | undefined,
  t: number,
): asserts value is NativeRecognition | undefined {
  if (!expected) {
    if (value !== undefined)
      throw new Error(
        "Unexpected local recognizer output without configured weights.",
      );
    return;
  }
  if (
    !isRecord(value) ||
    value.protocolVersion !== expected.protocolVersion ||
    value.recognizerId !== expected.recognizerId ||
    value.fingerprint !== expected.fingerprint ||
    !["warming", "active", "uncertain"].includes(String(value.state)) ||
    !Array.isArray(value.events) ||
    value.events.length > 8
  ) {
    throw new Error(
      "Configured local recognizer output is missing or incompatible. No fallback was used.",
    );
  }
  const ids = new Set<string>();
  for (const event of value.events) {
    if (
      !isRecord(event) ||
      typeof event.id !== "string" ||
      !event.id ||
      ids.has(event.id) ||
      !["left", "right"].includes(String(event.hand)) ||
      !["straight", "hook", "uppercut"].includes(String(event.family)) ||
      ![
        event.startMs,
        event.peakMs,
        event.endMs,
        event.detectedAtMs,
        event.score,
      ].every((v) => typeof v === "number" && Number.isFinite(v)) ||
      Number(event.startMs) < 0 ||
      Number(event.startMs) >= Number(event.endMs) ||
      Number(event.peakMs) < Number(event.startMs) ||
      Number(event.peakMs) > Number(event.endMs) ||
      Number(event.endMs) > Number(event.detectedAtMs) ||
      event.detectedAtMs !== t ||
      Number(event.score) < 0 ||
      Number(event.score) > 1
    ) {
      throw new Error(
        "Local recognizer returned invalid causal event evidence.",
      );
    }
    ids.add(event.id);
  }
}

function readModelInfo(value: unknown): LocalPoseModelInfo {
  if (
    !isRecord(value) ||
    !["rtmpose-m", "rtmw-l"].includes(String(value.id)) ||
    !["CPU", "CoreML+CPU"].includes(String(value.delegate)) ||
    value.backend !== "onnxruntime" ||
    !isRecord(value.estimator) ||
    value.estimator.id !== value.id ||
    value.estimator.scoreType !== "simcc" ||
    typeof value.estimator.minimumScore !== "number" ||
    !Number.isFinite(value.estimator.minimumScore) ||
    value.estimator.minimumScore <= 0 ||
    value.estimator.minimumScore >= 1 ||
    !isRecord(value.modelManifest)
  ) {
    throw new Error(
      "Local pose service returned unsupported model provenance.",
    );
  }
  for (const key of ["detector", "pose"]) {
    const item = value.modelManifest[key];
    if (
      !isRecord(item) ||
      typeof item.sha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(item.sha256) ||
      typeof item.sourceUrl !== "string" ||
      !item.sourceUrl.startsWith("https://")
    ) {
      throw new Error("Local pose model fingerprints are missing.");
    }
  }
  if (value.recognizer !== undefined) {
    const pose = value.modelManifest.pose as Record<string, unknown>;
    readRecognizerInfo(value.recognizer, String(pose.sha256));
  }
  return value as unknown as LocalPoseModelInfo;
}

/** Owned immutable pixels; close releases an unsent frame. Single use, same client. */
export interface PreparedLocalPoseFrame {
  close(): void;
}

interface PreparedImage {
  jpeg: Blob;
  width: number;
  height: number;
  t: number;
  started: number;
  captured: number;
  encoded: number;
  credentials: SessionCredentials;
}

/** Optional same-origin bridge to an explicitly started loopback service.
 * This client never starts a server, downloads weights, or contacts a cloud API.
 */
export class LocalPoseClient {
  private readonly baseUrl: string;
  private readonly fetcher: typeof fetch;
  private readonly timeoutMs: number;
  private credentials: SessionCredentials | null = null;
  private info: LocalPoseModelInfo | null = null;
  private initPromise: Promise<void> | null = null;
  private controller: AbortController | null = null;
  private disposed = false;
  private detecting = false;
  private encoding = false;
  private prepared = new Map<PreparedLocalPoseFrame, PreparedImage>();
  private lastTimestamp = -1;
  private encoder: FrameEncoder | null = null;

  constructor(
    options: {
      baseUrl?: string;
      fetch?: typeof fetch;
      timeoutMs?: number;
    } = {},
  ) {
    const location = globalThis.location?.href ?? "http://127.0.0.1:5173/";
    const base = new URL(options.baseUrl ?? "/local-pose", location);
    const page = new URL(location);
    if (
      base.origin !== page.origin ||
      base.protocol !== "http:" ||
      !["127.0.0.1", "localhost"].includes(base.hostname) ||
      base.search ||
      base.hash ||
      base.username ||
      base.password
    ) {
      throw new Error(
        "Local pose requests must use the same local application origin.",
      );
    }
    this.baseUrl = base.href.replace(/\/$/, "");
    this.fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.timeoutMs = options.timeoutMs ?? 10_000;
    if (
      !Number.isFinite(this.timeoutMs) ||
      this.timeoutMs <= 0 ||
      this.timeoutMs > 60_000
    )
      throw new Error(
        "Local pose request timeout must be between 1 and 60000 ms.",
      );
  }

  get delegate(): LocalPoseModelInfo["delegate"] | null {
    return this.info?.delegate ?? null;
  }
  get modelInfo(): LocalPoseModelInfo | null {
    return this.info;
  }

  init(): Promise<void> {
    if (this.disposed)
      return Promise.reject(new Error("Local pose client is disposed."));
    if (this.credentials) return Promise.resolve();
    this.initPromise ??= this.initialize().finally(() => {
      this.initPromise = null;
    });
    return this.initPromise;
  }

  private async initialize(): Promise<void> {
    const response = await this.request("/v1/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (
      !isRecord(response) ||
      response.protocolVersion !== PROTOCOL_VERSION ||
      typeof response.sessionId !== "string" ||
      !/^[a-zA-Z0-9_-]{16,100}$/.test(response.sessionId) ||
      typeof response.token !== "string" ||
      !/^[a-zA-Z0-9_-]{32,100}$/.test(response.token)
    ) {
      throw new Error("Local pose service returned an invalid session.");
    }
    const credentials = {
      sessionId: response.sessionId,
      token: response.token,
    };
    try {
      const info = readModelInfo(response.modelInfo);
      if (this.disposed)
        throw new Error("Local pose client stopped during initialization.");
      this.credentials = credentials;
      this.info = info;
    } catch (error) {
      this.closeSession(credentials);
      throw error;
    }
  }

  async detect(video: HTMLVideoElement, t: number): Promise<PoseFrame> {
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight)
      throw new Error("Video has no decoded frame available yet.");
    return this.detectImage(video, t);
  }

  /** Serial convenience path for warmup and offline analysis. */
  async detectImage(source: ImageBitmapSource, t: number): Promise<PoseFrame> {
    this.assertDetectionReady(t);
    const prepared = await this.prepareImage(source, t);
    try {
      return await this.detectPrepared(prepared, t);
    } finally {
      prepared.close();
    }
  }

  /** Capture immediately at the source callback, then encode off the UI thread.
   * One encoding may overlap one HTTP request. The caller owns the result and
   * must send or close it before preparing another waiting frame.
   */
  async prepareImage(
    source: ImageBitmapSource,
    t: number,
  ): Promise<PreparedLocalPoseFrame> {
    if (this.disposed) throw new Error("Local pose client is disposed.");
    const credentials = this.credentials;
    if (!credentials || !this.info)
      throw new Error("Initialize local pose before preparing frames.");
    this.validateTimestamp(t);
    if (this.encoding || this.prepared.size) throw new LocalPoseBusyError();
    this.encoding = true;
    const started = performance.now();
    let bitmap: ImageBitmap | undefined;
    try {
      // CSS mirroring is deliberately excluded from analysis pixels.
      bitmap = await createImageBitmap(source);
      if (
        !bitmap.width ||
        !bitmap.height ||
        bitmap.width * bitmap.height > 16_000_000
      )
        throw new Error(
          "Decoded image dimensions exceed the local capture limit.",
        );
      if (this.disposed)
        throw new Error("Local pose stopped while capturing the frame.");
      const scale = Math.min(
        1,
        1280 / Math.max(bitmap.width, bitmap.height),
        Math.sqrt(MAXIMUM_PIXELS / (bitmap.width * bitmap.height)),
      );
      // Rounding both dimensions upward can exceed the server's pixel cap.
      const width = Math.max(1, Math.floor(bitmap.width * scale));
      const height = Math.max(1, Math.floor(bitmap.height * scale));
      const captured = performance.now();
      let jpeg: Blob;
      if (typeof Worker !== "undefined") {
        this.encoder ??= new FrameEncoder();
        jpeg = await this.encoder.encode(bitmap, width, height);
        bitmap = undefined; // transferred ownership
      } else {
        // Environments without workers retain the same bounded pixel contract.
        const canvas = new OffscreenCanvas(width, height);
        const context = canvas.getContext("2d");
        if (!context)
          throw new Error("Local pose image capture is unavailable.");
        context.drawImage(bitmap, 0, 0, width, height);
        jpeg = await canvas.convertToBlob({
          type: "image/jpeg",
          quality: 0.95,
        });
      }
      if (jpeg.type !== "image/jpeg" || jpeg.size > MAXIMUM_IMAGE_BYTES)
        throw new Error(
          "Frame encoding exceeded the local pose transport limit.",
        );
      if (this.disposed)
        throw new Error("Local pose stopped while encoding the frame.");
      const encoded = performance.now();
      if (this.credentials !== credentials)
        throw new Error("Local pose session changed while encoding.");
      const prepared: PreparedLocalPoseFrame = Object.freeze({
        close: () => {
          this.prepared.delete(prepared);
        },
      });
      this.prepared.set(prepared, {
        jpeg,
        width,
        height,
        t,
        started,
        captured,
        encoded,
        credentials,
      });
      return prepared;
    } finally {
      bitmap?.close();
      this.encoding = false;
    }
  }

  /** Consume one prepared image without capturing again or changing its clock. */
  async detectPrepared(
    prepared: PreparedLocalPoseFrame,
    t: number,
  ): Promise<PoseFrame> {
    this.assertDetectionReady(t);
    const image = this.prepared.get(prepared);
    if (!image || image.credentials !== this.credentials)
      throw new Error(
        "Prepared frame is closed, consumed, or belongs to another session.",
      );
    if (image.t !== t)
      throw new Error("Prepared frame source timestamp changed.");
    this.prepared.delete(prepared);
    this.detecting = true;
    const { jpeg, width, height, started, captured, encoded, credentials } =
      image;
    const info = this.info!;
    const requested = performance.now();
    try {
      const response = await this.request(
        `/v1/sessions/${credentials.sessionId}/frame`,
        {
          method: "POST",
          headers: {
            "Content-Type": "image/jpeg",
            Authorization: `Bearer ${credentials.token}`,
            "X-Frame-Time-Ms": String(t),
          },
          body: jpeg,
        },
      );
      if (this.disposed || this.credentials !== credentials)
        throw new Error(
          "Local pose session closed before this result arrived.",
        );
      if (!isRecord(response) || !isRecord(response.frame))
        throw new Error("Local pose service returned no frame.");
      const frame = response.frame;
      if (
        frame.t !== t ||
        frame.width !== width ||
        frame.height !== height ||
        typeof frame.inferenceMs !== "number" ||
        !Number.isFinite(frame.inferenceMs) ||
        frame.inferenceMs < 0 ||
        !Array.isArray(frame.landmarks) ||
        ![0, 33].includes(frame.landmarks.length) ||
        !isRecord(frame.estimator) ||
        frame.estimator.id !== info.id ||
        frame.estimator.scoreType !== "simcc" ||
        frame.estimator.minimumScore !== info.estimator.minimumScore ||
        !Number.isInteger(response.personCount) ||
        Number(response.personCount) < 0 ||
        (response.personCount !== 1 && frame.landmarks.length !== 0)
      )
        throw new Error(
          "Local pose output dimensions, time or estimator identity changed.",
        );
      for (const point of frame.landmarks) {
        if (
          !isRecord(point) ||
          ![point.x, point.y, point.score].every(
            (value) => typeof value === "number" && Number.isFinite(value),
          ) ||
          "visibility" in point ||
          "presence" in point
        )
          throw new Error(
            "Local pose output must preserve finite native keypoint scores.",
          );
      }
      validateRecognition(frame.recognition, info.recognizer, t);
      this.lastTimestamp = t;
      const completed = performance.now();
      const queueMs = requested - encoded;
      return {
        ...frame,
        nativeKeypoints: response.nativeKeypoints,
        personCount: response.personCount,
        detectorObservations: response.detectorObservations,
        localTiming: {
          ...(isRecord(response.timing) ? response.timing : {}),
          captureMs: captured - started,
          encodeMs: encoded - captured,
          queueMs,
          requestMs: completed - requested,
          // Preserve active-work timing; source-to-result age is measured by the pump.
          totalMs: completed - started - queueMs,
          preparedAgeMs: completed - started,
          encodedBytes: jpeg.size,
        },
      } as unknown as PoseFrame;
    } finally {
      this.detecting = false;
    }
  }

  private validateTimestamp(t: number): void {
    if (!Number.isFinite(t) || t < 0 || t <= this.lastTimestamp)
      throw new Error(
        "Frame times must increase. Open a new local pose session after seeking.",
      );
  }

  private assertDetectionReady(t: number): void {
    if (this.disposed) throw new Error("Local pose client is disposed.");
    if (!this.credentials || !this.info)
      throw new Error("Initialize local pose before detecting frames.");
    if (this.detecting) throw new LocalPoseBusyError();
    this.validateTimestamp(t);
  }

  private async request(path: string, init: RequestInit): Promise<unknown> {
    const controller = new AbortController();
    this.controller = controller;
    const timer = setTimeout(
      () => controller.abort(new Error("Local pose request timed out.")),
      this.timeoutMs,
    );
    try {
      const response = await this.fetcher(this.baseUrl + path, {
        ...init,
        credentials: "omit",
        mode: "same-origin",
        redirect: "error",
        cache: "no-store",
        signal: controller.signal,
      });
      const text = await response.text();
      let body: unknown;
      try {
        body = JSON.parse(text);
      } catch {
        body = null;
      }
      if (response.status === 429) throw new LocalPoseBusyError();
      if (!response.ok)
        throw new Error(
          isRecord(body) && typeof body.error === "string"
            ? body.error
            : "Local pose service is unavailable. Start it from this project, then try again.",
        );
      return body;
    } catch (error) {
      if (error instanceof LocalPoseBusyError) throw error;
      if (controller.signal.aborted)
        throw new Error(
          this.disposed
            ? "Local pose client disposed."
            : "Local pose request timed out.",
        );
      throw error instanceof Error
        ? error
        : new Error("Local pose request failed.");
    } finally {
      clearTimeout(timer);
      if (this.controller === controller) this.controller = null;
    }
  }

  private closeSession(credentials: SessionCredentials): void {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2000);
    void this.fetcher(`${this.baseUrl}/v1/sessions/${credentials.sessionId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${credentials.token}` },
      credentials: "omit",
      mode: "same-origin",
      redirect: "error",
      cache: "no-store",
      keepalive: true,
      signal: controller.signal,
    })
      .catch(() => {
        /* An unreachable service expires abandoned sessions after 60 s. */
      })
      .finally(() => clearTimeout(timer));
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.controller?.abort();
    this.encoder?.dispose();
    this.prepared.clear();
    if (this.credentials) this.closeSession(this.credentials);
    this.credentials = null;
    this.info = null;
  }
}
