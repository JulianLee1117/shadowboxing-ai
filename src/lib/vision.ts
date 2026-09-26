import type { ModelVariant, PoseFrame } from "./types";
import type {
  PoseWorkerRequest,
  PoseWorkerReply,
} from "../workers/pose.worker";

type Delegate = "GPU" | "CPU";
interface PendingRequest {
  resolve: (message: PoseWorkerReply) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

/** One local worker per source session. All input coordinates remain unmirrored. */
export class VisionRunner {
  private worker: Worker | null = null;
  private pending = new Map<number, PendingRequest>();
  private nextRequestId = 1;
  private initPromise: Promise<void> | null = null;
  private initialized = false;
  private disposed = false;
  private detecting = false;
  private activeDelegate: Delegate | null = null;

  constructor(private readonly variant: ModelVariant) {}

  /** The actual selected delegate. CPU fallback is observable, never implied GPU. */
  get delegate(): Delegate | null {
    return this.activeDelegate;
  }

  init(): Promise<void> {
    if (this.disposed)
      return Promise.reject(new Error("Pose runner has been disposed."));
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.initialize();
    return this.initPromise;
  }

  private async initialize(): Promise<void> {
    let gpuError: unknown;
    try {
      await this.startWorker("GPU");
    } catch (error) {
      gpuError = error;
      this.stopWorker(new Error("Restarting pose worker with CPU inference."));
      if (this.disposed)
        throw new Error("Pose runner was disposed during initialization.");
      try {
        // A fresh worker also releases any partially initialized GPU/WASM state.
        await this.startWorker("CPU");
      } catch (cpuError) {
        this.stopWorker(new Error("Pose initialization failed."));
        throw new Error(
          `Local pose model could not start. GPU: ${messageOf(gpuError)} CPU: ${messageOf(cpuError)}`,
        );
      }
    }
    if (this.disposed)
      throw new Error("Pose runner was disposed during initialization.");
    this.initialized = true;
  }

  private async startWorker(delegate: Delegate): Promise<void> {
    const worker = new Worker(
      new URL("../workers/pose.worker.ts", import.meta.url),
      { type: "module" },
    );
    this.worker = worker;
    worker.onmessage = (event: MessageEvent<PoseWorkerReply>) => {
      if (this.worker !== worker) return;
      const message = event.data;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.pending.delete(message.id);
      if (message.type === "error") pending.reject(new Error(message.error));
      else pending.resolve(message);
    };
    worker.onerror = (event) => {
      event.preventDefault();
      if (this.worker === worker)
        this.stopWorker(
          new Error(event.message || "Pose worker failed to load."),
        );
    };
    worker.onmessageerror = () => {
      if (this.worker === worker)
        this.stopWorker(
          new Error("Pose worker returned an unreadable message."),
        );
    };
    const reply = await this.request(
      {
        type: "init",
        id: this.nextRequestId++,
        variant: this.variant,
        delegate,
        // The application is local, and these assets are always same-origin.
        modelUrl: new URL(
          `/models/pose_landmarker_${this.variant}.task`,
          window.location.href,
        ).href,
        wasmUrl: new URL("/wasm", window.location.href).href,
      },
      [],
      25_000,
    );
    if (reply.type !== "ready")
      throw new Error("Unexpected pose initialization response.");
    this.activeDelegate = reply.delegate;
  }

  async detect(video: HTMLVideoElement, t: number): Promise<PoseFrame> {
    if (this.disposed) throw new Error("Pose runner has been disposed.");
    if (!this.initialized || !this.worker)
      throw new Error("Initialize the pose runner before detecting frames.");
    if (this.detecting)
      throw new Error(
        "Pose inference already in progress; skip this frame instead of queueing it.",
      );
    if (!Number.isFinite(t) || t < 0)
      throw new Error(
        "Frame timestamp must be finite, source-relative milliseconds.",
      );
    if (
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      throw new Error("Video has no decoded frame available yet.");
    }
    this.detecting = true;
    let bitmap: ImageBitmap | undefined;
    try {
      // CSS preview mirroring does not affect this raw pixel capture.
      bitmap = await createImageBitmap(video);
      if (this.disposed || !this.worker)
        throw new Error("Pose runner stopped while capturing the frame.");
      const reply = await this.request(
        {
          type: "detect",
          id: this.nextRequestId++,
          bitmap,
          t,
          width: bitmap.width,
          height: bitmap.height,
        },
        [bitmap],
        10_000,
      );
      // Ownership transferred; the worker closes the bitmap in its finally block.
      bitmap = undefined;
      if (reply.type !== "result")
        throw new Error("Unexpected pose inference response.");
      return reply.frame;
    } finally {
      // Also handles capture/transfer failure and disposal during bitmap creation.
      bitmap?.close();
      this.detecting = false;
    }
  }

  private request(
    message: PoseWorkerRequest,
    transfer: Transferable[],
    timeoutMs: number,
  ): Promise<PoseWorkerReply> {
    const worker = this.worker;
    if (!worker || this.disposed)
      return Promise.reject(new Error("Pose worker is unavailable."));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.stopWorker(
          new Error(
            `${message.type === "init" ? "Pose initialization" : "Pose inference"} timed out. Check local model assets and browser support.`,
          ),
        );
      }, timeoutMs);
      this.pending.set(message.id, { resolve, reject, timer });
      try {
        worker.postMessage(message, transfer);
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(message.id);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  private stopWorker(error: Error): void {
    this.initialized = false;
    this.activeDelegate = null;
    this.worker?.terminate();
    this.worker = null;
    for (const request of this.pending.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    this.pending.clear();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.stopWorker(new Error("Pose runner disposed."));
  }
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
