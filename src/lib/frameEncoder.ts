import type { FrameEncoderReply } from "../workers/frameEncoder.worker";

/** Moves JPEG work away from rendering. One owned bitmap, no frame queue. */
export class FrameEncoder {
  private worker: Worker | null = null;
  private pending: {
    id: number;
    resolve: (blob: Blob) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  } | null = null;
  private nextId = 1;
  private disposed = false;

  encode(bitmap: ImageBitmap, width: number, height: number): Promise<Blob> {
    if (this.disposed || this.pending) {
      bitmap.close();
      return Promise.reject(new Error("Frame encoder is unavailable or busy."));
    }
    try {
      this.worker ??= this.createWorker();
    } catch (error) {
      bitmap.close();
      return Promise.reject(error);
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.fail(new Error("Local frame encoding timed out."));
      }, 10_000);
      this.pending = { id, resolve, reject, timer };
      try {
        // Ownership transfers immediately; the worker closes it after drawing.
        this.worker!.postMessage({ id, bitmap, width, height }, [bitmap]);
      } catch (error) {
        bitmap.close();
        this.fail(
          error instanceof Error ? error : new Error("Frame transfer failed."),
        );
      }
    });
  }

  private createWorker(): Worker {
    const worker = new Worker(
      new URL("../workers/frameEncoder.worker.ts", import.meta.url),
      { type: "module", name: "corner-frame-encoder" },
    );
    worker.onmessage = (message: MessageEvent<FrameEncoderReply>) => {
      const pending = this.pending;
      if (!pending || message.data.id !== pending.id) return;
      clearTimeout(pending.timer);
      this.pending = null;
      if (message.data.type === "encoded") pending.resolve(message.data.blob);
      else pending.reject(new Error(message.data.error));
    };
    worker.onerror = () => this.fail(new Error("Local frame encoder stopped."));
    worker.onmessageerror = () =>
      this.fail(new Error("Local encoded frame could not be read."));
    return worker;
  }

  private fail(error: Error): void {
    const pending = this.pending;
    this.pending = null;
    if (pending) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.worker?.terminate();
    this.worker = null;
  }

  dispose(): void {
    this.disposed = true;
    this.fail(new Error("Local frame encoder disposed."));
  }
}
