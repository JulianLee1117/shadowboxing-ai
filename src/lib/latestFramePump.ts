export interface LatestFramePumpOptions {
  /** Borrows the snapshot until this promise settles. Do not close or transfer it. */
  process: (
    bitmap: ImageBitmap,
    t: number,
    observedAt: number,
  ) => Promise<unknown>;
  onSkipped?: () => void;
  onError?: (error: Error) => void;
  snapshot?: (source: ImageBitmapSource) => Promise<ImageBitmap>;
}

interface CapturedFrame {
  bitmap: ImageBitmap;
  t: number;
  observedAt: number;
}

/** Bounded latest-captured-frame handoff, never a queue of old video references.
 * One inference and one snapshot operation may run concurrently. At most one
 * captured frame waits; replacement closes it before another capture starts.
 * Call push at the source callback so snapshot pixels retain that callback's
 * timestamp. Callbacks during snapshot creation are skipped, never captured
 * later under their old time. The pump owns every returned ImageBitmap.
 */
export class LatestFramePump {
  private capturing = false;
  private processing = false;
  private pending: CapturedFrame | null = null;
  private disposed = false;
  private captureEpoch = 0;
  private lastTimestamp = -Infinity;
  private idleWaiters = new Set<() => void>();
  private readonly snapshot: NonNullable<LatestFramePumpOptions["snapshot"]>;

  constructor(private readonly options: LatestFramePumpOptions) {
    this.snapshot = options.snapshot ?? ((source) => createImageBitmap(source));
  }

  /** Returns whether capture started. The source itself is never queued. */
  push(source: ImageBitmapSource, t: number, observedAt: number): boolean {
    if (this.disposed) return false;
    if (
      !Number.isFinite(t) ||
      t < 0 ||
      !Number.isFinite(observedAt) ||
      observedAt < 0
    ) {
      this.reportError(
        new Error(
          "Frame source and observation times must be finite and nonnegative.",
        ),
      );
      return false;
    }
    if (t <= this.lastTimestamp) {
      this.skipped();
      return false;
    }
    this.lastTimestamp = t;
    if (this.capturing) {
      this.skipped();
      return false;
    }
    if (this.pending) {
      const old = this.pending;
      this.pending = null;
      this.close(old.bitmap);
      this.skipped();
    }
    if (this.disposed) return false;
    this.capturing = true;
    void this.capture(source, t, observedAt, this.captureEpoch);
    return true;
  }

  /** Drop queued/in-flight capture evidence, but allow later source callbacks. */
  discardPending(): void {
    this.captureEpoch++;
    if (this.pending) {
      const old = this.pending;
      this.pending = null;
      this.close(old.bitmap);
      this.skipped();
    }
    this.resolveIdle();
  }

  /** Terminal stop. Current inference borrows its bitmap until it settles. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.discardPending();
  }

  /** Resolves after current capture, inference and any waiting snapshot drain. */
  whenIdle(): Promise<void> {
    if (!this.capturing && !this.processing && !this.pending)
      return Promise.resolve();
    return new Promise((resolve) => this.idleWaiters.add(resolve));
  }

  private async capture(
    source: ImageBitmapSource,
    t: number,
    observedAt: number,
    epoch: number,
  ): Promise<void> {
    let bitmap: ImageBitmap;
    try {
      // Invoke snapshot immediately, before the first await, while source/t match.
      bitmap = await this.snapshot(source);
    } catch (error) {
      this.capturing = false;
      if (!this.disposed && epoch === this.captureEpoch)
        this.reportError(error);
      this.resolveIdle();
      return;
    }
    this.capturing = false;
    if (this.disposed || epoch !== this.captureEpoch) {
      this.close(bitmap);
      this.skipped();
    } else {
      const frame = { bitmap, t, observedAt };
      if (this.processing) this.pending = frame;
      else void this.process(frame);
    }
    this.resolveIdle();
  }

  private async process(frame: CapturedFrame): Promise<void> {
    this.processing = true;
    try {
      await this.options.process(frame.bitmap, frame.t, frame.observedAt);
    } catch (error) {
      if (!this.disposed) this.reportError(error);
    } finally {
      this.close(frame.bitmap);
      this.processing = false;
      const next = this.pending;
      this.pending = null;
      if (next) {
        if (this.disposed) {
          this.close(next.bitmap);
          this.skipped();
        } else void this.process(next);
      }
      this.resolveIdle();
    }
  }

  private close(bitmap: ImageBitmap): void {
    try {
      bitmap.close();
    } catch (error) {
      this.reportError(error);
    }
  }

  private skipped(): void {
    try {
      this.options.onSkipped?.();
    } catch (error) {
      this.reportError(error);
    }
  }

  private reportError(error: unknown): void {
    try {
      this.options.onError?.(
        error instanceof Error ? error : new Error(String(error)),
      );
    } catch {
      // Consumer error reporting must not prevent owned bitmap cleanup.
    }
  }

  private resolveIdle(): void {
    if (this.capturing || this.processing || this.pending) return;
    const waiters = [...this.idleWaiters];
    this.idleWaiters.clear();
    waiters.forEach((resolve) => resolve());
  }
}
