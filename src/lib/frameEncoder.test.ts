import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FrameEncoder } from "./frameEncoder";

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onmessageerror: (() => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();
  constructor() {
    FakeWorker.instances.push(this);
  }
  reply(data: unknown) {
    this.onmessage?.({ data } as MessageEvent);
  }
}
const bitmap = () =>
  ({ width: 640, height: 360, close: vi.fn() }) as unknown as ImageBitmap;
beforeEach(() => {
  FakeWorker.instances = [];
  vi.stubGlobal("Worker", FakeWorker);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("owned local frame encoder", () => {
  it("transfers exact snapshots and reuses a single worker across frames", async () => {
    const encoder = new FrameEncoder(),
      first = bitmap();
    const a = encoder.encode(first, 640, 360),
      worker = FakeWorker.instances[0];
    expect(worker.postMessage).toHaveBeenCalledWith(
      { id: 1, bitmap: first, width: 640, height: 360 },
      [first],
    );
    const blob = new Blob(["jpeg"], { type: "image/jpeg" });
    worker.reply({ id: 1, type: "encoded", blob });
    expect(await a).toBe(blob);
    const second = bitmap(),
      b = encoder.encode(second, 640, 360);
    expect(FakeWorker.instances).toHaveLength(1);
    worker.reply({ id: 2, type: "encoded", blob });
    expect(await b).toBe(blob);
    encoder.dispose();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });
  it("rejects and closes another snapshot instead of queueing it", async () => {
    const encoder = new FrameEncoder(),
      a = encoder.encode(bitmap(), 640, 360),
      extra = bitmap();
    await expect(encoder.encode(extra, 640, 360)).rejects.toThrow("busy");
    expect(extra.close).toHaveBeenCalledOnce();
    encoder.dispose();
    await expect(a).rejects.toThrow("disposed");
  });
  it("rejects active encoding on disposal and never accepts late replies", async () => {
    const encoder = new FrameEncoder(),
      a = encoder.encode(bitmap(), 640, 360),
      worker = FakeWorker.instances[0];
    encoder.dispose();
    worker.reply({ id: 1, type: "encoded", blob: new Blob() });
    await expect(a).rejects.toThrow("disposed");
    const extra = bitmap();
    await expect(encoder.encode(extra, 640, 360)).rejects.toThrow(
      "unavailable",
    );
    expect(extra.close).toHaveBeenCalledOnce();
  });
  it("times out a lost worker response without leaving a pending frame", async () => {
    vi.useFakeTimers();
    const encoder = new FrameEncoder(),
      a = encoder.encode(bitmap(), 640, 360),
      worker = FakeWorker.instances[0];
    const rejection = expect(a).rejects.toThrow("timed out");
    await vi.advanceTimersByTimeAsync(10_000);
    await rejection;
    expect(worker.terminate).toHaveBeenCalledOnce();
    encoder.dispose();
  });
  it("closes an untransferred bitmap when transfer fails", async () => {
    const encoder = new FrameEncoder(),
      first = encoder.encode(bitmap(), 640, 360),
      worker = FakeWorker.instances[0];
    worker.reply({ id: 1, type: "error", error: "Encoding failed" });
    await expect(first).rejects.toThrow("Encoding failed");
    worker.postMessage.mockImplementationOnce(() => {
      throw new Error("transfer rejected");
    });
    const next = bitmap();
    await expect(encoder.encode(next, 640, 360)).rejects.toThrow(
      "transfer rejected",
    );
    expect(next.close).toHaveBeenCalledOnce();
    expect(worker.terminate).toHaveBeenCalledOnce();
    encoder.dispose();
  });
});
