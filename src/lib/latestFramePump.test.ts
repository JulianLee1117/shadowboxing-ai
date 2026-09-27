import { describe, expect, it, vi } from "vitest";
import { LatestFramePump } from "./latestFramePump";

const source = {} as ImageBitmapSource;
const bitmap = () => ({ close: vi.fn() }) as unknown as ImageBitmap;
function deferred<T = void>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
const tick = async () => {
  await Promise.resolve();
  await Promise.resolve();
};

describe("bounded latest-frame ownership", () => {
  it("replaces a waiting snapshot and retains its original source times", async () => {
    const first = bitmap(),
      old = bitmap(),
      latest = bitmap();
    const run = deferred();
    const skipped = vi.fn();
    const process = vi
      .fn()
      .mockReturnValueOnce(run.promise)
      .mockResolvedValue(undefined);
    const snapshot = vi
      .fn()
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce(old)
      .mockResolvedValueOnce(latest);
    const pump = new LatestFramePump({ snapshot, process, onSkipped: skipped });
    pump.push(source, 0, 100);
    await tick();
    pump.push(source, 33, 133);
    await tick();
    expect(process).toHaveBeenCalledTimes(1);
    pump.push(source, 66, 166);
    expect(old.close).toHaveBeenCalledTimes(1);
    await tick();
    expect(process).toHaveBeenCalledTimes(1);
    run.resolve();
    await pump.whenIdle();
    expect(process.mock.calls).toEqual([
      [first, 0, 100],
      [latest, 66, 166],
    ]);
    expect(skipped).toHaveBeenCalledTimes(1);
    for (const item of [first, old, latest])
      expect(item.close).toHaveBeenCalledTimes(1);
  });

  it("coalesces snapshot creation without later capturing a live source at an old timestamp", async () => {
    const capture = deferred<ImageBitmap>();
    const taken = bitmap();
    const process = vi.fn().mockResolvedValue(undefined);
    const snapshot = vi.fn().mockReturnValue(capture.promise);
    const skipped = vi.fn();
    const pump = new LatestFramePump({ snapshot, process, onSkipped: skipped });
    expect(pump.push(source, 10, 100)).toBe(true);
    for (let i = 1; i <= 100; i++)
      expect(pump.push(source, 10 + i, 100 + i)).toBe(false);
    expect(snapshot).toHaveBeenCalledTimes(1);
    capture.resolve(taken);
    await pump.whenIdle();
    expect(process.mock.calls).toEqual([[taken, 10, 100]]);
    expect(snapshot).toHaveBeenCalledTimes(1);
    expect(skipped).toHaveBeenCalledTimes(100);
    expect(taken.close).toHaveBeenCalledTimes(1);
  });

  it("keeps just one inference active and immediately hands off a ready snapshot", async () => {
    const first = deferred(),
      second = deferred();
    const a = bitmap(),
      b = bitmap();
    const process = vi
      .fn()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const pump = new LatestFramePump({
      process,
      snapshot: vi.fn().mockResolvedValueOnce(a).mockResolvedValueOnce(b),
    });
    pump.push(source, 0, 0);
    await tick();
    pump.push(source, 33, 33);
    await tick();
    const idle = vi.fn();
    void pump.whenIdle().then(idle);
    expect(process).toHaveBeenCalledTimes(1);
    first.resolve();
    await tick();
    expect(process).toHaveBeenCalledTimes(2);
    expect(idle).not.toHaveBeenCalled();
    expect(b.close).not.toHaveBeenCalled();
    second.resolve();
    await pump.whenIdle();
    expect(b.close).toHaveBeenCalledTimes(1);
  });

  it("disposal closes waiting work but does not close a bitmap borrowed by inference", async () => {
    const a = bitmap(),
      b = bitmap();
    const run = deferred();
    const process = vi.fn().mockReturnValue(run.promise);
    const pump = new LatestFramePump({
      process,
      snapshot: vi.fn().mockResolvedValueOnce(a).mockResolvedValueOnce(b),
    });
    pump.push(source, 0, 0);
    await tick();
    pump.push(source, 33, 33);
    await tick();
    pump.dispose();
    pump.dispose();
    expect(a.close).not.toHaveBeenCalled();
    expect(b.close).toHaveBeenCalledTimes(1);
    expect(pump.push(source, 66, 66)).toBe(false);
    const done = vi.fn();
    void pump.whenIdle().then(done);
    await tick();
    expect(done).not.toHaveBeenCalled();
    run.resolve();
    await pump.whenIdle();
    expect(process).toHaveBeenCalledTimes(1);
    expect(a.close).toHaveBeenCalledTimes(1);
  });

  it("disposal during capture closes its late result without inference", async () => {
    const capture = deferred<ImageBitmap>(),
      late = bitmap();
    const process = vi.fn();
    const pump = new LatestFramePump({
      process,
      snapshot: () => capture.promise,
    });
    pump.push(source, 0, 0);
    pump.dispose();
    const done = vi.fn();
    void pump.whenIdle().then(done);
    await tick();
    expect(done).not.toHaveBeenCalled();
    capture.resolve(late);
    await pump.whenIdle();
    expect(process).not.toHaveBeenCalled();
    expect(late.close).toHaveBeenCalledTimes(1);
  });

  it("discardPending invalidates capture and permits a subsequent round", async () => {
    const capture = deferred<ImageBitmap>(),
      old = bitmap(),
      next = bitmap();
    const process = vi.fn().mockResolvedValue(undefined);
    const snapshot = vi
      .fn()
      .mockReturnValueOnce(capture.promise)
      .mockResolvedValueOnce(next);
    const pump = new LatestFramePump({ process, snapshot });
    pump.push(source, 10, 100);
    pump.discardPending();
    capture.resolve(old);
    await pump.whenIdle();
    expect(old.close).toHaveBeenCalledTimes(1);
    expect(process).not.toHaveBeenCalled();
    expect(pump.push(source, 20, 110)).toBe(true);
    await pump.whenIdle();
    expect(process.mock.calls).toEqual([[next, 20, 110]]);
    expect(next.close).toHaveBeenCalledTimes(1);
  });

  it("snapshot and process errors still drain ownership and permit recovery", async () => {
    const failCapture = new Error("capture"),
      failProcess = new Error("inference");
    const a = bitmap(),
      b = bitmap();
    const errors = vi.fn();
    const snapshot = vi
      .fn()
      .mockRejectedValueOnce(failCapture)
      .mockResolvedValueOnce(a)
      .mockResolvedValueOnce(b);
    const process = vi
      .fn()
      .mockRejectedValueOnce(failProcess)
      .mockResolvedValue(undefined);
    const pump = new LatestFramePump({ snapshot, process, onError: errors });
    for (const t of [0, 1, 2]) {
      pump.push(source, t, t);
      await pump.whenIdle();
    }
    expect(errors.mock.calls).toEqual([[failCapture], [failProcess]]);
    expect(a.close).toHaveBeenCalledTimes(1);
    expect(b.close).toHaveBeenCalledTimes(1);
  });

  it("never delivers repeated or backwards source timestamps", async () => {
    const process = vi.fn().mockResolvedValue(undefined),
      skipped = vi.fn();
    const pump = new LatestFramePump({
      process,
      onSkipped: skipped,
      snapshot: async () => bitmap(),
    });
    pump.push(source, 10, 100);
    await pump.whenIdle();
    expect(pump.push(source, 10, 101)).toBe(false);
    expect(pump.push(source, 9, 102)).toBe(false);
    pump.push(source, 11, 103);
    await pump.whenIdle();
    expect(process.mock.calls.map((call) => call.slice(1))).toEqual([
      [10, 100],
      [11, 103],
    ]);
    expect(skipped).toHaveBeenCalledTimes(2);
  });
});
