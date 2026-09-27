import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocalPoseBusyError, LocalPoseClient } from "./localPoseClient";
import { LatestFramePump } from "./latestFramePump";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

const estimator = {
  id: "rtmpose-m",
  scoreType: "simcc",
  minimumScore: 0.55,
};
const response = (value: unknown) =>
  new Response(JSON.stringify(value), { status: 200 });
const ready = () => ({
  protocolVersion: "local-pose-1",
  sessionId: "session_1234567890123456",
  token: "token_" + "a".repeat(32),
  modelInfo: {
    id: "rtmpose-m",
    backend: "onnxruntime",
    delegate: "CoreML+CPU",
    estimator,
    modelManifest: Object.fromEntries(
      ["pose", "detector"].map((name) => [
        name,
        {
          sha256: "a".repeat(64),
          sourceUrl: "https://example.com/pinned-model",
        },
      ]),
    ),
  },
});
const frame = (t: number) => ({
  frame: {
    t,
    width: 640,
    height: 360,
    inferenceMs: 20,
    estimator,
    landmarks: [],
  },
  personCount: 0,
  nativeKeypoints: [],
  detectorObservations: [],
});
const source = {} as ImageBitmapSource;
const jpeg = new Blob(["fixed observed pixels"], { type: "image/jpeg" });
let captured: Array<{
  width: number;
  height: number;
  close: ReturnType<typeof vi.fn>;
}>;
let encode = vi.fn<() => Promise<Blob>>();
const clients: LocalPoseClient[] = [];

function transport(
  infer: (init: RequestInit) => Promise<Response> = async (init) =>
    response(
      frame(
        Number((init.headers as Record<string, string>)["X-Frame-Time-Ms"]),
      ),
    ),
) {
  return vi.fn(async (_url: string | URL | Request, init: RequestInit = {}) => {
    if (init.method === "DELETE") return response({ closed: true });
    if (
      (init.headers as Record<string, string>)?.["Content-Type"] ===
      "image/jpeg"
    )
      return infer(init);
    return response(ready());
  });
}

async function client(fetcher = transport()) {
  const value = new LocalPoseClient({ fetch: fetcher as typeof fetch });
  clients.push(value);
  await value.init();
  return value;
}

beforeEach(() => {
  captured = [];
  encode = vi.fn().mockResolvedValue(jpeg);
  vi.stubGlobal("Worker", undefined);
  vi.stubGlobal(
    "createImageBitmap",
    vi.fn().mockImplementation(async () => {
      const bitmap = { width: 640, height: 360, close: vi.fn() };
      captured.push(bitmap);
      return bitmap;
    }),
  );
  vi.stubGlobal(
    "OffscreenCanvas",
    class {
      getContext() {
        return { drawImage: vi.fn() };
      }
      convertToBlob() {
        return encode();
      }
    },
  );
});

afterEach(() => {
  for (const value of clients.splice(0)) value.dispose();
  vi.unstubAllGlobals();
});

describe("prepared local frame ownership and overlap", () => {
  it("captures immediately but sends nothing until the prepared frame is consumed", async () => {
    const fetcher = transport();
    const value = await client(fetcher);
    const preparing = value.prepareImage(source, 12.5);
    expect(createImageBitmap).toHaveBeenCalledWith(source);
    const prepared = await preparing;
    expect(fetcher).toHaveBeenCalledTimes(1);
    const result = await value.detectPrepared(prepared, 12.5);
    expect(result.t).toBe(12.5);
    const request = fetcher.mock.calls[1][1]!;
    expect(request.body).toBe(jpeg);
    expect(request.headers).toMatchObject({ "X-Frame-Time-Ms": "12.5" });
    prepared.close();
    expect(captured[0].close).toHaveBeenCalledTimes(1);
  });

  it("encodes the next observation during HTTP while preserving a single request", async () => {
    const firstResponse = deferred<Response>();
    let active = 0;
    let maximum = 0;
    let calls = 0;
    const fetcher = transport(async (init) => {
      active++;
      maximum = Math.max(maximum, active);
      try {
        return ++calls === 1
          ? await firstResponse.promise
          : response(
              frame(
                Number(
                  (init.headers as Record<string, string>)["X-Frame-Time-Ms"],
                ),
              ),
            );
      } finally {
        active--;
      }
    });
    const value = await client(fetcher);
    const first = await value.prepareImage(source, 0);
    const inFlight = value.detectPrepared(first, 0);
    expect(active).toBe(1);
    const next = await value.prepareImage(source, 33);
    expect(encode).toHaveBeenCalledTimes(2);
    expect(active).toBe(1);
    await expect(value.detectPrepared(next, 33)).rejects.toBeInstanceOf(
      LocalPoseBusyError,
    );
    expect(calls).toBe(1);
    firstResponse.resolve(response(frame(0)));
    await inFlight;
    await expect(value.detectPrepared(next, 33)).resolves.toMatchObject({
      t: 33,
    });
    expect(maximum).toBe(1);
    expect(calls).toBe(2);
    first.close();
    next.close();
  });

  it("keeps only one encoded waiting frame until it is closed or consumed", async () => {
    const value = await client();
    const prepared = await value.prepareImage(source, 0);
    await expect(value.prepareImage(source, 33)).rejects.toBeInstanceOf(
      LocalPoseBusyError,
    );
    expect(createImageBitmap).toHaveBeenCalledTimes(1);
    prepared.close();
    prepared.close();
    const next = await value.prepareImage(source, 33);
    await expect(value.detectPrepared(next, 33)).resolves.toMatchObject({
      t: 33,
    });
    next.close();
  });

  it("does not start a second encoder while capture or encoding is pending", async () => {
    const encoded = deferred<Blob>();
    encode.mockReturnValueOnce(encoded.promise);
    const value = await client();
    const first = value.prepareImage(source, 0);
    await expect(value.prepareImage(source, 33)).rejects.toBeInstanceOf(
      LocalPoseBusyError,
    );
    encoded.resolve(jpeg);
    const prepared = await first;
    expect(createImageBitmap).toHaveBeenCalledTimes(1);
    expect(encode).toHaveBeenCalledTimes(1);
    prepared.close();
  });

  it("rejects another client and changed timestamp without consuming the owner's frame", async () => {
    const owner = await client();
    const other = await client();
    const prepared = await owner.prepareImage(source, 20);
    await expect(other.detectPrepared(prepared, 20)).rejects.toThrow();
    await expect(owner.detectPrepared(prepared, 21)).rejects.toThrow();
    await expect(owner.detectPrepared(prepared, 20)).resolves.toMatchObject({
      t: 20,
    });
    await expect(owner.detectPrepared(prepared, 22)).rejects.toThrow();
    prepared.close();
  });

  it("never sends a closed waiting frame or revives one after disposal", async () => {
    const fetcher = transport();
    const value = await client(fetcher);
    const closed = await value.prepareImage(source, 20);
    closed.close();
    await expect(value.detectPrepared(closed, 20)).rejects.toThrow();
    const stale = await value.prepareImage(source, 30);
    value.dispose();
    await expect(value.detectPrepared(stale, 30)).rejects.toThrow();
    expect(
      fetcher.mock.calls.filter(([, init]) => init?.body instanceof Blob),
    ).toHaveLength(0);
    stale.close();
  });

  it("releases a late encoding result after disposal without making an inference request", async () => {
    const encoded = deferred<Blob>();
    encode.mockReturnValueOnce(encoded.promise);
    const fetcher = transport();
    const value = await client(fetcher);
    const preparing = value.prepareImage(source, 0);
    const rejection = expect(preparing).rejects.toThrow(/disposed|stopped/i);
    await Promise.resolve();
    value.dispose();
    encoded.resolve(jpeg);
    await rejection;
    expect(captured[0].close).toHaveBeenCalledTimes(1);
    expect(
      fetcher.mock.calls.filter(([, init]) => init?.body instanceof Blob),
    ).toHaveLength(0);
  });

  it("aborts HTTP and invalidates the separately prepared successor on disposal", async () => {
    let signal: AbortSignal | null = null;
    const fetcher = transport(async (init) => {
      signal = init.signal!;
      return new Promise<Response>((_resolve, reject) => {
        init.signal!.addEventListener(
          "abort",
          () => reject(new Error("aborted")),
          { once: true },
        );
      });
    });
    const value = await client(fetcher);
    const first = await value.prepareImage(source, 0);
    const running = value.detectPrepared(first, 0);
    const rejection = expect(running).rejects.toThrow(/disposed/i);
    const next = await value.prepareImage(source, 33);
    value.dispose();
    await rejection;
    expect(signal!.aborted).toBe(true);
    await expect(value.detectPrepared(next, 33)).rejects.toThrow();
    expect(
      fetcher.mock.calls.filter(([, init]) => init?.body instanceof Blob),
    ).toHaveLength(1);
    first.close();
    next.close();
  });

  it("replaces a waiting encoded frame without blocking the next callback or changing its timestamp", async () => {
    const firstResponse = deferred<Response>();
    const sent: number[] = [];
    const fetcher = transport(async (init) => {
      const t = Number(
        (init.headers as Record<string, string>)["X-Frame-Time-Ms"],
      );
      sent.push(t);
      return sent.length === 1 ? firstResponse.promise : response(frame(t));
    });
    const value = await client(fetcher);
    const prepare = vi.spyOn(value, "prepareImage");
    const errors = vi.fn();
    const pump = new LatestFramePump({
      snapshot: (pixels, t) => value.prepareImage(pixels, t),
      process: (prepared, t) => value.detectPrepared(prepared, t),
      onError: errors,
    });
    for (const [index, t] of [0, 33, 66].entries()) {
      expect(pump.push(source, t, t + 100)).toBe(true);
      await prepare.mock.results[index].value;
    }
    expect(sent).toEqual([0]);
    expect(encode).toHaveBeenCalledTimes(3);
    firstResponse.resolve(response(frame(0)));
    await pump.whenIdle();
    expect(sent).toEqual([0, 66]);
    expect(errors).not.toHaveBeenCalled();
    expect(
      captured.every((bitmap) => bitmap.close.mock.calls.length === 1),
    ).toBe(true);
    pump.dispose();
  });

  it("discards an encoding across a round boundary and can send the next observed frame", async () => {
    const encoded = deferred<Blob>();
    encode.mockReturnValueOnce(encoded.promise);
    const fetcher = transport();
    const value = await client(fetcher);
    const errors = vi.fn();
    const pump = new LatestFramePump({
      snapshot: (pixels, t) => value.prepareImage(pixels, t),
      process: (prepared, t) => value.detectPrepared(prepared, t),
      onError: errors,
    });
    expect(pump.push(source, 0, 100)).toBe(true);
    await Promise.resolve();
    pump.discardPending();
    encoded.resolve(jpeg);
    await pump.whenIdle();
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(pump.push(source, 33, 133)).toBe(true);
    await pump.whenIdle();
    const sent = fetcher.mock.calls.filter(
      ([, init]) => init?.body instanceof Blob,
    );
    expect(sent).toHaveLength(1);
    expect(sent[0][1]?.headers).toMatchObject({ "X-Frame-Time-Ms": "33" });
    expect(errors).not.toHaveBeenCalled();
    expect(
      captured.every((bitmap) => bitmap.close.mock.calls.length === 1),
    ).toBe(true);
    pump.dispose();
  });
});
