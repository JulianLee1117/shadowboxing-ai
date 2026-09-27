import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocalPoseBusyError, LocalPoseClient } from "./localPoseClient";

const metadata = () => ({
  id: "rtmpose-m",
  backend: "onnxruntime",
  delegate: "CoreML+CPU",
  estimator: { id: "rtmpose-m", scoreType: "simcc", minimumScore: 0.55 },
  modelManifest: Object.fromEntries(
    ["detector", "pose"].map((key) => [
      key,
      {
        sha256: "a".repeat(64),
        sourceUrl: "https://example.com/official-model",
      },
    ]),
  ),
});
const ready = () => ({
  protocolVersion: "local-pose-1",
  sessionId: "session_1234567890123456",
  token: "token_" + "a".repeat(32),
  modelInfo: metadata(),
});
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status });
const frame = (t: number) => ({
  frame: {
    t,
    width: 640,
    height: 360,
    inferenceMs: 27,
    estimator: metadata().estimator,
    landmarks: Array.from({ length: 33 }, () => ({
      x: 0.5,
      y: 0.4,
      score: 0.7,
    })),
  },
  personCount: 1,
  timing: { decodeMs: 2, detectorMs: 15, poseMs: 12 },
  nativeKeypoints: [],
  detectorObservations: [],
});
const recognizerInfo = () => ({
  protocolVersion: "shadowbox-recognition-v1",
  recognizerId: "personal-hybrid-v1",
  fingerprint: "b".repeat(64),
  featureVersion: "arm-offsets-native-v3",
  checkpointSha256: "c".repeat(64),
  externalModelSha256: "d".repeat(64),
  poseModelSha256: "a".repeat(64),
  developmentOnly: true,
  scoreSemantics: "uncalibrated_model_support",
});
const recognition = () => ({
  protocolVersion: "shadowbox-recognition-v1",
  recognizerId: "personal-hybrid-v1",
  fingerprint: "b".repeat(64),
  state: "active",
  events: [
    {
      id: "left-cycle-1",
      hand: "left",
      family: "hook",
      startMs: 10,
      peakMs: 30,
      endMs: 50,
      detectedAtMs: 60,
      score: 0.8,
    },
  ],
});
let close: ReturnType<typeof vi.fn>;
beforeEach(() => {
  close = vi.fn();
  vi.stubGlobal(
    "createImageBitmap",
    vi.fn().mockResolvedValue({ width: 640, height: 360, close }),
  );
  vi.stubGlobal(
    "OffscreenCanvas",
    class {
      getContext() {
        return { drawImage: vi.fn() };
      }
      convertToBlob() {
        return Promise.resolve(new Blob(["jpeg"], { type: "image/jpeg" }));
      }
    },
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("optional local pose transport", () => {
  it("preserves configured physical-hand family evidence in absolute source time", async () => {
    const config = ready();
    Object.assign(config.modelInfo, { recognizer: recognizerInfo() });
    const output = frame(60);
    Object.assign(output.frame, { recognition: recognition() });
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(config))
      .mockResolvedValueOnce(response(output))
      .mockResolvedValue(response({ closed: true }));
    const client = new LocalPoseClient({ fetch: fetcher });
    await client.init();
    const result = await client.detectImage({} as ImageBitmap, 60);
    expect(result.recognition).toEqual(recognition());
    expect(result.recognition?.events[0]).toMatchObject({
      hand: "left",
      family: "hook",
      startMs: 10,
      detectedAtMs: 60,
    });
    client.dispose();
  });

  it.each(["missing", "fingerprint", "future", "duplicate", "protocol"])(
    "rejects %s recognizer evidence without falling back to geometric detections",
    async (problem) => {
      const config = ready();
      Object.assign(config.modelInfo, { recognizer: recognizerInfo() });
      const output = frame(60),
        evidence = recognition();
      if (problem === "fingerprint") evidence.fingerprint = "e".repeat(64);
      if (problem === "protocol") evidence.protocolVersion = "future-protocol";
      if (problem === "future") evidence.events[0].endMs = 61;
      if (problem === "duplicate")
        evidence.events.push({ ...evidence.events[0] });
      if (problem !== "missing")
        Object.assign(output.frame, { recognition: evidence });
      const fetcher = vi
        .fn()
        .mockResolvedValueOnce(response(config))
        .mockResolvedValueOnce(response(output))
        .mockResolvedValue(response({ closed: true }));
      const client = new LocalPoseClient({ fetch: fetcher });
      await client.init();
      await expect(client.detectImage({} as ImageBitmap, 60)).rejects.toThrow(
        /recognizer/,
      );
      client.dispose();
    },
  );

  it("rejects unsolicited temporal outputs when the session has no configured recognizer", async () => {
    const output = frame(60);
    Object.assign(output.frame, { recognition: recognition() });
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(ready()))
      .mockResolvedValueOnce(response(output))
      .mockResolvedValue(response({ closed: true }));
    const client = new LocalPoseClient({ fetch: fetcher });
    await client.init();
    await expect(client.detectImage({} as ImageBitmap, 60)).rejects.toThrow(
      "without configured weights",
    );
    client.dispose();
  });

  it.each(["featureVersion", "poseModelSha256"])(
    "fails initialization on incompatible recognizer %s",
    async (key) => {
      const info = recognizerInfo();
      Object.assign(info, {
        [key]:
          key === "featureVersion" ? "incompatible-features" : "f".repeat(64),
      });
      const config = ready();
      Object.assign(config.modelInfo, { recognizer: info });
      const fetcher = vi
        .fn()
        .mockResolvedValueOnce(response(config))
        .mockResolvedValue(response({ closed: true }));
      const client = new LocalPoseClient({ fetch: fetcher });
      await expect(client.init()).rejects.toThrow("incompatible");
      expect(fetcher.mock.calls[1][1].method).toBe("DELETE");
    },
  );

  it("rejects remote and different-port destinations before making a request", () => {
    for (const baseUrl of [
      "https://remote.example",
      "http://127.0.0.1:8765",
      "http://localhost:5173",
    ]) {
      expect(() => new LocalPoseClient({ baseUrl })).toThrow("same local");
    }
  });

  it("deduplicates initialization and preserves native scores, source time and cleanup", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(ready()))
      .mockResolvedValueOnce(response(frame(12.5)))
      .mockResolvedValue(response({ closed: true }));
    const client = new LocalPoseClient({ fetch: fetcher });
    await Promise.all([client.init(), client.init()]);
    expect(fetcher).toHaveBeenCalledTimes(1);
    const result = await client.detectImage({} as ImageBitmap, 12.5);
    expect(result.t).toBe(12.5);
    expect(result.landmarks[15]).toEqual({ x: 0.5, y: 0.4, score: 0.7 });
    expect(result.estimator?.minimumScore).toBe(0.55);
    expect(fetcher.mock.calls[1][1]).toMatchObject({
      credentials: "omit",
      mode: "same-origin",
      redirect: "error",
      headers: { "Content-Type": "image/jpeg", "X-Frame-Time-Ms": "12.5" },
    });
    expect(fetcher.mock.calls[1][1].body).toBeInstanceOf(Blob);
    await expect(client.detectImage({} as ImageBitmap, 12.5)).rejects.toThrow(
      "increase",
    );
    client.dispose();
    expect(fetcher.mock.calls[2][1].method).toBe("DELETE");
    expect(client.modelInfo).toBeNull();
    expect(close).toHaveBeenCalledOnce();
  });

  it("keeps odd laptop aspect ratios below the server pixel-allocation limit", async () => {
    vi.mocked(createImageBitmap).mockResolvedValueOnce({
      width: 3024,
      height: 1964,
      close,
    } as unknown as ImageBitmap);
    const output = frame(1);
    output.frame.width = 1191;
    output.frame.height = 773;
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(ready()))
      .mockResolvedValueOnce(response(output))
      .mockResolvedValue(response({ closed: true }));
    const client = new LocalPoseClient({ fetch: fetcher });
    await client.init();
    const result = await client.detectImage({} as ImageBitmap, 1);
    expect(result.width * result.height).toBeLessThanOrEqual(921_600);
    expect(result.width / result.height).toBeCloseTo(3024 / 1964, 2);
    client.dispose();
  });

  it("keeps only one frame in flight and aborts it on disposal", async () => {
    let entered!: () => void;
    const started = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(ready()))
      .mockImplementation((_url, init) => {
        if (init.method === "DELETE")
          return Promise.resolve(response({ closed: true }));
        entered();
        return new Promise((_resolve, reject) =>
          init.signal.addEventListener("abort", () =>
            reject(new Error("aborted")),
          ),
        );
      });
    const client = new LocalPoseClient({ fetch: fetcher });
    await client.init();
    const pending = client.detectImage({} as ImageBitmap, 1);
    await started;
    await expect(
      client.detectImage({} as ImageBitmap, 2),
    ).rejects.toBeInstanceOf(LocalPoseBusyError);
    client.dispose();
    await expect(pending).rejects.toThrow("disposed");
    expect(close).toHaveBeenCalledOnce();
  });

  it("server backpressure skips a frame without advancing its source clock", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(ready()))
      .mockResolvedValueOnce(response({ error: "busy" }, 429))
      .mockResolvedValueOnce(response(frame(10)))
      .mockResolvedValue(response({ closed: true }));
    const client = new LocalPoseClient({ fetch: fetcher });
    await client.init();
    await expect(
      client.detectImage({} as ImageBitmap, 10),
    ).rejects.toBeInstanceOf(LocalPoseBusyError);
    expect((await client.detectImage({} as ImageBitmap, 10)).t).toBe(10);
    client.dispose();
  });

  it.each([0, 1, -1, 1.1])(
    "rejects invalid native-score policy %s and closes that session",
    async (minimumScore) => {
      const body = ready();
      body.modelInfo.estimator.minimumScore = minimumScore;
      const fetcher = vi
        .fn()
        .mockResolvedValueOnce(response(body))
        .mockResolvedValue(response({ closed: true }));
      const client = new LocalPoseClient({ fetch: fetcher });
      await expect(client.init()).rejects.toThrow("provenance");
      expect(fetcher.mock.calls[1][1].method).toBe("DELETE");
    },
  );

  it.each(["time", "estimator", "visibility"])(
    "rejects mutated %s rather than accepting mislabeled coordinates",
    async (mutation) => {
      const body = frame(10);
      if (mutation === "time") body.frame.t = 11;
      if (mutation === "estimator") body.frame.estimator.id = "rtmw-l";
      if (mutation === "visibility")
        Object.assign(body.frame.landmarks[15], { visibility: 0.99 });
      const fetcher = vi
        .fn()
        .mockResolvedValueOnce(response(ready()))
        .mockResolvedValueOnce(response(body))
        .mockResolvedValue(response({ closed: true }));
      const client = new LocalPoseClient({ fetch: fetcher });
      await client.init();
      await expect(client.detectImage({} as ImageBitmap, 10)).rejects.toThrow();
      client.dispose();
    },
  );

  it("closes a late initialization response even if the transport ignores abort", async () => {
    let resolve!: (value: Response) => void;
    const fetcher = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<Response>((done) => {
            resolve = done;
          }),
      )
      .mockResolvedValue(response({ closed: true }));
    const client = new LocalPoseClient({ fetch: fetcher });
    const pending = client.init();
    client.dispose();
    resolve(response(ready()));
    await expect(pending).rejects.toThrow("stopped");
    expect(fetcher.mock.calls[1][1].method).toBe("DELETE");
  });

  it("aborts a stalled local request within its configured deadline", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn(
      (_url, init) =>
        new Promise<Response>((_resolve, reject) =>
          init?.signal?.addEventListener("abort", () =>
            reject(new Error("aborted")),
          ),
        ),
    );
    const client = new LocalPoseClient({ fetch: fetcher, timeoutMs: 100 });
    const check = expect(client.init()).rejects.toThrow("timed out");
    await vi.advanceTimersByTimeAsync(100);
    await check;
  });
});
