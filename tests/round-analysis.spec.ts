import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

test.describe("local saved-video analysis service", () => {
  test.setTimeout(90_000);
  test("analyzes generated video sequentially with honest timestamps and releases every resource", async ({
    page,
  }) => {
    const external = await setup(page);
    const result = await page.evaluate(async () => {
      const modulePath = "/src/lib/roundAnalysis.ts";
      const { analyzeRound } = await import(modulePath);
      const fixture = (
        window as unknown as { analysisFixture: () => Promise<unknown> }
      ).analysisFixture;
      const session = await fixture();
      const original = JSON.stringify(session);
      const progress: unknown[] = [];
      const report = await analyzeRound(session, {
        onProgress: (p: unknown) => progress.push(p),
      });
      const diagnostics = (
        window as unknown as {
          analysisDiagnostics: {
            camera: number;
            workers: number;
            terminated: number;
            urls: number;
          };
        }
      ).analysisDiagnostics;
      return {
        report,
        progress,
        originalUnchanged: original === JSON.stringify(session),
        diagnostics,
        remainingVideos: document.querySelectorAll("video[data-round-analysis]")
          .length,
      };
    });
    expect(result.originalUnchanged).toBe(true);
    expect(result.report.reportType).toBe("round-video-analysis");
    expect(result.report.sourceSessionId).toBe("generated-source");
    expect(result.report.sourceFingerprint).toMatch(/^[a-f0-9]{64}$/);
    expect(result.report.model).toBe("full");
    expect(result.report.delegate).toBe("CPU");
    expect(result.report.frames.length).toBeGreaterThanOrEqual(4);
    expect(
      result.report.frames.every(
        (f: { t: number }, i: number, all: { t: number }[]) =>
          f.t >= 0 && f.t <= 600 && (i === 0 || f.t > all[i - 1].t),
      ),
    ).toBe(true);
    expect(result.report.provenance.videoOffsetMs).toBe(100);
    expect(result.report.provenance.interpolation).toBe(false);
    expect(
      result.report.cadence.decodedTimestampFrames +
        result.report.cadence.estimatedTimestampFrames,
    ).toBe(result.report.frames.length);
    expect(result.report.cadence.decodedTimestampFrames).toBe(
      result.report.frames.length,
    );
    expect(result.report.cadence.estimatedTimestampFrames).toBe(0);
    expect(result.report.provenance.timestampMode).toBe("decoded-media-time");
    expect(result.report.provenance.decoder).toBe(
      "mediabunny-1.60.0-webcodecs",
    );
    expect(result.report.events).toEqual([]);
    expect(result.report.uncertaintyIntervals.length).toBeGreaterThan(0);
    expect(result.diagnostics.camera).toBe(0);
    expect(result.diagnostics.workers).toBe(result.diagnostics.terminated);
    expect(result.diagnostics.urls).toBe(0);
    expect(result.remainingVideos).toBe(0);
    expect(external).toEqual([]);
  });

  test("aborts during analysis without mutating original evidence or keeping workers/video alive", async ({
    page,
  }) => {
    const external = await setup(page);
    const result = await page.evaluate(async () => {
      const modulePath = "/src/lib/roundAnalysis.ts";
      const { analyzeRound } = await import(modulePath);
      const w = window as unknown as {
        analysisFixture: () => Promise<unknown>;
        analysisDiagnostics: {
          camera: number;
          workers: number;
          terminated: number;
          urls: number;
        };
      };
      const session = await w.analysisFixture();
      const original = JSON.stringify(session);
      const controller = new AbortController();
      let name = "";
      let analyzed = 0;
      try {
        await analyzeRound(session, {
          signal: controller.signal,
          onProgress: (p: { phase: string; completedFrames: number }) => {
            if (p.phase === "analyzing" && p.completedFrames > 0) {
              analyzed = p.completedFrames;
              controller.abort();
            }
          },
        });
      } catch (error) {
        name = (error as Error).name;
      }
      return {
        name,
        analyzed,
        originalUnchanged: original === JSON.stringify(session),
        diagnostics: w.analysisDiagnostics,
        remainingVideos: document.querySelectorAll("video[data-round-analysis]")
          .length,
      };
    });
    expect(result.name).toBe("AbortError");
    expect(result.analyzed).toBe(1);
    expect(result.originalUnchanged).toBe(true);
    expect(result.diagnostics.camera).toBe(0);
    expect(result.diagnostics.workers).toBe(result.diagnostics.terminated);
    expect(result.diagnostics.urls).toBe(0);
    expect(result.remainingVideos).toBe(0);
    expect(external).toEqual([]);
  });
  test("cancellation from the final progress callback cannot publish a completed report", async ({
    page,
  }) => {
    await setup(page);
    const result = await page.evaluate(async () => {
      const modulePath = "/src/lib/roundAnalysis.ts",
        decoderPath = "/src/lib/videoFrames.ts";
      const { analyzeRound } = await import(modulePath);
      const { openVideoFrames } = await import(decoderPath);
      const w = window as unknown as {
        analysisFixture: () => Promise<{ video: Blob }>;
        analysisDiagnostics: {
          workers: number;
          terminated: number;
          urls: number;
        };
      };
      const session = await w.analysisFixture();
      const decoded = await openVideoFrames(session.video, {
        signal: new AbortController().signal,
        offsetMs: 100,
        durationMs: 600,
        maxPixels: 1000000,
      });
      let expected = 0;
      try {
        for await (const image of decoded.frames) {
          void image;
          expected++;
        }
      } finally {
        decoded.dispose();
      }
      const controller = new AbortController();
      let name = "",
        processed = 0;
      try {
        await analyzeRound(session, {
          signal: controller.signal,
          onProgress: (p: { completedFrames: number }) => {
            processed = p.completedFrames;
            if (processed === expected) controller.abort();
          },
        });
      } catch (error) {
        name = (error as Error).name;
      }
      return { name, processed, expected, diagnostics: w.analysisDiagnostics };
    });
    expect(result.expected).toBeGreaterThan(0);
    expect(result.processed).toBe(result.expected);
    expect(result.name).toBe("AbortError");
    expect(result.diagnostics.workers).toBe(result.diagnostics.terminated);
    expect(result.diagnostics.urls).toBe(0);
  });

  test("includes the final MediaRecorder frame when its encoded duration is missing", async ({
    page,
  }) => {
    await setup(page);
    const report = await page.evaluate(async () => {
      const modulePath = "/src/lib/roundAnalysis.ts";
      const { analyzeRound } = await import(modulePath);
      const fixture = (
        window as unknown as { analysisFixture: () => Promise<object> }
      ).analysisFixture;
      return analyzeRound({
        ...(await fixture()),
        videoOffsetMs: 0,
        durationMs: 1500,
      });
    });
    // Chromium MediaRecorder omits the final packet duration: demuxed duration
    // equals that last frame's PTS. It must still be decoded when in the round.
    expect(report.frames.at(-1).t).toBeCloseTo(
      report.provenance.decodedDurationMs,
      5,
    );
    expect(report.cadence.estimatedTimestampFrames).toBe(0);
    expect(report.completeness.status).toBe("partial");
    expect(report.completeness.reason).toContain("video ends before");
  });

  test("missing WebCodecs fails explicitly instead of substituting guessed seek timestamps", async ({
    page,
  }) => {
    const external = await setup(page);
    const result = await page.evaluate(async () => {
      const modulePath = "/src/lib/roundAnalysis.ts";
      const { analyzeRound } = await import(modulePath);
      Object.defineProperty(window, "VideoDecoder", {
        configurable: true,
        value: undefined,
      });
      const w = window as unknown as {
        analysisFixture: () => Promise<unknown>;
        analysisDiagnostics: {
          workers: number;
          terminated: number;
          urls: number;
        };
      };
      let message = "";
      try {
        await analyzeRound(await w.analysisFixture());
      } catch (error) {
        message = (error as Error).message;
      }
      return { message, diagnostics: w.analysisDiagnostics };
    });
    expect(result.message).toContain("WebCodecs");
    expect(result.diagnostics.workers).toBe(result.diagnostics.terminated);
    expect(result.diagnostics.urls).toBe(0);
    expect(external).toEqual([]);
  });

  test("a decode failure cleans up and leaves the saved session unchanged", async ({
    page,
  }) => {
    const external = await setup(page);
    const result = await page.evaluate(async () => {
      const modulePath = "/src/lib/roundAnalysis.ts";
      const { analyzeRound } = await import(modulePath);
      const w = window as unknown as {
        analysisFixture: () => Promise<object>;
        analysisDiagnostics: {
          camera: number;
          workers: number;
          terminated: number;
          urls: number;
        };
      };
      const original = await w.analysisFixture();
      const session = {
        ...original,
        video: new Blob(["invalid-video"], { type: "video/webm" }),
      };
      const snapshot = JSON.stringify(session);
      let message = "";
      try {
        await analyzeRound(session);
      } catch (error) {
        message = (error as Error).message;
      }
      return {
        message,
        unchanged: snapshot === JSON.stringify(session),
        diagnostics: w.analysisDiagnostics,
        remainingVideos: document.querySelectorAll("video[data-round-analysis]")
          .length,
      };
    });
    expect(result.message).toMatch(/video|format|parse|input/i);
    expect(result.unchanged).toBe(true);
    expect(result.diagnostics.camera).toBe(0);
    expect(result.diagnostics.workers).toBe(result.diagnostics.terminated);
    expect(result.diagnostics.urls).toBe(0);
    expect(result.remainingVideos).toBe(0);
    expect(external).toEqual([]);
  });
});

async function setup(page: Page) {
  const external: string[] = [];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== "http://127.0.0.1:5173") {
      external.push(url.href);
      return route.abort();
    }
    if (url.pathname === "/__round_analysis_test")
      return route.fulfill({
        contentType: "text/html",
        body: "<!doctype html><title>Isolated local round analysis test</title>",
      });
    return route.continue();
  });
  await page.addInitScript(() => {
    const w = window as unknown as {
      analysisDiagnostics: {
        camera: number;
        workers: number;
        terminated: number;
        urls: number;
      };
      analysisFixture: () => Promise<unknown>;
    };
    const d = (w.analysisDiagnostics = {
      camera: 0,
      workers: 0,
      terminated: 0,
      urls: 0,
    });
    navigator.mediaDevices.getUserMedia = () => {
      d.camera++;
      return Promise.reject(
        new Error("Physical camera must not be requested."),
      );
    };
    const create = URL.createObjectURL.bind(URL),
      revoke = URL.revokeObjectURL.bind(URL);
    const mediaUrls = new Set<string>();
    URL.createObjectURL = (blob) => {
      const url = create(blob);
      // Mediabunny caches a static JavaScript worker URL for later clips.
      // The privacy/lifecycle invariant here is no retained media URL.
      if (!(blob instanceof Blob) || !blob.type.includes("javascript"))
        mediaUrls.add(url);
      d.urls = mediaUrls.size;
      return url;
    };
    URL.revokeObjectURL = (url) => {
      mediaUrls.delete(url);
      d.urls = mediaUrls.size;
      revoke(url);
    };
    const OriginalWorker = Worker;
    window.Worker = class extends OriginalWorker {
      ended = false;
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        d.workers++;
      }
      override terminate() {
        if (!this.ended) {
          d.terminated++;
          this.ended = true;
        }
        super.terminate();
      }
      override postMessage(
        message: unknown,
        transferOrOptions?: Transferable[] | StructuredSerializeOptions,
      ) {
        const request = message as {
          type?: string;
          delegate?: string;
          id?: number;
        };
        if (request.type === "init" && request.delegate === "GPU") {
          queueMicrotask(() =>
            this.dispatchEvent(
              new MessageEvent("message", {
                data: {
                  type: "error",
                  id: request.id,
                  error:
                    "Injected GPU fallback for deterministic service test.",
                },
              }),
            ),
          );
          return;
        }
        if (Array.isArray(transferOrOptions))
          super.postMessage(message, transferOrOptions);
        else super.postMessage(message, transferOrOptions);
      }
    };
    w.analysisFixture = async () => {
      const canvas = document.createElement("canvas");
      canvas.width = 320;
      canvas.height = 240;
      const ctx = canvas.getContext("2d")!;
      const stream = canvas.captureStream(20);
      const recorder = new MediaRecorder(stream, {
        mimeType: "video/webm;codecs=vp8",
      });
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      let x = 0;
      const draw = () => {
        ctx.fillStyle = "#202026";
        ctx.fillRect(0, 0, 320, 240);
        ctx.fillStyle = "#84beab";
        ctx.fillRect(x++ % 200, 70, 40, 40);
      };
      draw();
      const timer = setInterval(draw, 50);
      try {
        recorder.start(100);
        await new Promise((r) => setTimeout(r, 1100));
        await new Promise<void>((r) => {
          recorder.onstop = () => r();
          recorder.stop();
        });
      } finally {
        clearInterval(timer);
        if (recorder.state !== "inactive") recorder.stop();
        stream.getTracks().forEach((t) => t.stop());
      }
      return Object.freeze({
        id: "generated-source",
        createdAt: "2026-01-01T00:00:00Z",
        source: "camera",
        stance: "orthodox",
        model: "full",
        drill: "Free practice",
        durationMs: 600,
        frames: Object.freeze([]),
        events: Object.freeze([]),
        annotations: Object.freeze([]),
        video: new Blob(chunks, { type: recorder.mimeType }),
        videoOffsetMs: 100,
        measuredFps: 10,
        inferenceP95: 0,
        skippedFrames: 0,
        schemaVersion: "1.0",
        capture: {
          width: 320,
          height: 240,
          deliveredFps: 10,
          timingSource: "fixture",
          delegate: "CPU",
        },
      });
    };
  });
  await page.goto("/__round_analysis_test");
  return external;
}
