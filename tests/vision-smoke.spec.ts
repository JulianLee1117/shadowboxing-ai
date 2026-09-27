import { expect, test } from "@playwright/test";
import type { Download, Page } from "@playwright/test";
import type { Session } from "../src/lib/types";

const appUrl = process.env.PLAYWRIGHT_BASE_URL ?? "http://127.0.0.1:5173";

// These tests use a synthetic blank canvas video. They never request a camera,
// and prove runtime plumbing only, not human pose or boxing accuracy.
test.describe("actual local MediaPipe runtime", () => {
  test.setTimeout(90_000);

  test("initializes the full model and detects unmirrored synthetic video frames", async ({
    page,
  }) => {
    const requests: string[] = [];
    page.on("request", (request) => requests.push(request.url()));
    await page.goto(appUrl);

    const result = await detectSyntheticFrames(page);

    expect(["GPU", "CPU"]).toContain(result.delegate);
    expect(result.frames).toHaveLength(2);
    for (const [index, frame] of result.frames.entries()) {
      expect(frame.t).toBe(index * (1000 / 30));
      expect(frame.width).toBe(640);
      expect(frame.height).toBe(480);
      expect(Array.isArray(frame.landmarks)).toBe(true);
      expect(Number.isFinite(frame.inferenceMs)).toBe(true);
      expect(frame.inferenceMs).toBeGreaterThanOrEqual(0);
    }
    expect(result.disposedError).toMatch(/disposed/i);
    expect(
      requests.some(
        (url) => new URL(url).pathname === "/models/pose_landmarker_full.task",
      ),
    ).toBe(true);
    expect(
      requests.some(
        (url) =>
          new URL(url).pathname === "/wasm/vision_wasm_module_internal.wasm",
      ),
    ).toBe(true);
    expect(externalRequests(requests)).toEqual([]);
  });

  test("restarts in a fresh CPU worker after an injected GPU initialization failure", async ({
    page,
  }) => {
    const requests: string[] = [];
    page.on("request", (request) => requests.push(request.url()));
    await forceCpuFallback(page);
    await page.goto(appUrl);

    const result = await detectSyntheticFrames(page);

    expect(result.delegate).toBe("CPU");
    expect(
      result.frames.every((frame) => Number.isFinite(frame.inferenceMs)),
    ).toBe(true);
    expect(
      await page.evaluate(
        () =>
          (window as Window & { forcedGpuFailures?: number }).forcedGpuFailures,
      ),
    ).toBe(1);
    expect(externalRequests(requests)).toEqual([]);
  });

  test("rejected imported playback stops cleanly and preserves the original video", async ({
    page,
  }) => {
    await forceCpuFallback(page);
    await page.goto(appUrl);
    const recording = await recordSyntheticVideo(page);
    const original = Buffer.from(recording.bytes);
    await page.locator('input[type="file"]').setInputFiles({
      name: "rejected-playback.webm",
      mimeType: recording.mimeType,
      buffer: original,
    });
    const analyze = page.getByRole("button", {
      name: "Analyze clip",
      exact: true,
    });
    await expect(analyze).toBeEnabled({ timeout: 45_000 });
    const capture = page.locator(".camera-stage video");
    await capture.evaluate((video: HTMLVideoElement) => {
      video.play = () =>
        Promise.reject(
          new DOMException("Injected playback rejection.", "NotAllowedError"),
        );
    });
    await analyze.click();
    await expect(page.getByRole("alert")).toContainText(
      "Could not play this video format",
    );
    await expect(
      page.getByRole("heading", { name: "Review", exact: true }),
    ).toBeVisible();
    await expect(capture).toHaveJSProperty("srcObject", null);
    await expect(capture).not.toHaveAttribute("src", /.+/);
    await expect(
      page.getByRole("button", { name: "Stop & save", exact: true }),
    ).toHaveCount(0);
    await expect
      .poll(async () => (await savedSessionSummaries(page)).length)
      .toBe(1);
    const [saved] = await savedSessionSummaries(page);
    expect(saved.source).toBe("file");
    expect(saved.videoBytes).toBe(original.byteLength);
    expect(saved.times).toEqual([]);
    await page.getByRole("button", { name: "Practice", exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Enable camera", exact: true }),
    ).toBeVisible();
    await expect(analyze).toHaveCount(0);
  });

  test("analyzes an uploaded synthetic video twice and preserves the original video export", async ({
    page,
  }) => {
    const requests: string[] = [];
    page.on("request", (request) => requests.push(request.url()));
    // Use actual CPU inference for repeatable lifecycle coverage on CI hosts
    // with software GPUs; the first smoke test still covers default selection.
    await forceCpuFallback(page);
    await page.goto(appUrl);
    const recording = await recordSyntheticVideo(page);
    const original = Buffer.from(recording.bytes);
    expect(original.byteLength).toBeGreaterThan(1000);

    for (let completed = 1; completed <= 2; completed++) {
      if (completed > 1)
        await page
          .getByRole("button", { name: "Practice", exact: true })
          .click();
      // Each completed round releases its source. Import the same original
      // again to exercise a fresh worker and a fresh media-time origin.
      // This canvas fixture never requests getUserMedia or camera permission.
      await page.locator('input[type="file"]').setInputFiles({
        name: "synthetic-motion.webm",
        mimeType: recording.mimeType,
        buffer: original,
      });
      const analyze = page.getByRole("button", {
        name: "Analyze clip",
        exact: true,
      });
      await expect(analyze).toBeEnabled({ timeout: 45_000 });
      await expect(
        page.getByText("Imported clip stays local", { exact: false }),
      ).toBeVisible();
      await analyze.click();
      await expect(
        page.getByRole("button", { name: "Stop & save", exact: true }),
      ).toBeVisible({ timeout: 45_000 });
      // Completion and Review must happen at the clip's end without another
      // click, rather than waiting for the 30-second camera-round timer.
      await expect(
        page.getByRole("heading", { name: "Review", exact: true }),
      ).toBeVisible({ timeout: 15_000 });
      await expect
        .poll(async () => (await savedSessionSummaries(page)).length)
        .toBe(completed);
      await expect(page.locator(".camera-stage video")).toHaveJSProperty(
        "srcObject",
        null,
      );
      await expect(page.locator(".camera-stage video")).not.toHaveAttribute(
        "src",
        /.+/,
      );
    }

    const saved = await savedSessionSummaries(page);
    await test.info().attach("uploaded-session-summaries", {
      body: JSON.stringify(saved, null, 2),
      contentType: "application/json",
    });
    expect(saved).toHaveLength(2);
    expect(saved[0].id).not.toBe(saved[1].id);
    for (const session of saved) {
      expect(session.source).toBe("file");
      expect(session.capture?.delegate).toBe("CPU");
      expect(session.videoBytes).toBe(original.byteLength);
      expect(session.durationMs).toBeGreaterThan(1500);
      expect(session.durationMs).toBeLessThan(4000);
      expect(session.videoOffsetMs).toBe(0);
      expect(
        session.times.length,
        `Saved ${session.id} without enough analyzed frames; skipped ${session.skippedFrames}.`,
      ).toBeGreaterThan(5);
      expect(session.times[0]).toBeLessThan(1000);
      expect(
        session.times.every(
          (t, i) =>
            Number.isFinite(t) &&
            t >= 0 &&
            t <= session.durationMs + 1 &&
            (i === 0 || t > session.times[i - 1]),
        ),
      ).toBe(true);
      expect(session.finiteTimings).toBe(true);
    }

    await expect(page.getByRole("heading", { name: "Review" })).toBeVisible();
    const replay = page.locator(".replay-stage video");
    await expect(replay).toBeVisible();
    await expect(replay).toHaveAttribute("src", /^blob:/);
    await expect
      .poll(() =>
        replay.evaluate((video: HTMLVideoElement) => video.readyState),
      )
      .toBeGreaterThanOrEqual(1);

    await page.getByText("Export & details", { exact: true }).click();
    const evidenceDownload = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Evidence JSON", exact: true })
      .click();
    const evidence = JSON.parse(
      (await downloadBytes(await evidenceDownload)).toString("utf8"),
    ) as Session;
    expect(evidence.id).toBe(saved[0].id);
    expect(evidence.frames.length).toBe(saved[0].times.length);
    expect(evidence.video).toBeUndefined(); // Blob stays in the separate video export.

    const videoDownload = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export video", exact: true })
      .click();
    expect((await downloadBytes(await videoDownload)).equals(original)).toBe(
      true,
    );

    page.once("dialog", (dialog) => dialog.accept());
    await page
      .getByRole("button", { name: "Delete this round", exact: true })
      .click();
    await expect
      .poll(async () => (await savedSessionSummaries(page)).length)
      .toBe(1);
    await expect(page.locator(".replay-stage video")).toHaveCount(0);
    expect(externalRequests(requests)).toEqual([]);
  });
});

async function forceCpuFallback(page: Page) {
  await page.addInitScript(() => {
    const OriginalWorker = window.Worker;
    const diagnosticWindow = window as Window & {
      forcedGpuFailures?: number;
    };
    diagnosticWindow.forcedGpuFailures = 0;
    window.Worker = class extends OriginalWorker {
      override postMessage(
        message: unknown,
        transferOrOptions?: Transferable[] | StructuredSerializeOptions,
      ): void {
        const request = message as {
          type?: string;
          delegate?: string;
          id?: number;
        };
        if (request?.type === "init" && request.delegate === "GPU") {
          diagnosticWindow.forcedGpuFailures =
            (diagnosticWindow.forcedGpuFailures ?? 0) + 1;
          // Exercise the runner's failure/restart path. The CPU model itself
          // still loads real local weights and performs actual inference.
          queueMicrotask(() =>
            this.dispatchEvent(
              new MessageEvent("message", {
                data: {
                  type: "error",
                  id: request.id,
                  error: "Injected unavailable GPU for fallback test.",
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
  });
}

async function detectSyntheticFrames(page: Page) {
  return page.evaluate(async () => {
    const modulePath = "/src/lib/vision.ts";
    const { VisionRunner } = await import(modulePath);
    const runner = new VisionRunner("full");
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Synthetic video canvas is unavailable.");
    context.fillStyle = "#202026";
    context.fillRect(0, 0, canvas.width, canvas.height);
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    const stream = canvas.captureStream(30);
    video.srcObject = stream;
    document.body.appendChild(video);
    try {
      await video.play();
      await runner.init();
      const delegate: string = runner.delegate;
      const frames: Array<{
        t: number;
        width: number;
        height: number;
        landmarks: unknown[];
        inferenceMs: number;
      }> = [];
      frames.push(await runner.detect(video, 0));
      frames.push(await runner.detect(video, 1000 / 30));
      runner.dispose();
      let disposedError = "";
      try {
        await runner.detect(video, 100);
      } catch (error) {
        disposedError = error instanceof Error ? error.message : String(error);
      }
      return { delegate, frames, disposedError };
    } finally {
      runner.dispose();
      stream.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
      video.remove();
    }
  });
}

async function recordSyntheticVideo(page: Page) {
  return page.evaluate(async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Synthetic recording canvas is unavailable.");
    const stream = canvas.captureStream(30);
    const mimeType = ["video/webm;codecs=vp8", "video/webm"].find((type) =>
      MediaRecorder.isTypeSupported(type),
    );
    if (!mimeType)
      throw new Error("This test browser cannot record a WebM fixture.");
    const recorder = new MediaRecorder(stream, { mimeType });
    const chunks: Blob[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    let frame = 0;
    const draw = () => {
      context.fillStyle = "#202026";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "#90baad";
      context.fillRect(40 + ((frame++ * 7) % 480), 180, 80, 80);
    };
    draw();
    const timer = window.setInterval(draw, 1000 / 30);
    try {
      recorder.start(200);
      await new Promise((resolve) => window.setTimeout(resolve, 2400));
      await new Promise<void>((resolve, reject) => {
        recorder.onstop = () => resolve();
        recorder.onerror = () =>
          reject(new Error("Synthetic fixture recording failed."));
        recorder.stop();
      });
      const blob = new Blob(chunks, { type: recorder.mimeType });
      return {
        mimeType: blob.type,
        bytes: Array.from(new Uint8Array(await blob.arrayBuffer())),
      };
    } finally {
      clearInterval(timer);
      if (recorder.state !== "inactive") recorder.stop();
      stream.getTracks().forEach((track) => track.stop());
    }
  });
}

async function savedSessionSummaries(page: Page) {
  return page.evaluate(async () => {
    // Read persisted browser evidence directly so this UI lifecycle test also
    // works against compiled production assets without a Vite source endpoint.
    const sessions = await new Promise<Session[]>((resolve, reject) => {
      const open = indexedDB.open("corner-local-v1", 2);
      open.onerror = () =>
        reject(new Error("Could not read the session database."));
      open.onblocked = () =>
        reject(new Error("The session database is blocked."));
      open.onupgradeneeded = () => {
        open.transaction?.abort();
        reject(
          new Error(
            "The application has not initialized its session database.",
          ),
        );
      };
      open.onsuccess = () => {
        const db = open.result;
        if (!db.objectStoreNames.contains("sessions")) {
          db.close();
          reject(new Error("The session object store is missing."));
          return;
        }
        const transaction = db.transaction("sessions", "readonly");
        const request = transaction.objectStore("sessions").getAll();
        transaction.oncomplete = () => {
          const records = request.result as Session[];
          db.close();
          resolve(
            records.sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
          );
        };
        transaction.onerror = transaction.onabort = () => {
          db.close();
          reject(new Error("Could not finish reading saved sessions."));
        };
      };
    });
    return sessions.map((session) => ({
      id: session.id,
      source: session.source,
      durationMs: session.durationMs,
      videoOffsetMs: session.videoOffsetMs,
      videoBytes: session.video?.size ?? 0,
      skippedFrames: session.skippedFrames,
      capture: session.capture,
      times: session.frames.map((frame) => frame.t),
      finiteTimings: session.frames.every(
        (frame) =>
          Number.isFinite(frame.inferenceMs) &&
          Number.isFinite(frame.frameAgeMs),
      ),
    }));
  });
}

async function downloadBytes(download: Download): Promise<Buffer> {
  const stream = await download.createReadStream();
  if (!stream) throw new Error("Exported download has no readable data.");
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks);
}

function externalRequests(requests: string[]): string[] {
  const appOrigin = new URL(appUrl).origin;
  return requests.filter(
    (url) => /^https?:/.test(url) && new URL(url).origin !== appOrigin,
  );
}
