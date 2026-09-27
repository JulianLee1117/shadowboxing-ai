import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import type { Session } from "../src/lib/types";
import { demoFrame } from "../src/lib/demo";
import { DETECTOR_VERSION } from "../src/lib/motion";
import type { DetectorRecheckReport } from "../src/lib/recheck";

const APP_ORIGIN = "http://127.0.0.1:5173";

async function denyPhysicalCamera(page: Page) {
  await page.addInitScript(() => {
    const state = window as unknown as {
      cameraRequests: MediaStreamConstraints[];
    };
    state.cameraRequests = [];
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      configurable: true,
      value: async (constraints: MediaStreamConstraints) => {
        state.cameraRequests.push(constraints);
        throw new DOMException(
          "Denied by the privacy test; no physical camera was requested.",
          "NotAllowedError",
        );
      },
    });
  });
}

async function readExport(page: Page): Promise<Session> {
  const button = page.getByRole("button", {
    name: "Evidence JSON",
    exact: true,
  });
  if (!(await button.isVisible()))
    await page.getByText("Export & details", { exact: true }).click();
  const pending = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Evidence JSON", exact: true })
    .click();
  const download = await pending;
  expect(download.suggestedFilename()).toMatch(/^corner-.+\.json$/);
  const path = await download.path();
  expect(path).toBeTruthy();
  return JSON.parse(await readFile(path!, "utf8")) as Session;
}

test("initial page acquires no camera and makes no external network requests", async ({
  page,
}) => {
  await denyPhysicalCamera(page);
  await page.setViewportSize({ width: 1512, height: 823 });
  const externalRequests: string[] = [];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (
      (url.protocol === "http:" || url.protocol === "https:") &&
      url.origin !== APP_ORIGIN
    ) {
      externalRequests.push(url.href);
      return route.abort();
    }
    return route.continue();
  });
  await page.goto(APP_ORIGIN);
  await expect(
    page.getByRole("heading", { name: "Practice", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Enable camera", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Record round", exact: true }),
  ).toHaveCount(0);
  await page.waitForLoadState("networkidle");
  expect(
    await page.evaluate(
      () =>
        (window as unknown as { cameraRequests: unknown[] }).cameraRequests
          .length,
    ),
  ).toBe(0);
  expect(externalRequests).toEqual([]);
  await page.screenshot({
    path: "artifacts/ui/practice-desktop.png",
    fullPage: true,
  });
});

test("camera denial is explained and leaves the camera off", async ({
  page,
}) => {
  await denyPhysicalCamera(page);
  await page.goto(APP_ORIGIN);
  await page
    .getByRole("button", { name: "Enable camera", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText(
    "Camera access was denied",
  );
  await expect(
    page.getByRole("button", { name: "Enable camera", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Record round", exact: true }),
  ).toHaveCount(0);
  const requests = await page.evaluate(
    () =>
      (window as unknown as { cameraRequests: MediaStreamConstraints[] })
        .cameraRequests,
  );
  expect(requests).toHaveLength(1);
  expect(requests[0].audio).toBe(false);
});

test("demo round opens review, survives export and reload, and permits annotation edits and deletion", async ({
  page,
}) => {
  await denyPhysicalCamera(page);
  // Automatic Stop & save -> Review must preserve the completed round's timestamps.
  await page.goto(APP_ORIGIN);
  await page.getByText("More options", { exact: true }).click();
  await page.getByRole("button", { name: "Try demo", exact: true }).click();
  await page.getByRole("button", { name: "Start demo", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Stop & save", exact: true }),
  ).toBeVisible();
  // Two known synthetic strokes occur during this interval. This is UI coverage,
  // never a real-camera accuracy or performance benchmark.
  await page.waitForTimeout(6200);
  await page.getByRole("button", { name: "Stop & save", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Review", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".round-library .session-item")).toHaveCount(1);

  const exported = await readExport(page);
  expect(exported.schemaVersion).toBe("1.0");
  expect(exported.source).toBe("demo");
  expect(exported.model).toBe("synthetic");
  expect(exported.durationMs).toBeGreaterThan(5000);
  expect(exported.durationMs).toBeLessThan(10_000);
  expect(exported.frames.length).toBeGreaterThan(30);
  expect(exported.events.length).toBeGreaterThanOrEqual(1);
  const lastFrameTime = exported.frames.at(-1)!.t;
  expect(lastFrameTime).toBeGreaterThan(5000);
  expect(exported.durationMs).toBeGreaterThanOrEqual(lastFrameTime);
  expect(exported.durationMs - lastFrameTime).toBeLessThan(500);
  expect(
    exported.events.every(
      (event) => event.experimental && event.endMs <= exported.durationMs,
    ),
  ).toBe(true);
  expect(exported).not.toHaveProperty("video");
  expect(exported.annotations).toEqual([]);
  expect(exported.annotationsComplete).toBe(false);

  // Reload closes the in-memory instance; the round must really exist in IndexedDB.
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await expect(page.locator(".round-library .session-item")).toHaveCount(1);
  await page.locator(".round-library .session-item").click();
  const persisted = await readExport(page);
  expect(persisted.id).toBe(exported.id);
  expect(persisted.durationMs).toBe(exported.durationMs);
  expect(persisted.frames.length).toBe(exported.frames.length);

  await expect(
    page.getByLabel("Annotation action", { exact: true }),
  ).not.toBeVisible();
  await page.getByText("Label this round (optional)", { exact: true }).click();
  await page
    .getByLabel("Annotation action", { exact: true })
    .getByRole("button", { name: "other", exact: true })
    .click();
  await page
    .getByLabel("Annotation hand", { exact: true })
    .getByRole("button", { name: "unknown", exact: true })
    .click();
  await page.getByLabel("Annotation start", { exact: true }).fill("0.1");
  await page.getByLabel("Annotation end", { exact: true }).fill("0.6");
  await page
    .getByLabel("Annotation note", { exact: true })
    .fill("UI test reference interval");
  await page.getByRole("button", { name: "Add label", exact: true }).click();
  await expect(page.locator(".annotation-row")).toHaveCount(1);
  await expect(page.locator(".annotation-row")).toContainText(
    "UI test reference interval",
  );
  const labeled = await readExport(page);
  expect(labeled.annotations).toHaveLength(1);
  expect(labeled.annotations[0]).toMatchObject({
    label: "other",
    hand: "unknown",
    startMs: 100,
    endMs: 600,
  });
  expect(labeled.annotationsComplete).toBe(false);

  await page
    .getByRole("button", { name: "Delete annotation", exact: true })
    .click();
  await expect(page.locator(".annotation-row")).toHaveCount(0);
  expect((await readExport(page)).annotations).toEqual([]);
  await page.getByRole("button", { name: /^Remove round from/ }).click();
  await expect(page.locator(".round-library .session-item")).toHaveCount(0);
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await expect(page.locator(".round-library .session-item")).toHaveCount(0);
  expect(
    await page.evaluate(
      () =>
        (window as unknown as { cameraRequests: unknown[] }).cameraRequests
          .length,
    ),
  ).toBe(0);
});

test("Stop camera ends a synthetic stream even while pose initialization is pending", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const state = window as unknown as {
      syntheticCamera: MediaStream | null;
      cameraRequests: number;
    };
    state.syntheticCamera = null;
    state.cameraRequests = 0;
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      configurable: true,
      value: async () => {
        state.cameraRequests += 1;
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 360;
        const context = canvas.getContext("2d")!;
        let value = 0;
        const draw = () => {
          context.fillStyle = value++ % 2 ? "#234" : "#345";
          context.fillRect(0, 0, 640, 360);
        };
        draw();
        const synthetic = canvas.captureStream(10);
        const timer = window.setInterval(draw, 100);
        synthetic
          .getTracks()
          .forEach((track) =>
            track.addEventListener("ended", () => window.clearInterval(timer)),
          );
        state.syntheticCamera = synthetic;
        return synthetic;
      },
    });
  });
  await page.route("**/models/pose_landmarker_*.task", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    await route.abort();
  });
  await page.goto(APP_ORIGIN);
  await page
    .getByRole("button", { name: "Enable camera", exact: true })
    .click();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (
            window as unknown as { syntheticCamera: MediaStream | null }
          ).syntheticCamera?.getVideoTracks()[0]?.readyState,
      ),
    )
    .toBe("live");
  await page.getByRole("button", { name: "Stop camera", exact: true }).click();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (
            window as unknown as { syntheticCamera: MediaStream | null }
          ).syntheticCamera?.getVideoTracks()[0]?.readyState,
      ),
    )
    .toBe("ended");
  await expect(
    page.getByRole("button", { name: "Enable camera", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => (window as unknown as { cameraRequests: number }).cameraRequests,
    ),
  ).toBe(1);
});

test("camera countdown can be cancelled and records once without visible pose", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await page.addInitScript(() => {
    const state = window as unknown as {
      syntheticCamera: MediaStream | null;
      cameraRequests: MediaStreamConstraints[];
      recorderStarts: number[];
    };
    state.syntheticCamera = null;
    state.cameraRequests = [];
    state.recorderStarts = [];
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      configurable: true,
      value: async (constraints: MediaStreamConstraints) => {
        state.cameraRequests.push(constraints);
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 360;
        const context = canvas.getContext("2d")!;
        let value = 0;
        const draw = () => {
          // No person or landmark fixture: tracking must remain unassessable.
          context.fillStyle = value++ % 2 ? "#234" : "#345";
          context.fillRect(0, 0, canvas.width, canvas.height);
        };
        draw();
        const stream = canvas.captureStream(20);
        const timer = window.setInterval(draw, 50);
        stream.getTracks().forEach((track) => {
          const originalStop = track.stop.bind(track);
          track.stop = () => {
            clearInterval(timer);
            originalStop();
          };
        });
        state.syntheticCamera = stream;
        return stream;
      },
    });
    const OriginalRecorder = window.MediaRecorder;
    window.MediaRecorder = class extends OriginalRecorder {
      override start(timeslice?: number) {
        state.recorderStarts.push(performance.now());
        super.start(timeslice);
      }
    };
    // Run actual local CPU inference for predictable blank-frame diagnostics.
    // The separate runtime smoke test covers ordinary GPU selection.
    const OriginalWorker = window.Worker;
    window.Worker = class extends OriginalWorker {
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
                  error: "Use actual CPU inference in this lifecycle test.",
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
  await page.goto(APP_ORIGIN);
  await page
    .getByRole("button", { name: "Enable camera", exact: true })
    .click();
  const record = page.getByRole("button", {
    name: "Record round",
    exact: true,
  });
  await expect(record).toBeEnabled({ timeout: 45_000 });
  await expect(page.getByLabel("Arm tracking")).toContainText("L · uncertain");
  await expect(page.getByLabel("Arm tracking")).toContainText("R · uncertain");

  await record.click();
  await expect(
    page.getByText("Step back into position", { exact: true }),
  ).toBeVisible();
  await page.waitForTimeout(500);
  await page
    .getByRole("button", { name: "Cancel countdown", exact: true })
    .click();
  await expect(record).toBeEnabled();
  expect(
    await page.evaluate(
      () => (window as unknown as { recorderStarts: number[] }).recorderStarts,
    ),
  ).toEqual([]);

  const clickedAt = await page.evaluate(() => performance.now());
  await record.click();
  await expect(page.locator(".countdown-number")).toHaveText("8");
  await expect(
    page.getByRole("button", { name: "Stop & save", exact: true }),
  ).toBeVisible({ timeout: 11_000 });
  await page.waitForTimeout(1200);
  const starts = await page.evaluate(
    () => (window as unknown as { recorderStarts: number[] }).recorderStarts,
  );
  expect(starts).toHaveLength(1);
  expect(starts[0] - clickedAt).toBeGreaterThanOrEqual(7900);
  expect(starts[0] - clickedAt).toBeLessThan(11_000);
  await expect(page.getByLabel("Arm tracking")).toContainText("uncertain");
  await page.getByRole("button", { name: "Stop & save", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Review", exact: true }),
  ).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (
            window as unknown as { syntheticCamera: MediaStream }
          ).syntheticCamera.getVideoTracks()[0].readyState,
      ),
    )
    .toBe("ended");
  await expect(page.locator(".replay-stage video")).toBeVisible();
  const saved = await readExport(page);
  expect(saved.source).toBe("camera");
  expect(saved.durationMs).toBeGreaterThan(1000);
  expect(saved.events).toEqual([]);
  const requests = await page.evaluate(
    () =>
      (window as unknown as { cameraRequests: MediaStreamConstraints[] })
        .cameraRequests,
  );
  expect(requests).toHaveLength(1);
  expect(requests[0].audio).toBe(false);
});

test("concurrent starts and late camera responses cannot replace the latest source", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await page.addInitScript(() => {
    const state = window as unknown as {
      pendingCameras: Array<(stream: MediaStream) => void>;
      syntheticCameras: MediaStream[];
      resolveCamera: (index: number) => void;
    };
    state.pendingCameras = [];
    state.syntheticCameras = [];
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      configurable: true,
      value: () =>
        new Promise<MediaStream>((resolve) => {
          state.pendingCameras.push(resolve);
        }),
    });
    state.resolveCamera = (index) => {
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 360;
      const context = canvas.getContext("2d")!;
      let value = 0;
      const draw = () => {
        context.fillStyle = value++ % 2 ? "#234" : "#345";
        context.fillRect(0, 0, canvas.width, canvas.height);
      };
      draw();
      const stream = canvas.captureStream(10);
      const timer = window.setInterval(draw, 100);
      stream.getTracks().forEach((track) => {
        const originalStop = track.stop.bind(track);
        track.stop = () => {
          clearInterval(timer);
          originalStop();
        };
      });
      state.syntheticCameras[index] = stream;
      state.pendingCameras[index](stream);
    };
  });
  await page.goto(APP_ORIGIN);
  const enable = page.getByRole("button", {
    name: "Enable camera",
    exact: true,
  });
  await enable.evaluate((button: HTMLButtonElement) => {
    // Two invocations before the first start() continuation yields to React.
    button.click();
    button.click();
  });
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as unknown as { pendingCameras: unknown[] }).pendingCameras
            .length,
      ),
    )
    .toBe(1);
  await page.getByRole("button", { name: "Stop camera", exact: true }).click();
  await enable.click();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as unknown as { pendingCameras: unknown[] }).pendingCameras
            .length,
      ),
    )
    .toBe(2);

  // The second source becomes ready while the first permission response is
  // still unresolved. Resolving the first later must not touch the second.
  await page.evaluate(() =>
    (
      window as unknown as { resolveCamera: (index: number) => void }
    ).resolveCamera(1),
  );
  await expect(
    page.getByRole("button", { name: "Record round", exact: true }),
  ).toBeEnabled({ timeout: 45_000 });
  await page.evaluate(() =>
    (
      window as unknown as { resolveCamera: (index: number) => void }
    ).resolveCamera(0),
  );
  await expect
    .poll(() =>
      page.evaluate(() =>
        (
          window as unknown as { syntheticCameras: MediaStream[] }
        ).syntheticCameras.map(
          (stream) => stream.getVideoTracks()[0].readyState,
        ),
      ),
    )
    .toEqual(["ended", "live"]);
  await expect(
    page.getByRole("button", { name: "Record round", exact: true }),
  ).toBeEnabled();
  expect(
    await page
      .locator(".camera-stage video")
      .evaluate(
        (video: HTMLVideoElement) =>
          video.srcObject ===
          (window as unknown as { syntheticCameras: MediaStream[] })
            .syntheticCameras[1],
      ),
  ).toBe(true);
  await page.getByRole("button", { name: "Stop camera", exact: true }).click();
  await expect
    .poll(() =>
      page.evaluate(() =>
        (
          window as unknown as { syntheticCameras: MediaStream[] }
        ).syntheticCameras.map(
          (stream) => stream.getVideoTracks()[0].readyState,
        ),
      ),
    )
    .toEqual(["ended", "ended"]);
});

test("camera round ends on time when no video frame callbacks arrive", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await page.addInitScript(() => {
    const state = window as unknown as { syntheticCamera: MediaStream | null };
    state.syntheticCamera = null;
    // Simulate a suspended frame callback stream after setup. Timer completion
    // must not depend on a successful pose result or another decoded frame.
    HTMLVideoElement.prototype.requestVideoFrameCallback = () => 1;
    HTMLVideoElement.prototype.cancelVideoFrameCallback = () => {};
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      configurable: true,
      value: async () => {
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 360;
        const context = canvas.getContext("2d")!;
        let value = 0;
        const draw = () => {
          context.fillStyle = value++ % 2 ? "#234" : "#345";
          context.fillRect(0, 0, canvas.width, canvas.height);
        };
        draw();
        const stream = canvas.captureStream(20);
        const timer = window.setInterval(draw, 50);
        stream.getTracks().forEach((track) => {
          const originalStop = track.stop.bind(track);
          track.stop = () => {
            clearInterval(timer);
            originalStop();
          };
        });
        state.syntheticCamera = stream;
        return stream;
      },
    });
  });
  await page.goto(APP_ORIGIN);
  await page
    .getByRole("button", { name: "Enable camera", exact: true })
    .click();
  const record = page.getByRole("button", {
    name: "Record round",
    exact: true,
  });
  await expect(record).toBeEnabled({ timeout: 45_000 });
  await expect(
    page
      .getByLabel("Round duration", { exact: true })
      .getByRole("button", { name: "30s", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.clock.install();
  await record.click();
  await page.clock.fastForward(8100);
  await expect(
    page.getByRole("button", { name: "Stop & save", exact: true }),
  ).toBeVisible();
  // Allow the real MediaRecorder to produce a media chunk; the app clock is
  // then advanced independently of pose callbacks and real encoding time.
  await page.waitForTimeout(300);
  await page.clock.fastForward(30_100);
  await expect(
    page.getByRole("heading", { name: "Review", exact: true }),
  ).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (
            window as unknown as { syntheticCamera: MediaStream }
          ).syntheticCamera.getVideoTracks()[0].readyState,
      ),
    )
    .toBe("ended");
  const saved = await readExport(page);
  expect(saved.source).toBe("camera");
  expect(saved.frames).toEqual([]);
  expect(saved.events).toEqual([]);
  expect(saved.durationMs).toBeGreaterThanOrEqual(29_900);
  expect(saved.durationMs).toBeLessThan(31_000);
});

test("390px mobile layout stays within the viewport", async ({ page }) => {
  await denyPhysicalCamera(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(APP_ORIGIN);
  await expect(
    page.getByRole("heading", { name: "Practice", exact: true }),
  ).toBeVisible();
  const overflow = () =>
    page.evaluate(() => ({
      viewport: window.innerWidth,
      document: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
    }));
  let dimensions = await overflow();
  expect(dimensions.document).toBeLessThanOrEqual(dimensions.viewport);
  expect(dimensions.body).toBeLessThanOrEqual(dimensions.viewport);
  await page.screenshot({
    path: "artifacts/ui/practice-mobile.png",
    fullPage: true,
  });
  await page.getByText("More options", { exact: true }).click();
  await page.getByRole("button", { name: "Try demo", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Start demo", exact: true }),
  ).toBeEnabled();
  dimensions = await overflow();
  expect(dimensions.document).toBeLessThanOrEqual(dimensions.viewport);
  expect(dimensions.body).toBeLessThanOrEqual(dimensions.viewport);
});

test("retained-video replay bounds native seeking and hides pose samples across gaps", async ({
  page,
}) => {
  await denyPhysicalCamera(page);
  await page.goto(APP_ORIGIN);
  // A canvas-only recording supplies a real seekable media element without a
  // physical camera. Only 600 ms of its 1.8-second duration is analyzed.
  await page.evaluate(
    async (frames) => {
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 360;
      const context = canvas.getContext("2d")!;
      let index = 0;
      const draw = () => {
        context.fillStyle = index++ % 2 ? "#234" : "#345";
        context.fillRect(0, 0, canvas.width, canvas.height);
      };
      draw();
      const stream = canvas.captureStream(20);
      const recorder = new MediaRecorder(stream, {
        mimeType: "video/webm;codecs=vp8",
      });
      const chunks: BlobPart[] = [];
      const recording = new Promise<Blob>((resolve, reject) => {
        recorder.ondataavailable = (event) => {
          if (event.data.size) chunks.push(event.data);
        };
        recorder.onstop = () =>
          resolve(new Blob(chunks, { type: recorder.mimeType }));
        recorder.onerror = () =>
          reject(new Error("Could not create the canvas-only replay fixture."));
      });
      const timer = window.setInterval(draw, 50);
      recorder.start(100);
      await new Promise((resolve) => setTimeout(resolve, 1800));
      recorder.stop();
      const video = await recording;
      clearInterval(timer);
      stream.getTracks().forEach((track) => track.stop());

      const session: Session = {
        id: "replay-interval-fixture",
        createdAt: new Date().toISOString(),
        source: "file",
        stance: "orthodox",
        model: "full",
        drill: "one-two",
        durationMs: 600,
        videoOffsetMs: 250,
        video,
        frames,
        events: [],
        annotations: [],
        measuredFps: 0,
        inferenceP95: 0,
        skippedFrames: 0,
        schemaVersion: "1.0",
        annotationsComplete: false,
      };
      await new Promise<void>((resolve, reject) => {
        const request = indexedDB.open("corner-local-v1");
        request.onupgradeneeded = () =>
          request.result.createObjectStore("sessions", { keyPath: "id" });
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const database = request.result;
          const transaction = database.transaction("sessions", "readwrite");
          transaction.objectStore("sessions").put(session);
          transaction.oncomplete = () => {
            database.close();
            resolve();
          };
          transaction.onerror = transaction.onabort = () => {
            database.close();
            reject(transaction.error);
          };
        };
      });
    },
    [demoFrame(0), demoFrame(50), demoFrame(600)],
  );
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await page.locator(".round-library .session-item").click();
  const video = page.locator(".replay-stage video");
  const tracking = page
    .locator(".review")
    .getByRole("checkbox", { name: "Show tracking", exact: true });
  await expect(tracking).toBeChecked();
  await tracking.check();
  await expect
    .poll(() =>
      video.evaluate((element: HTMLVideoElement) => element.readyState),
    )
    .toBeGreaterThanOrEqual(2);
  await expect
    .poll(() =>
      video.evaluate((element: HTMLVideoElement) => element.currentTime),
    )
    .toBeCloseTo(0.25, 2);

  // Native controls and external seeks must obey both interval boundaries.
  await video.evaluate((element: HTMLVideoElement) => {
    element.currentTime = 0.05;
  });
  await expect
    .poll(() =>
      video.evaluate((element: HTMLVideoElement) => element.currentTime),
    )
    .toBeCloseTo(0.25, 2);
  await video.evaluate(async (element: HTMLVideoElement) => {
    element.muted = true;
    await element.play();
    element.currentTime = 1.4;
  });
  await expect
    .poll(() =>
      video.evaluate((element: HTMLVideoElement) => ({
        time: element.currentTime,
        paused: element.paused,
      })),
    )
    .toEqual({ time: 0.85, paused: true });
  await expect(page.getByLabel("Replay position", { exact: true })).toHaveValue(
    "600",
  );

  // Playing normally through the analysis end must also pause before the rest
  // of the retained original can be mistaken for analyzed material.
  await video.evaluate(async (element: HTMLVideoElement) => {
    element.currentTime = 0.7;
    await element.play();
  });
  await expect
    .poll(() =>
      video.evaluate((element: HTMLVideoElement) => ({
        time: element.currentTime,
        paused: element.paused,
      })),
    )
    .toEqual({ time: 0.85, paused: true });

  await video.evaluate((element: HTMLVideoElement) => {
    element.currentTime = 0.55;
  });
  await expect(page.getByLabel("Replay position", { exact: true })).toHaveValue(
    "300",
  );
  await expect(page.locator(".replay-stage .pose-overlay")).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("No recent pose sample");
  await video.evaluate((element: HTMLVideoElement) => {
    element.currentTime = 0.3;
  });
  await expect(page.locator(".replay-stage .pose-overlay")).toHaveCount(1);
  await expect(page.getByRole("status")).toHaveCount(0);
  const speed = page.getByLabel("Playback speed", { exact: true });
  await expect(
    speed.getByRole("button", { name: "1×", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await speed.getByRole("button", { name: "0.5×", exact: true }).click();
  await expect(video).toHaveJSProperty("playbackRate", 0.5);
  await page.getByRole("button", { name: "Play replay", exact: true }).click();
  await expect(video).toHaveJSProperty("paused", false);
  await page
    .getByRole("button", { name: "Previous frame", exact: true })
    .click();
  await expect(video).toHaveJSProperty("paused", true);
  await expect(
    page.getByRole("button", { name: "Play replay", exact: true }),
  ).toBeVisible();
  await speed.getByRole("button", { name: "1×", exact: true }).click();
  await expect(video).toHaveJSProperty("playbackRate", 1);
  expect(
    await page.evaluate(
      () =>
        (window as unknown as { cameraRequests: unknown[] }).cameraRequests
          .length,
    ),
  ).toBe(0);
});

test("motion-only replay supports half speed and frame stepping pauses playback", async ({
  page,
}) => {
  await denyPhysicalCamera(page);
  await page.goto(APP_ORIGIN);
  await page.evaluate(
    async (frames) => {
      const session: Session = {
        id: "motion-speed-fixture",
        createdAt: new Date().toISOString(),
        source: "demo",
        stance: "orthodox",
        model: "synthetic",
        drill: "open",
        durationMs: 5000,
        frames,
        events: [],
        annotations: [],
        measuredFps: 10,
        inferenceP95: 0,
        skippedFrames: 0,
        schemaVersion: "1.0",
        annotationsComplete: false,
      };
      await new Promise<void>((resolve, reject) => {
        const request = indexedDB.open("corner-local-v1");
        request.onupgradeneeded = () =>
          request.result.createObjectStore("sessions", { keyPath: "id" });
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const database = request.result;
          const transaction = database.transaction("sessions", "readwrite");
          const store = transaction.objectStore("sessions");
          store.put(session);
          store.put({
            ...session,
            id: "motion-speed-second",
            createdAt: new Date(0).toISOString(),
          });
          transaction.oncomplete = () => {
            database.close();
            resolve();
          };
          transaction.onerror = transaction.onabort = () => {
            database.close();
            reject(transaction.error);
          };
        };
      });
    },
    Array.from({ length: 51 }, (_, index) => demoFrame(index * 100)),
  );
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await page.locator(".round-library .session-item").first().click();
  await expect(page.locator(".replay-stage video")).toHaveCount(0);
  const speed = page.getByLabel("Playback speed", { exact: true });
  const position = page.getByLabel("Replay position", { exact: true });
  await expect(
    speed.getByRole("button", { name: "1×", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await speed.getByRole("button", { name: "0.5×", exact: true }).click();
  await page.clock.install();
  await page.getByRole("button", { name: "Play replay", exact: true }).click();
  await page.clock.runFor(1000);
  const halfSpeedTime = Number(await position.inputValue());
  expect(halfSpeedTime).toBeGreaterThanOrEqual(480);
  expect(halfSpeedTime).toBeLessThanOrEqual(520);
  await speed.getByRole("button", { name: "1×", exact: true }).click();
  await page.clock.runFor(1000);
  const normalSpeedTime = Number(await position.inputValue());
  expect(normalSpeedTime - halfSpeedTime).toBeGreaterThanOrEqual(960);
  expect(normalSpeedTime - halfSpeedTime).toBeLessThanOrEqual(1040);
  await page
    .getByRole("button", { name: "Previous frame", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Play replay", exact: true }),
  ).toBeVisible();
  const pausedAt = await position.inputValue();
  await page.clock.runFor(1000);
  await expect(position).toHaveValue(pausedAt);
  await speed.getByRole("button", { name: "0.5×", exact: true }).click();
  await page.locator(".round-library .session-item").last().click();
  await expect(
    speed.getByRole("button", { name: "1×", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(position).toHaveValue("0");
});

test("detection recheck is temporary and exports separately from original evidence", async ({
  page,
}) => {
  await denyPhysicalCamera(page);
  await page.goto(APP_ORIGIN);
  await page.evaluate(
    async ({ frames, currentVersion }) => {
      const original: Session = {
        id: "recheck-original",
        createdAt: new Date().toISOString(),
        source: "demo",
        stance: "orthodox",
        model: "synthetic",
        drill: "open",
        durationMs: 6400,
        frames,
        events: [],
        annotations: [],
        measuredFps: 25,
        inferenceP95: 0,
        skippedFrames: 0,
        schemaVersion: "1.0",
        detectorVersion: "saved-older-detector",
      };
      await new Promise<void>((resolve, reject) => {
        const request = indexedDB.open("corner-local-v1");
        request.onupgradeneeded = () =>
          request.result.createObjectStore("sessions", { keyPath: "id" });
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const database = request.result;
          const transaction = database.transaction("sessions", "readwrite");
          const store = transaction.objectStore("sessions");
          store.put(original);
          store.put({
            ...original,
            id: "already-current",
            createdAt: new Date(0).toISOString(),
            detectorVersion: currentVersion,
          });
          transaction.oncomplete = () => {
            database.close();
            resolve();
          };
          transaction.onerror = transaction.onabort = () => {
            database.close();
            reject(transaction.error);
          };
        };
      });
    },
    {
      frames: Array.from({ length: 161 }, (_, index) => demoFrame(index * 40)),
      currentVersion: DETECTOR_VERSION,
    },
  );
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await page.locator(".round-library .session-item").first().click();
  const original = await readExport(page);
  expect(original.events).toEqual([]);
  await page
    .getByRole("checkbox", { name: "Show detections", exact: true })
    .check();
  await expect(page.locator(".detections h2")).toHaveText("0 detected punches");
  await page
    .getByRole("button", { name: "Recheck detections", exact: true })
    .click();
  await expect(
    page.getByText("Updated analysis · Experimental", { exact: true }),
  ).toBeVisible();
  expect(await page.locator(".event-chip").count()).toBeGreaterThan(0);
  await page.screenshot({
    path: "artifacts/ui/recheck-desktop.png",
    fullPage: true,
  });
  const downloading = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export updated analysis", exact: true })
    .click();
  const download = await downloading;
  expect(download.suggestedFilename()).toBe(
    "corner-recheck-original-updated-analysis.json",
  );
  const report = JSON.parse(
    await readFile((await download.path())!, "utf8"),
  ) as DetectorRecheckReport;
  expect(report).toMatchObject({
    reportType: "detector-recheck",
    sessionId: original.id,
    detectorVersion: DETECTOR_VERSION,
    sourceDetectorVersion: "saved-older-detector",
    trackingSource: "saved-frames",
    frameCount: original.frames.length,
  });
  expect(report.events.length).toBeGreaterThan(0);
  expect(report).not.toHaveProperty("frames");
  expect(report).not.toHaveProperty("video");
  expect(await readExport(page)).toEqual(original);

  await page
    .getByRole("button", { name: "Use saved detections", exact: true })
    .click();
  await expect(page.locator(".detections h2")).toHaveText("0 detected punches");
  await expect(
    page.getByRole("button", { name: "Export updated analysis", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Recheck detections", exact: true })
    .click();
  await page.locator(".round-library .session-item").last().click();
  await expect(
    page.getByRole("button", { name: "Export updated analysis", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("checkbox", { name: "Show detections", exact: true }),
  ).toBeChecked();
  await page
    .getByRole("checkbox", { name: "Show detections", exact: true })
    .check();
  await expect(
    page.getByRole("button", { name: "Recheck detections", exact: true }),
  ).toHaveCount(0);
  await page.locator(".round-library .session-item").first().click();
  await expect(
    page.getByRole("checkbox", { name: "Show detections", exact: true }),
  ).toBeChecked();
  await page
    .getByRole("checkbox", { name: "Show detections", exact: true })
    .check();
  await expect(
    page.getByRole("button", { name: "Recheck detections", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".detections h2")).toHaveText("0 detected punches");

  // Close the component and read the persisted session again: rechecks cannot
  // silently rewrite original events or provenance in IndexedDB.
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await page.locator(".round-library .session-item").first().click();
  expect(await readExport(page)).toEqual(original);
});

test("a late in-flight camera frame stays inside the frozen media duration", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const state = window as unknown as {
      mediaMs: number;
      callback?: VideoFrameRequestCallback;
      pendingReply?: () => void;
    };
    state.mediaMs = 10_000;
    Object.defineProperty(HTMLVideoElement.prototype, "currentTime", {
      configurable: true,
      get: () => state.mediaMs / 1000,
      set: () => {},
    });
    HTMLVideoElement.prototype.requestVideoFrameCallback = (callback) => {
      state.callback = callback;
      return 1;
    };
    HTMLVideoElement.prototype.cancelVideoFrameCallback = () => {};
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      configurable: true,
      value: async () => {
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 360;
        canvas.getContext("2d")!.fillRect(0, 0, 640, 360);
        return canvas.captureStream(30);
      },
    });
    const OriginalWorker = window.Worker;
    window.Worker = class extends OriginalWorker {
      override postMessage(message: unknown) {
        const request = message as {
          type: string;
          id: number;
          delegate: string;
          t: number;
          width: number;
          height: number;
          bitmap: ImageBitmap;
        };
        if (request.type === "init") {
          queueMicrotask(() =>
            this.dispatchEvent(
              new MessageEvent("message", {
                data: {
                  type: "ready",
                  id: request.id,
                  delegate: request.delegate,
                },
              }),
            ),
          );
        } else {
          request.bitmap.close();
          state.pendingReply = () =>
            this.dispatchEvent(
              new MessageEvent("message", {
                data: {
                  type: "result",
                  id: request.id,
                  frame: {
                    t: request.t,
                    width: request.width,
                    height: request.height,
                    landmarks: [],
                    inferenceMs: 10,
                  },
                },
              }),
            );
        }
      }
    };
  });
  await page.goto(APP_ORIGIN);
  await page
    .getByRole("button", { name: "Enable camera", exact: true })
    .click();
  const record = page.getByRole("button", {
    name: "Record round",
    exact: true,
  });
  await expect(record).toBeEnabled();
  await page.clock.install();
  await record.click();
  await page.clock.fastForward(8100);
  await expect(
    page.getByRole("button", { name: "Stop & save", exact: true }),
  ).toBeVisible();
  await page.clock.fastForward(29_850);
  await page.evaluate(() => {
    const state = window as unknown as {
      mediaMs: number;
      callback: VideoFrameRequestCallback;
    };
    state.mediaMs = 40_040;
    state.callback(performance.now(), {
      mediaTime: 40.015,
    } as VideoFrameCallbackMetadata);
  });
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          !!(window as unknown as { pendingReply?: () => void }).pendingReply,
      ),
    )
    .toBe(true);
  // The frame is captured before closing, but its result arrives after the
  // wall-clock duration has frozen. Its media time must still fit the export.
  await page.clock.fastForward(200);
  await page.evaluate(() =>
    (window as unknown as { pendingReply: () => void }).pendingReply(),
  );
  await expect(
    page.getByRole("heading", { name: "Review", exact: true }),
  ).toBeVisible();
  const saved = await readExport(page);
  expect(saved.frames).toHaveLength(1);
  expect(saved.frames[0].t).toBeCloseTo(30_015);
  expect(saved.durationMs).toBeGreaterThanOrEqual(saved.frames[0].t);
  expect(saved.durationMs).toBeLessThan(30_200);
});

test("every consecutive 20 Hz pose result reaches the visible skeleton", async ({
  page,
}) => {
  // Exercise the real capture pump and useStudio receive path. Only inference
  // is synthetic; there is no physical camera, GPU model or local service.
  await page.route("**/src/lib/vision.ts*", (route) =>
    route.fulfill({
      contentType: "application/javascript",
      body: `export class VisionRunner {
      delegate='CPU'; modelInfo=null;
      async init(){}
      async prepareImage(_image,t){return {t,close(){}};}
      async detectPrepared(image,t){
        if(image.t!==t)throw Error('Source timestamp changed');
        return this.detectImage(null,t);
      }
      async detectImage(_image,t){
        const index=window.__poseResults.length;
        const frame=structuredClone(window.__poseFixture);
        frame.t=t;frame.landmarks[15].x=.30+index*.015;
        window.__poseResults.push({t,at:performance.now(),x:frame.landmarks[15].x});
        return frame;
      }
      dispose(){}
    }`,
    }),
  );
  await page.addInitScript(
    (frame) => {
      const state = window as any;
      state.__poseFixture = frame;
      state.__poseResults = [];
      state.__poseCallback = null;
      HTMLVideoElement.prototype.requestVideoFrameCallback = (callback) => {
        state.__poseCallback = callback;
        return 1;
      };
      HTMLVideoElement.prototype.cancelVideoFrameCallback = () => {
        state.__poseCallback = null;
      };
      navigator.mediaDevices.getUserMedia = async () => {
        const canvas = document.createElement("canvas");
        canvas.width = frame.width;
        canvas.height = frame.height;
        canvas.getContext("2d")!.fillRect(0, 0, canvas.width, canvas.height);
        state.__poseStream = canvas.captureStream(20);
        return state.__poseStream;
      };
    },
    { ...demoFrame(0), width: 640, height: 360 },
  );
  await page.goto(APP_ORIGIN);
  await page
    .getByRole("button", { name: "Enable camera", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Record round", exact: true }),
  ).toBeEnabled();
  await page.clock.install({ time: new Date("2030-01-01T00:00:00Z") });
  await page.clock.pauseAt(new Date("2030-01-01T00:00:01Z"));
  const wrist = page.locator(
    ".camera-stage circle.pose-joint.hand-left[r='9']",
  );
  for (let index = 0; index < 8; index++) {
    await page.clock.runFor(50);
    await page.evaluate((index) => {
      (window as any).__poseCallback(performance.now(), {
        mediaTime: 1 + index * 0.05,
      });
    }, index);
    await expect
      .poll(() => page.evaluate(() => (window as any).__poseResults.length))
      .toBe(index + 1);
    // This waits for a real React/SVG commit, not merely an inference callback.
    // With the former 65ms gate the second 50ms observation never appears.
    await expect
      .poll(async () => Number(await wrist.getAttribute("cx")), {
        timeout: 2000,
      })
      .toBeCloseTo((0.3 + index * 0.015) * 640, 6);
  }
  const observed = await page.evaluate(
    () =>
      (window as any).__poseResults as { t: number; at: number; x: number }[],
  );
  expect(observed).toHaveLength(8);
  for (let index = 1; index < observed.length; index++) {
    expect(observed[index].at - observed[index - 1].at).toBeCloseTo(50, 5);
    expect(observed[index].t - observed[index - 1].t).toBeCloseTo(50, 5);
  }
  await page.getByRole("button", { name: "Stop camera", exact: true }).click();
  await expect(page.locator(".camera-stage .pose-overlay")).toHaveCount(0);
  expect(
    await page.evaluate(
      () => (window as any).__poseStream.getVideoTracks()[0].readyState,
    ),
  ).toBe("ended");
});
