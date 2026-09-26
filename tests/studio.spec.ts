import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import type { Session } from "../src/lib/types";
import { demoFrame } from "../src/lib/demo";

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
    page.getByRole("heading", { name: "Your next good round." }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Enable camera", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Start round", exact: true }),
  ).toBeDisabled();
  await page.waitForLoadState("networkidle");
  expect(
    await page.evaluate(
      () =>
        (window as unknown as { cameraRequests: unknown[] }).cameraRequests
          .length,
    ),
  ).toBe(0);
  expect(externalRequests).toEqual([]);
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
    page.getByRole("button", { name: "Start round", exact: true }),
  ).toBeDisabled();
  const requests = await page.evaluate(
    () =>
      (window as unknown as { cameraRequests: MediaStreamConstraints[] })
        .cameraRequests,
  );
  expect(requests).toHaveLength(1);
  expect(requests[0].audio).toBe(false);
});

test("demo round survives immediate navigation, export, reload, annotation edits, and deletion", async ({
  page,
}) => {
  await denyPhysicalCamera(page);
  // Exercise Finish -> Review while asynchronous metadata finalization is pending.
  // This must not replace a completed round's timestamp with a cleared source time.
  await page.route("**/models/manifest.json", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 450));
    await route.continue();
  });
  await page.goto(APP_ORIGIN);
  await page.getByRole("button", { name: "Explore a simulated round" }).click();
  await page.getByRole("button", { name: "Start round", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Finish round", exact: true }),
  ).toBeVisible();
  // Two known synthetic strokes occur during this interval. This is UI coverage,
  // never a real-camera accuracy or performance benchmark.
  await page.waitForTimeout(6200);
  const candidateCounts = await page
    .locator(".round-count strong")
    .allTextContents();
  expect(
    candidateCounts.reduce((sum, value) => sum + Number(value), 0),
  ).toBeGreaterThanOrEqual(1);
  await page.getByRole("button", { name: "Finish round", exact: true }).click();
  await page.getByRole("button", { name: /^Round review/ }).click();
  await expect(
    page.getByRole("button", { name: "Evidence JSON", exact: true }),
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
  await page.getByRole("button", { name: /^Round review/ }).click();
  await expect(page.locator(".round-library .session-item")).toHaveCount(1);
  await page.locator(".round-library .session-item").click();
  const persisted = await readExport(page);
  expect(persisted.id).toBe(exported.id);
  expect(persisted.durationMs).toBe(exported.durationMs);
  expect(persisted.frames.length).toBe(exported.frames.length);

  await page
    .getByLabel("Annotation action", { exact: true })
    .selectOption("other");
  await page
    .getByLabel("Annotation hand", { exact: true })
    .selectOption("unknown");
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
  await page
    .getByRole("button", { name: "Delete this round", exact: true })
    .click();
  await expect(page.locator(".round-library .session-item")).toHaveCount(0);
  await page.reload();
  await page.getByRole("button", { name: /^Round review/ }).click();
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

test("390px mobile layout stays within the viewport", async ({ page }) => {
  await denyPhysicalCamera(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(APP_ORIGIN);
  await expect(
    page.getByRole("heading", { name: "Your next good round." }),
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
  await page.getByRole("button", { name: "Explore a simulated round" }).click();
  await expect(
    page.getByRole("button", { name: "Start round", exact: true }),
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
        const request = indexedDB.open("corner-local-v1", 1);
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
  await page.getByRole("button", { name: /^Round review/ }).click();
  await page.locator(".round-library .session-item").click();
  const video = page.locator(".replay-stage video");
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
  expect(
    await page.evaluate(
      () =>
        (window as unknown as { cameraRequests: unknown[] }).cameraRequests
          .length,
    ),
  ).toBe(0);
});
