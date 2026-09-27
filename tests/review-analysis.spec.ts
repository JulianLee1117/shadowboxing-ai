import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

const origin = "http://127.0.0.1:5173";
// Controlled lifecycle service: it deliberately ignores AbortSignal. A late
// result must be discarded by the real hook, independently of service behavior.
const analysisStub = `
export const ANALYSIS_VERSION = 'test-analysis-v1';
export const REPORT_VERSION = 'round-video-analysis-v1';
window.__analysisJobs = [];
export function analyzeRound(source, options) {
  return new Promise(resolve => {
    window.__analysisJobs.push({ source, signal: options.signal, complete: async () => {
      const hash = await crypto.subtle.digest('SHA-256', await source.video.arrayBuffer());
      const sourceFingerprint = Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2,'0')).join('');
      const report = { id:'analysis-'+source.id, reportType:'round-video-analysis', reportVersion:REPORT_VERSION,
        sourceSessionId:source.id, sourceDetectorVersion:source.detectorVersion, sourceFingerprint,
        analysisVersion:ANALYSIS_VERSION,detectorVersion:'stub-detector',createdAt:new Date().toISOString(),stance:source.stance,
        frames:[],events:[{id:'new-analysis-event',hand:'right',role:'rear',label:'cross',startMs:300,peakMs:450,endMs:650,detectedAtMs:680,score:.8,extension:.6,guardReturn:'unassessable',experimental:true}],
        model:'full',delegate:'CPU',modelManifest:{fixture:true},
        provenance:{source:'saved-video',videoBytes:source.video.size,videoOffsetMs:source.videoOffsetMs??0,sourceDurationMs:source.durationMs,decodedDurationMs:source.durationMs,timestampMode:'decoded-media-time',interpolation:false,mirrored:false,sequentialInference:true},
        cadence:{requestedFps:30,cadenceSource:'test',observedFps:30,maximumGapMs:34,decodedTimestampFrames:0,estimatedTimestampFrames:0,duplicateFramesSkipped:0,processingMs:1},
        uncertaintyIntervals:[],warnings:[],completeness:{status:'complete',reason:null,processedFrames:0,plannedFrames:0,coveredUntilMs:source.durationMs}};
      if(window.__partialReport) report.completeness={...report.completeness,status:'partial',reason:'Stopped before the end of the recording.',coveredUntilMs:500};
      window.__lastReport=report; resolve(report);
    }});
    options.onProgress?.({phase:'analyzing',completedFrames:1,plannedFrames:10,fraction:.1});
  });
}`;
const studioStub = `
const studio={videoRef:{current:null},status:'off',source:'camera',running:false,elapsed:0,events:[],frame:null,error:null,
  start:async()=>{},stop:async()=>{},beginRound:async()=>{},clearError:()=>{}};
export function useStudio(onComplete){window.__completeRound=onComplete;return studio;}
`;

async function setup(page: Page, count = 1) {
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    if (url.pathname === "/__lifecycle_seed")
      return route.fulfill({
        contentType: "text/html",
        body: "<!doctype html><title>Fixture</title>",
      });
    if (url.pathname === "/src/lib/roundAnalysis.ts")
      return route.fulfill({
        contentType: "text/javascript",
        body: analysisStub,
      });
    if (url.pathname === "/src/hooks/useStudio.ts")
      return route.fulfill({
        contentType: "text/javascript",
        body: studioStub,
      });
    return route.continue();
  });
  await page.addInitScript(() => {
    const w = window as any;
    w.__makeSession = (id: string, createdAt = "2026-01-01T00:00:00.000Z") => ({
      schemaVersion: "1.0",
      id,
      createdAt,
      source: "camera",
      stance: "orthodox",
      model: "full",
      drill: "double-jab",
      durationMs: 1000,
      frames: [],
      events: [
        {
          id: "original-event",
          hand: "left",
          role: "lead",
          label: "jab",
          startMs: 100,
          peakMs: 200,
          endMs: 400,
          score: 0.7,
          extension: 0.5,
          guardReturn: "returned",
          experimental: true,
        },
      ],
      annotations: [
        {
          id: "original-label",
          hand: "left",
          label: "jab",
          startMs: 100,
          endMs: 400,
          note: "Preserve reference label",
        },
      ],
      annotationsComplete: true,
      measuredFps: 30,
      inferenceP95: 10,
      skippedFrames: 0,
      detectorVersion: "original-detector",
      video: new Blob([new Uint8Array([1, 2, 3])], { type: "video/webm" }),
    });
    navigator.mediaDevices.getUserMedia = async () => {
      throw new Error("Physical camera forbidden in lifecycle tests");
    };
  });
  await page.goto(`${origin}/__lifecycle_seed`);
  await page.evaluate(async (count) => {
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("corner-local-v1", 1);
      request.onupgradeneeded = () =>
        request.result.createObjectStore("sessions", { keyPath: "id" });
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result,
          tx = db.transaction("sessions", "readwrite");
        for (let i = 0; i < count; i++)
          tx.objectStore("sessions").put(
            (window as any).__makeSession(
              `original-${i}`,
              `2026-01-0${i + 1}T00:00:00.000Z`,
            ),
          );
        tx.oncomplete = () => {
          db.close();
          resolve();
        };
      };
    });
  }, count);
  await page.goto(origin);
  await expect(
    page.getByRole("button", { name: /^Saved rounds/ }),
  ).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => (window as any).__analysisJobs.length))
    .toBe(0);
}
async function openReview(page: Page) {
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await expect(page.locator(".analysis-panel")).toBeVisible();
}
async function finish(page: Page, index: number) {
  await page.evaluate(async (i) => {
    await (window as any).__analysisJobs[i].complete();
  }, index);
}
async function dbState(page: Page) {
  return page.evaluate(async () => {
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const r = indexedDB.open("corner-local-v1");
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    });
    const version = db.version;
    const values = await Promise.all(
      ["sessions", "analyses"].map(
        (name) =>
          new Promise<any[]>((resolve, reject) => {
            const tx = db.transaction(name);
            const r = tx.objectStore(name).getAll();
            tx.oncomplete = () => resolve(r.result);
            tx.onerror = () => reject(tx.error);
          }),
      ),
    );
    db.close();
    return {
      version,
      sessions: values[0].map((s) => ({
        ...s,
        videoBytes: s.video.size,
        video: undefined,
      })),
      analyses: values[1],
    };
  });
}
async function exported(page: Page, name: string) {
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name, exact: true }).click();
  const download = await pending;
  return JSON.parse(await readFile((await download.path())!, "utf8"));
}

test("v1 migration preserves originals; separate analysis and focus survive reload and exports", async ({
  page,
}) => {
  await setup(page);
  const before = await dbState(page);
  expect(before.version).toBe(2);
  expect(before.analyses).toEqual([]);
  await openReview(page);
  await expect(
    page.getByText("Focus: Double jab", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
  await finish(page, 0);
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toBeVisible();
  await expect.poll(async () => (await dbState(page)).analyses.length).toBe(1);
  expect((await dbState(page)).sessions).toEqual(before.sessions);
  await expect(page.locator(".detection-heading")).toContainText(
    "Original detections",
  );
  await expect(page.locator(".detection-heading h2")).toContainText("1 jab");
  await page
    .getByRole("button", { name: "Show video analysis", exact: true })
    .click();
  await expect(page.locator(".detection-heading")).toContainText(
    "Video analysis",
  );
  await expect(page.locator(".detection-heading h2")).toContainText("1 cross");
  await page.reload();
  await openReview(page);
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toBeVisible();
  expect(await page.evaluate(() => (window as any).__analysisJobs.length)).toBe(
    0,
  );
  await expect(page.locator(".detection-heading")).toContainText(
    "Original detections",
  );
  await page.getByText("Export & details", { exact: true }).click();
  const evidence = await exported(page, "Evidence JSON"),
    analysis = await exported(page, "Export video analysis");
  expect(evidence.events[0].id).toBe("original-event");
  expect(evidence.annotations[0].note).toBe("Preserve reference label");
  expect(evidence.drill).toBe("double-jab");
  expect(analysis.events[0].id).toBe("new-analysis-event");
  expect(analysis).not.toHaveProperty("annotations");
});

test("cancel, round switch and unmount discard service results that ignore abort", async ({
  page,
}) => {
  await setup(page, 2);
  await openReview(page);
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Cancel analysis", exact: true })
    .click();
  await finish(page, 0);
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
  await page.locator(".round-library .session-item").last().click();
  await finish(page, 1);
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Analyze recording", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
  await page.getByRole("button", { name: "Practice", exact: true }).click();
  await finish(page, 2);
  await expect(
    page.getByRole("heading", { name: "Practice", exact: true }),
  ).toBeVisible();
  expect((await dbState(page)).analyses).toEqual([]);
  expect(
    await page.evaluate(() =>
      (window as any).__analysisJobs.every((j: any) => j.signal.aborted),
    ),
  ).toBe(true);
});

test("atomic deletion removes derived analysis and refuses a late resurrection", async ({
  page,
}) => {
  await setup(page);
  await openReview(page);
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
  await finish(page, 0);
  await expect.poll(async () => (await dbState(page)).analyses.length).toBe(1);
  await page.getByText("Export & details", { exact: true }).click();
  page.once("dialog", (d) => d.accept());
  await page
    .getByRole("button", { name: "Delete this round", exact: true })
    .click();
  await expect(page.locator(".round-library .session-item")).toHaveCount(0);
  const error = await page.evaluate(async () => {
    const path = "/src/lib/storage.ts";
    const { saveRoundAnalysis } = await import(path);
    try {
      await saveRoundAnalysis((window as any).__lastReport);
      return null;
    } catch (e) {
      return String(e);
    }
  });
  expect(error).toContain("Save the original round");
  const state = await dbState(page);
  expect(state.sessions).toEqual([]);
  expect(state.analyses).toEqual([]);
});

test("only a newly completed video round auto-analyzes; reopening stored rounds does not start jobs", async ({
  page,
}) => {
  await setup(page);
  await openReview(page);
  expect(await page.evaluate(() => (window as any).__analysisJobs.length)).toBe(
    0,
  );
  await page.getByRole("button", { name: "Practice", exact: true }).click();
  await page.evaluate(() =>
    (window as any).__completeRound(
      (window as any).__makeSession("new-round", "2026-02-01T00:00:00.000Z"),
    ),
  );
  await expect(
    page.getByRole("heading", { name: "Review", exact: true }),
  ).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => (window as any).__analysisJobs.length))
    .toBe(1);
  await finish(page, 0);
  await expect.poll(async () => (await dbState(page)).analyses.length).toBe(1);
  await page.reload();
  expect(await page.evaluate(() => (window as any).__analysisJobs.length)).toBe(
    0,
  );
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toBeVisible();
  expect(await page.evaluate(() => (window as any).__analysisJobs.length)).toBe(
    0,
  );
});

test("same-sized replacement footage and changed timing cannot reuse a cached analysis", async ({
  page,
}) => {
  await setup(page);
  await openReview(page);
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
  await finish(page, 0);
  await expect.poll(async () => (await dbState(page)).analyses.length).toBe(1);
  const firstHash = (await dbState(page)).analyses[0].sourceFingerprint;
  await page.evaluate(() => {
    const w = window as any;
    const changed = w.__makeSession("original-0");
    changed.video = new Blob([new Uint8Array([3, 2, 1])], {
      type: "video/webm",
    });
    w.__completeRound(changed);
  });
  await expect
    .poll(() => page.evaluate(() => (window as any).__analysisJobs.length))
    .toBe(2);
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toHaveCount(0);
  await finish(page, 1);
  await expect
    .poll(async () => (await dbState(page)).analyses[0]?.sourceFingerprint)
    .not.toBe(firstHash);
  await page.evaluate(() => {
    const w = window as any;
    // Reuse the actual Blob object: timing changes must independently invalidate.
    w.__completeRound({
      ...w.__analysisJobs[1].source,
      videoOffsetMs: 100,
      durationMs: 900,
    });
  });
  await expect
    .poll(() => page.evaluate(() => (window as any).__analysisJobs.length))
    .toBe(3);
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toHaveCount(0);
  await finish(page, 2);
  await expect
    .poll(
      async () => (await dbState(page)).analyses[0]?.provenance.videoOffsetMs,
    )
    .toBe(100);
  expect((await dbState(page)).analyses[0].provenance.sourceDurationMs).toBe(
    900,
  );
});

test("partial reports show their limitation and require an explicit switch from original counts", async ({
  page,
}) => {
  await setup(page);
  await openReview(page);
  await page.evaluate(() => {
    (window as any).__partialReport = true;
  });
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
  await finish(page, 0);
  await expect(
    page.getByText("Partial analysis available", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".analysis-panel")).toContainText(
    "Stopped before the end of the recording.",
  );
  await expect(page.locator(".detection-heading")).toContainText(
    "Original detections",
  );
  await expect(page.locator(".detection-heading h2")).toContainText("1 jab");
  await page
    .getByRole("button", { name: "Show video analysis", exact: true })
    .click();
  await expect(page.locator(".detection-heading")).toContainText(
    "Partial video analysis",
  );
  await expect(page.locator(".detection-heading h2")).toContainText("1 cross");
  const original = (await dbState(page)).sessions[0];
  expect(original.events[0].id).toBe("original-event");
});

test("original detections cannot form combinations across wholly missing saved tracking", async ({
  page,
}) => {
  await setup(page);
  await page.evaluate(async () => {
    const source = (window as any).__makeSession("original-0");
    source.events.push({
      ...source.events[0],
      id: "original-cross",
      hand: "right",
      role: "rear",
      label: "cross",
      startMs: 420,
      peakMs: 550,
      endMs: 750,
    });
    const path = "/src/lib/storage.ts";
    const { saveSession } = await import(path);
    await saveSession(source);
  });
  await page.reload();
  await openReview(page);
  await page.getByLabel("Show detections", { exact: true }).check();
  await expect(page.locator(".detection-heading h2")).toContainText("1 jab");
  await expect(page.locator(".detection-heading h2")).toContainText("1 cross");
  await expect(
    page.getByRole("heading", { name: "Combinations", exact: true }),
  ).toHaveCount(0);
  await expect(page.getByText(/Uncertain tracking ·/)).toBeVisible();
});
