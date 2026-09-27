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

async function setup(page: Page, count = 1, version: 1 | 2 = 1) {
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
  await page.evaluate(
    async ({ count, version }) => {
      const sessions = Array.from({ length: count }, (_, i) =>
        (window as any).__makeSession(
          `original-${i}`,
          new Date(Date.UTC(2026, 0, i + 1)).toISOString(),
        ),
      );
      let savedReport: any;
      if (version === 2) {
        const path = "/src/lib/roundAnalysis.ts";
        const { analyzeRound } = await import(path);
        const pending = analyzeRound(sessions[0], {});
        await (window as any).__analysisJobs[0].complete();
        savedReport = await pending;
      }
      await new Promise<void>((resolve, reject) => {
        const request = indexedDB.open("corner-local-v1", version);
        request.onupgradeneeded = () => {
          request.result.createObjectStore("sessions", { keyPath: "id" });
          if (version === 2)
            request.result.createObjectStore("analyses", {
              keyPath: "sourceSessionId",
            });
        };
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const db = request.result,
            tx = db.transaction(
              version === 2 ? ["sessions", "analyses"] : ["sessions"],
              "readwrite",
            );
          for (const session of sessions)
            tx.objectStore("sessions").put(session);
          if (savedReport) tx.objectStore("analyses").put(savedReport);
          tx.oncomplete = () => {
            db.close();
            resolve();
          };
        };
      });
    },
    { count, version },
  );
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
  await page
    .getByRole("button", { name: "Re-run analysis", exact: true })
    .click();
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
      ["sessions", "analyses", "trash"].map(
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
      sessions: await Promise.all(
        values[0].map(async (s) => ({
          ...s,
          videoBytes: s.video.size,
          videoContents: Array.from(
            new Uint8Array(await s.video.arrayBuffer()),
          ),
          video: undefined,
        })),
      ),
      analyses: values[1],
      trash: values[2],
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
  expect(before.version).toBe(3);
  expect(before.analyses).toEqual([]);
  await openReview(page);
  await page.locator(".replay-stage video").evaluate((video) => {
    (window as any).__replayNode = video;
    (window as any).__replaySrc = (video as HTMLVideoElement).src;
  });
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
  await expect(page.locator(".review-count.punch-1 strong")).toHaveText("1");
  await page
    .getByRole("button", { name: "Show video analysis", exact: true })
    .click();
  await expect(page.locator(".detection-heading")).toContainText(
    "Video analysis",
  );
  await expect(page.locator(".review-count.punch-2 strong")).toHaveText("1");
  await page
    .getByRole("checkbox", { name: "Show tracking", exact: true })
    .uncheck();
  await page
    .getByRole("checkbox", { name: "Show tracking", exact: true })
    .check();
  await page.getByLabel("Show detections", { exact: true }).uncheck();
  await page.getByLabel("Show detections", { exact: true }).check();
  expect(
    await page
      .locator(".replay-stage video")
      .evaluate(
        (video) =>
          video === (window as any).__replayNode &&
          (video as HTMLVideoElement).src === (window as any).__replaySrc,
      ),
  ).toBe(true);
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
  const rerun = page.getByRole("button", {
    name: "Analyze again",
    exact: true,
  });
  await expect(rerun).toHaveCount(1);
  await expect(
    page
      .locator(".analysis-actions")
      .getByRole("button", { name: "Analyze again", exact: true }),
  ).toBeVisible();
  await rerun.click();
  await expect
    .poll(() => page.evaluate(() => (window as any).__analysisJobs.length))
    .toBe(1);
  await expect(
    page.getByRole("button", { name: "Cancel analysis", exact: true }),
  ).toBeVisible();
  await finish(page, 0);
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toBeVisible();
  expect((await dbState(page)).sessions).toEqual(before.sessions);
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

test("v2 migration, Undo and restore preserve original video, labels and cached analysis", async ({
  page,
}) => {
  await setup(page, 1, 2);
  const original = await dbState(page);
  expect(original.version).toBe(3);
  expect(original.analyses).toHaveLength(1);
  expect(original.trash).toEqual([]);
  await openReview(page);
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /^Remove round from/ }).click();
  await expect(page.locator(".session-item")).toHaveCount(0);
  const removed = await dbState(page);
  expect(removed.sessions).toEqual(original.sessions);
  expect(removed.analyses).toEqual(original.analyses);
  expect(removed.trash).toHaveLength(1);
  const rejected = await page.evaluate(
    async ({ report }) => {
      const path = "/src/lib/storage.ts";
      const {
        saveRoundAnalysis,
        saveSession,
        listSessions,
        loadRoundAnalysis,
      } = await import(path);
      const errors = [];
      for (const operation of [
        () => saveRoundAnalysis(report),
        () => saveSession((window as any).__makeSession("original-0")),
      ]) {
        try {
          await operation();
        } catch (e) {
          errors.push(String(e));
        }
      }
      return {
        errors,
        active: await listSessions(),
        report: await loadRoundAnalysis("original-0"),
      };
    },
    { report: original.analyses[0] },
  );
  expect(rejected.errors).toHaveLength(2);
  expect(
    rejected.errors.every((error) => error.includes("Recently deleted")),
  ).toBe(true);
  expect(rejected.active).toEqual([]);
  expect(rejected.report).toBeUndefined();
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(page.locator(".session-item")).toHaveCount(1);
  await page
    .getByRole("button", { name: "Re-run analysis", exact: true })
    .click();
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toBeVisible();
  expect(await dbState(page)).toEqual(original);
  await page.getByRole("button", { name: /^Remove round from/ }).click();
  await expect(page.locator(".session-item")).toHaveCount(0);
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await expect(page.locator(".session-item")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Undo", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Recently deleted (1)", exact: true })
    .click();
  await page.getByRole("button", { name: "Restore", exact: true }).click();
  await expect(page.locator(".session-item")).toHaveCount(1);
  await page
    .getByRole("button", { name: "Re-run analysis", exact: true })
    .click();
  await expect(
    page.getByText("Round analysis ready", { exact: true }),
  ).toBeVisible();
  expect(await dbState(page)).toEqual(original);
  expect(await page.evaluate(() => (window as any).__analysisJobs.length)).toBe(
    0,
  );
});

test("confirmed permanent deletion removes original and analysis atomically; late jobs cannot restore it", async ({
  page,
}) => {
  await setup(page, 1, 2);
  const original = await dbState(page);
  await openReview(page);
  await page
    .getByRole("button", { name: "Analyze again", exact: true })
    .click();
  await page.getByRole("button", { name: /^Remove round from/ }).click();
  await expect(page.locator(".session-item")).toHaveCount(0);
  expect(
    await page.evaluate(() => (window as any).__analysisJobs[0].signal.aborted),
  ).toBe(true);
  await finish(page, 0);
  expect((await dbState(page)).analyses).toEqual(original.analyses);
  await page
    .getByRole("button", { name: "Recently deleted (1)", exact: true })
    .click();
  page.once("dialog", (dialog) => dialog.dismiss());
  await page
    .getByRole("button", { name: "Delete permanently", exact: true })
    .click();
  expect((await dbState(page)).sessions).toEqual(original.sessions);
  page.once("dialog", (dialog) => dialog.accept());
  await page
    .getByRole("button", { name: "Delete permanently", exact: true })
    .click();
  await expect(page.locator(".trashed-round")).toHaveCount(0);
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
  expect(state.trash).toEqual([]);
});

test("an annotation edit in a stale tab cannot recreate a permanently deleted round", async ({
  page,
}) => {
  await setup(page, 1, 2);
  await openReview(page);
  // Another connection changes storage while this tab still holds the old round.
  await page.evaluate(async () => {
    const path = "/src/lib/storage.ts";
    const { moveSessionToTrash, deleteSession } = await import(path);
    await moveSessionToTrash("original-0");
    await deleteSession("original-0");
  });
  await page.getByText("Label this round (optional)", { exact: true }).click();
  await page
    .getByLabel("Annotation note", { exact: true })
    .fill("Unsaved edit after external deletion");
  await page.getByRole("button", { name: "Add label", exact: true }).click();
  await expect(
    page.getByText(
      "This round was deleted in another tab. Export your edits before closing.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /^Remove round from/ }),
  ).toBeDisabled();
  expect((await dbState(page)).sessions).toEqual([]);
  await page.getByText("Export & details", { exact: true }).click();
  const evidence = await exported(page, "Evidence JSON");
  expect(evidence.annotations).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ note: "Unsaved edit after external deletion" }),
    ]),
  );
});

test("saving rounds cannot be removed, and a save failure preserves the in-memory evidence", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/src/lib/storage.ts*", async (route) => {
    if (new URL(route.request().url()).searchParams.has("unwrapped"))
      return route.fallback();
    return route.fulfill({
      contentType: "text/javascript",
      body: `
    export * from '/src/lib/storage.ts?unwrapped';
    import {saveSession as saveOriginal} from '/src/lib/storage.ts?unwrapped';
    export async function saveSession(session, options) {
      if(session.id === 'pending-save') await new Promise((resolve,reject)=>{window.__rejectSave=()=>reject(new Error('Fixture: storage full'));});
      return saveOriginal(session, options);
    }
  `,
    });
  });
  await page.reload();
  await page.evaluate(() => {
    const w = window as any;
    w.__completeRound(
      w.__makeSession("pending-save", "2026-02-01T00:00:00.000Z"),
    );
  });
  const pending = page.locator(".session-card").first();
  await expect(pending.locator(".session-remove")).toBeDisabled();
  await page.locator(".session-item").last().click();
  await expect(pending.locator(".session-remove")).toBeDisabled();
  await page.evaluate(() => (window as any).__rejectSave());
  await expect(
    page.getByText("Fixture: storage full", { exact: true }),
  ).toBeVisible();
  await expect(pending.locator(".session-remove")).toBeDisabled();
  await expect(pending).toContainText("Not saved");
  await expect(page.locator(".session-item")).toHaveCount(2);
  await pending.locator(".session-item").click();
  await expect(
    page.getByText("Not saved — export before closing", { exact: true }),
  ).toBeVisible();
  await page.getByText("Export & details", { exact: true }).click();
  const evidence = await exported(page, "Evidence JSON");
  expect(evidence.id).toBe("pending-save");
  expect(evidence.annotations[0].note).toBe("Preserve reference label");
  expect((await dbState(page)).sessions.map((s) => s.id)).toEqual([
    "original-0",
  ]);
});

test("completed rounds show live results immediately and never start a second pass without a click", async ({
  page,
}) => {
  await setup(page);
  await page.evaluate(() =>
    (window as any).__completeRound(
      (window as any).__makeSession("new-round", "2026-02-01T00:00:00.000Z"),
    ),
  );
  await expect(
    page.getByRole("heading", { name: "Review", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".review-count.punch-1 strong")).toHaveText("1");
  await expect(page.locator(".detection-heading")).toContainText(
    "Original detections",
  );
  await expect(
    page.getByRole("checkbox", { name: "Show tracking", exact: true }),
  ).toBeChecked();
  await expect(page.locator("select,details")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Analyze recording", exact: true }),
  ).not.toBeVisible();
  await expect.poll(async () => (await dbState(page)).sessions.length).toBe(2);
  expect(await page.evaluate(() => (window as any).__analysisJobs.length)).toBe(
    0,
  );
  expect((await dbState(page)).analyses).toEqual([]);
  await page
    .getByRole("button", { name: "Re-run analysis", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
  await finish(page, 0);
  await expect.poll(async () => (await dbState(page)).analyses.length).toBe(1);
  await page.reload();
  await openReview(page);
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
  await expect(
    page.getByRole("button", { name: "Analyze recording", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
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
  await expect(
    page.getByRole("button", { name: "Analyze recording", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
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
  await expect(page.locator(".review-count.punch-1 strong")).toHaveText("1");
  await page
    .getByRole("button", { name: "Show video analysis", exact: true })
    .click();
  await expect(page.locator(".detection-heading")).toContainText(
    "Partial video analysis",
  );
  await expect(page.locator(".review-count.punch-2 strong")).toHaveText("1");
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
  await expect(page.locator(".review-count.punch-1 strong")).toHaveText("1");
  await expect(page.locator(".review-count.punch-2 strong")).toHaveText("1");
  await expect(
    page.getByRole("heading", { name: "Combinations", exact: true }),
  ).toHaveCount(0);
  await expect(page.getByText(/Uncertain tracking ·/)).toBeVisible();
});

test("zero-length failed import is preserved without starting automatic analysis", async ({
  page,
}) => {
  await setup(page);
  await page.evaluate(() => {
    const w = window as any;
    const failed = w.__makeSession("failed-import", "2026-02-02T00:00:00.000Z");
    failed.source = "file";
    failed.durationMs = 0;
    failed.frames = [];
    failed.events = [];
    failed.annotations = [];
    w.__completeRound(failed);
  });
  await expect(
    page.getByRole("heading", { name: "Review", exact: true }),
  ).toBeVisible();
  await expect.poll(async () => (await dbState(page)).sessions.length).toBe(2);
  await page
    .getByRole("button", { name: "Re-run analysis", exact: true })
    .click();
  // Cache loading must never start inference.
  await expect(
    page.getByRole("button", { name: "Analyze recording", exact: true }),
  ).toBeEnabled();
  expect(await page.evaluate(() => (window as any).__analysisJobs.length)).toBe(
    0,
  );
  await expect(
    page.getByText("Analyzing locally…", { exact: true }),
  ).toHaveCount(0);
  await expect(page.locator(".analysis-error")).toHaveCount(0);
  await page.getByText("Export & details", { exact: true }).click();
  const evidence = await exported(page, "Evidence JSON");
  expect(evidence).toMatchObject({
    id: "failed-import",
    source: "file",
    durationMs: 0,
    frames: [],
    events: [],
    annotations: [],
  });
  const stored = (await dbState(page)).sessions.find(
    (s: { id: string }) => s.id === "failed-import",
  );
  expect(stored?.videoBytes).toBe(3);
  expect((await dbState(page)).analyses).toEqual([]);
});

test("a playable recording stays mounted and visible across optional analysis and result switches", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1512, height: 768 });
  await setup(page, 12);
  await page.evaluate(async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 360;
    const context = canvas.getContext("2d")!;
    const stream = canvas.captureStream(15);
    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
    const done = new Promise<Blob>((resolve) => {
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = () => resolve(new Blob(chunks, { type: "video/webm" }));
    });
    recorder.start();
    for (let i = 0; i < 18; i++) {
      context.fillStyle = "#213d55";
      context.fillRect(0, 0, 640, 360);
      context.fillStyle = "#80d8ff";
      context.fillRect(100 + i * 10, 80, 60, 150);
      await new Promise((resolve) => setTimeout(resolve, 70));
    }
    recorder.stop();
    const video = await done;
    stream.getTracks().forEach((track) => track.stop());
    const w = window as any;
    w.__completeRound({
      ...w.__makeSession("playable", "2026-02-01T00:00:00.000Z"),
      video,
    });
  });
  const video = page.locator(".replay-stage video");
  await expect(video).toHaveJSProperty("readyState", 4);
  await expect(
    page.getByRole("button", { name: "Play replay", exact: true }),
  ).toBeInViewport({ ratio: 1 });
  await expect(video).toBeInViewport({ ratio: 1 });
  await video.evaluate((node) => {
    (window as any).__originalPlayer = node;
    (window as any).__originalUrl = (node as HTMLVideoElement).src;
    (node as HTMLVideoElement).currentTime = 0.4;
  });
  await expect(page.getByLabel("Replay position")).toHaveValue("400");
  await expect(page.locator(".session-item")).toHaveCount(13);
  await expect(page.locator(".session-item").first()).toContainText(
    "1 detected",
  );
  await page.locator(".session-card").nth(1).locator(".session-remove").click();
  await expect(page.locator(".session-item")).toHaveCount(12);
  await expect(page.getByLabel("Replay position")).toHaveValue("400");
  expect(
    await video.evaluate(
      (node) =>
        node === (window as any).__originalPlayer &&
        (node as HTMLVideoElement).src === (window as any).__originalUrl,
    ),
  ).toBe(true);
  expect(await page.evaluate(() => (window as any).__analysisJobs.length)).toBe(
    0,
  );
  await page
    .getByRole("button", { name: "Re-run analysis", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Analyze recording", exact: true })
    .click();
  await finish(page, 0);
  await page
    .getByRole("button", { name: "Show video analysis", exact: true })
    .click();
  await expect(video).toBeInViewport({ ratio: 1 });
  await expect(video).toHaveJSProperty("readyState", 4);
  expect(
    await video.evaluate(
      (node) =>
        node === (window as any).__originalPlayer &&
        (node as HTMLVideoElement).src === (window as any).__originalUrl,
    ),
  ).toBe(true);
  await expect(page.getByLabel("Replay position")).toHaveValue("400");
  await page
    .getByRole("button", { name: "Show original", exact: true })
    .click();
  await page
    .getByRole("checkbox", { name: "Show tracking", exact: true })
    .uncheck();
  await page
    .getByRole("checkbox", { name: "Show tracking", exact: true })
    .check();
  expect(
    await video.evaluate(
      (node) =>
        node === (window as any).__originalPlayer &&
        (node as HTMLVideoElement).src === (window as any).__originalUrl,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "Play replay", exact: true }).click();
  await expect(video).toHaveJSProperty("paused", false);
  await page.getByRole("button", { name: "Pause replay", exact: true }).click();
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  await expect(video).toBeInViewport({ ratio: 1 });
  await expect(
    page.getByRole("button", { name: "Play replay", exact: true }),
  ).toBeInViewport({ ratio: 1 });
  await expect(
    page.getByRole("checkbox", { name: "Show tracking", exact: true }),
  ).toBeInViewport({ ratio: 1 });
  const strip = await page.locator(".round-library").boundingBox();
  expect(strip!.height).toBeLessThan(90);
  await page.screenshot({
    path: test.info().outputPath("replay-stable-desktop.png"),
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: test.info().outputPath("replay-stable-mobile.png"),
  });
  const beforeRemoval = await dbState(page);
  const savedVideo = beforeRemoval.sessions.find((s) => s.id === "playable");
  const savedReport = beforeRemoval.analyses.find(
    (r) => r.sourceSessionId === "playable",
  );
  await page.locator(".session-card.is-selected .session-remove").click();
  await expect(page.locator(".session-item")).toHaveCount(11);
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await page
    .getByRole("button", { name: "Recently deleted (2)", exact: true })
    .click();
  await page
    .locator(".trashed-round")
    .first()
    .getByRole("button", { name: "Restore", exact: true })
    .click();
  await expect(video).toHaveJSProperty("readyState", 4);
  await page.getByRole("button", { name: "Play replay", exact: true }).click();
  await expect(video).toHaveJSProperty("paused", false);
  await page.getByRole("button", { name: "Pause replay", exact: true }).click();
  const restored = await dbState(page);
  expect(restored.sessions.find((s) => s.id === "playable")).toEqual(
    savedVideo,
  );
  expect(
    restored.analyses.find((r) => r.sourceSessionId === "playable"),
  ).toEqual(savedReport);
  expect(await page.evaluate(() => (window as any).__analysisJobs.length)).toBe(
    0,
  );
});
