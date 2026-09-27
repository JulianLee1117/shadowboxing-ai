import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import type { Session } from "../src/lib/types";
import type { CoachReviewData } from "../src/lib/coachReview";

// A real, seekable canvas recording in an isolated browser context. This is
// UI/storage evidence only; the image and event proposals are synthetic.
async function openCoachFixture(page: Page) {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      throw new Error("Physical camera forbidden in coach-review tests");
    };
  });
  await page.goto("/");
  await page.evaluate(async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext("2d")!;
    let frame = 0;
    const draw = () => {
      ctx.fillStyle = "#182d26";
      ctx.fillRect(0, 0, 640, 360);
      ctx.fillStyle = "#dae7d5";
      ctx.font = "22px sans-serif";
      ctx.fillText("Synthetic playback fixture", 180, 150);
      ctx.fillText(String(frame++), 300, 200);
    };
    draw();
    const stream = canvas.captureStream(20);
    const recorder = new MediaRecorder(stream, {
      mimeType: "video/webm;codecs=vp8",
    });
    const chunks: BlobPart[] = [];
    const recorded = new Promise<Blob>((resolve, reject) => {
      recorder.ondataavailable = (event) => chunks.push(event.data);
      recorder.onstop = () =>
        resolve(new Blob(chunks, { type: recorder.mimeType }));
      recorder.onerror = () => reject(new Error("Fixture recording failed"));
    });
    const interval = setInterval(draw, 50);
    recorder.start(100);
    await new Promise((resolve) => setTimeout(resolve, 2800));
    recorder.stop();
    const video = await recorded;
    clearInterval(interval);
    stream.getTracks().forEach((track) => track.stop());
    const session: Session = {
      schemaVersion: "1.0",
      id: "coach-fixture",
      createdAt: "2026-01-01T00:00:00Z",
      source: "file",
      stance: "orthodox",
      model: "full",
      drill: "open",
      durationMs: 2400,
      videoOffsetMs: 150,
      video,
      frames: [],
      events: [400, 1500].map((startMs, i) => ({
        id: `proposal-${i}`,
        hand: i ? "right" : "left",
        role: i ? "rear" : "lead",
        label: i ? "cross" : "jab",
        startMs,
        peakMs: startMs + 150,
        endMs: startMs + 300,
        detectedAtMs: startMs + 450,
        score: 0.8,
        guardReturn: "unassessable",
        extension: 0,
        experimental: true,
      })),
      annotations: [],
      measuredFps: 20,
      inferenceP95: 0,
      skippedFrames: 0,
      detectorVersion: "synthetic-proposals",
    };
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("corner-local-v1");
      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction("sessions", "readwrite");
        tx.objectStore("sessions").put(session);
        tx.oncomplete = () => {
          db.close();
          resolve();
        };
        tx.onerror = () => {
          db.close();
          reject(tx.error);
        };
      };
      request.onerror = () => reject(request.error);
    });
  });
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await page
    .getByRole("button", { name: "Correct punches", exact: true })
    .click();
  await expect(
    page.getByRole("group", { name: "Actual punch identity" }),
  ).toBeVisible();
  await expect
    .poll(() =>
      page
        .locator(".coach-player video")
        .evaluate((v: HTMLVideoElement) => v.readyState),
    )
    .toBeGreaterThanOrEqual(2);
}

async function storedSession(page: Page) {
  return page.evaluate(
    () =>
      new Promise<Session>((resolve, reject) => {
        const request = indexedDB.open("corner-local-v1");
        request.onsuccess = () => {
          const db = request.result;
          const get = db
            .transaction("sessions")
            .objectStore("sessions")
            .get("coach-fixture");
          get.onsuccess = () => {
            db.close();
            resolve(get.result);
          };
          get.onerror = () => {
            db.close();
            reject(get.error);
          };
        };
        request.onerror = () => reject(request.error);
      }),
  );
}

test("coach cards persist judgments, undo and missed moments without changing original evidence", async ({
  page,
}) => {
  await openCoachFixture(page);
  const original = await storedSession(page);
  const originalMediaUrl = await page
    .locator(".coach-player video")
    .getAttribute("src");
  await page.getByRole("textbox", { name: "Coach notes" }).fill("Draft cue");
  await expect(
    page.getByRole("button", { name: "Practice", exact: true }),
  ).toBeDisabled();
  await page.getByRole("link", { name: "Corner home" }).click();
  await expect(page.getByRole("textbox", { name: "Coach notes" })).toHaveValue(
    "Draft cue",
  );
  await page.getByRole("button", { name: "Discard note", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Practice", exact: true }),
  ).toBeEnabled();
  const identity = page.getByRole("group", { name: "Actual punch identity" });
  const guard = page.getByRole("group", { name: "Non-punching hand guard" });
  await expect(
    page.locator(".pose-overlay, .replay-detection, select, details"),
  ).toHaveCount(0);
  await expect(identity.locator('[aria-pressed="true"]')).toHaveCount(0);
  await expect(guard).toHaveCount(0);
  await page
    .getByRole("button", { name: "Play coaching clip", exact: true })
    .click();
  await expect
    .poll(() =>
      page
        .locator(".coach-player video")
        .evaluate((v: HTMLVideoElement) => v.currentTime),
    )
    .toBeGreaterThan(0.2);
  await page
    .getByRole("button", { name: "Pause coaching clip", exact: true })
    .click();
  await identity.getByRole("button", { name: "Left jab", exact: true }).click();
  await expect(guard).toBeVisible();
  await page
    .getByRole("button", { name: "Change action", exact: true })
    .click();
  await identity.getByRole("button", { name: "Left jab", exact: true }).click();
  await expect(guard).toBeVisible();
  await expect(page.locator(".coach-confirmed-identity")).toContainText(
    "Left jab",
  );
  await guard.getByRole("button", { name: "Guard held", exact: true }).click();
  await expect(page.locator(".coach-progress")).toContainText("Clip 2 of 2");
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(page.locator(".coach-progress")).toContainText("Clip 1 of 2");
  await expect(page.locator(".coach-player video")).toHaveAttribute(
    "src",
    originalMediaUrl!,
  );
  await expect(guard.locator('[aria-pressed="true"]')).toHaveCount(0);
  await guard
    .getByRole("button", { name: "Not this drill/style", exact: true })
    .click();
  await expect(page.locator(".coach-progress")).toContainText("Clip 2 of 2");
  await identity
    .getByRole("button", { name: "Right hook", exact: true })
    .click();
  await expect(page.locator(".coach-confirmed-identity")).toContainText(
    "Right hook",
  );
  await expect(guard).toHaveCount(0);
  await page
    .getByRole("button", { name: "Find a missed punch", exact: true })
    .click();
  await page.getByRole("slider", { name: "Full video position" }).focus();
  await page.keyboard.press("End");
  await page
    .getByRole("button", { name: "Review this moment", exact: true })
    .click();
  await expect(page.locator(".coach-progress")).toContainText("Clip 3 of 3");
  await expect(identity.locator('[aria-pressed="true"]')).toHaveCount(0);
  await identity
    .getByRole("button", { name: "Right cross", exact: true })
    .click();
  await guard.getByRole("button", { name: "Can't tell", exact: true }).click();
  await expect(page.locator(".coach-progress")).toContainText(
    "3 actions reviewed",
  );
  const downloadEvent = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export labels", exact: true })
    .click();
  const download = await downloadEvent;
  const exported: CoachReviewData & { exportState: string } = JSON.parse(
    await readFile((await download.path())!, "utf8"),
  );
  expect(exported.schemaVersion).toBe("local-coach-review-2");
  expect(exported.source.videoSha256).toMatch(/^[a-f0-9]{64}$/);
  expect(exported.reviewer).toEqual({
    id: "local-user",
    expertise: "user_review",
  });
  expect(exported.provenance.formJudgments).toBe(
    "provisional_not_correct_form_references",
  );
  expect(exported.cards[0].guard?.answer).toBe("not-applicable");
  expect(exported.cards[1].identity).toBe("right-hook");
  expect(exported.cards[1].proposal.kind).toBe("detector-navigation");
  expect(exported.cards[1].guard).toBeNull();
  expect(exported.cards[2].proposal.kind).toBe("manual-moment");
  expect(exported.cards[2].guard?.answer).toBe("unclear");
  await page
    .getByRole("button", { name: "Close punch review", exact: true })
    .click();
  await expect(page.locator(".replay-stage video")).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await page
    .getByRole("button", { name: "Correct punches", exact: true })
    .click();
  await expect(page.locator(".coach-progress")).toContainText(
    "3 actions reviewed",
  );
  const saved = await storedSession(page);
  expect(saved.events).toEqual(original.events);
  expect(saved.annotations).toEqual(original.annotations);
  const { exportState, ...savedLabels } = exported;
  expect(exportState).toBe("current_review");
  expect(saved.coachReview).toEqual(savedLabels);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: test.info().outputPath("coach-review-mobile.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 1512, height: 900 });
  await page.screenshot({
    path: test.info().outputPath("coach-review-desktop.png"),
    fullPage: true,
  });
});

test("action-only corrections survive reopening without requiring or implying correct form", async ({
  page,
}) => {
  await openCoachFixture(page);
  await expect(page.locator(".coach-intro")).toContainText(
    "do not make your movement a correct-form example",
  );
  await page
    .getByRole("group", { name: "Actual punch identity" })
    .getByRole("button", { name: "Left jab", exact: true })
    .click();
  await expect(page.locator(".coach-progress")).toContainText(
    "1 action reviewed",
  );
  await expect(
    page.getByRole("group", { name: "Non-punching hand guard" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.locator(".coach-progress")).toContainText("Clip 2 of 2");
  await page
    .getByRole("button", { name: "Close punch review", exact: true })
    .click();
  await page.reload();
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await page
    .getByRole("button", { name: "Correct punches", exact: true })
    .click();
  await expect(page.locator(".coach-progress")).toContainText("Clip 2 of 2");
  const review = (await storedSession(page)).coachReview!;
  expect(review.cards[0].identity).toBe("left-straight");
  expect(review.cards[0].guard).toBeNull();
  expect(review.cards[1].identity).toBeNull();
  expect(review.reviewer.expertise).toBe("user_review");
});

test("a failed coach-label write does not commit or advance and can be retried", async ({
  page,
}) => {
  await openCoachFixture(page);
  await page.evaluate(() => {
    const put = IDBObjectStore.prototype.put;
    let fail = true;
    IDBObjectStore.prototype.put = function (
      value: unknown,
      key?: IDBValidKey,
    ) {
      if (fail && this.name === "sessions" && (value as Session).coachReview) {
        fail = false;
        throw new DOMException("Fixture storage full", "QuotaExceededError");
      }
      return key === undefined
        ? put.call(this, value)
        : put.call(this, value, key);
    };
  });
  const identity = page.getByRole("group", { name: "Actual punch identity" });
  await identity
    .getByRole("button", { name: "Right hook", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Retry save", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".coach-progress")).toContainText("Clip 1 of 2");
  await expect(identity.locator('[aria-pressed="true"]')).toHaveCount(0);
  expect((await storedSession(page)).coachReview).toBeUndefined();
  await expect(
    page.getByRole("button", { name: "Practice", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Export unsaved labels", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Retry save", exact: true }).click();
  await expect(page.locator(".coach-progress")).toContainText("Clip 2 of 2");
  expect((await storedSession(page)).coachReview?.cards[0].identity).toBe(
    "right-hook",
  );
});
