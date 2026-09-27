import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { demoFrame } from "../src/lib/demo";
import type { PunchEvent, Session } from "../src/lib/types";

const origin = "http://127.0.0.1:5173";
const names = [
  "Jab",
  "Cross",
  "Lead hook",
  "Rear hook",
  "Lead uppercut",
  "Rear uppercut",
];
// Synthetic display fixture, deliberately southpaw: role is not physical hand.
// It tests presentation and evidence preservation, not detector accuracy.
const events: PunchEvent[] = [
  "jab",
  "cross",
  "hook",
  "hook",
  "uppercut",
  "uppercut",
].map((label, i) => ({
  id: `fixture-${i}`,
  label: label as PunchEvent["label"],
  role: i % 2 === 0 ? "lead" : "rear",
  hand: i % 2 === 0 ? "right" : "left",
  startMs: 1000 + i * 1200,
  peakMs: 1250 + i * 1200,
  endMs: 1500 + i * 1200,
  detectedAtMs: 1540 + i * 1200,
  extension: 0.6,
  score: 0.7,
  guardReturn: "returned",
  experimental: true,
}));
const fixture: Session = {
  schemaVersion: "1.0",
  id: "six-punch-review-fixture",
  createdAt: "2026-01-01T00:00:00.000Z",
  source: "demo",
  stance: "southpaw",
  model: "synthetic",
  drill: "free",
  durationMs: 10000,
  frames: Array.from({ length: 251 }, (_, i) => demoFrame(i * 40)),
  events,
  annotations: [],
  measuredFps: 25,
  inferenceP95: 0,
  skippedFrames: 0,
  detectorVersion: "fixture-saved-events",
};

async function openFixture(page: Page) {
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    if (url.pathname === "/__six_punch_seed")
      return route.fulfill({
        contentType: "text/html",
        body: "<!doctype html><title>Synthetic fixture</title>",
      });
    return route.continue();
  });
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      throw new Error("Physical camera forbidden in review UI tests");
    };
  });
  await page.goto(`${origin}/__six_punch_seed`);
  await page.evaluate(async (session) => {
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("corner-local-v1", 2);
      request.onupgradeneeded = () => {
        request.result.createObjectStore("sessions", { keyPath: "id" });
        request.result.createObjectStore("analyses", {
          keyPath: "sourceSessionId",
        });
      };
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction("sessions", "readwrite");
        tx.objectStore("sessions").put(session);
        tx.oncomplete = () => {
          db.close();
          resolve();
        };
        tx.onerror = () => reject(tx.error);
      };
    });
  }, fixture);
  await page.goto(origin);
  await page.getByRole("button", { name: /^Saved rounds/ }).click();
  await expect(
    page.getByRole("heading", { name: "6 detected punches", exact: true }),
  ).toBeVisible();
}

test("review preserves six punch identities, physical hands and original evidence", async ({
  page,
}) => {
  await openFixture(page);
  const counts = page.locator(".review-punch-counts .review-count");
  await expect(counts).toHaveCount(6);
  for (let i = 0; i < names.length; i++) {
    await expect(counts.nth(i)).toContainText(names[i]);
    await expect(counts.nth(i).locator("strong")).toHaveText("1");
  }
  await expect(page.getByLabel("Detected punches")).toHaveCount(0);
  await page.getByLabel("Show detections").check();
  const cards = page.getByLabel("Detected punches").getByRole("button");
  await expect(cards).toHaveCount(6);
  await expect(cards.locator(".review-punch-number")).toHaveText([
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
  ]);
  for (let i = 0; i < names.length; i++) {
    await expect(cards.nth(i).locator("strong")).toHaveText(names[i]);
    await expect(cards.nth(i)).toContainText(events[i].hand);
  }
  await page.getByRole("button", { name: "Play replay", exact: true }).click();
  await cards.nth(2).click();
  await expect(page.getByLabel("Replay position")).toHaveValue("3400");
  const currentPunch = page.getByLabel("Current detected punches");
  await expect(currentPunch).toContainText("Lead hook");
  await expect(currentPunch).toContainText("right hand");
  await expect(
    page.getByRole("button", { name: "Play replay", exact: true }),
  ).toBeVisible();
  await page
    .getByLabel("Punch timeline")
    .getByRole("button", {
      name: "Rear uppercut at 7.25 seconds",
      exact: true,
    })
    .click();
  await expect(page.getByLabel("Replay position")).toHaveValue("7000");
  await expect(currentPunch).toContainText("Rear uppercut");
  await expect(currentPunch).toContainText("left hand");
  await page.getByRole("button", { name: "Next frame", exact: true }).click();
  await expect(page.getByLabel("Replay position")).toHaveValue("7040");
  await expect(currentPunch).toContainText("Rear uppercut");
  await page.getByLabel("Replay position").focus();
  await page.keyboard.press("End");
  await expect(page.getByLabel("Replay position")).toHaveValue("10000");
  await expect(currentPunch).toHaveCount(0);
  await cards.nth(0).click();
  await expect(currentPunch).toContainText("Jab");
  await page.getByLabel("Show detections").uncheck();
  await expect(currentPunch).toHaveCount(0);
  await page.getByText("Export & details", { exact: true }).click();
  const downloadPromise = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Evidence JSON", exact: true })
    .click();
  const download = await downloadPromise;
  const saved = JSON.parse(await readFile((await download.path())!, "utf8"));
  expect(saved.events).toEqual(events);
  expect(saved.stance).toBe("southpaw");
  expect(saved.detectorVersion).toBe(fixture.detectorVersion);
});

test("review focus keeps replay controls and restores layout after native or fallback exit", async ({
  page,
}) => {
  await openFixture(page);
  const player = page.locator(".review-player");
  await page.getByLabel("Show detections").check();
  await page
    .getByLabel("Punch timeline")
    .getByRole("button", { name: "Rear uppercut at 7.25 seconds", exact: true })
    .click();
  await page.getByRole("button", { name: "Focus video", exact: true }).click();
  await expect(
    page.getByRole("dialog", { name: "Round replay" }),
  ).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          document.fullscreenElement?.classList.contains("review-player") ??
          false,
      ),
    )
    .toBe(true);
  await page.getByRole("button", { name: "Next frame", exact: true }).click();
  await expect(page.getByLabel("Replay position")).toHaveValue("7040");
  await expect(player.getByLabel("Current detected punches")).toContainText(
    "Rear uppercut",
  );
  await expect(player.getByLabel("Current detected punches")).toContainText(
    "left hand",
  );
  await page.getByLabel("Show detections").uncheck();
  await expect(player.getByLabel("Current detected punches")).toHaveCount(0);
  await page.getByRole("button", { name: "Exit focus", exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => document.fullscreenElement === null))
    .toBe(true);
  await expect(player).not.toHaveClass(/is-focused/);

  await page.evaluate(() => {
    HTMLElement.prototype.requestFullscreen = async () => {
      throw new Error("Fullscreen unavailable in fallback fixture");
    };
  });
  await page.getByRole("button", { name: "Focus video", exact: true }).click();
  await expect(player).toHaveClass(/is-focused/);
  await page.keyboard.press("Escape");
  await expect(player).not.toHaveClass(/is-focused/);
  expect(await page.evaluate(() => document.body.style.overflow)).toBe("");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  await expect(
    page.getByRole("button", { name: "Focus video", exact: true }),
  ).toBeVisible();
});
