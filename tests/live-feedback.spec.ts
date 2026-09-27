import { expect, test } from "@playwright/test";

test("large live feedback and focus controls work without opening a camera", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1512, height: 823 });
  await page.addInitScript(() => {
    const state = window as typeof window & { cameraCalls: number };
    state.cameraCalls = 0;
    navigator.mediaDevices.getUserMedia = async () => {
      state.cameraCalls += 1;
      throw new Error("This test must not open a camera");
    };
  });
  await page.goto("/");
  const stage = page.locator(".camera-stage");
  const initial = await stage.boundingBox();
  expect(initial?.width).toBeGreaterThan(1400);
  expect(initial?.height).toBeGreaterThan(550);
  await page.screenshot({
    path: test.info().outputPath("practice-desktop.png"),
  });
  await page.getByText("More options", { exact: true }).click();
  await page.getByRole("button", { name: "Try demo", exact: true }).click();
  await page.getByRole("button", { name: "Start demo", exact: true }).click();
  await expect(page.getByTestId("live-punch-count")).toHaveText("0");
  await expect(page.locator(".live-punch-name")).toHaveText("Find your rhythm");
  await expect(
    page.getByRole("button", { name: "Stop & save", exact: true }),
  ).toBeInViewport({ ratio: 1 });
  await page.getByRole("button", { name: "Focus view", exact: true }).click();
  await expect(page.locator(".practice-arena")).toHaveClass(/is-focused/);
  await expect(
    page.getByRole("timer", { name: "Round time remaining" }),
  ).toBeVisible();
  await expect
    .poll(async () =>
      Number(await page.getByTestId("live-punch-count").textContent()),
    )
    .toBeGreaterThan(0);
  await expect(page.locator(".live-punch-name.detected")).toBeVisible();
  await expect(page.locator(".live-punch-identity")).toHaveText(
    /^(Left|Right) hand · (Lead|Rear)$/,
  );
  expect(
    await page
      .locator(".live-punch-name")
      .evaluate((el) => parseFloat(getComputedStyle(el).fontSize)),
  ).toBeGreaterThanOrEqual(60);
  await page.screenshot({
    path: test.info().outputPath("practice-focused-desktop.png"),
  });
  await page.getByRole("button", { name: "Stop & save", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Review", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".practice-arena")).not.toHaveClass(/is-focused/);
  expect(await page.evaluate(() => document.fullscreenElement)).toBeNull();
  expect(
    await page.evaluate(
      () => (window as typeof window & { cameraCalls: number }).cameraCalls,
    ),
  ).toBe(0);
  await page.getByRole("button", { name: "Practice", exact: true }).click();
  await page.getByText("More options", { exact: true }).click();
  await page.getByRole("button", { name: "Try demo", exact: true }).click();
  await page.getByRole("button", { name: "Start demo", exact: true }).click();
  await expect(page.getByTestId("live-punch-count")).toHaveText("0");
  await expect(page.locator(".live-punch-name")).toHaveText("Find your rhythm");
  await expect(page.locator(".live-punch-name.detected")).toHaveCount(0);
  await page.getByRole("button", { name: "Stop & save", exact: true }).click();
});

test("mobile practice and focus view stay within the viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.addInitScript(() => {
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: undefined,
    });
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Enable camera", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.getByRole("button", { name: "Focus view", exact: true }).click();
  await expect(page.locator(".practice-arena")).toHaveClass(/is-focused/);
  await page.screenshot({
    path: test.info().outputPath("practice-focused-mobile.png"),
  });
  await page
    .getByRole("button", { name: "Exit focus view", exact: true })
    .focus();
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("button", { name: "Enable camera", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(
    page.getByRole("button", { name: "Exit focus view", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.locator(".practice-arena")).not.toHaveClass(/is-focused/);
  await page.getByRole("button", { name: "Focus view", exact: true }).click();
  await page
    .getByRole("button", { name: "Exit focus view", exact: true })
    .click();
  await expect(page.locator(".practice-arena")).not.toHaveClass(/is-focused/);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
});

test("an untracked round explains missing detections while keeping recording controls visible", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1512, height: 823 });
  // Replace only synthetic pose input. The actual app and recording lifecycle
  // remain intact; no physical camera or model inference is used here.
  await page.route("**/src/lib/demo.ts*", (route) =>
    route.fulfill({
      contentType: "application/javascript",
      body: "export function demoFrame(t) { return { t, width: 1280, height: 720, landmarks: [], inferenceMs: 0 }; }",
    }),
  );
  await page.goto("/");
  await page.getByText("More options", { exact: true }).click();
  await page.getByRole("button", { name: "Try demo", exact: true }).click();
  await page.getByRole("button", { name: "Start demo", exact: true }).click();
  await expect(page.getByLabel("Arm tracking")).toContainText("L · uncertain");
  await expect(page.getByLabel("Arm tracking")).toContainText("R · uncertain");
  await expect(page.locator(".live-punch-name")).toHaveText("Tracking unclear");
  await expect(page.locator(".live-punch-identity")).toHaveText(
    "Keep your arms and torso in view",
  );
  await expect(
    page.getByRole("timer", { name: "Round time remaining" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Stop & save", exact: true }),
  ).toBeInViewport({ ratio: 1 });
  const tracking = await page.getByLabel("Arm tracking").boundingBox();
  const timer = await page
    .getByRole("timer", { name: "Round time remaining" })
    .boundingBox();
  const feedback = await page.getByLabel("Live punch feedback").boundingBox();
  expect(tracking!.x).toBeGreaterThan(timer!.x + timer!.width);
  expect(tracking!.y + tracking!.height).toBeLessThan(feedback!.y);
  await page.screenshot({
    path: test.info().outputPath("practice-tracking-uncertain.png"),
  });
  await page.getByRole("button", { name: "Stop & save", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Review", exact: true }),
  ).toBeVisible();
});
