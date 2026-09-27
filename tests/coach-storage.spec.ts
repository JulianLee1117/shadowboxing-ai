import { expect, test, type Page } from "@playwright/test";
import type { Session } from "../src/lib/types";
import type { CoachReviewData } from "../src/lib/coachReview";

interface StorageFixture {
  session: Session;
  review: CoachReviewData;
  storage: typeof import("../src/lib/storage");
  coach: typeof import("../src/lib/coachReview");
}
declare global {
  interface Window {
    coachStorageFixture: StorageFixture;
  }
}

// Exercise real IDB transactions without recording or decoding any media.
async function prepare(page: Page) {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      throw new Error("Physical camera forbidden in storage tests");
    };
  });
  await page.goto("/");
  await page.evaluate(async () => {
    const storagePath = "/src/lib/storage.ts";
    const coachPath = "/src/lib/coachReview.ts";
    const fingerprintPath = "/src/lib/mediaFingerprint.ts";
    const storage: StorageFixture["storage"] = await import(storagePath);
    const coach: StorageFixture["coach"] = await import(coachPath);
    const { fingerprintVideo } = await import(fingerprintPath);
    const session: Session = {
      id: "atomic-coach-storage",
      createdAt: "2026-01-01T00:00:00.000Z",
      schemaVersion: "1.0",
      source: "file",
      stance: "orthodox",
      model: "full",
      drill: "open",
      durationMs: 2000,
      videoOffsetMs: 150,
      video: new Blob(["immutable-original-video"], { type: "video/webm" }),
      frames: [
        { t: 500, width: 640, height: 360, landmarks: [], inferenceMs: 5 },
      ],
      events: [
        {
          id: "proposal-0",
          hand: "left",
          role: "lead",
          label: "jab",
          startMs: 400,
          peakMs: 550,
          endMs: 700,
          detectedAtMs: 850,
          score: 0.8,
          guardReturn: "unassessable",
          extension: null,
          experimental: true,
        },
      ],
      annotations: [],
      measuredFps: 30,
      inferenceP95: 5,
      skippedFrames: 0,
      detectorVersion: "storage-fixture",
      modelManifest: { version: "fixture-1" },
      capture: {
        width: 640,
        height: 360,
        timingSource: "source",
        delegate: "CPU",
      },
    };
    await storage.saveSession(session);
    const initial = coach.createCoachReview(
      session,
      await fingerprintVideo(session.video),
      await coach.fingerprintCoachSource(session),
    );
    const review = coach.answerIdentity(
      initial,
      initial.cards[0].id,
      "left-hook",
      "first view",
    );
    window.coachStorageFixture = { session, review, storage, coach };
  });
}

test("coach patch retains newer annotations and the stored original evidence", async ({
  page,
}) => {
  await prepare(page);
  const result = await page.evaluate(async () => {
    const { session, review, storage, coach } = window.coachStorageFixture;
    const latest: Session = {
      ...session,
      annotations: [
        {
          id: "newer-label",
          startMs: 350,
          endMs: 750,
          label: "hook",
          hand: "left",
          note: "another view",
        },
      ],
      annotationsComplete: true,
    };
    // Each API opens its own connection. The caller still holds the older round.
    await storage.saveSession(latest, { requireExisting: true });
    const merged = await storage.saveCoachReview(session, review, undefined);
    const [stored] = await storage.listSessions();
    return {
      annotations: merged.annotations,
      annotationsComplete: merged.annotationsComplete,
      storedMatchesReturned: JSON.stringify(stored) === JSON.stringify(merged),
      evidenceUnchanged:
        coach.serializeCoachSource(stored) ===
        coach.serializeCoachSource(session),
      originalBytes: await stored.video!.text(),
      identity: stored.coachReview!.cards[0].identity,
    };
  });
  expect(result.annotations).toEqual([
    {
      id: "newer-label",
      startMs: 350,
      endMs: 750,
      label: "hook",
      hand: "left",
      note: "another view",
    },
  ]);
  expect(result.annotationsComplete).toBe(true);
  expect(result.storedMatchesReturned).toBe(true);
  expect(result.evidenceUnchanged).toBe(true);
  expect(result.originalBytes).toBe("immutable-original-video");
  expect(result.identity).toBe("left-hook");
});

test("concurrent coach writers commit once and reject the stale draft", async ({
  page,
}) => {
  await prepare(page);
  const result = await page.evaluate(async () => {
    const { session, review, storage, coach } = window.coachStorageFixture;
    const initial = await storage.saveCoachReview(session, review, undefined);
    const first = coach.answerIdentity(
      review,
      review.cards[0].id,
      "left-uppercut",
      "writer one",
    );
    const second = coach.answerIdentity(
      review,
      review.cards[0].id,
      "not-punch",
      "writer two",
    );
    const results = await Promise.allSettled([
      storage.saveCoachReview(initial, first, review),
      storage.saveCoachReview(initial, second, review),
    ]);
    const [stored] = await storage.listSessions();
    const success = results.find((r) => r.status === "fulfilled");
    const rejected = results.find((r) => r.status === "rejected");
    return {
      statuses: results.map((r) => r.status).sort(),
      error: rejected?.status === "rejected" ? String(rejected.reason) : null,
      stored: stored.coachReview,
      committed:
        success?.status === "fulfilled" ? success.value.coachReview : null,
    };
  });
  expect(result.statuses).toEqual(["fulfilled", "rejected"]);
  expect(result.error).toMatch(/changed in another view/);
  expect(result.stored).toEqual(result.committed);
});

test("stale annotation writes preserve coach labels committed by another view", async ({
  page,
}) => {
  await prepare(page);
  const result = await page.evaluate(async () => {
    const { session, review, storage, coach } = window.coachStorageFixture;
    const originalCoach = await storage.saveCoachReview(
      session,
      review,
      undefined,
    );
    const latestReview = coach.answerIdentity(
      review,
      review.cards[0].id,
      "right-uppercut",
      "newer coach edit",
    );
    await storage.saveCoachReview(originalCoach, latestReview, review);
    // Both a snapshot without labels and one carrying an older coach edit must
    // retain the current labels. Each update uses an independent DB connection.
    const checks = [];
    for (const stale of [session, originalCoach]) {
      const annotationEdit: Session = {
        ...stale,
        annotations: [
          {
            id: "annotation-after-coach",
            startMs: 400,
            endMs: 700,
            label: "uppercut",
            hand: "right",
            note: "new annotation",
          },
        ],
        annotationsComplete: true,
      };
      const returned = await storage.saveSession(annotationEdit, {
        requireExisting: true,
      });
      const [stored] = await storage.listSessions();
      checks.push({
        returnedReview: returned.coachReview,
        storedReview: stored.coachReview,
        annotations: stored.annotations,
        complete: stored.annotationsComplete,
        sourceUnchanged:
          coach.serializeCoachSource(stored) ===
          coach.serializeCoachSource(session),
        videoBytes: await stored.video!.text(),
      });
    }
    return { checks, latestReview };
  });
  for (const check of result.checks) {
    expect(check.returnedReview).toEqual(result.latestReview);
    expect(check.storedReview).toEqual(result.latestReview);
    expect(check.annotations).toEqual([
      {
        id: "annotation-after-coach",
        startMs: 400,
        endMs: 700,
        label: "uppercut",
        hand: "right",
        note: "new annotation",
      },
    ]);
    expect(check.complete).toBe(true);
    expect(check.sourceUnchanged).toBe(true);
    expect(check.videoBytes).toBe("immutable-original-video");
  }
});

test("coach patch rejects changed source context and Blob metadata", async ({
  page,
}) => {
  await prepare(page);
  const result = await page.evaluate(async () => {
    const { session, review, storage } = window.coachStorageFixture;
    const mutations: Array<[string, (s: Session) => void]> = [
      [
        "capture time",
        (s) => {
          s.createdAt = "2026-01-02T00:00:00.000Z";
        },
      ],
      [
        "source",
        (s) => {
          s.source = "camera";
        },
      ],
      [
        "stance",
        (s) => {
          s.stance = "southpaw";
        },
      ],
      [
        "drill",
        (s) => {
          s.drill = "different";
        },
      ],
      [
        "model",
        (s) => {
          s.model = "heavy";
        },
      ],
      [
        "duration",
        (s) => {
          s.durationMs += 1;
        },
      ],
      [
        "video offset",
        (s) => {
          s.videoOffsetMs = 200;
        },
      ],
      [
        "detector",
        (s) => {
          s.detectorVersion = "changed";
        },
      ],
      [
        "model provenance",
        (s) => {
          s.modelManifest = { version: "changed" };
        },
      ],
      [
        "capture",
        (s) => {
          s.capture!.width = 1280;
        },
      ],
      [
        "frames",
        (s) => {
          s.frames[0].t += 1;
        },
      ],
      [
        "events",
        (s) => {
          s.events[0].peakMs += 1;
        },
      ],
      [
        "video size",
        (s) => {
          s.video = new Blob(["different"], { type: "video/webm" });
        },
      ],
      [
        "video type",
        (s) => {
          s.video = new Blob(["immutable-original-video"], {
            type: "video/mp4",
          });
        },
      ],
    ];
    const checks = [];
    for (const [name, mutate] of mutations) {
      const newer = structuredClone(session);
      mutate(newer);
      await storage.saveSession(newer, { requireExisting: true });
      let error = "";
      try {
        await storage.saveCoachReview(session, review, undefined);
      } catch (reason) {
        error = String(reason);
      }
      const [stored] = await storage.listSessions();
      checks.push({
        name,
        error,
        noLabelsWritten: stored.coachReview === undefined,
      });
    }
    return checks;
  });
  expect(result).toHaveLength(14);
  for (const check of result) {
    expect(check.error, check.name).toMatch(/original evidence changed/);
    expect(check.noLabelsWritten, check.name).toBe(true);
  }
});

test("coach patch never recreates a round removed in another view", async ({
  page,
}) => {
  await prepare(page);
  const result = await page.evaluate(async () => {
    const { session, review, storage } = window.coachStorageFixture;
    await storage.moveSessionToTrash(session.id);
    let error = "";
    try {
      await storage.saveCoachReview(session, review, undefined);
    } catch (reason) {
      error = String(reason);
    }
    return {
      error,
      visible: (await storage.listSessions()).length,
      trash: (await storage.listTrashedRounds()).length,
    };
  });
  expect(result.error).toMatch(/Restore this round/);
  expect(result.visible).toBe(0);
  expect(result.trash).toBe(1);
});
