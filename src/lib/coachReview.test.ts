import { describe, expect, it, vi } from "vitest";
import {
  addManualCard,
  actionReviewed,
  answerGuard,
  answerIdentity,
  cardComplete,
  clipBounds,
  createCoachReview,
  fingerprintCoachSource,
  identityName,
  identityOptions,
  persistCoachReview,
  saveCardNote,
  validateCoachReview,
  type CoachReviewData,
} from "./coachReview";
import type { Session, PunchEvent } from "./types";

const videoSha = "a".repeat(64),
  sourceSha = "b".repeat(64);
const event: PunchEvent = {
  id: "proposal-1",
  hand: "right",
  role: "rear",
  label: "cross",
  startMs: 100,
  peakMs: 300,
  endMs: 500,
  score: 0.97,
  extension: null,
  guardReturn: "unassessable",
  experimental: true,
};
function fixture(): Session {
  return {
    id: "round-1",
    createdAt: "2026-09-27T00:00:00Z",
    source: "camera",
    stance: "orthodox",
    model: "rtmpose-m",
    drill: "open",
    durationMs: 3000,
    frames: [],
    events: [{ ...event }],
    annotations: [
      {
        id: "reference-kept",
        startMs: 100,
        endMs: 500,
        label: "jab",
        hand: "left",
        note: "Independent reference",
      },
    ],
    video: new Blob(["local video"], { type: "video/webm" }),
    videoOffsetMs: 1200,
    measuredFps: 20,
    inferenceP95: 50,
    skippedFrames: 0,
    schemaVersion: "1.0",
    detectorVersion: "fixture-recognizer",
  };
}
function review(session = fixture()) {
  return createCoachReview(
    session,
    videoSha,
    sourceSha,
    "2026-09-27T00:01:00Z",
  );
}

describe("local coach labels", () => {
  it("attributes new answers to the user without inventing coaching expertise or form gold", () => {
    const session = fixture();
    const data = review(session);
    expect(data.schemaVersion).toBe("local-coach-review-2");
    expect(data.reviewer).toEqual({
      id: "local-user",
      expertise: "user_review",
    });
    expect(data.provenance.formJudgments).toBe(
      "provisional_not_correct_form_references",
    );
    expect(validateCoachReview(data, session)).toBe(data);
    expect(() =>
      validateCoachReview(
        {
          ...data,
          reviewer: {
            id: "local-user-coach",
            expertise: "coach_self_reported",
          },
        },
        session,
      ),
    ).toThrow(/Unsupported/);
    expect(() =>
      validateCoachReview(
        {
          ...data,
          provenance: { ...data.provenance, formJudgments: undefined },
        },
        session,
      ),
    ).toThrow(/Unsupported/);
  });
  it("reads and edits legacy records without rewriting their historical authorship", async () => {
    const session = fixture();
    const initial = review(session);
    const { formJudgments: _unused, ...provenance } = initial.provenance;
    const legacy: CoachReviewData = {
      ...initial,
      schemaVersion: "local-coach-review-1",
      reviewer: { id: "local-user-coach", expertise: "coach_self_reported" },
      provenance,
    };
    const before = JSON.stringify(legacy);
    expect(validateCoachReview(legacy, session)).toBe(legacy);
    session.coachReview = legacy;
    const edited = answerIdentity(
      legacy,
      legacy.cards[0].id,
      "right-straight",
      "",
    );
    const save = vi.fn(async () => {});
    await persistCoachReview(session, edited, save);
    expect(edited.schemaVersion).toBe("local-coach-review-1");
    expect(edited.reviewer).toEqual(legacy.reviewer);
    expect(edited.provenance).toEqual(provenance);
    expect(JSON.stringify(legacy)).toBe(before);
    expect(save.mock.calls[0]).toEqual([
      { ...session, coachReview: edited },
      legacy,
    ]);
  });
  it("resumes action review after a saved straight without manufacturing an optional guard answer", () => {
    const data = review({
      ...fixture(),
      events: [
        event,
        { ...event, id: "second", startMs: 1500, peakMs: 1700, endMs: 1900 },
      ],
    });
    const next = answerIdentity(
      data,
      data.cards[0].id,
      "right-straight",
      "Technique needs work",
    );
    expect(next.cards.filter(actionReviewed)).toHaveLength(1);
    expect(next.cards.findIndex((card) => !actionReviewed(card))).toBe(1);
    expect(next.cards[0].guard).toBeNull();
    expect(cardComplete(next.cards[0])).toBe(false);
    expect(data.cards[0].identity).toBeNull();
  });
  it("uses detections only to find clips and never prefills user judgments", () => {
    const session = fixture();
    const original = JSON.stringify(session.events);
    const data = review(session);
    expect(data.cards).toHaveLength(1);
    expect(data.cards[0]).toMatchObject({
      windowStartMs: 0,
      windowEndMs: 1300,
      identity: null,
      guard: null,
      updatedAt: null,
    });
    expect(data.cards[0].proposal).toMatchObject({
      kind: "detector-navigation",
      eventId: "proposal-1",
      label: "cross",
    });
    const corrected = answerIdentity(
      data,
      data.cards[0].id,
      "left-hook",
      "Actually a left hook",
    );
    expect(corrected.cards[0].identity).toBe("left-hook");
    expect(corrected.cards[0].guard).toBeNull();
    expect(cardComplete(corrected.cards[0])).toBe(true);
    expect(data.cards[0].identity).toBeNull();
    expect(JSON.stringify(session.events)).toBe(original);
  });
  it("bounds source-time clips independently of the nonzero video offset", () => {
    expect(clipBounds(50, 3000)).toEqual({
      windowStartMs: 0,
      windowEndMs: 850,
    });
    expect(clipBounds(2999, 3000)).toEqual({
      windowStartMs: 2199,
      windowEndMs: 3000,
    });
    expect(review().source.videoOffsetMs).toBe(1200);
    expect(() => clipBounds(NaN, 3000)).toThrow();
    expect(() => clipBounds(1, 0)).toThrow();
    expect(() => review({ ...fixture(), videoOffsetMs: -1 })).toThrow();
  });
  it("omits broken navigation proposals and deduplicates IDs without inventing actions", () => {
    const session = fixture();
    session.events = [
      event,
      { ...event },
      { ...event, id: "outside", endMs: 3500 },
      { ...event, id: "reverse", startMs: 600 },
    ];
    expect(review(session).cards.map((c) => c.id)).toEqual([
      "event:proposal-1",
    ]);
  });
  it("maps both stances while retaining physical hand identity", () => {
    expect(identityOptions("southpaw").slice(0, 2)).toEqual([
      "right-straight",
      "left-straight",
    ]);
    expect(identityName("left-straight", "southpaw")).toBe("Left cross");
    expect(identityName("right-hook", "orthodox")).toBe("Right hook");
  });
  it("requires actual straight confirmation before form labels and clears form when identity changes", () => {
    const data = review(),
      id = data.cards[0].id;
    expect(() => answerGuard(data, id, "needs-work", "")).toThrow(
      /Confirm a straight/,
    );
    const straight = answerIdentity(data, id, "left-straight", "");
    expect(cardComplete(straight.cards[0])).toBe(false);
    const rated = answerGuard(
      straight,
      id,
      "needs-work",
      "Keep the other hand at the chosen guard",
    );
    expect(rated.cards[0].guard).toMatchObject({
      answer: "needs-work",
      context: "eligible",
      view: "adequate",
      outcomeObserved: true,
      feedbackWarranted: "yes",
    });
    const revised = answerIdentity(rated, id, "unclear", "");
    expect(revised.cards[0].guard).toBeNull();
    expect(revised.cards[0].identity).toBe("unclear");
  });
  it("keeps uncertainty, inapplicability and an unanswered card separate", () => {
    const data = review(),
      id = data.cards[0].id;
    const straight = answerIdentity(data, id, "right-straight", "");
    expect(data.cards[0].identity).toBeNull();
    expect(
      answerGuard(straight, id, "unclear", "").cards[0].guard,
    ).toMatchObject({
      context: "unknown",
      outcomeObserved: false,
      feedbackWarranted: "uncertain",
    });
    expect(
      answerGuard(straight, id, "not-applicable", "Intentional parry").cards[0]
        .guard,
    ).toMatchObject({
      context: "not_applicable",
      outcomeObserved: false,
      feedbackWarranted: "no",
    });
    expect(() =>
      validateCoachReview(
        {
          ...straight,
          cards: [
            {
              ...straight.cards[0],
              guard: {
                ...answerGuard(straight, id, "held", "").cards[0].guard!,
                view: "unknown",
              },
            },
          ],
        },
        fixture(),
      ),
    ).toThrow(/contradict/);
  });
  it("adds missed moments even when there are no detections, without asserting full coverage", () => {
    const session = { ...fixture(), events: [] };
    const data = review(session);
    expect(data.cards).toHaveLength(0);
    const added = addManualCard(data, 2900, "manual-one");
    expect(added.cards[0]).toMatchObject({
      windowStartMs: 2100,
      windowEndMs: 3000,
      identity: null,
      proposal: { kind: "manual-moment", selectedAtMs: 2900 },
    });
    expect(added.provenance.coverage).toBe(
      "selected_clips_not_complete_recording",
    );
    expect(data.cards).toHaveLength(0);
    expect(() => addManualCard(added, 100, "manual-one")).toThrow();
  });
  it("rejects reuse for different video, evidence, stance or interval", () => {
    const data = review();
    expect(() =>
      validateCoachReview(data, fixture(), "c".repeat(64), sourceSha),
    ).toThrow(/match/);
    expect(() =>
      validateCoachReview(data, fixture(), videoSha, "d".repeat(64)),
    ).toThrow(/match/);
    expect(() =>
      validateCoachReview(data, { ...fixture(), stance: "southpaw" }),
    ).toThrow(/match/);
    expect(() =>
      validateCoachReview(data, { ...fixture(), durationMs: 2999 }),
    ).toThrow(/match/);
  });
  it("waits for persistence and preserves original video, events and reference annotations", async () => {
    const session = fixture(),
      data = answerIdentity(
        review(session),
        "event:proposal-1",
        "not-punch",
        "",
      );
    let release!: () => void;
    let settled = false;
    let saved: Session | undefined;
    const onUpdate = vi.fn((next: Session) => {
      saved = next;
      return new Promise<void>((resolve) => {
        release = resolve;
      });
    });
    const pending = persistCoachReview(session, data, onUpdate).then(() => {
      settled = true;
    });
    await Promise.resolve();
    expect(settled).toBe(false);
    expect(session.coachReview).toBeUndefined();
    expect(saved!.events).toBe(session.events);
    expect(saved!.annotations).toBe(session.annotations);
    expect(saved!.video).toBe(session.video);
    release();
    await pending;
    expect(settled).toBe(true);
    expect(saved!.coachReview).toBe(data);
  });
  it("passes the prior committed label version for atomic stale-tab checks", async () => {
    const session = fixture();
    const committed = review(session);
    session.coachReview = committed;
    const next = answerIdentity(
      committed,
      "event:proposal-1",
      "left-straight",
      "",
    );
    const save = vi.fn(async () => {});
    await persistCoachReview(session, next, save);
    expect(save.mock.calls[0]).toEqual([
      { ...session, coachReview: next },
      committed,
    ]);
    expect(session.coachReview).toBe(committed);
  });
  it("propagates failed saves without turning the prior review into a successful answer", async () => {
    const session = fixture(),
      before = review(session),
      after = answerIdentity(before, "event:proposal-1", "left-straight", "");
    await expect(
      persistCoachReview(session, after, async () => {
        throw new Error("Disk full");
      }),
    ).rejects.toThrow("Disk full");
    expect(before.cards[0].identity).toBeNull();
    expect(session.coachReview).toBeUndefined();
  });
  it("fingerprints original evidence but not notes or later independent annotation edits", async () => {
    const session = fixture();
    const first = await fingerprintCoachSource(session);
    expect(
      await fingerprintCoachSource({
        ...session,
        coachReview: saveCardNote(
          review(session),
          "event:proposal-1",
          "Good variation",
        ),
        annotations: [],
      }),
    ).toBe(first);
    expect(
      await fingerprintCoachSource({
        ...session,
        events: [{ ...event, peakMs: 301 }],
      }),
    ).not.toBe(first);
  });
});
