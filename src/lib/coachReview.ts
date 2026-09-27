import type { Session, Stance } from "./types";

export const COACH_REVIEW_VERSION = "local-coach-review-1" as const;
export const GUARD_RUBRIC_VERSION = "isolated-high-guard-user-v1" as const;
export type ActionIdentity =
  | "left-straight"
  | "right-straight"
  | "left-hook"
  | "right-hook"
  | "left-uppercut"
  | "right-uppercut"
  | "not-punch"
  | "unclear";
export type GuardAnswer = "held" | "needs-work" | "unclear" | "not-applicable";
export interface CoachCard {
  id: string;
  windowStartMs: number;
  windowEndMs: number;
  proposal:
    | {
        kind: "detector-navigation";
        eventId: string;
        detectorVersion: string | null;
        label: string;
        hand: "left" | "right";
        startMs: number;
        peakMs: number;
        endMs: number;
      }
    | { kind: "manual-moment"; selectedAtMs: number };
  identity: ActionIdentity | null;
  guard: {
    criterion: "non_punching_hand_guard";
    rubricVersion: typeof GUARD_RUBRIC_VERSION;
    answer: GuardAnswer;
    context: "eligible" | "unknown" | "not_applicable";
    view: "adequate" | "unknown";
    outcomeObserved: boolean;
    feedbackWarranted: "yes" | "no" | "uncertain";
  } | null;
  notes: string;
  updatedAt: string | null;
}
export interface CoachReviewData {
  schemaVersion: typeof COACH_REVIEW_VERSION;
  source: {
    sessionId: string;
    sessionFingerprint: string;
    videoSha256: string;
    stance: Stance;
    durationMs: number;
    videoOffsetMs: number;
  };
  reviewer: { id: "local-user-coach"; expertise: "coach_self_reported" };
  provenance: {
    labels: "user-authored";
    proposalUse: "navigation_only_not_ground_truth";
    independence: "not_independent_validation";
    coverage: "selected_clips_not_complete_recording";
    bounds: "review_windows_not_verified_action_boundaries";
    training: "not_activated_or_automatically_used";
  };
  createdAt: string;
  updatedAt: string;
  cards: CoachCard[];
}
const IDENTITIES: readonly ActionIdentity[] = [
  "left-straight",
  "right-straight",
  "left-hook",
  "right-hook",
  "left-uppercut",
  "right-uppercut",
  "not-punch",
  "unclear",
];
const GUARDS: readonly GuardAnswer[] = [
  "held",
  "needs-work",
  "unclear",
  "not-applicable",
];
const finite = (n: number) => Number.isFinite(n);
const shaPattern = /^[a-f0-9]{64}$/;
export function clipBounds(momentMs: number, durationMs: number) {
  if (!finite(momentMs) || !finite(durationMs) || durationMs <= 0)
    throw new Error("A valid video interval is required.");
  const center = Math.max(0, Math.min(durationMs, momentMs));
  return {
    windowStartMs: Math.max(0, center - 800),
    windowEndMs: Math.min(durationMs, center + 800),
  };
}
export function identityName(identity: ActionIdentity, stance: Stance) {
  if (identity === "not-punch") return "Not a punch";
  if (identity === "unclear") return "Can't tell";
  const [hand, family] = identity.split("-");
  const lead = hand === (stance === "orthodox" ? "left" : "right");
  const name = family === "straight" ? (lead ? "jab" : "cross") : family;
  return `${hand === "left" ? "Left" : "Right"} ${name}`;
}
export function identityOptions(stance: Stance): ActionIdentity[] {
  const lead = stance === "orthodox" ? "left" : "right";
  const rear = lead === "left" ? "right" : "left";
  return [
    `${lead}-straight`,
    `${rear}-straight`,
    `${lead}-hook`,
    `${rear}-hook`,
    `${lead}-uppercut`,
    `${rear}-uppercut`,
    "not-punch",
    "unclear",
  ];
}
export function isStraight(identity: ActionIdentity | null) {
  return identity?.endsWith("-straight") === true;
}
export function cardComplete(card: CoachCard) {
  return (
    card.identity !== null &&
    (!isStraight(card.identity) || card.guard !== null)
  );
}
/** Shared exact evidence/context serialization for hashing and atomic save comparison. */
export function serializeCoachSource(session: Session): string {
  return JSON.stringify({
    id: session.id,
    createdAt: session.createdAt,
    source: session.source,
    stance: session.stance,
    model: session.model,
    drill: session.drill,
    durationMs: session.durationMs,
    videoOffsetMs: session.videoOffsetMs ?? 0,
    detectorVersion: session.detectorVersion ?? null,
    modelManifest: session.modelManifest ?? null,
    capture: session.capture ?? null,
    frames: session.frames,
    events: session.events,
  });
}
/** Original motion evidence is fingerprinted; user labels are stored separately. */
export async function fingerprintCoachSource(session: Session) {
  const hash = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(serializeCoachSource(session)),
  );
  return Array.from(new Uint8Array(hash), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}
export function createCoachReview(
  session: Session,
  videoSha256: string,
  sessionFingerprint: string,
  now = new Date().toISOString(),
): CoachReviewData {
  if (
    !session.video ||
    session.source === "demo" ||
    !finite(session.durationMs) ||
    session.durationMs <= 0 ||
    !finite(session.videoOffsetMs ?? 0) ||
    (session.videoOffsetMs ?? 0) < 0 ||
    !shaPattern.test(videoSha256) ||
    !shaPattern.test(sessionFingerprint)
  )
    throw new Error(
      "Saved original video and valid fingerprints are required.",
    );
  const seen = new Set<string>();
  const cards: CoachCard[] = [];
  for (const event of [...session.events].sort(
    (a, b) => a.startMs - b.startMs,
  )) {
    if (
      seen.has(event.id) ||
      ![event.startMs, event.peakMs, event.endMs].every(finite) ||
      event.startMs < 0 ||
      event.endMs > session.durationMs ||
      !(
        event.startMs <= event.peakMs &&
        event.peakMs <= event.endMs &&
        event.startMs < event.endMs
      )
    )
      continue;
    seen.add(event.id);
    cards.push({
      id: `event:${event.id}`,
      windowStartMs: Math.max(0, event.startMs - 800),
      windowEndMs: Math.min(session.durationMs, event.endMs + 800),
      proposal: {
        kind: "detector-navigation",
        eventId: event.id,
        detectorVersion: session.detectorVersion ?? null,
        label: event.label,
        hand: event.hand,
        startMs: event.startMs,
        peakMs: event.peakMs,
        endMs: event.endMs,
      },
      identity: null,
      guard: null,
      notes: "",
      updatedAt: null,
    });
  }
  return {
    schemaVersion: COACH_REVIEW_VERSION,
    source: {
      sessionId: session.id,
      sessionFingerprint,
      videoSha256,
      stance: session.stance,
      durationMs: session.durationMs,
      videoOffsetMs: session.videoOffsetMs ?? 0,
    },
    reviewer: { id: "local-user-coach", expertise: "coach_self_reported" },
    provenance: {
      labels: "user-authored",
      proposalUse: "navigation_only_not_ground_truth",
      independence: "not_independent_validation",
      coverage: "selected_clips_not_complete_recording",
      bounds: "review_windows_not_verified_action_boundaries",
      training: "not_activated_or_automatically_used",
    },
    createdAt: now,
    updatedAt: now,
    cards,
  };
}
export function validateCoachReview(
  data: CoachReviewData,
  session: Session,
  videoSha256?: string,
  sessionFingerprint?: string,
) {
  if (
    !data ||
    data.schemaVersion !== COACH_REVIEW_VERSION ||
    data.source?.sessionId !== session.id ||
    data.source.stance !== session.stance ||
    data.source.durationMs !== session.durationMs ||
    data.source.videoOffsetMs !== (session.videoOffsetMs ?? 0) ||
    !shaPattern.test(data.source.videoSha256) ||
    !shaPattern.test(data.source.sessionFingerprint) ||
    (videoSha256 && data.source.videoSha256 !== videoSha256) ||
    (sessionFingerprint &&
      data.source.sessionFingerprint !== sessionFingerprint)
  )
    throw new Error("These coach labels do not match this original recording.");
  if (
    !Array.isArray(data.cards) ||
    data.reviewer?.expertise !== "coach_self_reported" ||
    data.provenance?.coverage !== "selected_clips_not_complete_recording"
  )
    throw new Error("Unsupported coach review data.");
  const ids = new Set<string>();
  for (const card of data.cards) {
    if (
      !card.id ||
      ids.has(card.id) ||
      !finite(card.windowStartMs) ||
      !finite(card.windowEndMs) ||
      card.windowStartMs < 0 ||
      card.windowStartMs >= card.windowEndMs ||
      card.windowEndMs > session.durationMs ||
      (card.identity !== null && !IDENTITIES.includes(card.identity)) ||
      typeof card.notes !== "string" ||
      card.notes.length > 2000
    )
      throw new Error("Invalid coach review clip.");
    ids.add(card.id);
    if (
      card.guard &&
      (!isStraight(card.identity) ||
        !GUARDS.includes(card.guard.answer) ||
        card.guard.criterion !== "non_punching_hand_guard" ||
        card.guard.rubricVersion !== GUARD_RUBRIC_VERSION)
    )
      throw new Error("Guard labels require a confirmed straight punch.");
    if (card.guard) {
      const decisive =
        card.guard.answer === "held" || card.guard.answer === "needs-work";
      const context = decisive
        ? "eligible"
        : card.guard.answer === "not-applicable"
          ? "not_applicable"
          : "unknown";
      const feedback =
        card.guard.answer === "needs-work"
          ? "yes"
          : card.guard.answer === "unclear"
            ? "uncertain"
            : "no";
      if (
        card.guard.context !== context ||
        card.guard.view !== (decisive ? "adequate" : "unknown") ||
        card.guard.outcomeObserved !== decisive ||
        card.guard.feedbackWarranted !== feedback
      )
        throw new Error(
          "Guard judgment and evidence eligibility contradict each other.",
        );
    }
    if (
      card.proposal?.kind !== "manual-moment" &&
      card.proposal?.kind !== "detector-navigation"
    )
      throw new Error("Missing clip navigation provenance.");
  }
  return data;
}
function replaceCard(
  data: CoachReviewData,
  id: string,
  update: (c: CoachCard) => CoachCard,
  now: string,
) {
  if (!data.cards.some((c) => c.id === id))
    throw new Error("This clip is no longer available.");
  return {
    ...data,
    updatedAt: now,
    cards: data.cards.map((c) => (c.id === id ? update(c) : c)),
  };
}
export function answerIdentity(
  data: CoachReviewData,
  id: string,
  identity: ActionIdentity,
  notes: string,
  now = new Date().toISOString(),
) {
  if (!IDENTITIES.includes(identity))
    throw new Error("Choose the action you actually see.");
  return replaceCard(
    data,
    id,
    (c) => ({
      ...c,
      identity,
      guard: c.identity === identity ? c.guard : null,
      notes: notes.slice(0, 2000),
      updatedAt: now,
    }),
    now,
  );
}
export function answerGuard(
  data: CoachReviewData,
  id: string,
  answer: GuardAnswer,
  notes: string,
  now = new Date().toISOString(),
) {
  if (!GUARDS.includes(answer)) throw new Error("Choose a guard judgment.");
  return replaceCard(
    data,
    id,
    (c) => {
      if (!isStraight(c.identity))
        throw new Error("Confirm a straight punch before reviewing its guard.");
      const decisive = answer === "held" || answer === "needs-work";
      return {
        ...c,
        notes: notes.slice(0, 2000),
        updatedAt: now,
        guard: {
          criterion: "non_punching_hand_guard",
          rubricVersion: GUARD_RUBRIC_VERSION,
          answer,
          context: decisive
            ? "eligible"
            : answer === "not-applicable"
              ? "not_applicable"
              : "unknown",
          view: decisive ? "adequate" : "unknown",
          outcomeObserved: decisive,
          feedbackWarranted:
            answer === "needs-work"
              ? "yes"
              : answer === "unclear"
                ? "uncertain"
                : "no",
        },
      };
    },
    now,
  );
}
export function saveCardNote(
  data: CoachReviewData,
  id: string,
  notes: string,
  now = new Date().toISOString(),
) {
  return replaceCard(
    data,
    id,
    (c) => ({ ...c, notes: notes.slice(0, 2000), updatedAt: now }),
    now,
  );
}
export function addManualCard(
  data: CoachReviewData,
  momentMs: number,
  id: string,
  now = new Date().toISOString(),
) {
  if (!id || data.cards.some((c) => c.id === id))
    throw new Error("A new clip ID is required.");
  const bounds = clipBounds(momentMs, data.source.durationMs);
  return {
    ...data,
    updatedAt: now,
    cards: [
      ...data.cards,
      {
        id,
        ...bounds,
        proposal: {
          kind: "manual-moment" as const,
          selectedAtMs: Math.max(0, Math.min(data.source.durationMs, momentMs)),
        },
        identity: null,
        guard: null,
        notes: "",
        updatedAt: null,
      },
    ],
  };
}
/** Await the actual save. A rejected write never counts as a completed review step. */
export async function persistCoachReview(
  session: Session,
  data: CoachReviewData,
  onUpdate: (next: Session, expectedReview?: CoachReviewData) => Promise<void>,
) {
  validateCoachReview(data, session);
  const updated = { ...session, coachReview: data };
  await onUpdate(updated, session.coachReview);
  return data;
}
