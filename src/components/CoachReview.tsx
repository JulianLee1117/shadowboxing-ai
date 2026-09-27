import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  Pause,
  Play,
  RotateCcw,
  X,
} from "lucide-react";
import { fingerprintVideo } from "../lib/mediaFingerprint";
import { downloadBlob, formatTime } from "../lib/storage";
import {
  addManualCard,
  answerGuard,
  answerIdentity,
  cardComplete,
  createCoachReview,
  fingerprintCoachSource,
  identityName,
  identityOptions,
  isStraight,
  persistCoachReview,
  saveCardNote,
  validateCoachReview,
  type ActionIdentity,
  type CoachReviewData,
  type GuardAnswer,
} from "../lib/coachReview";
import type { Session } from "../lib/types";
import "./CoachReview.css";

type Props = {
  session: Session;
  onUpdate: (
    session: Session,
    expectedReview?: CoachReviewData,
  ) => Promise<void>;
  onClose: () => void;
  onNavigationLockChange?: (locked: boolean) => void;
};

export function CoachReview({
  session,
  onUpdate,
  onClose,
  onNavigationLockChange,
}: Props) {
  const [data, setData] = useState<CoachReviewData | null>(null);
  const [index, setIndex] = useState(0);
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [browsing, setBrowsing] = useState(false);
  const [time, setTime] = useState(0);
  const [speed, setSpeed] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [note, setNote] = useState("");
  const [editingIdentity, setEditingIdentity] = useState(false);
  const [undo, setUndo] = useState<{
    data: CoachReviewData;
    index: number;
  } | null>(null);
  const [retry, setRetry] = useState<{
    data: CoachReviewData;
    index: number;
  } | null>(null);
  const [notice, setNotice] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const busyRef = useRef(false);
  const mounted = useRef(true);
  const latest = useRef(session);
  latest.current = session;
  const sourceKey = `${session.id}:${session.stance}:${session.durationMs}:${session.videoOffsetMs ?? 0}`;
  const card =
    data?.source.sessionId === session.id ? data.cards[index] : undefined;
  const dirty = !!card && note !== card.notes;
  const navigationLocked = dirty || saving || !!retry;
  const navigationCallback = useRef(onNavigationLockChange);
  navigationCallback.current = onNavigationLockChange;
  const disabled = loading || saving || !!retry || !videoReady;
  const intervalStart = browsing ? 0 : (card?.windowStartMs ?? 0);
  const intervalEnd = browsing
    ? session.durationMs
    : (card?.windowEndMs ?? session.durationMs);
  const offset = session.videoOffsetMs ?? 0;
  const bounds = useRef({
    start: intervalStart,
    end: intervalEnd,
    offset,
    browsing,
  });
  bounds.current = { start: intervalStart, end: intervalEnd, offset, browsing };

  useEffect(() => {
    onNavigationLockChange?.(navigationLocked);
  }, [navigationLocked, onNavigationLockChange]);
  useEffect(() => {
    if (!navigationLocked) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [navigationLocked]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      navigationCallback.current?.(false);
    };
  }, []);
  useEffect(() => {
    let canceled = false;
    setData(null);
    setIndex(0);
    setUndo(null);
    setRetry(null);
    setError("");
    setNotice("");
    setLoading(true);
    setBrowsing(false);
    setNote("");
    setVideoReady(false);
    const source = latest.current;
    if (!source.video || source.source === "demo") {
      setError("This review needs the saved original video.");
      setLoading(false);
      return;
    }
    const objectUrl = URL.createObjectURL(source.video);
    setUrl(objectUrl);
    void Promise.all([
      fingerprintVideo(source.video),
      fingerprintCoachSource(source),
    ])
      .then(([videoSha, sessionSha]) => {
        if (canceled) return;
        const review = source.coachReview
          ? validateCoachReview(
              source.coachReview,
              source,
              videoSha,
              sessionSha,
            )
          : createCoachReview(source, videoSha, sessionSha);
        setData(review);
        const next = review.cards.findIndex((c) => !cardComplete(c));
        setIndex(Math.max(0, next));
        setBrowsing(review.cards.length === 0);
        setLoading(false);
      })
      .catch((e) => {
        if (!canceled) {
          setError(
            e instanceof Error ? e.message : "Could not open coach review.",
          );
          setLoading(false);
        }
      });
    return () => {
      canceled = true;
      videoRef.current?.pause();
      URL.revokeObjectURL(objectUrl);
    };
  }, [sourceKey, session.video]);
  useEffect(() => {
    setNote(card?.notes ?? "");
    setEditingIdentity(false);
  }, [card?.id, card?.notes, card?.identity]);
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = speed;
  }, [speed, url]);
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    setPlaying(false);
    if (video.readyState >= 1) {
      video.currentTime = (intervalStart + offset) / 1000;
      setTime(intervalStart);
    }
  }, [card?.id, browsing, url, intervalStart, offset]);

  function syncTime() {
    const video = videoRef.current;
    if (!video) return;
    const { start, end, offset: videoOffset, browsing: whole } = bounds.current;
    const t = video.currentTime * 1000 - videoOffset;
    if (t < start - 2) video.currentTime = (start + videoOffset) / 1000;
    else if (t >= end - 2) {
      if (!whole && !video.paused) {
        video.currentTime = (start + videoOffset) / 1000;
      } else {
        video.pause();
        if (t > end + 2) video.currentTime = (end + videoOffset) / 1000;
      }
    }
    setTime(
      Math.max(start, Math.min(end, video.currentTime * 1000 - videoOffset)),
    );
  }
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !url) return;
    let handle = 0;
    let stopped = false;
    const tick = () => {
      if (stopped) return;
      syncTime();
      handle = video.requestVideoFrameCallback(tick);
    };
    if (video.requestVideoFrameCallback)
      handle = video.requestVideoFrameCallback(tick);
    return () => {
      stopped = true;
      if (handle) video.cancelVideoFrameCallback(handle);
    };
  }, [url]);
  async function togglePlay() {
    const video = videoRef.current;
    if (!video) return;
    if (!video.paused) {
      video.pause();
      return;
    }
    if (time >= intervalEnd - 5)
      video.currentTime = (intervalStart + offset) / 1000;
    try {
      await video.play();
    } catch {
      setError("Playback could not start. Press Play to try again.");
    }
  }
  function seek(value: number) {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    const t = Math.max(intervalStart, Math.min(intervalEnd, value));
    video.currentTime = (t + offset) / 1000;
    setTime(t);
  }

  async function commit(
    next: CoachReviewData,
    nextIndex = index,
    undoing = false,
  ) {
    if (busyRef.current || !data) return;
    busyRef.current = true;
    setSaving(true);
    setError("");
    setNotice("");
    const previous = { data, index };
    try {
      await persistCoachReview(latest.current, next, onUpdate);
      if (!mounted.current) return;
      setData(next);
      setIndex(Math.max(0, Math.min(next.cards.length - 1, nextIndex)));
      setBrowsing(false);
      setRetry(null);
      setUndo(undoing ? null : previous);
      setNotice("Saved on this device");
    } catch (e) {
      if (mounted.current) {
        setError(
          e instanceof Error
            ? e.message
            : "Labels could not be saved. Your original round is unchanged.",
        );
        setRetry({ data: next, index: nextIndex });
      }
    } finally {
      busyRef.current = false;
      if (mounted.current) setSaving(false);
    }
  }
  function identify(identity: ActionIdentity) {
    if (!data || !card || disabled) return;
    const next = answerIdentity(data, card.id, identity, note);
    void commit(
      next,
      isStraight(identity) ? index : Math.min(index + 1, data.cards.length - 1),
    );
  }
  function rateGuard(answer: GuardAnswer) {
    if (data && card && !disabled)
      void commit(
        answerGuard(data, card.id, answer, note),
        Math.min(index + 1, data.cards.length - 1),
      );
  }
  function addMoment() {
    if (data && !disabled)
      void commit(
        addManualCard(data, time, `manual:${crypto.randomUUID()}`),
        data.cards.length,
      );
  }
  function changeCard(next: number) {
    if (disabled || dirty || !data) return;
    setIndex(Math.max(0, Math.min(data.cards.length - 1, next)));
    setBrowsing(false);
    setNotice("");
  }
  function exportLabels(review = data, unsaved = false) {
    if (!review) return;
    downloadBlob(
      new Blob(
        [
          JSON.stringify(
            {
              ...review,
              exportState: unsaved ? "unsaved_attempt" : "current_review",
            },
            null,
            2,
          ),
        ],
        { type: "application/json" },
      ),
      `coach-labels-${unsaved ? "unsaved-" : ""}${session.id}.json`,
    );
  }
  const completed = data?.cards.filter(cardComplete).length ?? 0;
  const focusMoment =
    card?.proposal.kind === "detector-navigation"
      ? card.proposal.peakMs
      : card?.proposal.selectedAtMs;

  return (
    <section className="coach-review" aria-label="Coach review">
      <header className="coach-review-header">
        <div>
          <span className="eyebrow">Your coaching labels · local only</span>
          <h2>Review the movement</h2>
        </div>
        <button
          type="button"
          className="icon-button"
          onClick={onClose}
          disabled={navigationLocked}
          aria-label="Close coach review"
        >
          <X size={20} />
        </button>
      </header>
      <p className="coach-intro">
        Confirm the action, then judge one guard cue. Labels stay local and do
        not train the model automatically.
      </p>
      {error && (
        <p className="coach-error" role="alert">
          {error}{" "}
          {retry && (
            <>
              <button
                type="button"
                disabled={saving}
                onClick={() => void commit(retry.data, retry.index)}
              >
                Retry save
              </button>
              <button
                type="button"
                disabled={saving}
                onClick={() => exportLabels(retry.data, true)}
              >
                Export unsaved labels
              </button>
              <button
                type="button"
                disabled={saving}
                onClick={() => {
                  setRetry(null);
                  setError("");
                  setNote(card?.notes ?? "");
                }}
              >
                Discard unsaved changes
              </button>
            </>
          )}
        </p>
      )}
      {loading && <p role="status">Checking this recording…</p>}
      <div className="coach-workspace">
        <div className="coach-media">
          {url && (
            <div className="coach-player">
              <video
                ref={videoRef}
                src={url}
                playsInline
                preload="auto"
                aria-label="Original coaching clip"
                onLoadedMetadata={() => {
                  const v = videoRef.current;
                  if (v) {
                    v.playbackRate = speed;
                    v.currentTime = (intervalStart + offset) / 1000;
                    setTime(intervalStart);
                  }
                }}
                onLoadedData={() => setVideoReady(true)}
                onError={() => {
                  setVideoReady(false);
                  setError(
                    "This video could not be decoded. No labels have been changed.",
                  );
                }}
                onTimeUpdate={syncTime}
                onSeeked={syncTime}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => {
                  const v = videoRef.current;
                  if (v && !browsing && card) {
                    v.currentTime = (intervalStart + offset) / 1000;
                    void v.play().catch(() => setPlaying(false));
                  }
                }}
              />
              <span className="coach-video-label">
                Original video · no predictions
              </span>
            </div>
          )}
          {url && (
            <div className="coach-playback">
              <button
                type="button"
                onClick={() => void togglePlay()}
                aria-label={
                  playing ? "Pause coaching clip" : "Play coaching clip"
                }
              >
                {playing ? <Pause size={17} /> : <Play size={17} />}{" "}
                {playing ? "Pause" : "Play"}
              </button>
              <input
                aria-label={
                  browsing ? "Full video position" : "Coaching clip position"
                }
                type="range"
                min={intervalStart}
                max={intervalEnd}
                step={1}
                value={Math.max(intervalStart, Math.min(intervalEnd, time))}
                onChange={(e) => seek(Number(e.target.value))}
              />
              <span>
                {formatTime(time)} / {formatTime(intervalEnd)}
              </span>
              <div
                className="coach-speed"
                role="group"
                aria-label="Coaching playback speed"
              >
                {[0.5, 1].map((v) => (
                  <button
                    type="button"
                    key={v}
                    aria-pressed={speed === v}
                    onClick={() => setSpeed(v)}
                  >
                    {v}×
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="coach-answers">
          {data && (
            <>
              <div className="coach-progress">
                <span>
                  {browsing
                    ? "Find a missed movement"
                    : data.cards.length
                      ? `Clip ${index + 1} of ${data.cards.length}`
                      : "No clips yet"}{" "}
                  · {completed} reviewed
                </span>
                <button
                  type="button"
                  disabled={disabled || dirty}
                  onClick={() => {
                    setBrowsing(!browsing);
                    setNotice("");
                  }}
                >
                  {browsing && card ? "Back to clip" : "Find a missed punch"}
                </button>
              </div>
              {browsing ? (
                <div className="coach-question">
                  <h3>Choose a moment to review</h3>
                  <p>
                    Play or scrub the full video. This adds a clip around the
                    playhead; it does not label it as a punch.
                  </p>
                  <button
                    type="button"
                    className="primary"
                    disabled={disabled}
                    onClick={addMoment}
                  >
                    Review this moment
                  </button>
                </div>
              ) : card ? (
                <>
                  <div className="coach-focus-moment">
                    <span>
                      Focus at {((focusMoment ?? 0) / 1000).toFixed(1)}s ·
                      navigation marker only
                    </span>
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() => seek(focusMoment ?? intervalStart)}
                    >
                      Jump to moment
                    </button>
                  </div>
                  {card.identity && !editingIdentity ? (
                    <div className="coach-confirmed-identity">
                      <div>
                        <span>Your action label</span>
                        <strong>
                          {identityName(card.identity, session.stance)}
                        </strong>
                      </div>
                      <button
                        type="button"
                        disabled={disabled || dirty}
                        onClick={() => setEditingIdentity(true)}
                      >
                        Change action
                      </button>
                    </div>
                  ) : (
                    <div className="coach-question">
                      <h3>What actually happened?</h3>
                      <p>
                        Use the physical hand, regardless of screen mirroring.
                        If this clip contains several actions, judge the
                        selected moment or add a more suitable clip.
                      </p>
                      <div
                        className="coach-identities"
                        role="group"
                        aria-label="Actual punch identity"
                      >
                        {identityOptions(session.stance).map((identity) => (
                          <button
                            type="button"
                            key={identity}
                            disabled={disabled}
                            aria-pressed={card.identity === identity}
                            onClick={() => identify(identity)}
                          >
                            {identityName(identity, session.stance)}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {isStraight(card.identity) && !editingIdentity && (
                    <div className="coach-question coach-guard">
                      <h3>
                        Did the other hand stay in your intended high guard?
                      </h3>
                      <p>
                        For an <strong>isolated straight</strong>. “Guard held”
                        and “Needs work” mean this drill applies and you could
                        see the hand clearly. “Needs work” means you would give
                        a correction. Choose “Not this drill/style” for
                        deliberate defense, another guard or a competing action.
                      </p>
                      <div
                        className="coach-guard-choices"
                        role="group"
                        aria-label="Non-punching hand guard"
                      >
                        {(
                          [
                            ["held", "Guard held"],
                            ["needs-work", "Needs work"],
                            ["unclear", "Can't tell"],
                            ["not-applicable", "Not this drill/style"],
                          ] as const
                        ).map(([value, label]) => (
                          <button
                            type="button"
                            key={value}
                            disabled={disabled}
                            aria-pressed={card.guard?.answer === value}
                            onClick={() => rateGuard(value)}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {card.identity && !isStraight(card.identity) && (
                    <p className="coach-small">
                      {card.identity === "unclear"
                        ? "Identity is uncertain. No form judgment recorded."
                        : card.identity === "not-punch"
                          ? "Marked as not a punch. No form judgment recorded."
                          : "Action confirmed. Hook and uppercut form criteria are not enabled."}
                    </p>
                  )}
                  <label className="coach-notes">
                    Notes{" "}
                    <span>
                      (optional — acceptable variation or the cue you would
                      give)
                    </span>
                    <textarea
                      aria-label="Coach notes"
                      value={note}
                      maxLength={2000}
                      disabled={disabled}
                      rows={2}
                      onChange={(e) => setNote(e.target.value)}
                    />
                  </label>
                  {dirty && (
                    <div className="coach-note-save">
                      <span>Save your note before leaving this clip.</span>
                      <button
                        type="button"
                        disabled={disabled}
                        onClick={() =>
                          void commit(saveCardNote(data, card.id, note))
                        }
                      >
                        Save note
                      </button>
                      <button
                        type="button"
                        disabled={saving}
                        onClick={() => setNote(card.notes)}
                      >
                        Discard note
                      </button>
                    </div>
                  )}
                </>
              ) : (
                <p>
                  No detected clips were available. Use “Find a missed punch” to
                  review the video.
                </p>
              )}
              <footer className="coach-review-footer">
                <div className="coach-nav">
                  <button
                    type="button"
                    disabled={disabled || dirty || index <= 0}
                    onClick={() => changeCard(index - 1)}
                  >
                    <ArrowLeft size={16} /> Back
                  </button>
                  <button
                    type="button"
                    disabled={
                      disabled ||
                      dirty ||
                      !card ||
                      index >= data.cards.length - 1
                    }
                    onClick={() => changeCard(index + 1)}
                  >
                    Next <ArrowRight size={16} />
                  </button>
                  <button
                    type="button"
                    disabled={disabled || dirty || !undo}
                    onClick={() =>
                      undo && void commit(undo.data, undo.index, true)
                    }
                  >
                    <RotateCcw size={16} /> Undo
                  </button>
                </div>
                <span className="coach-save-status" role="status">
                  {saving
                    ? "Saving…"
                    : notice || "Next leaves unanswered clips unreviewed"}
                </span>
                <button
                  type="button"
                  disabled={disabled || dirty}
                  onClick={() => exportLabels()}
                >
                  <Download size={16} /> Export labels
                </button>
              </footer>
              <p className="coach-small">
                Selected clips are not a complete punch inventory. Your own
                judgments are training material only after a separate, explicit
                data review. Original video, events and reference annotations
                stay intact.
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
