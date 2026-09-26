import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Download,
  Play,
  Pause,
  Trash2,
  X,
} from "lucide-react";
import { PoseOverlay } from "./PoseOverlay";
import { assessArmTracking } from "../lib/motion";
import {
  deleteSession,
  downloadBlob,
  exportSession,
  formatTime,
} from "../lib/storage";
import type { Session, SessionAnnotation } from "../lib/types";

type Props = {
  sessions: Session[];
  selected: Session | null;
  saveState: "saving" | "saved" | "error";
  onSelect: (session: Session) => void;
  onUpdate: (session: Session) => Promise<void>;
  onDeleted: (id: string) => void;
  onNotice: (message: string) => void;
  onPractice: () => void;
};

export function RoundReview({
  sessions,
  selected,
  saveState,
  onSelect,
  onUpdate,
  onDeleted,
  onNotice,
  onPractice,
}: Props) {
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [overlay, setOverlay] = useState(false);
  const [showPredictions, setShowPredictions] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const offset = selected?.videoOffsetMs ?? 0;
  const duration = selected?.durationMs ?? 0;
  const sessionId = selected?.id;
  const videoBlob = selected?.video;
  useEffect(() => {
    setTime(0);
    setPlaying(false);
    setOverlay(!videoBlob);
    setShowPredictions(false);
    if (!videoBlob) {
      setVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(videoBlob);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [sessionId, videoBlob]);
  const sync = useCallback(
    (video: HTMLVideoElement) => {
      const start = offset / 1000,
        end = (offset + duration) / 1000;
      const next = Math.max(start, Math.min(end, video.currentTime));
      if (video.currentTime >= end) video.pause();
      if (Math.abs(video.currentTime - next) > 0.0001) video.currentTime = next;
      setTime(Math.max(0, Math.min(duration, next * 1000 - offset)));
    },
    [offset, duration],
  );
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;
    let callback: number | null = null,
      raf = 0,
      stopped = false;
    const schedule = () => {
      if (video.requestVideoFrameCallback)
        callback = video.requestVideoFrameCallback(() => {
          if (!stopped) {
            sync(video);
            schedule();
          }
        });
      else
        raf = requestAnimationFrame(() => {
          if (!stopped) {
            if (!video.paused) sync(video);
            schedule();
          }
        });
    };
    schedule();
    return () => {
      stopped = true;
      if (callback !== null) video.cancelVideoFrameCallback?.(callback);
      cancelAnimationFrame(raf);
    };
  }, [videoUrl, sync]);
  useEffect(() => {
    if (!playing || videoUrl) return;
    let previous = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now(),
        delta = now - previous;
      previous = now;
      setTime((current) => Math.min(duration, current + delta));
    }, 33);
    return () => clearInterval(timer);
  }, [playing, videoUrl, duration]);
  useEffect(() => {
    if (time >= duration) setPlaying(false);
  }, [time, duration]);
  const seek = (ms: number) => {
    const next = Math.max(0, Math.min(duration, ms));
    setTime(next);
    if (videoRef.current && videoUrl)
      videoRef.current.currentTime = (next + offset) / 1000;
  };
  const stepTo = (ms: number) => {
    videoRef.current?.pause();
    setPlaying(false);
    seek(ms);
  };
  const togglePlay = () => {
    if (time >= duration - 1) seek(0);
    if (videoRef.current && videoUrl) {
      if (videoRef.current.paused)
        void videoRef.current
          .play()
          .catch(() =>
            onNotice("Could not play this video. Try exporting it."),
          );
      else videoRef.current.pause();
    } else setPlaying((old) => !old);
  };
  const nearest = selected?.frames.reduce(
    (best, f) =>
      !best || Math.abs(f.t - time) < Math.abs(best.t - time) ? f : best,
    selected.frames[0],
  );
  const frame = nearest && Math.abs(nearest.t - time) <= 100 ? nearest : null;
  const tracking = frame ? assessArmTracking(frame) : null;
  const remove = async () => {
    if (
      !selected ||
      !window.confirm(
        "Delete this round and its video from this browser? Export first if you want a copy.",
      )
    )
      return;
    try {
      await deleteSession(selected.id);
      onDeleted(selected.id);
    } catch {
      onNotice("Could not delete this round from local storage.");
    }
  };

  return (
    <section className="review">
      <div className="page-heading">
        <div>
          <h1>Review</h1>
          <p>
            {selected
              ? `${formatTime(duration)} round · ${selected.stance === "orthodox" ? "Left" : "Right"} hand leads`
              : "Your rounds, saved on this device."}
          </p>
        </div>
        <button className="button secondary" onClick={onPractice}>
          New round
        </button>
      </div>
      {sessions.length > 0 && (
        <div className="round-library" aria-label="Saved rounds">
          {sessions.map((session) => (
            <button
              className={`session-item ${session.id === sessionId ? "selected" : ""}`}
              aria-pressed={session.id === sessionId}
              key={session.id}
              onClick={() => onSelect(session)}
            >
              <span>
                {new Date(session.createdAt).toLocaleString([], {
                  month: "short",
                  day: "numeric",
                  hour: "numeric",
                  minute: "2-digit",
                })}
              </span>
              <small>
                {formatTime(session.durationMs)} ·{" "}
                {session.source === "demo"
                  ? "Demo"
                  : session.video
                    ? "Video"
                    : "Motion only"}
              </small>
            </button>
          ))}
        </div>
      )}
      {!selected ? (
        <div className="review-empty">
          <h2>
            {sessions.length
              ? "Choose a round above."
              : "Your first round starts here."}
          </h2>
          <p>
            Record a short round, then compare your movement with its tracking.
          </p>
          <button className="button primary" onClick={onPractice}>
            Go to practice
          </button>
        </div>
      ) : (
        <>
          <div className="replay-stage" key={selected.id}>
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                playsInline
                onLoadedMetadata={(e) => {
                  e.currentTarget.currentTime = offset / 1000;
                  sync(e.currentTarget);
                }}
                onSeeking={(e) => sync(e.currentTarget)}
                onSeeked={(e) => sync(e.currentTarget)}
                onTimeUpdate={(e) => sync(e.currentTarget)}
                onPlay={(e) => {
                  setPlaying(true);
                  sync(e.currentTarget);
                }}
                onPause={() => setPlaying(false)}
              />
            ) : (
              <div className="replay-label">
                {selected.source === "demo"
                  ? "Simulated replay"
                  : "Motion only — video was not recorded"}
              </div>
            )}
            {overlay && frame && (
              <PoseOverlay
                frame={frame}
                mirror={false}
                silhouette={!videoUrl}
              />
            )}
          </div>
          <div className="replay-controls">
            <button
              className="icon-button"
              aria-label={playing ? "Pause replay" : "Play replay"}
              onClick={togglePlay}
            >
              {playing ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button
              className="icon-button"
              aria-label="Previous frame"
              onClick={() =>
                stepTo(
                  selected.frames.filter((f) => f.t < time - 1).at(-1)?.t ?? 0,
                )
              }
            >
              <ChevronLeft size={18} />
            </button>
            <button
              className="icon-button"
              aria-label="Next frame"
              onClick={() =>
                stepTo(
                  selected.frames.find((f) => f.t > time + 1)?.t ?? duration,
                )
              }
            >
              <ChevronRight size={18} />
            </button>
            <span>{(time / 1000).toFixed(1)}s</span>
            <input
              type="range"
              aria-label="Replay position"
              min={0}
              max={duration || 1}
              step={1}
              value={Math.min(time, duration)}
              onChange={(e) => seek(Number(e.target.value))}
            />
            <span>{formatTime(duration)}</span>
          </div>
          <div className="review-toggles">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={overlay}
                onChange={(e) => setOverlay(e.target.checked)}
              />
              Show tracking
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={showPredictions}
                onChange={(e) => setShowPredictions(e.target.checked)}
              />
              Show detections
            </label>
            <span
              className={`save-status ${saveState === "error" ? "save-error" : ""}`}
            >
              {saveState === "saving"
                ? "Saving…"
                : saveState === "error"
                  ? "Not saved — export before closing"
                  : "Saved on this device"}
            </span>
          </div>
          {overlay && !frame && (
            <p className="muted" role="status">
              No recent pose sample at this moment. The overlay is hidden.
            </p>
          )}
          {overlay && frame && (
            <p className="tracking-description">
              L · {tracking?.left.assessable ? "tracked" : "uncertain"}{" "}
              <span> / </span> R ·{" "}
              {tracking?.right.assessable ? "tracked" : "uncertain"}. Check L /
              R follow your physical hands. Visibility does not confirm hand
              identity or technique.
            </p>
          )}
          {showPredictions && (
            <div className="detections">
              <div className="detection-heading">
                <h2>
                  {selected.events.filter((e) => e.label === "jab").length} jabs{" "}
                  <span>/</span>{" "}
                  {selected.events.filter((e) => e.label === "cross").length}{" "}
                  crosses
                </h2>
                <span>Experimental</span>
              </div>
              <p>
                These counts can miss punches or count other movement. Compare
                with the video.
              </p>
              <div className="event-timeline">
                {selected.events.length ? (
                  selected.events.map((event) => (
                    <button
                      key={event.id}
                      className={`event-chip ${event.label}`}
                      onClick={() => seek(event.startMs)}
                    >
                      <strong>{event.label === "jab" ? "Jab" : "Cross"}</strong>
                      <span>
                        {(event.peakMs / 1000).toFixed(1)}s · {event.hand}
                      </span>
                    </button>
                  ))
                ) : (
                  <span className="muted">
                    No straight punches were detected.
                  </span>
                )}
              </div>
            </div>
          )}
          {!selected.video && selected.source !== "demo" && (
            <p className="legacy-note">
              This round has tracking only; no video was saved. Record a new
              camera round to check missed punches against the original footage.
            </p>
          )}
          <ReferenceLabels
            key={`labels-${selected.id}`}
            session={selected}
            onUpdate={onUpdate}
            onNotice={onNotice}
            seek={seek}
          />
          <details className="options" key={`export-${selected.id}`}>
            <summary>Export & details</summary>
            <div className="export-buttons">
              <button
                className="button secondary"
                onClick={() => exportSession(selected)}
              >
                <Download size={15} />
                Evidence JSON
              </button>
              {selected.video && (
                <button
                  className="button secondary"
                  onClick={() =>
                    downloadBlob(
                      selected.video!,
                      `corner-${selected.id}.${selected.video!.type.includes("mp4") ? "mp4" : "webm"}`,
                    )
                  }
                >
                  <Download size={15} />
                  Export video
                </button>
              )}
              <button
                className="text-button delete"
                onClick={() => void remove()}
              >
                <Trash2 size={15} />
                Delete this round
              </button>
            </div>
            <dl className="technical-details">
              <div>
                <dt>Model</dt>
                <dd>{selected.model}</dd>
              </div>
              <div>
                <dt>Processed frames</dt>
                <dd>{selected.frames.length}</dd>
              </div>
              <div>
                <dt>Pose rate</dt>
                <dd>{selected.measuredFps.toFixed(1)} fps</dd>
              </div>
              <div>
                <dt>Inference p95</dt>
                <dd>{Math.round(selected.inferenceP95)} ms</dd>
              </div>
              <div>
                <dt>Detector</dt>
                <dd>{selected.detectorVersion ?? "Original baseline (v1)"}</dd>
              </div>
            </dl>
            <p>
              Exports contain this round’s original detections. Software updates
              do not rewrite saved results. Local browser storage is not a
              backup.
            </p>
          </details>
        </>
      )}
    </section>
  );
}

function ReferenceLabels({
  session,
  onUpdate,
  onNotice,
  seek,
}: {
  session: Session;
  onUpdate: Props["onUpdate"];
  onNotice: Props["onNotice"];
  seek: (time: number) => void;
}) {
  const [label, setLabel] = useState<SessionAnnotation["label"]>("jab");
  const [hand, setHand] = useState<SessionAnnotation["hand"]>("left");
  const [start, setStart] = useState(0),
    [end, setEnd] = useState(1),
    [note, setNote] = useState("");
  const add = () => {
    if (
      !Number.isFinite(start) ||
      !Number.isFinite(end) ||
      start < 0 ||
      end <= start ||
      end * 1000 > session.durationMs + 1
    ) {
      onNotice("Choose a valid start and end inside this round.");
      return;
    }
    void onUpdate({
      ...session,
      annotationsComplete: false,
      annotations: [
        ...session.annotations,
        {
          id: crypto.randomUUID(),
          label,
          hand,
          startMs: start * 1000,
          endMs: end * 1000,
          note: note.trim(),
        },
      ].sort((a, b) => a.startMs - b.startMs),
    });
    setNote("");
  };
  return (
    <details className="options">
      <summary>Label this round (optional)</summary>
      <p>
        Label the original video independently, including missed punches. A
        skeleton alone cannot validate the tracking.
      </p>
      <div className="annotation-fields">
        <label>
          Action
          <select
            aria-label="Annotation action"
            value={label}
            onChange={(e) =>
              setLabel(e.target.value as SessionAnnotation["label"])
            }
          >
            {["jab", "cross", "hook", "uppercut", "other", "unobservable"].map(
              (v) => (
                <option key={v}>{v}</option>
              ),
            )}
          </select>
        </label>
        <label>
          Hand
          <select
            aria-label="Annotation hand"
            value={hand}
            onChange={(e) =>
              setHand(e.target.value as SessionAnnotation["hand"])
            }
          >
            {["left", "right", "unknown"].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          Start (s)
          <input
            aria-label="Annotation start"
            type="number"
            min="0"
            step="0.01"
            value={start}
            onChange={(e) => setStart(Number(e.target.value))}
          />
        </label>
        <label>
          End (s)
          <input
            aria-label="Annotation end"
            type="number"
            min="0"
            step="0.01"
            value={end}
            onChange={(e) => setEnd(Number(e.target.value))}
          />
        </label>
      </div>
      <div className="annotation-note">
        <input
          aria-label="Annotation note"
          placeholder="What happened in the video?"
          value={note}
          maxLength={1000}
          onChange={(e) => setNote(e.target.value)}
        />
        <button className="button secondary" onClick={add}>
          Add label
        </button>
      </div>
      {session.annotations.map((a) => (
        <div className="annotation-row" key={a.id}>
          <button onClick={() => seek(a.startMs)}>
            <strong>
              {a.label} · {a.hand}
            </strong>
            <span>
              {(a.startMs / 1000).toFixed(2)}–{(a.endMs / 1000).toFixed(2)}s
            </span>
            <small>{a.note}</small>
          </button>
          <button
            className="icon-button"
            aria-label="Delete annotation"
            onClick={() =>
              void onUpdate({
                ...session,
                annotationsComplete: false,
                annotations: session.annotations.filter(
                  (item) => item.id !== a.id,
                ),
              })
            }
          >
            <X size={15} />
          </button>
        </div>
      ))}
      <label className="checkbox completeness">
        <input
          type="checkbox"
          checked={!!session.annotationsComplete}
          onChange={(e) =>
            void onUpdate({ ...session, annotationsComplete: e.target.checked })
          }
        />
        I reviewed the entire round and labeled every action, including missed
        detections.
      </label>
    </details>
  );
}
