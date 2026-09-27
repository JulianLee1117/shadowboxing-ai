import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Download,
  Play,
  Pause,
  Trash2,
  X,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { PoseOverlay } from "./PoseOverlay";
import { punchName, punchNotation } from "../lib/punches";
import "./RoundReview.css";
import { assessArmTracking, DETECTOR_VERSION } from "../lib/motion";
import { recheckDetections, type DetectorRecheckReport } from "../lib/recheck";
import { useRoundAnalysis } from "../hooks/useRoundAnalysis";
import { groupCombinations } from "../lib/combinations";
import { summarizeTrackingTrust } from "../lib/trackingTrust";
import { combinationLabel, DRILLS, type DrillId } from "../lib/drills";
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
  autoAnalyze: boolean;
  onAutoAnalysisHandled: () => void;
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
  autoAnalyze,
  onAutoAnalysisHandled,
  saveState,
  onSelect,
  onUpdate,
  onDeleted,
  onNotice,
  onPractice,
}: Props) {
  const [time, setTime] = useState(0);
  const [focused, setFocused] = useState(false);
  const focusButtonRef = useRef<HTMLButtonElement>(null);
  const playerRef = useRef<HTMLDivElement>(null);
  const wasFullscreen = useRef(false);
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [overlay, setOverlay] = useState(false);
  const [showPredictions, setShowPredictions] = useState(false);
  const [recheck, setRecheck] = useState<DetectorRecheckReport | null>(null);
  const analysis = useRoundAnalysis(selected);
  const [useVideoAnalysis, setUseVideoAnalysis] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const offset = selected?.videoOffsetMs ?? 0;
  const duration = selected?.durationMs ?? 0;
  const sessionId = selected?.id;
  const videoBlob = selected?.video;
  // The identity check also prevents stale results during a round-switch render.
  const updated =
    recheck &&
    recheck.sessionId === sessionId &&
    recheck.detectorVersion === DETECTOR_VERSION
      ? recheck
      : null;
  const videoReport = useVideoAnalysis ? analysis.report : null;
  const detectionEvents =
    videoReport?.events ?? updated?.events ?? selected?.events ?? [];
  const visibleDetections = showPredictions
    ? detectionEvents.filter(
        // Native video seeking can round a fractional onset down. Allow one
        // millisecond for display only; saved evidence and event bounds stay exact.
        (event) => time + 1 >= event.startMs && time <= event.endMs,
      )
    : [];
  const trackingFrames = videoReport?.frames ?? selected?.frames ?? [];
  const savedUncertainty = useMemo(
    () => summarizeTrackingTrust(selected?.frames ?? [], duration),
    [selected?.frames, duration],
  );
  const uncertaintyIntervals =
    videoReport?.uncertaintyIntervals ?? savedUncertainty;
  const combinations = useMemo(
    () =>
      groupCombinations(detectionEvents, {
        stance: selected?.stance ?? "orthodox",
        uncertaintyIntervals,
      }),
    [detectionEvents, selected?.stance, uncertaintyIntervals],
  );
  const punchCounts = [
    { notation: "1", name: "Jab", plural: "jabs" },
    { notation: "2", name: "Cross", plural: "crosses" },
    { notation: "3", name: "Lead hook", plural: "lead hooks" },
    { notation: "4", name: "Rear hook", plural: "rear hooks" },
    { notation: "5", name: "Lead uppercut", plural: "lead uppercuts" },
    { notation: "6", name: "Rear uppercut", plural: "rear uppercuts" },
  ].map((identity) => ({
    ...identity,
    count: detectionEvents.filter(
      (event) => punchNotation(event) === identity.notation,
    ).length,
  }));
  const leaveFocus = useCallback(() => {
    setFocused(false);
    if (document.fullscreenElement === playerRef.current)
      void document.exitFullscreen().catch(() => {});
  }, []);
  const toggleFocus = async () => {
    if (focused) return leaveFocus();
    setFocused(true);
    try {
      await playerRef.current?.requestFullscreen?.();
    } catch {
      // The full-viewport focus view remains available without fullscreen support.
    }
  };
  useEffect(() => {
    leaveFocus();
  }, [sessionId, leaveFocus]);
  useEffect(() => {
    const changed = () => {
      const current = document.fullscreenElement === playerRef.current;
      if (wasFullscreen.current && !current) setFocused(false);
      wasFullscreen.current = current;
    };
    document.addEventListener("fullscreenchange", changed);
    return () => document.removeEventListener("fullscreenchange", changed);
  }, []);
  useEffect(() => {
    if (!focused) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        leaveFocus();
        focusButtonRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [focused, leaveFocus]);
  useEffect(() => setRecheck(null), [sessionId, DETECTOR_VERSION]);
  useEffect(() => setUseVideoAnalysis(false), [sessionId]);
  useEffect(() => {
    if (
      !autoAnalyze ||
      analysis.loadingSaved ||
      analysis.running ||
      saveState === "saving"
    )
      return;
    onAutoAnalysisHandled();
    if (!analysis.report) void analysis.start();
  }, [
    autoAnalyze,
    analysis.loadingSaved,
    analysis.running,
    analysis.report,
    analysis.start,
    saveState,
    onAutoAnalysisHandled,
  ]);
  useEffect(() => {
    if (analysis.report) {
      // Fresh decoding can change tracking and has not beaten saved-pose replay
      // on the development set. Keep it a deliberate comparison, never silently
      // replace original or rechecked counts when a background job finishes.
      setUseVideoAnalysis(false);
      setShowPredictions(true);
    }
  }, [analysis.report]);
  useEffect(() => {
    setTime(0);
    setPlaying(false);
    setPlaybackRate(1);
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
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }, [videoUrl, playbackRate]);
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
      setTime((current) => Math.min(duration, current + delta * playbackRate));
    }, 33);
    return () => clearInterval(timer);
  }, [playing, videoUrl, duration, playbackRate]);
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
  const nearest = trackingFrames.reduce(
    (best, f) =>
      !best || Math.abs(f.t - time) < Math.abs(best.t - time) ? f : best,
    trackingFrames[0],
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
    <section
      className={`review round-review ${focused ? "review-is-focused" : ""}`}
    >
      <div className="review-intro" inert={focused}>
        <div className="page-heading">
          <div>
            <h1>Review</h1>
            <p>
              {selected
                ? `${formatTime(duration)} round · ${selected.stance === "orthodox" ? "Left" : "Right"} hand leads`
                : "Your rounds, saved on this device."}
            </p>
            {selected &&
              selected.drill !== "open" &&
              selected.drill in DRILLS && (
                <p className="round-focus">
                  Focus: {DRILLS[selected.drill as DrillId].label}
                </p>
              )}
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
      </div>
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
          <div
            ref={playerRef}
            className={`review-player ${focused ? "is-focused" : ""}`}
            role={focused ? "dialog" : undefined}
            aria-modal={focused ? true : undefined}
            aria-label={focused ? "Round replay" : undefined}
            onKeyDown={(event) => {
              if (!focused || event.key !== "Tab") return;
              const controls = Array.from(
                event.currentTarget.querySelectorAll<HTMLElement>(
                  "button:not(:disabled), input:not(:disabled), select:not(:disabled), video[controls]",
                ),
              );
              const first = controls[0],
                last = controls.at(-1);
              if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last?.focus();
              } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first?.focus();
              }
            }}
          >
            <div className="review-player-heading">
              <span>{selected.video ? "Recording" : "Motion replay"}</span>
              <button
                ref={focusButtonRef}
                className="text-button review-focus-button"
                aria-pressed={focused}
                onClick={() => void toggleFocus()}
              >
                {focused ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                {focused ? "Exit focus" : "Focus video"}
              </button>
            </div>
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
              {visibleDetections.length > 0 && (
                <div
                  className="replay-detection"
                  aria-label="Current detected punches"
                >
                  <small>Detected</small>
                  {visibleDetections.map((event) => (
                    <div key={event.id}>
                      <strong>{punchName(event)}</strong>
                      <span>{event.hand} hand</span>
                    </div>
                  ))}
                </div>
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
                    trackingFrames.filter((f) => f.t < time - 1).at(-1)?.t ?? 0,
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
                    trackingFrames.find((f) => f.t > time + 1)?.t ?? duration,
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
              <label>
                Speed
                <select
                  aria-label="Playback speed"
                  value={playbackRate}
                  onChange={(e) => setPlaybackRate(Number(e.target.value))}
                >
                  <option value={0.5}>0.5×</option>
                  <option value={1}>1×</option>
                </select>
              </label>
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
          </div>
          <div className="review-below" inert={focused}>
            {selected.video && selected.source !== "demo" && (
              <div className="analysis-panel">
                <div className="analysis-row">
                  <div>
                    <strong>
                      {analysis.running
                        ? "Analyzing locally…"
                        : analysis.report
                          ? analysis.report.completeness.status === "partial"
                            ? "Partial analysis available"
                            : "Round analysis ready"
                          : "Review this recording"}
                    </strong>
                    <p>
                      {analysis.running
                        ? "You can watch while analysis runs."
                        : analysis.report
                          ? analysis.report.completeness.status === "partial"
                            ? analysis.report.completeness.reason
                            : "An experimental second pass, available to compare."
                          : "Analyze punches and combinations on this device."}
                    </p>
                  </div>
                  <div className="analysis-actions">
                    {analysis.running ? (
                      <>
                        <span className="analysis-progress-text">
                          {Math.round((analysis.progress?.fraction ?? 0) * 100)}
                          %
                        </span>
                        <button
                          className="text-button"
                          onClick={analysis.cancel}
                        >
                          Cancel analysis
                        </button>
                      </>
                    ) : (
                      <>
                        {analysis.report && (
                          <button
                            className="text-button"
                            onClick={() => {
                              if (useVideoAnalysis) setRecheck(null);
                              setUseVideoAnalysis((old) => !old);
                              setShowPredictions(true);
                            }}
                          >
                            {useVideoAnalysis
                              ? "Show original"
                              : "Show video analysis"}
                          </button>
                        )}
                        {!analysis.report && (
                          <button
                            className="button secondary"
                            disabled={analysis.loadingSaved}
                            onClick={() => void analysis.start()}
                          >
                            Analyze recording
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </div>
                {analysis.running && (
                  <progress
                    aria-label="Round analysis progress"
                    max={1}
                    value={analysis.progress?.fraction ?? 0}
                  />
                )}
                {analysis.error && (
                  <p className="analysis-error" role="alert">
                    {analysis.error}
                  </p>
                )}
                {analysis.report && !analysis.saved && (
                  <p className="analysis-error">
                    Analysis is available in this tab. Export it before closing.
                  </p>
                )}
              </div>
            )}
            {overlay && !frame && (
              <p className="muted" role="status">
                No recent pose sample at this moment. The overlay is hidden.
              </p>
            )}
            {overlay && frame && (
              <p className="tracking-description">
                L · {tracking?.left.assessable ? "tracked" : "uncertain"}{" "}
                <span> / </span> R ·{" "}
                {tracking?.right.assessable ? "tracked" : "uncertain"}. Check L
                / R follow your physical hands. Visibility does not confirm hand
                identity or technique.
              </p>
            )}
            <div className="detections">
              <div className="detection-heading">
                <h2>
                  {detectionEvents.length} detected{" "}
                  {detectionEvents.length === 1 ? "punch" : "punches"}
                </h2>
                <span>
                  {videoReport
                    ? videoReport.completeness.status === "partial"
                      ? "Partial video analysis · Experimental"
                      : "Video analysis · Experimental"
                    : updated
                      ? "Updated analysis · Experimental"
                      : "Original detections · Experimental"}
                </span>
              </div>
              <p>
                These counts can miss punches or count other movement. Compare
                with the video.
              </p>
              {!videoReport &&
                (updated || selected.detectorVersion !== DETECTOR_VERSION) && (
                  <>
                    <button
                      className="text-button"
                      onClick={() =>
                        setRecheck(updated ? null : recheckDetections(selected))
                      }
                      disabled={!updated && selected.frames.length === 0}
                    >
                      {updated ? "Use saved detections" : "Recheck detections"}
                    </button>
                    <p>Uses saved tracking; does not rerun pose.</p>
                  </>
                )}
              <div className="review-punch-counts" aria-label="Punch counts">
                {punchCounts.map((punch) => (
                  <div
                    key={punch.notation}
                    className={`review-count punch-${punch.notation}`}
                  >
                    <span className="review-count-name">
                      <span>{punch.notation}</span>
                      {punch.name}
                    </span>
                    <strong>{punch.count}</strong>
                  </div>
                ))}
              </div>
              {showPredictions && (
                <>
                  {detectionEvents.length > 0 && duration > 0 && (
                    <div
                      className="review-punch-timeline"
                      aria-label="Punch timeline"
                    >
                      <div className="review-timeline-track">
                        {detectionEvents.map((event) => (
                          <button
                            key={event.id}
                            className={`review-punch-tick punch-${punchNotation(event)}`}
                            style={{
                              left: `${Math.max(0, Math.min(100, (event.peakMs / duration) * 100))}%`,
                            }}
                            aria-label={`${punchName(event)} at ${(event.peakMs / 1000).toFixed(2)} seconds`}
                            title={`${punchNotation(event)} · ${punchName(event)} · ${(event.peakMs / 1000).toFixed(2)}s`}
                            onClick={() => stepTo(event.startMs)}
                          />
                        ))}
                        <span
                          className="review-timeline-cursor"
                          aria-hidden="true"
                          style={{
                            left: `${Math.max(0, Math.min(100, (time / duration) * 100))}%`,
                          }}
                        />
                      </div>
                      <div className="review-timeline-labels">
                        <span>0:00</span>
                        <span>Click a punch to replay it</span>
                        <span>{formatTime(duration)}</span>
                      </div>
                    </div>
                  )}
                  <div
                    className="event-timeline review-punch-list"
                    aria-label="Detected punches"
                  >
                    {detectionEvents.length ? (
                      detectionEvents.map((event) => (
                        <button
                          key={event.id}
                          className={`event-chip ${event.label} punch-${punchNotation(event)} ${visibleDetections.includes(event) ? "is-current" : ""}`}
                          onClick={() => stepTo(event.startMs)}
                        >
                          <b className="review-punch-number" aria-hidden="true">
                            {punchNotation(event)}
                          </b>
                          <strong>{punchName(event)}</strong>
                          <span>
                            {(event.peakMs / 1000).toFixed(2)}s · {event.hand}
                          </span>
                        </button>
                      ))
                    ) : (
                      <span className="muted">
                        No punches were detected. Missed actions may still be
                        visible in the video.
                      </span>
                    )}
                  </div>
                </>
              )}
              {showPredictions && combinations.length > 0 && (
                <div className="combination-section">
                  <h3>Combinations</h3>
                  <p>
                    Sequences of detected punches; this does not grade
                    technique.
                  </p>
                  <div
                    className="event-timeline"
                    aria-label="Detected combinations"
                  >
                    {combinations.map((combo) => (
                      <button
                        key={combo.id}
                        className="event-chip combo"
                        onClick={() => stepTo(combo.startMs)}
                      >
                        <strong>{combinationLabel(combo.name)}</strong>
                        <span>{(combo.startMs / 1000).toFixed(1)}s</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {showPredictions && !!uncertaintyIntervals.length && (
                <details
                  className="uncertainty-details"
                  key={`uncertainty-${videoReport?.id ?? selected.id}`}
                >
                  <summary>
                    Uncertain tracking · {uncertaintyIntervals.length} moments
                  </summary>
                  <p>
                    Check the video at these moments. An uncertain track cannot
                    establish a technique error.
                  </p>
                  <div className="event-timeline">
                    {uncertaintyIntervals
                      .slice(0, 12)
                      .map((interval, index) => (
                        <button
                          key={index}
                          className="event-chip uncertain-moment"
                          onClick={() => stepTo(interval.startMs)}
                        >
                          <strong>
                            {interval.hand === "both"
                              ? "Both arms"
                              : interval.hand === "left"
                                ? "Left arm"
                                : "Right arm"}
                          </strong>
                          <span>
                            {(interval.startMs / 1000).toFixed(1)}–
                            {(interval.endMs / 1000).toFixed(1)}s
                          </span>
                        </button>
                      ))}
                  </div>
                  {uncertaintyIntervals.length > 12 && (
                    <p>
                      Showing the first 12 moments.
                      {videoReport &&
                        " The full timeline is included in the analysis export."}
                    </p>
                  )}
                </details>
              )}
            </div>
            {!selected.video && selected.source !== "demo" && (
              <p className="legacy-note">
                This round has tracking only; no video was saved. Record a new
                camera round to check missed punches against the original
                footage.
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
                {analysis.report && (
                  <button
                    className="button secondary"
                    onClick={() =>
                      downloadBlob(
                        new Blob(
                          [
                            JSON.stringify({
                              ...analysis.report,
                              combinations: groupCombinations(
                                analysis.report!.events,
                                {
                                  stance: selected.stance,
                                  uncertaintyIntervals:
                                    analysis.report!.uncertaintyIntervals,
                                },
                              ),
                            }),
                          ],
                          { type: "application/json" },
                        ),
                        `corner-${selected.id}-video-analysis.json`,
                      )
                    }
                  >
                    <Download size={15} />
                    Export video analysis
                  </button>
                )}
                <button
                  className="button secondary"
                  onClick={() => exportSession(selected)}
                >
                  <Download size={15} />
                  Evidence JSON
                </button>
                {updated && (
                  <button
                    className="button secondary"
                    onClick={() =>
                      downloadBlob(
                        new Blob([JSON.stringify(updated)], {
                          type: "application/json",
                        }),
                        `corner-${selected.id}-updated-analysis.json`,
                      )
                    }
                  >
                    <Download size={15} />
                    Export updated analysis
                  </button>
                )}
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
              {analysis.report && (
                <div className="analysis-details">
                  <p>
                    Video analysis uses its own tracking. Counts can differ from
                    capture and may still be wrong.
                  </p>
                  <p>
                    Video pass: {analysis.report.frames.length} frames ·{" "}
                    {(analysis.report.cadence.processingMs / 1000).toFixed(1)}s{" "}
                    processing · {analysis.report.model} /{" "}
                    {analysis.report.delegate}. Detector:{" "}
                    {analysis.report.detectorVersion}.
                  </p>
                  {analysis.report.completeness.status === "partial" && (
                    <p>
                      Partial analysis: {analysis.report.completeness.reason}
                    </p>
                  )}
                  {analysis.report.warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                  <button
                    className="text-button"
                    disabled={analysis.running}
                    onClick={() => void analysis.start()}
                  >
                    Analyze again
                  </button>
                </div>
              )}
              {updated && (
                <p>
                  Updated analysis exports detections only. Evidence JSON keeps
                  the original saved results.
                </p>
              )}
              <p>Original capture details</p>
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
                  <dd>
                    {selected.detectorVersion ?? "Original baseline (v1)"}
                  </dd>
                </div>
              </dl>
              <p>
                Evidence JSON contains this round’s original detections.
                Software updates do not rewrite saved results. Local browser
                storage is not a backup.
              </p>
            </details>
          </div>
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
