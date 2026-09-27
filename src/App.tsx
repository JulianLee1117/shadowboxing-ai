import { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  Circle,
  Maximize2,
  Minimize2,
  Square,
  Upload,
  X,
} from "lucide-react";
import { useStudio } from "./hooks/useStudio";
import { PoseOverlay } from "./components/PoseOverlay";
import { RoundReview } from "./components/RoundReview";
import { LiveFeedback } from "./components/LiveFeedback";
import { assessArmTracking } from "./lib/motion";
import { ReadinessGate } from "./lib/readiness";
import { DRILLS, type DrillId } from "./lib/drills";
import { formatTime, listSessions, saveSession } from "./lib/storage";
import type { ModelVariant, Session, Stance } from "./lib/types";

function App() {
  const [view, setView] = useState<"practice" | "review">("practice");
  const [stance, setStance] = useState<Stance>("orthodox");
  const [duration, setDuration] = useState(30);
  const [drill, setDrill] = useState<DrillId>("open");
  const [model, setModel] = useState<ModelVariant>("full");
  const [overlay, setOverlay] = useState(false);
  const [focused, setFocused] = useState(false);
  const [sound, setSound] = useState(true);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<Session | null>(null);
  const [autoAnalyzeId, setAutoAnalyzeId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [saveStates, setSaveStates] = useState<
    Record<string, "saving" | "saved" | "error">
  >({});
  const saveRevisions = useRef<Record<string, number>>({});
  const [finishing, setFinishing] = useState(false);
  const gate = useRef(new ReadinessGate());
  const [readiness, setReadiness] = useState(() => gate.current.cancel());
  const uploadRef = useRef<HTMLInputElement>(null);
  const audio = useRef<AudioContext | null>(null);
  const lastBeep = useRef(-1);
  const arenaRef = useRef<HTMLDivElement>(null);
  const wasFullscreen = useRef(false);
  const persist = useCallback(async (session: Session) => {
    const revision = (saveRevisions.current[session.id] ?? 0) + 1;
    saveRevisions.current[session.id] = revision;
    setSaveStates((old) => ({ ...old, [session.id]: "saving" }));
    try {
      await saveSession(session);
      if (saveRevisions.current[session.id] === revision)
        setSaveStates((old) => ({ ...old, [session.id]: "saved" }));
    } catch (e) {
      if (saveRevisions.current[session.id] !== revision) return;
      setSaveStates((old) => ({ ...old, [session.id]: "error" }));
      setNotice(
        e instanceof Error
          ? e.message
          : "Could not save. Export this round before closing.",
      );
    }
  }, []);

  const onComplete = useCallback(
    (session: Session) => {
      setSessions((old) => [
        session,
        ...old.filter((s) => s.id !== session.id),
      ]);
      setSelected(session);
      // A failed import may still preserve its original video in a zero-length
      // round. Retain that evidence without launching another failing analysis.
      setAutoAnalyzeId(
        session.video &&
          session.source !== "demo" &&
          Number.isFinite(session.durationMs) &&
          session.durationMs > 0
          ? session.id
          : null,
      );
      setView("review");
      void persist(session);
    },
    [persist],
  );
  const studio = useStudio(onComplete);

  const leaveFocus = useCallback(() => {
    setFocused(false);
    if (document.fullscreenElement === arenaRef.current)
      void document.exitFullscreen().catch(() => {});
  }, []);
  const toggleFocus = async () => {
    if (focused) return leaveFocus();
    setFocused(true);
    // The viewport-sized focus view also works when native fullscreen is
    // unavailable or declined. No camera permission is requested here.
    try {
      await arenaRef.current?.requestFullscreen?.();
    } catch {
      /* Keep the in-page focus view. */
    }
  };
  useEffect(() => {
    const changed = () => {
      const current = document.fullscreenElement === arenaRef.current;
      if (wasFullscreen.current && !current) setFocused(false);
      wasFullscreen.current = current;
    };
    document.addEventListener("fullscreenchange", changed);
    return () => document.removeEventListener("fullscreenchange", changed);
  }, []);
  useEffect(() => {
    if (!focused) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") leaveFocus();
      if (event.key === "Tab") {
        const controls = Array.from(
          arenaRef.current?.querySelectorAll<HTMLElement>(
            "button:not(:disabled), select:not(:disabled), input:not(:disabled), [tabindex='0']",
          ) ?? [],
        ).filter((element) => element.getClientRects().length > 0);
        const first = controls[0];
        const last = controls.at(-1);
        if (
          first &&
          last &&
          (!arenaRef.current?.contains(document.activeElement) ||
            (event.shiftKey && document.activeElement === first) ||
            (!event.shiftKey && document.activeElement === last))
        ) {
          event.preventDefault();
          (event.shiftKey ? last : first).focus();
        }
      }
    };
    document.addEventListener("keydown", keydown);
    return () => {
      document.body.style.overflow = previous;
      document.removeEventListener("keydown", keydown);
    };
  }, [focused, leaveFocus]);
  useEffect(() => {
    if (view !== "practice") leaveFocus();
  }, [view, leaveFocus]);

  useEffect(() => {
    void listSessions()
      .then((saved) =>
        setSessions((current) => [
          ...current,
          ...saved.filter((s) => !current.some((c) => c.id === s.id)),
        ]),
      )
      .catch((e: Error) => setNotice(e.message));
    return () => {
      void audio.current?.close();
    };
  }, []);
  // Completion can come from the timer, a clip ending, or Stop & save.
  // Always release capture before showing the saved round.
  useEffect(() => {
    if (view === "review" && studio.status !== "off") void studio.stop();
  }, [view, studio.status, studio.stop]);

  const cancelCountdown = useCallback(() => {
    setReadiness(gate.current.cancel());
    lastBeep.current = -1;
  }, []);
  const beep = useCallback(
    (start = false) => {
      if (!sound || !audio.current || audio.current.state !== "running") return;
      const context = audio.current;
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.frequency.value = start ? 880 : 540;
      gain.gain.setValueAtTime(0.12, context.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.2);
      oscillator.connect(gain).connect(context.destination);
      oscillator.start();
      oscillator.stop(context.currentTime + 0.22);
    },
    [sound],
  );
  const begin = useCallback(async () => {
    beep(true);
    await studio.beginRound({
      stance,
      drill,
      durationSeconds: duration,
      record: true,
      // Stance is explicitly selected. Per-arm observability gates detections,
      // never video recording; this is not a technique calibration.
      calibrated: true,
    });
  }, [beep, studio.beginRound, stance, duration, drill]);
  const latestBegin = useRef(begin);
  latestBegin.current = begin;
  useEffect(() => {
    if (!readiness.pending) return;
    const tick = () => {
      const next = gate.current.update({ nowMs: performance.now() });
      setReadiness(next);
      if (
        next.countdownSeconds > 0 &&
        next.countdownSeconds <= 3 &&
        next.countdownSeconds !== lastBeep.current
      ) {
        lastBeep.current = next.countdownSeconds;
        beep();
      }
      if (next.start) void latestBegin.current();
    };
    const timer = window.setInterval(tick, 80);
    const hidden = () => {
      if (document.hidden) cancelCountdown();
    };
    document.addEventListener("visibilitychange", hidden);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", hidden);
    };
  }, [readiness.pending, beep, cancelCountdown]);
  useEffect(() => {
    if (studio.status !== "ready") cancelCountdown();
  }, [studio.status, cancelCountdown]);

  const startSource = async (kind: "camera" | "demo" | "file", file?: File) => {
    cancelCountdown();
    setNotice(null);
    await studio.start(kind, model, file);
  };
  const record = () => {
    if (sound) {
      try {
        audio.current ??= new AudioContext();
        void audio.current.resume().catch(() => {});
      } catch {
        /* Visual countdown remains available if audio is unsupported. */
      }
    }
    if (studio.source === "camera")
      setReadiness(gate.current.arm(performance.now()));
    else void begin();
  };
  const stop = async () => {
    cancelCountdown();
    setFinishing(true);
    try {
      await studio.stop();
    } finally {
      setFinishing(false);
    }
  };
  const navigate = async (next: "practice" | "review") => {
    cancelCountdown();
    await studio.stop();
    if (next === "review" && !selected) setSelected(sessions[0] ?? null);
    setView(next);
  };
  const updateSession = async (session: Session) => {
    setSelected(session);
    setSessions((old) => old.map((s) => (s.id === session.id ? session : s)));
    await persist(session);
  };
  const tracking = studio.frame ? assessArmTracking(studio.frame) : null;
  const active = studio.running || readiness.pending || finishing;

  return (
    <div
      className={`app-shell ${view === "practice" ? "practice-shell" : "review-shell"}`}
    >
      <header className="app-header">
        <a
          className="wordmark"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            if (!active && studio.status !== "loading")
              void navigate("practice");
          }}
          aria-label="Corner home"
        >
          <span className="corner-mark" />
          Corner<span className="preview-tag">preview</span>
        </a>
        <nav aria-label="Main navigation">
          <button
            aria-current={view === "practice" ? "page" : undefined}
            onClick={() => void navigate("practice")}
            disabled={active || studio.status === "loading"}
          >
            Practice
          </button>
          <button
            aria-current={view === "review" ? "page" : undefined}
            onClick={() => void navigate("review")}
            disabled={active || studio.status === "loading"}
          >
            Saved rounds
            {sessions.length > 0 && (
              <span className="nav-count">{sessions.length}</span>
            )}
          </button>
        </nav>
      </header>
      <main>
        {(notice || studio.error) && (
          <div className="notice" role="alert">
            <span>{notice || studio.error}</span>
            <button
              aria-label="Dismiss message"
              onClick={() => {
                setNotice(null);
                studio.clearError();
              }}
            >
              <X size={16} />
            </button>
          </div>
        )}
        {/* Keep the capture element mounted for source lifecycle and ended events. */}
        <section hidden={view !== "practice"} className="practice">
          <div className="page-heading">
            <div>
              <h1>Practice</h1>
              <p>{DRILLS[drill].prompt}</p>
            </div>
            <span className="local-badge">
              <span /> On your device
            </span>
          </div>
          <div
            ref={arenaRef}
            className={`practice-arena ${focused ? "is-focused" : ""} ${studio.running ? "is-recording" : ""}`}
          >
            <div
              className={`camera-stage ${studio.source === "demo" ? "demo-stage" : ""}`}
            >
              <video
                ref={studio.videoRef}
                muted
                playsInline
                className={
                  studio.source === "demo" || studio.status === "off"
                    ? "capture-hidden"
                    : ""
                }
                style={{
                  transform:
                    studio.source === "camera" ? "scaleX(-1)" : undefined,
                }}
              />
              {(overlay || studio.source === "demo") &&
                studio.status === "ready" && (
                  <PoseOverlay
                    frame={studio.frame}
                    mirror={studio.source === "camera"}
                    silhouette={studio.source === "demo"}
                  />
                )}
              {studio.status === "off" && (
                <div className="stage-empty">
                  <div className="camera-symbol">
                    <Camera size={28} strokeWidth={1.4} />
                  </div>
                  <h2>Your space to practice.</h2>
                  <p>Enable the webcam to get started.</p>
                  <button
                    className="button primary"
                    onClick={() => void startSource("camera")}
                  >
                    <Camera size={17} />
                    Enable camera
                  </button>
                </div>
              )}
              {studio.status === "loading" && (
                <div className="stage-message">
                  <span className="spinner" />
                  <h2>Preparing the camera & tracking…</h2>
                  <p>This can take a moment the first time.</p>
                </div>
              )}
              <button
                className="focus-toggle"
                onClick={() => void toggleFocus()}
                aria-label={focused ? "Exit focus view" : "Focus view"}
                aria-pressed={focused}
                title={focused ? "Exit focus view (Esc)" : "Focus view"}
              >
                {focused ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
                <span>{focused ? "Exit focus" : "Focus view"}</span>
              </button>
              {readiness.pending && (
                <div className="countdown-overlay" aria-live="polite">
                  <span className="countdown-number">
                    {readiness.countdownSeconds}
                  </span>
                  <h2>Step back into position</h2>
                  <p>
                    Recording starts automatically. Keep your hands in view.
                  </p>
                </div>
              )}
              {studio.status === "ready" && !readiness.pending && (
                <div className="stage-topline">
                  <span className="stage-pill">
                    {studio.running ? (
                      <>
                        <i className="record-dot" />
                        {studio.source === "camera" ? "Recording" : "Running"}
                      </>
                    ) : studio.source === "demo" ? (
                      "Simulated demo"
                    ) : studio.source === "file" ? (
                      "Imported clip"
                    ) : (
                      "Camera ready"
                    )}
                  </span>
                  {studio.running && (
                    <div
                      className="round-clock"
                      role="timer"
                      aria-label="Round time remaining"
                    >
                      <span>
                        {formatTime(
                          Math.max(0, duration * 1000 - studio.elapsed),
                        )}
                      </span>
                      <small>remaining</small>
                    </div>
                  )}
                </div>
              )}
              {studio.running && !readiness.pending && (
                <LiveFeedback
                  events={studio.events}
                  elapsedMs={studio.elapsed}
                  trackingUnclear={
                    !tracking?.left.assessable && !tracking?.right.assessable
                  }
                />
              )}
              {studio.status === "ready" && !readiness.pending && (
                <div
                  className={`tracking-strip ${studio.running ? "tracking-live" : ""}`}
                  aria-label="Arm tracking"
                >
                  <span
                    className={
                      tracking?.left.assessable ? "tracked" : "uncertain"
                    }
                  >
                    L · {tracking?.left.assessable ? "tracked" : "uncertain"}
                  </span>
                  <span
                    className={
                      tracking?.right.assessable ? "tracked" : "uncertain"
                    }
                  >
                    R · {tracking?.right.assessable ? "tracked" : "uncertain"}
                  </span>
                </div>
              )}
            </div>
            <div className="practice-controls">
              <div className="round-settings">
                <label>
                  Lead hand
                  <select
                    aria-label="Lead hand"
                    value={stance}
                    disabled={active || studio.status === "loading"}
                    onChange={(e) => setStance(e.target.value as Stance)}
                  >
                    <option value="orthodox">Left hand leads</option>
                    <option value="southpaw">Right hand leads</option>
                  </select>
                </label>
                <label>
                  Round
                  <select
                    aria-label="Round duration"
                    value={duration}
                    disabled={active || studio.status === "loading"}
                    onChange={(e) => setDuration(Number(e.target.value))}
                  >
                    <option value={30}>30 seconds</option>
                    <option value={60}>1 minute</option>
                    <option value={120}>2 minutes</option>
                    <option value={180}>3 minutes</option>
                  </select>
                </label>
              </div>
              <div className="action-group">
                {readiness.pending ? (
                  <button
                    className="button secondary"
                    onClick={cancelCountdown}
                  >
                    Cancel countdown
                  </button>
                ) : studio.running || finishing ? (
                  <button
                    className="button stop"
                    disabled={finishing}
                    onClick={() => void stop()}
                  >
                    <Square size={15} fill="currentColor" />
                    {finishing ? "Saving…" : "Stop & save"}
                  </button>
                ) : studio.status === "ready" ? (
                  <button className="button primary" onClick={record}>
                    <Circle size={14} fill="currentColor" />
                    {studio.source === "camera"
                      ? "Record round"
                      : studio.source === "file"
                        ? "Analyze clip"
                        : "Start demo"}
                  </button>
                ) : null}
                {studio.status !== "off" && !active && (
                  <button className="text-button" onClick={() => void stop()}>
                    {studio.source === "camera"
                      ? "Stop camera"
                      : "Close source"}
                  </button>
                )}
              </div>
            </div>
          </div>
          <div className="practice-note">
            {studio.source === "file"
              ? "Imported clip stays local. Analysis saves the original video with its tracking."
              : studio.source === "demo"
                ? "Synthetic motion for trying the controls. No camera or accuracy measurement."
                : "Record round gives you 8 seconds to step back, then saves video + tracking locally. No microphone."}
          </div>
          <details className="options" open={active ? false : undefined}>
            <summary>More options</summary>
            <div className="options-content">
              <label>
                Practice focus
                <select
                  aria-label="Practice focus"
                  value={drill}
                  disabled={active || studio.status === "loading"}
                  onChange={(e) => setDrill(e.target.value as DrillId)}
                >
                  {Object.entries(DRILLS).map(([id, value]) => (
                    <option value={id} key={id}>
                      {value.label}
                    </option>
                  ))}
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
                  checked={sound}
                  disabled={active || studio.status === "loading"}
                  onChange={(e) => setSound(e.target.checked)}
                />
                Countdown sound
              </label>
              <label>
                Model
                <select
                  aria-label="Model"
                  value={model}
                  disabled={studio.status !== "off" || active}
                  onChange={(e) => setModel(e.target.value as ModelVariant)}
                >
                  <option value="full">Full</option>
                  <option value="lite">Lite</option>
                  <option value="heavy">Heavy</option>
                </select>
              </label>
              <button
                className="text-button"
                disabled={active || studio.status === "loading"}
                onClick={() => uploadRef.current?.click()}
              >
                <Upload size={15} />
                Open video
              </button>
              <button
                className="text-button"
                disabled={active || studio.status === "loading"}
                onClick={() => void startSource("demo")}
              >
                Try demo
              </button>
            </div>
            <p>
              Frame your head through hips, leaving room for both arms to
              extend. L / R are the model’s hand labels: check they follow your
              physical hands. Amber means uncertain tracking. A visible hand can
              still be mislabeled.
            </p>
          </details>
          <input
            ref={uploadRef}
            type="file"
            accept="video/*"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file && !active && studio.status !== "loading")
                void startSource("file", file);
            }}
          />
        </section>
        {view === "review" && (
          <RoundReview
            sessions={sessions}
            selected={selected}
            autoAnalyze={selected?.id === autoAnalyzeId}
            onAutoAnalysisHandled={() => setAutoAnalyzeId(null)}
            saveState={
              selected ? (saveStates[selected.id] ?? "saved") : "saved"
            }
            onSelect={setSelected}
            onUpdate={updateSession}
            onDeleted={(id) => {
              setSessions((old) => old.filter((s) => s.id !== id));
              setSelected(null);
            }}
            onNotice={setNotice}
            onPractice={() => void navigate("practice")}
          />
        )}
      </main>
      <footer>Local video. Experimental punch recognition.</footer>
    </div>
  );
}
export default App;
