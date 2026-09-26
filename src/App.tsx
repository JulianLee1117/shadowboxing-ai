import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Camera,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Circle,
  Crosshair,
  Film,
  FlaskConical,
  FolderOpen,
  Headphones,
  Info,
  Maximize2,
  Play,
  Plus,
  ScanLine,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Timer,
  Trash2,
  Upload,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { useStudio } from "./hooks/useStudio";
import { PoseOverlay } from "./components/PoseOverlay";
import { demoFrame } from "./lib/demo";
import {
  deleteSession,
  downloadBlob,
  exportSession,
  formatTime,
  listSessions,
  percentile,
  saveSession,
} from "./lib/storage";
import type {
  ModelVariant,
  Session,
  SessionAnnotation,
  Stance,
} from "./lib/types";

const DRILLS = [
  {
    id: "jab",
    title: "Find your jab",
    subtitle: "Single lead straights",
    number: "01",
    tags: "FOUNDATIONS",
    sequence: ["Jab"],
    tip: "Throw a comfortable lead straight, then settle back into your chosen guard. Leave a little space between repetitions.",
  },
  {
    id: "cross",
    title: "Build your cross",
    subtitle: "Single rear straights",
    number: "02",
    tags: "FOUNDATIONS",
    sequence: ["Cross"],
    tip: "Practice a controlled rear straight. Reset between repetitions and keep both hands inside the camera view.",
  },
  {
    id: "one-two",
    title: "Connect the 1–2",
    subtitle: "Jab → cross",
    number: "03",
    tags: "COMBINATIONS",
    sequence: ["Jab", "Cross"],
    tip: "Link a jab and a cross at a comfortable pace. Pause briefly between combinations. The timeline shows what was detected.",
  },
  {
    id: "open",
    title: "Move your way",
    subtitle: "Open practice & capture",
    number: "04",
    tags: "EXPLORATION",
    sequence: ["Move", "Reset"],
    tip: "Use the camera and replay to explore your movement. Only experimental straight-punch events are currently recognized; other techniques need manual labels.",
  },
];
type View = "studio" | "review" | "lab";

function App() {
  const [view, setView] = useState<View>("studio");
  const [drill, setDrill] = useState("one-two");
  const [stance, setStance] = useState<Stance>("orthodox");
  const [duration, setDuration] = useState(120);
  const [model, setModel] = useState<ModelVariant>("full");
  const [mirror, setMirror] = useState(true);
  const [overlay, setOverlay] = useState(true);
  const [record, setRecord] = useState(false);
  const [sound, setSound] = useState(false);
  const [sideConfirmed, setSideConfirmed] = useState(false);
  const [calibrated, setCalibrated] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<Session | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [completeId, setCompleteId] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<"saving" | "saved" | "error">(
    "saved",
  );
  const [showPredictions, setShowPredictions] = useState(false);
  const [reviewTime, setReviewTime] = useState(0);
  const [reviewVideo, setReviewVideo] = useState<string | null>(null);
  const [annotationLabel, setAnnotationLabel] =
    useState<SessionAnnotation["label"]>("jab");
  const [annotationHand, setAnnotationHand] =
    useState<SessionAnnotation["hand"]>("left");
  const [annotationStart, setAnnotationStart] = useState(0);
  const [annotationEnd, setAnnotationEnd] = useState(1);
  const [annotationNote, setAnnotationNote] = useState("");
  const [help, setHelp] = useState(false);
  const uploadRef = useRef<HTMLInputElement>(null);
  const reviewRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const activeDrill = DRILLS.find((d) => d.id === drill)!;
  const replayOffsetMs = selected?.videoOffsetMs ?? 0;
  const replayDurationMs = selected?.durationMs ?? 0;
  const syncReviewVideo = useCallback(
    (video: HTMLVideoElement) => {
      const start = replayOffsetMs / 1000;
      const end = (replayOffsetMs + replayDurationMs) / 1000;
      const bounded = Math.max(start, Math.min(end, video.currentTime));
      if (video.currentTime >= end) video.pause();
      if (Math.abs(video.currentTime - bounded) > 0.0001)
        video.currentTime = bounded;
      setReviewTime(
        Math.max(
          0,
          Math.min(replayDurationMs, bounded * 1000 - replayOffsetMs),
        ),
      );
    },
    [replayOffsetMs, replayDurationMs],
  );

  const onComplete = useCallback((session: Session) => {
    setSessions((old) => [session, ...old.filter((s) => s.id !== session.id)]);
    setSelected(session);
    setCompleteId(session.id);
    setSaveState("saving");
    void saveSession(session)
      .then(() => setSaveState("saved"))
      .catch((e) => {
        setSaveState("error");
        setNotice(e.message);
      });
  }, []);
  const studio = useStudio(onComplete);
  useEffect(() => {
    void listSessions()
      .then(setSessions)
      .catch((e) => setNotice(e.message));
  }, []);
  useEffect(() => {
    if (!selected?.video) {
      setReviewVideo(null);
      return;
    }
    const url = URL.createObjectURL(selected.video);
    setReviewVideo(url);
    return () => URL.revokeObjectURL(url);
  }, [selected?.id, selected?.video]);
  useEffect(() => {
    setReviewTime(0);
    setShowPredictions(false);
  }, [selected?.id]);
  useEffect(() => {
    const video = reviewRef.current;
    if (view !== "review" || !video || !reviewVideo) return;
    let stopped = false;
    let frameCallback: number | null = null;
    let animationFrame = 0;
    const schedule = () => {
      if (video.requestVideoFrameCallback) {
        frameCallback = video.requestVideoFrameCallback(() => {
          if (stopped) return;
          syncReviewVideo(video);
          schedule();
        });
      } else {
        animationFrame = requestAnimationFrame(() => {
          if (stopped) return;
          if (!video.paused) syncReviewVideo(video);
          schedule();
        });
      }
    };
    schedule();
    return () => {
      stopped = true;
      if (frameCallback !== null)
        video.cancelVideoFrameCallback?.(frameCallback);
      cancelAnimationFrame(animationFrame);
    };
  }, [view, reviewVideo, syncReviewVideo]);
  useEffect(() => {
    if (!sound || !studio.running || !("speechSynthesis" in window)) return;
    const announce = () => {
      const voices = window.speechSynthesis.getVoices();
      const voice =
        voices.find((v) => v.localService && v.lang.startsWith("en")) ??
        voices.find((v) => v.localService);
      if (!voice) {
        setNotice(
          "No local speech voice is available. Visual drill prompts are still available.",
        );
        return;
      }
      const words = new SpeechSynthesisUtterance(
        activeDrill.sequence.join(", "),
      );
      words.voice = voice;
      words.rate = 0.9;
      window.speechSynthesis.speak(words);
    };
    announce();
    const timer = window.setInterval(announce, 6500);
    return () => {
      clearInterval(timer);
      window.speechSynthesis.cancel();
    };
  }, [sound, studio.running, activeDrill]);

  const navigate = async (next: View) => {
    if (next !== "studio") {
      await studio.stop();
      setCalibrated(false);
      setSideConfirmed(false);
    }
    setView(next);
  };
  const startSource = async (source: "camera" | "demo", file?: File) => {
    setCompleteId(null);
    setCalibrated(false);
    setSideConfirmed(false);
    await studio.start(file ? "file" : source, model, file);
  };
  const importVideo = async (file?: File) => {
    if (!file) return;
    if (file.size > 500 * 1024 * 1024) {
      setNotice(
        "Choose a clip smaller than 500 MB. Short clips are easier to inspect.",
      );
      return;
    }
    setView("studio");
    await startSource("camera", file);
  };
  const begin = () => {
    setCompleteId(null);
    studio.beginRound({
      stance,
      drill,
      durationSeconds: duration,
      record,
      calibrated,
    });
  };
  const seek = (ms: number) => {
    if (!selected) return;
    const next = Math.max(0, Math.min(selected.durationMs, ms));
    setReviewTime(next);
    if (reviewRef.current && reviewVideo)
      reviewRef.current.currentTime =
        (next + (selected.videoOffsetMs ?? 0)) / 1000;
  };
  const updateSession = async (session: Session) => {
    setSelected(session);
    setSessions((old) => old.map((s) => (s.id === session.id ? session : s)));
    try {
      await saveSession(session);
    } catch (e) {
      setNotice(
        e instanceof Error
          ? e.message
          : "Could not save this change. Export to preserve your labels.",
      );
    }
  };
  const addAnnotation = () => {
    if (!selected) return;
    if (
      !Number.isFinite(annotationStart) ||
      !Number.isFinite(annotationEnd) ||
      annotationStart < 0 ||
      annotationEnd <= annotationStart ||
      annotationEnd * 1000 > selected.durationMs + 1
    ) {
      setNotice("Choose a valid start and end inside this round.");
      return;
    }
    const annotation: SessionAnnotation = {
      id: crypto.randomUUID(),
      startMs: annotationStart * 1000,
      endMs: annotationEnd * 1000,
      label: annotationLabel,
      hand: annotationHand,
      note: annotationNote.trim(),
    };
    void updateSession({
      ...selected,
      annotationsComplete: false,
      annotations: [...selected.annotations, annotation].sort(
        (a, b) => a.startMs - b.startMs,
      ),
    });
    setAnnotationNote("");
  };
  const remove = async () => {
    if (!selected) return;
    try {
      await deleteSession(selected.id);
      setSessions((old) => old.filter((s) => s.id !== selected.id));
      setSelected(null);
    } catch {
      setNotice("Could not delete this round from local storage.");
    }
  };
  const nearestFrame = selected?.frames.reduce(
    (best, f) =>
      Math.abs(f.t - reviewTime) < Math.abs(best.t - reviewTime) ? f : best,
    selected.frames[0],
  );
  // A frozen pose is misleading over video that was not actually observed.
  const currentFrame =
    nearestFrame && Math.abs(nearestFrame.t - reviewTime) <= 100
      ? nearestFrame
      : null;
  const leadCount = studio.events.filter((e) => e.role === "lead").length;
  const rearCount = studio.events.filter((e) => e.role === "rear").length;
  const canBegin =
    studio.status === "ready" &&
    (studio.source === "demo" || studio.source === "file" || calibrated);
  const motionOnly = !selected?.video;
  const title =
    view === "studio"
      ? "Your next good round."
      : view === "review"
        ? "See what happened."
        : "Earn every observation.";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#studio"
          onClick={(e) => {
            e.preventDefault();
            void navigate("studio");
          }}
        >
          <span className="brand-mark">
            <i />
            <i />
          </span>
          <span>
            corner<span className="brand-dot">.</span>
          </span>
        </a>
        <div className="sidebar-caption">YOUR PRACTICE SPACE</div>
        <nav aria-label="Main navigation">
          <button
            className={`nav-item ${view === "studio" ? "active" : ""}`}
            onClick={() => void navigate("studio")}
          >
            <ScanLine size={19} /> <span>Training studio</span>
            <span className="nav-active-dot" />
          </button>
          <button
            className={`nav-item ${view === "review" ? "active" : ""}`}
            onClick={() => void navigate("review")}
          >
            <Film size={19} /> <span>Round review</span>
            {sessions.length > 0 && (
              <span className="nav-count">{sessions.length}</span>
            )}
          </button>
          <button
            className={`nav-item ${view === "lab" ? "active" : ""}`}
            onClick={() => void navigate("lab")}
          >
            <FlaskConical size={19} /> <span>Measurement lab</span>
          </button>
        </nav>
        <div className="sidebar-note">
          <span className="tiny-label">A LITTLE BETTER, EVERY ROUND</span>
          <p>
            One focus.
            <br />A few good reps.
            <br />
            Something to build on.
          </p>
          <div className="round-lines">
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
        </div>
        <div className="sidebar-bottom">
          <div className="local-badge">
            <ShieldCheck size={17} />
            <span>Private by default</span>
          </div>
          <p>
            Camera analysis stays
            <br />
            on this device.
          </p>
          <button className="text-button" onClick={() => setHelp(true)}>
            About this prototype <ArrowRight size={14} />
          </button>
          <div className="version">CORNER / LOCAL PREVIEW 0.1</div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div className="breadcrumb">
            WORKSPACE <span>/</span>{" "}
            <strong>
              {view === "studio"
                ? "TRAINING"
                : view === "review"
                  ? "REVIEW"
                  : "RESEARCH"}
            </strong>
          </div>
          <div className="topbar-status">
            <span className="status-dot" /> LOCAL SESSION{" "}
            <span className="topbar-divider" /> <span>NO ACCOUNT NEEDED</span>
          </div>
        </header>
        <div className="page-heading">
          <div>
            <span className="eyebrow">
              {view === "studio"
                ? "SHOW UP. FIND YOUR RHYTHM."
                : view === "review"
                  ? "OBSERVE. LABEL. LEARN."
                  : "MEASURE BEFORE YOU TRUST."}
            </span>
            <h1>{title}</h1>
            <p>
              {view === "studio"
                ? "A focused space for shadowboxing, with movement you can replay."
                : view === "review"
                  ? "Your rounds stay on this device. Inspect a moment, add a label, take the evidence with you."
                  : "Understand what the camera sees, and where the prototype still needs validation."}
            </p>
          </div>
          <button
            className="button secondary heading-help"
            onClick={() => setHelp(true)}
          >
            <Info size={16} /> Getting started
          </button>
        </div>
        {(notice || studio.error) && (
          <div className="notice" role="alert">
            <Info size={18} />
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

        <section hidden={view !== "studio"} className="studio-layout">
          <div className="studio-main">
            <div className="camera-stage" ref={stageRef}>
              <video
                ref={studio.videoRef}
                muted
                playsInline
                className={`camera-video ${mirror ? "mirrored" : ""}`}
                style={{
                  visibility:
                    studio.source === "camera" || studio.source === "file"
                      ? "visible"
                      : "hidden",
                }}
              />
              {!studio.source && (
                <div className="stage-idle">
                  <div className="stage-grid" />
                  <div className="idle-target">
                    <span />
                    <span />
                    <span />
                    <span />
                  </div>
                  <PoseOverlay frame={demoFrame(0)} mirror={false} silhouette />
                  <div className="idle-copy">
                    <span className="tiny-label">A SPACE TO GET BETTER</span>
                    <h2>Step into your corner.</h2>
                    <p>
                      Frame your head, hands, and full reach.
                      <br />
                      We’ll take it one round at a time.
                    </p>
                    <button
                      className="button primary"
                      onClick={() => void startSource("camera")}
                    >
                      <Camera size={18} /> Enable camera
                    </button>
                    <button
                      className="demo-link"
                      onClick={() => void startSource("demo")}
                    >
                      Explore a simulated round <ArrowRight size={14} />
                    </button>
                  </div>
                </div>
              )}
              {studio.source === "demo" && (
                <>
                  <div className="stage-grid" />
                  <div className="demo-watermark">
                    MOVEMENT STUDY <span>01 / SYNTHETIC</span>
                  </div>
                </>
              )}
              {overlay && studio.source && (
                <PoseOverlay
                  frame={studio.frame}
                  mirror={studio.source === "demo" ? false : mirror}
                  silhouette={studio.source === "demo"}
                />
              )}
              <div className="stage-top">
                <span
                  className={`stage-source ${studio.source === "demo" ? "demo" : ""}`}
                >
                  <span
                    className={`status-dot ${studio.status === "off" ? "off" : ""}`}
                  />
                  {studio.source === "demo"
                    ? "SIMULATED DEMO"
                    : studio.source === "file"
                      ? "LOCAL VIDEO"
                      : studio.source === "camera"
                        ? "YOUR CAMERA"
                        : "CAMERA OFF"}
                </span>
                <div className="stage-top-right">
                  {studio.status === "ready" && (
                    <span className="stage-fps">
                      {studio.source === "demo"
                        ? "SYNTHETIC"
                        : `${Math.round(studio.fps)} POSE FPS`}
                    </span>
                  )}
                  <button
                    aria-label="Fullscreen camera"
                    onClick={() =>
                      void stageRef.current
                        ?.requestFullscreen?.()
                        .catch(() =>
                          setNotice(
                            "Fullscreen is unavailable in this browser.",
                          ),
                        )
                    }
                  >
                    <Maximize2 size={17} />
                  </button>
                </div>
              </div>
              {studio.status === "loading" && (
                <div className="stage-loading">
                  <span className="spinner" />
                  <h3>Preparing your space</h3>
                  <p>Loading the local pose model…</p>
                </div>
              )}
              {studio.source && studio.status === "ready" && (
                <div className="stage-bottom">
                  <div className="tracking-status">
                    <span
                      className={`status-dot ${studio.quality.assessable ? "" : "warning"}`}
                    />
                    <div>
                      <strong>
                        {studio.source === "demo"
                          ? "Simulated body tracking"
                          : studio.quality.label}
                      </strong>
                      <span>
                        {studio.source === "demo"
                          ? "Interface demo · not an accuracy test"
                          : studio.quality.reasons[0] ||
                            "Visible upper-body landmarks · experimental analysis"}
                      </span>
                    </div>
                  </div>
                  <button
                    className={`overlay-button ${overlay ? "enabled" : ""}`}
                    aria-label="Toggle skeleton overlay"
                    aria-pressed={overlay}
                    onClick={() => setOverlay((v) => !v)}
                  >
                    <ScanLine size={17} /> Skeleton
                  </button>
                </div>
              )}
            </div>
            <div className="camera-toolbar">
              <div className="camera-toolbar-left">
                <ShieldCheck size={16} />
                <span>
                  {studio.source === "file"
                    ? "Imported clip stays local"
                    : record && studio.running && studio.source === "camera"
                      ? "Recording video locally"
                      : "On-device analysis"}
                  <span className="toolbar-dot">·</span>
                  {studio.source === "demo"
                    ? "No camera used"
                    : studio.source === "file"
                      ? "Original saved with analysis"
                      : record
                        ? "Video saves with the round"
                        : "Video recording off"}
                </span>
              </div>
              <div>
                <button
                  className="icon-text"
                  onClick={() => uploadRef.current?.click()}
                  disabled={studio.running}
                >
                  <Upload size={15} /> Import clip
                </button>
                {studio.source && (
                  <button
                    className="icon-text"
                    onClick={() => {
                      void studio.stop();
                      setCalibrated(false);
                    }}
                  >
                    <Square size={13} /> Stop{" "}
                    {studio.source === "camera" ? "camera" : "source"}
                  </button>
                )}
              </div>
            </div>
            <input
              type="file"
              ref={uploadRef}
              accept="video/*"
              hidden
              onChange={(e) => {
                void importVideo(e.target.files?.[0]);
                e.target.value = "";
              }}
            />

            <div className="round-strip">
              <div className="round-clock">
                <Timer size={19} />
                <strong>
                  {formatTime(
                    studio.running
                      ? Math.max(0, duration * 1000 - studio.elapsed)
                      : duration * 1000,
                  )}
                </strong>
                <span>{studio.running ? "REMAINING" : "ROUND LENGTH"}</span>
              </div>
              <div className="round-count">
                <strong>{leadCount.toString().padStart(2, "0")}</strong>
                <span>JAB CANDIDATES</span>
              </div>
              <div className="round-count">
                <strong>{rearCount.toString().padStart(2, "0")}</strong>
                <span>CROSS CANDIDATES</span>
              </div>
              <button
                className={`button ${studio.running ? "danger" : "primary"} round-start`}
                disabled={!studio.running && !canBegin}
                onClick={() =>
                  studio.running ? void studio.finishRound() : begin()
                }
              >
                {studio.running ? <Square size={16} /> : <Play size={17} />}{" "}
                {studio.running
                  ? "Finish round"
                  : studio.source === "file"
                    ? "Analyze clip"
                    : "Start round"}
              </button>
            </div>
            {completeId && (
              <div className="completed-banner">
                <CheckCircle2 size={20} />
                <div>
                  <strong>Your round is ready to review.</strong>
                  <span>
                    {saveState === "saving"
                      ? "Saving to this browser…"
                      : saveState === "saved"
                        ? `Saved locally with movement data${selected?.video ? " and video" : ""}.`
                        : "Local save failed. Export this round before closing."}
                  </span>
                </div>
                <button
                  className="text-button"
                  onClick={() => void navigate("review")}
                >
                  Review round <ArrowRight size={16} />
                </button>
              </div>
            )}

            <div className="section-label">
              <h2>Choose your focus</h2>
              <span>BUILD THE FUNDAMENTALS</span>
            </div>
            <div className="drill-grid">
              {DRILLS.map((d) => (
                <button
                  key={d.id}
                  className={`drill-card ${drill === d.id ? "selected" : ""}`}
                  disabled={studio.running}
                  onClick={() => setDrill(d.id)}
                >
                  <div className="drill-card-top">
                    <span>{d.number}</span>
                    {drill === d.id ? (
                      <CheckCircle2 size={18} />
                    ) : (
                      <ArrowRight size={17} />
                    )}
                  </div>
                  <span className="drill-tag">{d.tags}</span>
                  <h3>{d.title}</h3>
                  <p>{d.subtitle}</p>
                </button>
              ))}
            </div>
            <div className="evidence-note">
              <FlaskConical size={16} />
              <p>
                <strong>Built to show its work.</strong> Punch events are
                experimental candidates. Technique corrections remain off until
                coach-reviewed validation.
              </p>
            </div>
          </div>

          <aside className="session-panel">
            <div className="panel-title">
              <span className="tiny-label">THIS ROUND</span>
              <span className="live-preview">PREVIEW</span>
            </div>
            <h2>{activeDrill.title}</h2>
            <p className="panel-description">{activeDrill.tip}</p>
            <div className="sequence">
              {activeDrill.sequence.map((s, i) => (
                <span key={s}>
                  {i > 0 && <ArrowRight size={15} />}
                  <b>
                    {s === "Jab"
                      ? "1"
                      : s === "Cross"
                        ? "2"
                        : s === "Move"
                          ? "∞"
                          : "↺"}
                  </b>
                  <small>{s}</small>
                </span>
              ))}
            </div>
            <div className="panel-divider" />
            <label className="field-label">YOUR STANCE</label>
            <div className="segmented">
              <button
                aria-pressed={stance === "orthodox"}
                disabled={studio.running}
                onClick={() => {
                  setStance("orthodox");
                  setCalibrated(false);
                }}
              >
                Orthodox
              </button>
              <button
                aria-pressed={stance === "southpaw"}
                disabled={studio.running}
                onClick={() => {
                  setStance("southpaw");
                  setCalibrated(false);
                }}
              >
                Southpaw
              </button>
            </div>
            <p className="field-hint">
              {stance === "orthodox"
                ? "Left hand leads · right hand follows"
                : "Right hand leads · left hand follows"}
            </p>
            <div className="settings-row">
              <label htmlFor="duration">Round length</label>
              <select
                id="duration"
                value={duration}
                disabled={studio.running}
                onChange={(e) => setDuration(Number(e.target.value))}
              >
                <option value={60}>1 minute</option>
                <option value={120}>2 minutes</option>
                <option value={180}>3 minutes</option>
                <option value={600}>10 minutes</option>
              </select>
            </div>
            <div className="settings-row">
              <label htmlFor="model">Pose model</label>
              <select
                id="model"
                value={model}
                disabled={!!studio.source}
                onChange={(e) => setModel(e.target.value as ModelVariant)}
              >
                <option value="full">Full · balanced</option>
                <option value="heavy">Heavy · detailed</option>
                <option value="lite">Lite · lighter</option>
              </select>
            </div>
            <label className="switch-row">
              <span>Mirror preview</span>
              <input
                type="checkbox"
                checked={mirror}
                onChange={(e) => setMirror(e.target.checked)}
              />
              <span className="switch" />
            </label>
            <label className="switch-row">
              <span>
                Save round video <small>Optional · local only</small>
              </span>
              <input
                type="checkbox"
                checked={record}
                disabled={studio.running}
                onChange={(e) => setRecord(e.target.checked)}
              />
              <span className="switch" />
            </label>
            <label className="switch-row">
              <span>
                {sound ? <Volume2 size={15} /> : <VolumeX size={15} />} Drill
                callouts <small>Prompts, not technique judgments</small>
              </span>
              <input
                type="checkbox"
                checked={sound}
                onChange={(e) => setSound(e.target.checked)}
              />
              <span className="switch" />
            </label>
            <div className="panel-divider" />
            <div className="setup-title">
              <Crosshair size={16} />
              <h3>Before the bell</h3>
              {calibrated && <CheckCircle2 size={16} />}
            </div>
            <ol className="setup-list">
              <li>
                <span>01</span>Keep your head and full arm reach in frame.
              </li>
              <li>
                <span>02</span>Use even light. Leave space around your hands.
              </li>
              <li>
                <span>03</span>Confirm the stance and anatomical side mapping.
              </li>
            </ol>
            {studio.source === "camera" &&
              studio.status === "ready" &&
              !calibrated && (
                <div className="calibration">
                  <label>
                    <input
                      type="checkbox"
                      checked={sideConfirmed}
                      onChange={(e) => setSideConfirmed(e.target.checked)}
                    />
                    I raised my left hand and confirmed the overlay follows it.
                  </label>
                  <button
                    className="button secondary"
                    disabled={!sideConfirmed || !studio.quality.assessable}
                    onClick={() => setCalibrated(true)}
                  >
                    <Check size={15} /> Use this setup
                  </button>
                  <small>
                    {!studio.quality.assessable
                      ? "Keep both wrists, elbows, shoulders and hips visible."
                      : "This confirms tracking setup, not correct technique."}
                  </small>
                </div>
              )}
            {calibrated && (
              <div className="setup-ready">
                <Check size={16} /> Setup confirmed. Settle into guard.
              </div>
            )}
            {studio.source === "file" && (
              <div className="calibration">
                <label>
                  <input
                    type="checkbox"
                    checked={calibrated}
                    onChange={(e) => setCalibrated(e.target.checked)}
                  />
                  I confirmed this clip’s stance and unmirrored anatomical
                  sides.
                </label>
                <small>
                  Unconfirmed clips save pose data without classifying punches.
                </small>
              </div>
            )}
            <div className="listen-note">
              <Headphones size={17} />
              <p>
                Keep your eyes on your practice. Review the details after the
                round.
              </p>
            </div>
          </aside>
        </section>

        {view === "review" && (
          <section className="review-layout">
            <aside className="round-library">
              <div className="section-label">
                <h2>Your rounds</h2>
                <span>{sessions.length.toString().padStart(2, "0")}</span>
              </div>
              {sessions.length === 0 ? (
                <div className="library-empty">
                  <Film size={28} />
                  <p>
                    A little practice.
                    <br />
                    Something to look back on.
                  </p>
                  <button
                    className="text-button"
                    onClick={() => void navigate("studio")}
                  >
                    Start a round <ArrowRight size={14} />
                  </button>
                </div>
              ) : (
                sessions.map((s) => (
                  <button
                    key={s.id}
                    className={`session-item ${selected?.id === s.id ? "selected" : ""}`}
                    onClick={() => setSelected(s)}
                  >
                    <span className="session-symbol">
                      {s.source === "demo" ? (
                        <FlaskConical size={18} />
                      ) : (
                        <Film size={18} />
                      )}
                    </span>
                    <span>
                      <strong>
                        {DRILLS.find((d) => d.id === s.drill)?.title || s.drill}
                      </strong>
                      <small>
                        {new Date(s.createdAt).toLocaleDateString(undefined, {
                          month: "short",
                          day: "numeric",
                        })}{" "}
                        · {formatTime(s.durationMs)} ·{" "}
                        {s.source === "demo"
                          ? "Demo"
                          : s.video
                            ? "Video + motion"
                            : "Motion only"}
                      </small>
                    </span>
                    <ChevronRight size={15} />
                  </button>
                ))
              )}
            </aside>
            {selected ? (
              <div className="review-main">
                <div className="review-heading">
                  <div>
                    <span className="tiny-label">
                      {selected.source === "demo"
                        ? "SIMULATED DEMONSTRATION"
                        : "LOCAL ROUND"}{" "}
                      / {selected.stance.toUpperCase()}
                    </span>
                    <h2>
                      {DRILLS.find((d) => d.id === selected.drill)?.title ||
                        selected.drill}
                    </h2>
                  </div>
                  <div className="button-group">
                    <button
                      className="button secondary"
                      onClick={() => exportSession(selected)}
                    >
                      <ArrowDownToLine size={15} /> Evidence JSON
                    </button>
                    {selected.video && (
                      <button
                        className="icon-button"
                        title="Export video"
                        aria-label="Export video"
                        onClick={() =>
                          downloadBlob(
                            selected.video!,
                            `corner-${selected.id}.${selected.video!.type.includes("mp4") ? "mp4" : "webm"}`,
                          )
                        }
                      >
                        <Film size={17} />
                      </button>
                    )}
                    <button
                      className="icon-button delete"
                      aria-label="Delete this round"
                      title="Delete this round"
                      onClick={() => void remove()}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
                <div className="replay-stage">
                  {reviewVideo ? (
                    <video
                      ref={reviewRef}
                      src={reviewVideo}
                      controls
                      playsInline
                      onLoadedMetadata={(e) => {
                        e.currentTarget.currentTime =
                          (selected.videoOffsetMs ?? 0) / 1000;
                        syncReviewVideo(e.currentTarget);
                      }}
                      onSeeking={(e) => syncReviewVideo(e.currentTarget)}
                      onSeeked={(e) => syncReviewVideo(e.currentTarget)}
                      onPlay={(e) => syncReviewVideo(e.currentTarget)}
                      onTimeUpdate={(e) => syncReviewVideo(e.currentTarget)}
                    />
                  ) : (
                    <>
                      <div className="stage-grid" />
                      <div className="replay-label">
                        {selected.source === "demo"
                          ? "SIMULATED REPLAY"
                          : "MOTION REPLAY"}
                        <span>
                          {" "}
                          {selected.source === "demo"
                            ? "Synthetic landmarks · not a benchmark"
                            : "Video was not recorded"}
                        </span>
                      </div>
                    </>
                  )}
                  {currentFrame && (
                    <PoseOverlay
                      frame={currentFrame}
                      mirror={false}
                      silhouette={motionOnly}
                    />
                  )}
                </div>
                {!currentFrame && (
                  <p className="section-description" role="status">
                    No recent pose sample at this moment. The overlay is hidden.
                  </p>
                )}
                <div className="replay-controls">
                  <button
                    className="icon-button"
                    aria-label="Previous frame"
                    onClick={() =>
                      seek(
                        selected.frames
                          .filter((f) => f.t < reviewTime - 1)
                          .at(-1)?.t ?? 0,
                      )
                    }
                  >
                    <ChevronLeft size={18} />
                  </button>
                  <button
                    className="icon-button"
                    aria-label="Next frame"
                    onClick={() =>
                      seek(
                        selected.frames.find((f) => f.t > reviewTime + 1)?.t ??
                          selected.durationMs,
                      )
                    }
                  >
                    <ChevronRight size={18} />
                  </button>
                  <span>{(reviewTime / 1000).toFixed(2)}s</span>
                  <input
                    aria-label="Replay position"
                    type="range"
                    min={0}
                    max={selected.durationMs || 1}
                    step={1}
                    value={Math.min(reviewTime, selected.durationMs)}
                    onChange={(e) => seek(Number(e.target.value))}
                  />
                  <span>{formatTime(selected.durationMs)}</span>
                </div>
                <div className="review-stats">
                  <div>
                    <strong>{selected.events.length}</strong>
                    <span>Straight candidates</span>
                  </div>
                  <div>
                    <strong>
                      {selected.source === "demo"
                        ? "—"
                        : Math.round(selected.measuredFps)}
                    </strong>
                    <span>Processed pose FPS</span>
                  </div>
                  <div>
                    <strong>
                      {selected.source === "demo"
                        ? "—"
                        : `${Math.round(selected.inferenceP95)} ms`}
                    </strong>
                    <span>p95 model inference</span>
                  </div>
                  <div>
                    <strong>{selected.annotations.length}</strong>
                    <span>Manual labels</span>
                  </div>
                </div>
                <div className="section-label">
                  <h2>Movement timeline</h2>
                  <button
                    className="text-button"
                    aria-pressed={showPredictions}
                    onClick={() => setShowPredictions((v) => !v)}
                  >
                    {showPredictions
                      ? "Hide detections"
                      : "Show experimental detections"}
                  </button>
                </div>
                <p className="section-description">
                  Model suggestions can bias annotation. Keep detections hidden
                  when creating reference labels. Select an event to inspect its
                  motion. These are heuristic straight-punch candidates, not
                  validated technique assessments.
                </p>
                <div className="event-timeline" hidden={!showPredictions}>
                  {selected.events.length ? (
                    selected.events.map((event, i) => (
                      <button
                        key={event.id}
                        className="event-chip"
                        onClick={() => {
                          seek(event.startMs);
                          setAnnotationStart(event.startMs / 1000);
                          setAnnotationEnd(event.endMs / 1000);
                        }}
                      >
                        <span className={`event-hand ${event.role}`}>
                          {event.label === "jab" ? "1" : "2"}
                        </span>
                        <span>
                          <strong>
                            {event.label === "jab" ? "Jab" : "Cross"}{" "}
                            <small>candidate {i + 1}</small>
                          </strong>
                          <small>
                            {(event.startMs / 1000).toFixed(2)}–
                            {(event.endMs / 1000).toFixed(2)}s · {event.hand}{" "}
                            hand
                          </small>
                        </span>
                        <ArrowRight size={15} />
                      </button>
                    ))
                  ) : (
                    <div className="empty-events">
                      <Activity size={22} />
                      <span>
                        No straight-punch events were accepted. Inspect the
                        replay and add labels; uncertain movement should not
                        become a guessed punch.
                      </span>
                    </div>
                  )}
                </div>
                <div className="annotation-panel">
                  <div className="section-label">
                    <h2>Add a reference label</h2>
                    <span>MANUAL ANNOTATION</span>
                  </div>
                  <p className="section-description">
                    Label the original video where available. Mark definite
                    false detections “other”; use “unobservable” only when
                    judgment is impossible. Motion-only replays cannot
                    independently validate the pose model. Technique correctness
                    needs coach review.
                  </p>
                  <div className="annotation-fields">
                    <label>
                      Action
                      <select
                        aria-label="Annotation action"
                        value={annotationLabel}
                        onChange={(e) =>
                          setAnnotationLabel(
                            e.target.value as SessionAnnotation["label"],
                          )
                        }
                      >
                        {[
                          "jab",
                          "cross",
                          "hook",
                          "uppercut",
                          "other",
                          "unobservable",
                        ].map((v) => (
                          <option key={v} value={v}>
                            {v}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Hand
                      <select
                        aria-label="Annotation hand"
                        value={annotationHand}
                        onChange={(e) =>
                          setAnnotationHand(
                            e.target.value as SessionAnnotation["hand"],
                          )
                        }
                      >
                        <option>left</option>
                        <option>right</option>
                        <option>unknown</option>
                      </select>
                    </label>
                    <label>
                      Start (s)
                      <input
                        aria-label="Annotation start"
                        type="number"
                        min="0"
                        step="0.01"
                        value={annotationStart}
                        onChange={(e) =>
                          setAnnotationStart(Number(e.target.value))
                        }
                      />
                    </label>
                    <label>
                      End (s)
                      <input
                        aria-label="Annotation end"
                        type="number"
                        min="0"
                        step="0.01"
                        value={annotationEnd}
                        onChange={(e) =>
                          setAnnotationEnd(Number(e.target.value))
                        }
                      />
                    </label>
                  </div>
                  <div className="annotation-note">
                    <input
                      aria-label="Annotation note"
                      placeholder="What is visible? Any uncertainty?"
                      value={annotationNote}
                      maxLength={1000}
                      onChange={(e) => setAnnotationNote(e.target.value)}
                    />
                    <button className="button primary" onClick={addAnnotation}>
                      <Plus size={16} /> Add label
                    </button>
                  </div>
                  {selected.annotations.map((a) => (
                    <div className="annotation-row" key={a.id}>
                      <button onClick={() => seek(a.startMs)}>
                        <span>
                          {(a.startMs / 1000).toFixed(2)}–
                          {(a.endMs / 1000).toFixed(2)}s
                        </span>
                        <strong>
                          {a.label} · {a.hand}
                        </strong>
                        <small>{a.note || "No note"}</small>
                      </button>
                      <button
                        className="icon-button"
                        aria-label="Delete annotation"
                        onClick={() =>
                          void updateSession({
                            ...selected,
                            annotationsComplete: false,
                            annotations: selected.annotations.filter(
                              (x) => x.id !== a.id,
                            ),
                          })
                        }
                      >
                        <X size={15} />
                      </button>
                    </div>
                  ))}
                  <label className="complete-annotation">
                    <input
                      type="checkbox"
                      checked={!!selected.annotationsComplete}
                      onChange={(e) =>
                        void updateSession({
                          ...selected,
                          annotationsComplete: e.target.checked,
                        })
                      }
                    />{" "}
                    I reviewed the entire round and labeled every action,
                    including missed detections.
                  </label>
                </div>
              </div>
            ) : (
              <div className="review-placeholder">
                <FolderOpen size={42} />
                <h2>Your evidence lives here.</h2>
                <p>
                  Record a round or explore the simulated demo.
                  <br />
                  Then replay, label, and export it here.
                </p>
                <button
                  className="button primary"
                  onClick={() => void navigate("studio")}
                >
                  Go to training <ArrowRight size={17} />
                </button>
              </div>
            )}
          </section>
        )}

        {view === "lab" && (
          <section className="lab-layout">
            <div className="lab-banner">
              <div className="lab-icon">
                <FlaskConical size={28} />
              </div>
              <div>
                <span className="tiny-label">MEASUREMENT BEFORE JUDGMENT</span>
                <h2>A coach should know what it can see.</h2>
                <p>
                  This build captures local pose data and proposes
                  straight-punch events. Its detection thresholds are
                  experimental. No validated technique corrections or accuracy
                  claims are enabled.
                </p>
              </div>
              <span className="lab-status">
                <Circle size={10} /> VALIDATION PENDING
              </span>
            </div>
            <div className="lab-grid">
              <article className="lab-card">
                <ScanLine size={22} />
                <h3>Real local perception</h3>
                <p>
                  MediaPipe Pose Landmarker runs in a dedicated worker, using
                  assets served from this device. Full, Heavy, and Lite are
                  available for comparison.
                </p>
                <dl>
                  <div>
                    <dt>Execution</dt>
                    <dd>GPU with CPU fallback</dd>
                  </div>
                  <div>
                    <dt>Coordinates</dt>
                    <dd>Unmirrored, timestamped</dd>
                  </div>
                  <div>
                    <dt>Data path</dt>
                    <dd>Camera → local worker</dd>
                  </div>
                </dl>
              </article>
              <article className="lab-card">
                <Activity size={22} />
                <h3>Transparent event logic</h3>
                <p>
                  A causal motion baseline watches extension and recovery. It
                  uses actual anatomical sides and rejects missing joints or
                  interrupted trajectories.
                </p>
                <dl>
                  <div>
                    <dt>Vocabulary</dt>
                    <dd>Jab / cross candidates</dd>
                  </div>
                  <div>
                    <dt>Technique scores</dt>
                    <dd>Not enabled</dd>
                  </div>
                  <div>
                    <dt>Calibration</dt>
                    <dd>Manual setup confirmation</dd>
                  </div>
                </dl>
              </article>
              <article className="lab-card">
                <ShieldCheck size={22} />
                <h3>Your footage, your choice</h3>
                <p>
                  No account, analytics, or cloud video upload. Video recording
                  is opt-in. Saved rounds and labels live in this browser’s
                  local database.
                </p>
                <dl>
                  <div>
                    <dt>Export</dt>
                    <dd>Evidence JSON + video</dd>
                  </div>
                  <div>
                    <dt>Cloud review</dt>
                    <dd>Not connected</dd>
                  </div>
                  <div>
                    <dt>Delete</dt>
                    <dd>Per round in Review</dd>
                  </div>
                </dl>
              </article>
            </div>
            <div className="lab-bottom">
              <article className="benchmark-panel">
                <div className="section-label">
                  <h2>Build your benchmark</h2>
                  <span>THE NEXT USEFUL STEP</span>
                </div>
                <div className="benchmark-steps">
                  <div>
                    <span>01</span>
                    <div>
                      <h3>Capture across days</h3>
                      <p>
                        Short jab/cross drills, idle movement, different views,
                        and natural mistakes. Keep later sessions untouched for
                        testing.
                      </p>
                    </div>
                  </div>
                  <div>
                    <span>02</span>
                    <div>
                      <h3>Label the full round</h3>
                      <p>
                        Use Review to label actions independently. Include false
                        detections, missed events, and unobservable movement.
                      </p>
                    </div>
                  </div>
                  <div>
                    <span>03</span>
                    <div>
                      <h3>Compare evidence</h3>
                      <p>
                        Export JSON for the offline evaluator. It reports event
                        precision/recall and refuses to call synthetic data a
                        real benchmark.
                      </p>
                    </div>
                  </div>
                </div>
                <button
                  className="button secondary"
                  onClick={() => void navigate("review")}
                >
                  Open round review <ArrowRight size={16} />
                </button>
              </article>
              <article className="coverage-panel">
                <div className="section-label">
                  <h2>Latest round</h2>
                  <span>OBSERVED, NOT INFERRED</span>
                </div>
                {selected ? (
                  <>
                    <p>
                      {selected.source === "demo"
                        ? "Synthetic demo · metrics excluded from accuracy claims"
                        : `${selected.frames.length} processed frames · ${selected.model} model`}
                    </p>
                    <dl>
                      <div>
                        <dt>Duration</dt>
                        <dd>{formatTime(selected.durationMs)}</dd>
                      </div>
                      <div>
                        <dt>p95 observed frame age</dt>
                        <dd>
                          {selected.source === "demo" ||
                          !selected.frames.some((f) =>
                            Number.isFinite(f.frameAgeMs),
                          )
                            ? "—"
                            : `${Math.round(
                                percentile(
                                  selected.frames.flatMap((f) =>
                                    Number.isFinite(f.frameAgeMs)
                                      ? [f.frameAgeMs!]
                                      : [],
                                  ),
                                  0.95,
                                ),
                              )} ms`}
                        </dd>
                      </div>
                      <div>
                        <dt>Skipped while busy</dt>
                        <dd>{selected.skippedFrames}</dd>
                      </div>
                      <div>
                        <dt>Manual labels</dt>
                        <dd>{selected.annotations.length}</dd>
                      </div>
                    </dl>
                    <small>
                      Frame age starts when the browser loop observes a frame.
                      It is not sensor-to-screen latency.
                    </small>
                  </>
                ) : (
                  <div className="library-empty">
                    <SlidersHorizontal size={26} />
                    <p>Finish a round to see measured telemetry here.</p>
                  </div>
                )}
              </article>
            </div>
          </section>
        )}
        <footer>
          <span>
            CORNER <span className="footer-slash">/</span> PRACTICE WITH PURPOSE
          </span>
          <span>
            LOCAL PREVIEW <span className="status-dot" />
          </span>
        </footer>
      </main>
      {help && (
        <div className="modal-backdrop" onClick={() => setHelp(false)}>
          <section
            className="help-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="help-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close icon-button"
              aria-label="Close getting started"
              onClick={() => setHelp(false)}
            >
              <X size={20} />
            </button>
            <span className="tiny-label">WELCOME TO YOUR CORNER</span>
            <h2 id="help-title">A good place to start.</h2>
            <p>
              Enable your camera, choose a stance and drill, and keep your head,
              hips, elbows, and hands visible. Confirm your setup, then start a
              round.
            </p>
            <div className="help-step">
              <Camera />
              <div>
                <strong>You control the camera.</strong>
                <p>
                  Nothing starts until you enable it. Video recording is off by
                  default and no microphone is requested.
                </p>
              </div>
            </div>
            <div className="help-step">
              <Film />
              <div>
                <strong>Review the evidence.</strong>
                <p>
                  Rounds save motion data locally. Turn on “Save round video”
                  before a round if you want video replay too.
                </p>
              </div>
            </div>
            <div className="help-step">
              <FlaskConical />
              <div>
                <strong>This is a research preview.</strong>
                <p>
                  Straight-punch candidates may be missed or misclassified.
                  Other punches need manual labels. Technique critique is
                  waiting for coach-reviewed validation.
                </p>
              </div>
            </div>
            <button className="button primary" onClick={() => setHelp(false)}>
              Find my rhythm <ArrowRight size={17} />
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
export default App;
