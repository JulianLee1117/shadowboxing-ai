import { useCallback, useEffect, useRef, useState } from "react";
import {
  analyzeRound,
  ANALYSIS_VERSION,
  type RoundAnalysisReport,
  type AnalysisProgress,
} from "../lib/roundAnalysis";
import { loadRoundAnalysis, saveRoundAnalysis } from "../lib/storage";
import { fingerprintVideo } from "../lib/mediaFingerprint";
import type { ModelVariant, Session } from "../lib/types";

/** One selected-round job. Navigation and unmount invalidate every async callback. */
export function useRoundAnalysis(session: Session | null) {
  const [report, setReport] = useState<RoundAnalysisReport | null>(null);
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [loadingSaved, setLoadingSaved] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(true);
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const latestSession = useRef(session);
  const reportVideo = useRef<Blob | undefined>(undefined);
  latestSession.current = session;
  const id = session?.id;
  const video = session?.video;
  const stance = session?.stance;
  const offsetMs = session?.videoOffsetMs ?? 0;
  const durationMs = session?.durationMs;

  useEffect(() => {
    const current = ++generation.current;
    controller.current?.abort();
    controller.current = null;
    setReport(null);
    reportVideo.current = undefined;
    setProgress(null);
    setRunning(false);
    setError(null);
    setSaved(true);
    setLoadingSaved(!!id);
    if (id)
      void loadRoundAnalysis(id)
        .then(async (loaded) => {
          if (generation.current !== current) return;
          if (
            loaded?.sourceSessionId === id &&
            loaded.stance === stance &&
            loaded.analysisVersion === ANALYSIS_VERSION &&
            loaded.provenance.videoBytes === video?.size &&
            loaded.provenance.videoOffsetMs === offsetMs &&
            loaded.provenance.sourceDurationMs === durationMs &&
            video &&
            loaded.sourceFingerprint === (await fingerprintVideo(video)) &&
            generation.current === current
          ) {
            reportVideo.current = video;
            setReport(loaded);
          }
        })
        .catch(() => {
          // A failed cache read does not prevent a fresh analysis of in-memory video.
          if (generation.current === current)
            setError(
              "Saved analysis could not be loaded. You can analyze the recording again.",
            );
        })
        .finally(() => {
          if (generation.current === current) setLoadingSaved(false);
        });
    return () => {
      generation.current++;
      controller.current?.abort();
      controller.current = null;
    };
  }, [id, video, stance, offsetMs, durationMs, ANALYSIS_VERSION]);

  const cancel = useCallback(() => {
    generation.current++;
    controller.current?.abort();
    controller.current = null;
    setRunning(false);
    setProgress(null);
  }, []);

  const start = useCallback(async (model?: ModelVariant) => {
    const source = latestSession.current;
    if (!source?.video || source.source === "demo" || controller.current)
      return;
    const current = ++generation.current;
    const abort = new AbortController();
    controller.current = abort;
    setRunning(true);
    setLoadingSaved(false);
    setProgress(null);
    setError(null);
    try {
      const result = await analyzeRound(source, {
        model,
        signal: abort.signal,
        onProgress: (next) => {
          if (generation.current === current) setProgress(next);
        },
      });
      if (generation.current !== current || abort.signal.aborted) return;
      reportVideo.current = source.video;
      setReport(result);
      setSaved(false);
      try {
        await saveRoundAnalysis(result);
        if (generation.current === current) setSaved(true);
      } catch (saveError) {
        if (generation.current === current)
          setError(
            saveError instanceof Error
              ? saveError.message
              : "Analysis could not be saved. Export it before closing.",
          );
      }
    } catch (failure) {
      if (generation.current === current && !abort.signal.aborted)
        setError(
          failure instanceof Error
            ? failure.message
            : "Could not analyze this recording.",
        );
    } finally {
      if (generation.current === current) {
        controller.current = null;
        setRunning(false);
        setProgress(null);
      }
    }
  }, []);

  return {
    report:
      report &&
      report.sourceSessionId === id &&
      report.stance === stance &&
      reportVideo.current === video &&
      report.analysisVersion === ANALYSIS_VERSION &&
      report.provenance.videoOffsetMs === offsetMs &&
      report.provenance.sourceDurationMs === durationMs
        ? report
        : null,
    progress,
    running,
    loadingSaved,
    error,
    saved,
    start,
    cancel,
  };
}
