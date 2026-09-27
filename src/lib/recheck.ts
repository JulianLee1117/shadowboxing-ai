import { DETECTOR_VERSION, MotionEngine } from "./motion";
import type { PunchEvent, Session, Stance } from "./types";

/** A derived report, deliberately separate from a saved Session/evidence export. */
export interface DetectorRecheckReport {
  reportType: "detector-recheck";
  sessionId: string;
  detectorVersion: string;
  sourceDetectorVersion: string | null;
  createdAt: string;
  trackingSource: "saved-frames";
  stance: Stance;
  frameCount: number;
  events: PunchEvent[];
}

/** Reuse saved tracking in its original order; never mutate or persist the round. */
export function recheckDetections(
  session: Pick<Session, "id" | "stance" | "detectorVersion" | "frames">,
): DetectorRecheckReport {
  if (session.frames.some((frame) => frame.recognition))
    throw new Error(
      "Saved learned decisions cannot be recomputed from tracking alone. Analyze the video with the local recognizer.",
    );
  const engine = new MotionEngine({ stance: session.stance, calibrated: true });
  const events = session.frames.flatMap((frame) => engine.update(frame).events);
  return {
    reportType: "detector-recheck",
    sessionId: session.id,
    detectorVersion: DETECTOR_VERSION,
    sourceDetectorVersion: session.detectorVersion ?? null,
    createdAt: new Date().toISOString(),
    trackingSource: "saved-frames",
    stance: session.stance,
    frameCount: session.frames.length,
    events,
  };
}
