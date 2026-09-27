export type Stance = "orthodox" | "southpaw";
export type ModelVariant = "lite" | "full" | "heavy";
export type SourceKind = "camera" | "file" | "demo";
export interface Landmark {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
  presence?: number;
}
export interface PoseFrame {
  t: number; // source-relative milliseconds, unmirrored
  width: number;
  height: number;
  landmarks: Landmark[];
  worldLandmarks?: Landmark[];
  inferenceMs: number;
  frameAgeMs?: number; // from observed browser frame callback, not sensor exposure
}
export interface PunchEvent {
  id: string;
  hand: "left" | "right";
  role: "lead" | "rear";
  label: "jab" | "cross";
  startMs: number;
  peakMs: number;
  endMs: number;
  score: number; // heuristic signal score, NOT calibrated confidence
  extension: number;
  // Spatial return to this event's origin observed by detectedAtMs; later return
  // is not assessed, and this is not a technique-quality judgment.
  guardReturn: "returned" | "not-observed" | "unassessable";
  experimental: true;
  detectedAtMs?: number; // source time when the causal engine finalized the event
}
export interface QualityState {
  assessable: boolean;
  label: string;
  reasons: string[];
  visibleJoints: number;
  totalJoints: number;
}
export interface EngineResult {
  quality: QualityState;
  events: PunchEvent[];
  activeHand: "left" | "right" | null;
}
export interface EngineOptions {
  stance: Stance;
  calibrated: boolean;
}
export interface SessionAnnotation {
  id: string;
  startMs: number;
  endMs: number;
  label: "jab" | "cross" | "hook" | "uppercut" | "other" | "unobservable";
  hand: "left" | "right" | "unknown";
  note: string;
}
export interface Session {
  id: string;
  createdAt: string;
  source: SourceKind;
  stance: Stance;
  model: ModelVariant | "synthetic";
  drill: string;
  durationMs: number;
  frames: PoseFrame[];
  events: PunchEvent[];
  annotations: SessionAnnotation[];
  video?: Blob;
  videoOffsetMs?: number;
  measuredFps: number;
  inferenceP95: number;
  skippedFrames: number;
  schemaVersion: "1.0";
  annotationsComplete?: boolean;
  modelManifest?: unknown;
  detectorVersion?: string;
  capture?: {
    width: number;
    height: number;
    requestedFps?: number;
    deliveredFps?: number;
    timingSource: string;
    delegate: string;
  };
}
export const JOINT = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
} as const;
