import type { PoseFrame } from "../lib/types";
import { MOTION_LIMITS } from "../lib/motion";
import { observationScore, observationThreshold } from "../lib/poseConfidence";

const edges = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [27, 31],
  [28, 32],
];
export function PoseOverlay({
  frame,
  mirror,
  silhouette = false,
}: {
  frame: PoseFrame | null;
  mirror: boolean;
  silhouette?: boolean;
}) {
  if (!frame) return null;
  const uncertain = (i: number) => {
    const p = frame.landmarks[i];
    if (
      observationScore(frame, p) <
      observationThreshold(frame, MOTION_LIMITS.minimumVisibility)
    )
      return true;
    if (
      !frame.estimator &&
      p?.presence !== undefined &&
      (!Number.isFinite(p.presence) || p.presence < 0.5)
    )
      return true;
    return false;
  };
  const point = (i: number) => {
    const p = frame.landmarks[i];
    // Visualization has a lower floor than recognition. Show weak observed
    // coordinates as uncertain; never fill missing joints or change evidence.
    const displayFloor = frame.estimator ? 0.2 : 0.3;
    return p &&
      Number.isFinite(p.x) &&
      Number.isFinite(p.y) &&
      p.x >= 0 &&
      p.x <= 1 &&
      p.y >= 0 &&
      p.y <= 1 &&
      observationScore(frame, p) >= displayFloor &&
      (p.presence === undefined ||
        (Number.isFinite(p.presence) && p.presence >= 0.3))
      ? p
      : null;
  };
  return (
    <svg
      className={`pose-overlay ${silhouette ? "pose-demo" : ""}`}
      viewBox={`0 0 ${frame.width} ${frame.height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
      style={{ transform: mirror ? "scaleX(-1)" : undefined }}
    >
      {silhouette && point(0) && (
        <ellipse
          cx={point(0)!.x * frame.width}
          cy={point(0)!.y * frame.height}
          rx="28"
          ry="35"
          className="demo-head"
        />
      )}
      {edges.map(([a, b]) => {
        const p = point(a),
          q = point(b);
        return p && q ? (
          <g key={`${a}-${b}`}>
            <line
              className="pose-halo"
              x1={p.x * frame.width}
              y1={p.y * frame.height}
              x2={q.x * frame.width}
              y2={q.y * frame.height}
            />
            <line
              className={`pose-line ${a % 2 === b % 2 ? (a % 2 ? "hand-left" : "hand-right") : "body"} ${uncertain(a) || uncertain(b) ? "uncertain" : ""}`}
              x1={p.x * frame.width}
              y1={p.y * frame.height}
              x2={q.x * frame.width}
              y2={q.y * frame.height}
            />
          </g>
        ) : null;
      })}
      {[11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].map((i) => {
        const p = point(i);
        return p ? (
          <circle
            key={i}
            cx={p.x * frame.width}
            cy={p.y * frame.height}
            r={i === 15 || i === 16 ? 9 : 5}
            className={`pose-joint ${i % 2 ? "hand-left" : "hand-right"} ${uncertain(i) ? "uncertain" : ""}`}
          />
        ) : null;
      })}
      {[15, 16].map((i) => {
        const p = point(i);
        return p ? (
          <text
            key={`label-${i}`}
            transform={`translate(${p.x * frame.width} ${p.y * frame.height - 20}) scale(${mirror ? -1 : 1} 1)`}
            fill={uncertain(i) ? "#ffd28c" : i === 15 ? "#80d8ff" : "#ffac75"}
            stroke="#101820"
            strokeWidth="3"
            paintOrder="stroke"
            textAnchor="middle"
            fontSize="18"
            fontFamily="monospace"
          >
            {i === 15 ? "L" : "R"}
          </text>
        ) : null;
      })}
    </svg>
  );
}
