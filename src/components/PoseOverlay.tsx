import type { PoseFrame } from "../lib/types";

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
  const point = (i: number) => {
    const p = frame.landmarks[i];
    return p &&
      (p.visibility ?? 0) > 0.5 &&
      Number.isFinite(p.x) &&
      Number.isFinite(p.y)
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
              className={`pose-line ${a % 2 ? "lead" : "rear"}`}
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
            className={i % 2 ? "pose-joint lead" : "pose-joint rear"}
          />
        ) : null;
      })}
      {[15, 16].map((i) => {
        const p = point(i);
        return p ? (
          <text
            key={`label-${i}`}
            transform={`translate(${p.x * frame.width} ${p.y * frame.height - 20}) scale(${mirror ? -1 : 1} 1)`}
            fill="#eff9e1"
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
