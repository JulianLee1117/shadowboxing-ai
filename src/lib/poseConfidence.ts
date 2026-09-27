import type { Landmark, PoseFrame } from "./types";

/** Native scores and MediaPipe visibility have different meanings. A native
 * frame must supply its observation policy rather than inheriting MediaPipe's.
 * Passing this gate never proves joint identity or coordinate accuracy.
 */
export function observationScore(
  frame: PoseFrame,
  point: Landmark | undefined,
): number {
  const score = frame.estimator ? point?.score : point?.visibility;
  return Number.isFinite(score) ? score! : 0;
}

export function observationThreshold(
  frame: PoseFrame,
  mediaPipeThreshold = 0.65,
): number {
  if (!frame.estimator) return mediaPipeThreshold;
  const policy = frame.estimator;
  return policy.scoreType === "simcc" &&
    ["rtmpose-m", "rtmw-l"].includes(policy.id) &&
    Number.isFinite(policy.minimumScore) &&
    policy.minimumScore > 0 &&
    policy.minimumScore < 1
    ? policy.minimumScore
    : Infinity;
}

export function observedPoint(
  frame: PoseFrame,
  point: Landmark | undefined,
  mediaPipeThreshold = 0.65,
): point is Landmark {
  return (
    !!point &&
    Number.isFinite(point.x) &&
    Number.isFinite(point.y) &&
    point.x >= 0 &&
    point.x <= 1 &&
    point.y >= 0 &&
    point.y <= 1 &&
    observationScore(frame, point) >=
      observationThreshold(frame, mediaPipeThreshold) &&
    (frame.estimator !== undefined ||
      point.presence === undefined ||
      (Number.isFinite(point.presence) && point.presence >= 0.5))
  );
}
