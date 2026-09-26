import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "vite";
import { replayEvidence } from "./lib/replay-evidence.mjs";

let server, options, source;
before(async () => {
  server = await createServer({
    configFile: false,
    server: { middlewareMode: true, watch: null },
    appType: "custom",
    logLevel: "error",
  });
  const motion = await server.ssrLoadModule("/src/lib/motion.ts");
  const { demoFrame } = await server.ssrLoadModule("/src/lib/demo.ts");
  options = {
    MotionEngine: motion.MotionEngine,
    detectorVersion: motion.DETECTOR_VERSION,
    detectorFingerprint: "a".repeat(64),
    sessionFingerprint: "b".repeat(64),
  };
  source = {
    id: "fixture",
    schemaVersion: "1.0",
    source: "demo",
    model: "synthetic",
    stance: "orthodox",
    durationMs: 6400,
    frames: Array.from({ length: 161 }, (_, i) => demoFrame(i * 40)),
    events: [{ id: "original-only" }],
    annotations: [],
    annotationsComplete: false,
    detectorVersion: "saved-version",
  };
});
after(async () => server?.close());

function deepFreeze(obj) {
  if (obj && typeof obj === "object") {
    Object.freeze(obj);
    for (const value of Object.values(obj)) deepFreeze(value);
  }
  return obj;
}
function replacement() {
  return {
    artifactType: "research-pose-series",
    schemaVersion: "1.0",
    source: { sha256: "c".repeat(64) },
    modelManifest: { family: "fixture-foreign-pose", sha256: "d".repeat(64) },
    timestampMode: "decoded-pts",
    truncatedByFrameLimit: false,
    frames: structuredClone(source.frames),
  };
}
function replaceOptions(poses = replacement()) {
  return {
    ...options,
    poses,
    posesFingerprint: "e".repeat(64),
    videoFingerprint: "c".repeat(64),
    modelId: "fixture-foreign",
  };
}
test("real engine replay keeps frozen evidence/labels unchanged and does not invent completeness", () => {
  const original = deepFreeze(structuredClone(source));
  const output = replayEvidence(original, options);
  assert.ok(output.events.length > 0);
  assert.equal(output.annotationsComplete, false);
  assert.equal(output.source, "demo");
  assert.equal(output.model, "synthetic");
  assert.deepEqual(original.events, [{ id: "original-only" }]);
  assert.notEqual(output.id, original.id);
  assert.equal(output.benchmark.sourceDetectorVersion, "saved-version");
  assert.deepEqual(replayEvidence(original, options).events, output.events);
});
test("external poses retain honest model/provenance and unknown capture skips", () => {
  const output = replayEvidence(source, replaceOptions());
  assert.equal(output.model, "fixture-foreign");
  assert.equal(output.source, "demo"); // synthetic cannot become a real benchmark
  assert.equal(output.skippedFrames, null);
  assert.equal(output.benchmark.trackingSource, "replacement-pose-series");
  assert.equal(output.benchmark.sourceVideoSha256, "c".repeat(64));
  assert.equal(output.modelManifest.family, "fixture-foreign-pose");
});
test("rejects wrong videos, truncated runs, missing provenance and browser-model impersonation", () => {
  for (const patch of [
    { videoFingerprint: "f".repeat(64) },
    { modelId: "full" },
    { posesFingerprint: undefined },
    { modelId: "" },
  ])
    assert.throws(() =>
      replayEvidence(source, { ...replaceOptions(), ...patch }),
    );
  for (const patch of [
    { truncatedByFrameLimit: true },
    { truncatedByFrameLimit: undefined },
    { modelManifest: {} },
    { timestampMode: undefined },
  ])
    assert.throws(() =>
      replayEvidence(source, replaceOptions({ ...replacement(), ...patch })),
    );
});
test("rejects short or invalid series instead of scoring missing data as a full run", () => {
  assert.throws(
    () =>
      replayEvidence(
        source,
        replaceOptions({
          ...replacement(),
          frames: source.frames.slice(0, 40),
        }),
      ),
    /cover the full round/,
  );
  const frames = structuredClone(source.frames);
  frames[5].t = frames[4].t;
  assert.throws(
    () => replayEvidence({ ...source, frames }, options),
    /strictly increasing/,
  );
  frames[5].t = 200;
  frames[5].landmarks[0].x = NaN;
  assert.throws(
    () => replayEvidence({ ...source, frames }, options),
    /landmark/,
  );
  const malformed = replacement();
  malformed.frames[5].t = NaN;
  assert.throws(
    () => replayEvidence(source, replaceOptions(malformed)),
    /finite, nonnegative and strictly increasing/,
  );
});
test("explicit timestamp alignment preserves the supplied origin and exact replay events", () => {
  const poses = replacement();
  poses.frames = poses.frames.map((f) => ({ ...f, t: f.t + 500 }));
  const output = replayEvidence(source, {
    ...replaceOptions(poses),
    poseTimeOffsetMs: -500,
  });
  assert.equal(output.frames[0].t, 0);
  assert.equal(output.benchmark.poseTimeOffsetMs, -500);
  assert.deepEqual(output.events, replayEvidence(source, options).events);
  assert.throws(
    () => replayEvidence(source, replaceOptions(poses)),
    /cover the full round/,
  );
});
