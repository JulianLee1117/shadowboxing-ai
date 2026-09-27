import { parseArgs } from "node:util";
import { readFile, writeFile, mkdir, realpath } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { fingerprint, replayEvidence } from "./lib/replay-evidence.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    output: { type: "string" },
    poses: { type: "string" },
    model: { type: "string" },
    video: { type: "string" },
    "pose-offset-ms": { type: "string", default: "0" },
    help: { type: "boolean" },
  },
});
if (values.help || positionals.length !== 1 || !values.output) {
  console.log(
    "Usage: npm run replay -- session.json --output derived.json [--poses research-poses.json --model model-id --video original.webm --pose-offset-ms 0]\nReplays current counting rules locally. Never overwrites an existing file. Then use python3 -m ml.evaluate derived.json --output report.json.",
  );
  process.exit(values.help ? 0 : 2);
}
let server;
try {
  const sessionPath = await realpath(positionals[0]);
  const input = await readFile(sessionPath);
  const session = JSON.parse(input);
  const posesBytes = values.poses ? await readFile(values.poses) : undefined;
  const videoBytes = values.video ? await readFile(values.video) : undefined;
  if (values.poses && !videoBytes)
    throw new Error("Replacement poses require --video for checksum matching.");
  if (
    values.poses &&
    (session.videoOffsetMs ?? 0) !== 0 &&
    !process.argv.some(
      (arg) =>
        arg === "--pose-offset-ms" || arg.startsWith("--pose-offset-ms="),
    )
  )
    throw new Error(
      "This session has a video offset. Supply the verified pose-to-session offset explicitly.",
    );
  server = await createServer({
    root,
    configFile: false,
    server: { middlewareMode: true, watch: null },
    appType: "custom",
    logLevel: "error",
  });
  const { MotionEngine, DETECTOR_VERSION } =
    await server.ssrLoadModule("/src/lib/motion.ts");
  const motion = await readFile(path.join(root, "src/lib/motion.ts"));
  const types = await readFile(path.join(root, "src/lib/types.ts"));
  const curves = await readFile(path.join(root, "src/lib/curvedMotion.ts"));
  const confidence = await readFile(
    path.join(root, "src/lib/poseConfidence.ts"),
  );
  const recognition = await readFile(
    path.join(root, "src/lib/nativeRecognition.ts"),
  );
  const output = replayEvidence(session, {
    MotionEngine,
    detectorVersion: DETECTOR_VERSION,
    detectorFingerprint: fingerprint(
      Buffer.concat([motion, types, curves, confidence, recognition]),
    ),
    sessionFingerprint: fingerprint(input),
    poses: posesBytes ? JSON.parse(posesBytes) : undefined,
    posesFingerprint: posesBytes ? fingerprint(posesBytes) : undefined,
    videoFingerprint: videoBytes ? fingerprint(videoBytes) : undefined,
    modelId: values.model,
    poseTimeOffsetMs: Number(values["pose-offset-ms"]),
  });
  await mkdir(path.dirname(path.resolve(values.output)), { recursive: true });
  await writeFile(values.output, JSON.stringify(output) + "\n", { flag: "wx" });
  console.log(
    `${DETECTOR_VERSION}: ${output.events.length} events from ${output.frames.length} frames → ${values.output}. Original evidence unchanged.`,
  );
} catch (error) {
  console.error(`Replay failed: ${error.message}`);
  process.exitCode = 2;
} finally {
  await server?.close();
}
