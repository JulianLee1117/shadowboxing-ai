import { parseArgs } from "node:util";
import { promisify } from "node:util";
import { execFile } from "node:child_process";
import { readFile, writeFile, mkdir, realpath } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const execute = promisify(execFile);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    "output-dir": { type: "string" },
    python: { type: "string", default: "python3" },
    labels: { type: "string", default: "jab,cross,hook,uppercut" },
    help: { type: "boolean" },
  },
});
if (values.help || !positionals.length || !values["output-dir"]) {
  console.log(
    "Usage: npm run benchmark -- labeled-a.json labeled-b.json --output-dir data/pilot/runs/unique-run [--python python3] [--labels jab,cross,hook,uppercut]\nCompares saved detections with current rules on the same frames and labels. The default recall scope includes all six punches. Requires complete real-session annotations and a new output directory. No uploads or model downloads.",
  );
  process.exit(values.help ? 0 : 2);
}
try {
  const inputs = await Promise.all(positionals.map((p) => realpath(p)));
  for (const input of inputs) {
    const session = JSON.parse(await readFile(input, "utf8"));
    if (session.annotationsComplete !== true)
      throw new Error(`Complete independent annotations required: ${input}`);
    if (session.source === "demo" || session.model === "synthetic")
      throw new Error("Synthetic sessions are not a real-video benchmark.");
  }
  const outputDir = path.resolve(values["output-dir"]);
  await mkdir(path.dirname(outputDir), { recursive: true });
  // Exclusive directory creation preserves every previous experiment.
  await mkdir(outputDir);
  const baselinePath = path.join(outputDir, "saved-report.json");
  await execute(
    values.python,
    [
      "-m",
      "ml.evaluate",
      ...inputs,
      "--labels",
      values.labels,
      "--output",
      baselinePath,
    ],
    { cwd: root },
  );
  const derived = [];
  let detectorHash;
  for (const [index, input] of inputs.entries()) {
    const output = path.join(outputDir, `replay-${index + 1}.json`);
    await execute(
      process.execPath,
      [
        path.join(root, "scripts/replay-session.mjs"),
        input,
        "--output",
        output,
      ],
      { cwd: root },
    );
    const session = JSON.parse(await readFile(output, "utf8"));
    const currentHash = session.benchmark.detectorSourceSha256;
    if (detectorHash && currentHash !== detectorHash)
      throw new Error(
        "Detector source changed during the run; retry in a new directory after editing finishes.",
      );
    detectorHash = currentHash;
    derived.push(output);
  }
  const currentPath = path.join(outputDir, "current-report.json");
  await execute(
    values.python,
    [
      "-m",
      "ml.evaluate",
      ...derived,
      "--labels",
      values.labels,
      "--output",
      currentPath,
    ],
    { cwd: root },
  );
  const saved = JSON.parse(await readFile(baselinePath, "utf8"));
  const current = JSON.parse(await readFile(currentPath, "utf8"));
  const comparisons = saved.sessions.map((before, index) => {
    const after = current.sessions[index];
    const beforeIds = new Set(
      before.eventMetrics.matches.map((m) => m.annotationId),
    );
    const afterIds = new Set(
      after.eventMetrics.matches.map((m) => m.annotationId),
    );
    return {
      sourceSessionId: before.sessionId,
      saved: before.eventMetrics,
      current: after.eventMetrics,
      recoveredAnnotationIds: [...afterIds].filter((id) => !beforeIds.has(id)),
      lostAnnotationIds: [...beforeIds].filter((id) => !afterIds.has(id)),
    };
  });
  const summary = {
    artifactType: "detector-development-comparison",
    createdAt: new Date().toISOString(),
    detectorSourceSha256: detectorHash,
    protocol: saved.protocol,
    limitations:
      "Development regression on unchanged poses and labels; not held-out accuracy, model retraining, or coaching validation. Saved sessions may contain different detector versions.",
    saved: saved.eventMetrics,
    current: current.eventMetrics,
    sessions: comparisons,
  };
  await writeFile(
    path.join(outputDir, "comparison.json"),
    JSON.stringify(summary, null, 2) + "\n",
    { flag: "wx" },
  );
  console.log(
    `Saved: ${saved.eventMetrics.tp} matched, ${saved.eventMetrics.fp} unmatched, ${saved.eventMetrics.fn} missed.`,
  );
  console.log(
    `Current: ${current.eventMetrics.tp} matched, ${current.eventMetrics.fp} unmatched, ${current.eventMetrics.fn} missed.`,
  );
  console.log(`Development comparison complete → ${outputDir}`);
} catch (error) {
  console.error(`Benchmark failed: ${error.stderr || error.message}`);
  process.exitCode = 2;
}
