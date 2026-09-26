import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execute = promisify(execFile);
test("batch CLI preserves evidence, scores unchanged labels and refuses to replace an experiment", async () => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "corner-benchmark-test-"),
  );
  try {
    const input = path.join(directory, "fixture.json");
    // Format/integration fixture only: an empty observation, not real footage or
    // a recognition-accuracy claim. No synthetic punches are labeled as real.
    const fixture = {
      id: "empty-file-format-fixture",
      schemaVersion: "1.0",
      source: "file",
      model: "full",
      stance: "orthodox",
      durationMs: 1000,
      frames: [
        { t: 0, width: 640, height: 360, inferenceMs: 0, landmarks: [] },
      ],
      events: [],
      annotations: [],
      annotationsComplete: true,
    };
    const original = JSON.stringify(fixture);
    await writeFile(input, original);
    const output = path.join(directory, "run");
    const args = ["scripts/benchmark.mjs", input, "--output-dir", output];
    const result = await execute(process.execPath, args);
    assert.match(result.stdout, /comparison complete/);
    const report = JSON.parse(
      await readFile(path.join(output, "comparison.json")),
    );
    assert.equal(report.current.tp, 0);
    assert.equal(report.current.recall, null);
    assert.equal(report.sessions[0].sourceSessionId, fixture.id);
    assert.deepEqual(report.sessions[0].lostAnnotationIds, []);
    assert.equal(await readFile(input, "utf8"), original);
    await assert.rejects(execute(process.execPath, args), /EEXIST/);
    fixture.annotationsComplete = false;
    await writeFile(input, JSON.stringify(fixture));
    await assert.rejects(
      execute(process.execPath, [
        "scripts/benchmark.mjs",
        input,
        "--output-dir",
        path.join(directory, "incomplete"),
      ]),
      /Complete independent annotations required/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
