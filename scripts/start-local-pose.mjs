#!/usr/bin/env node
// Explicit foreground process only. No installs, downloads or login items.
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python =
  process.env.LOCAL_POSE_PYTHON || path.join(root, ".venv/bin/python");
if (!existsSync(python)) {
  console.error(
    "Local Python environment is missing. Install ml/requirements-extract.txt in .venv first, or set LOCAL_POSE_PYTHON to an existing Python executable.",
  );
  process.exit(2);
}
const child = spawn(
  python,
  ["-m", "ml.pose_service", ...process.argv.slice(2)],
  {
    cwd: root,
    stdio: "inherit",
  },
);
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}
child.on("error", (error) => {
  console.error(error.message);
  process.exitCode = 2;
});
child.on("exit", (code, signal) => {
  process.exitCode = code ?? (signal ? 1 : 0);
});
