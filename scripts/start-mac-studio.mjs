#!/usr/bin/env node
// Foreground supervisor only: no installation, download, unref, or login item.
import { spawn } from "node:child_process";
import { constants } from "node:fs";
import { access, readFile, stat } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const keys = new Set(["manifest", "provider", "minimumScore", "recognizer"]);

export const HELP = `Usage: node scripts/start-mac-studio.mjs [options]

Starts the local pose service and Vite together in this terminal. Ctrl+C stops
both. Existing processes are never reused or stopped. Ports 5173 and 8765 must
be free; these match the app's loopback proxy and allowed origins.

  --manifest FILE       Existing local model manifest (default ml/models.example.json)
  --provider coreml|cpu  Explicit execution provider (default coreml)
  --minimum-score N     Experimental native score policy, 0 < N < 1 (default 0.55)
  --config FILE         Local JSON configuration (default data/local-studio.json if present)
  --recognizer FILE     Optional existing personal-recognizer JSON manifest
  --help                Show this help

Configuration keys: manifest, provider, minimumScore, recognizer. CLI overrides configuration.
Relative paths resolve from the repository root. data/ is ignored by Git.
Models and .venv must already exist; nothing is installed or downloaded.
LOCAL_POSE_PYTHON can select an existing Python executable.
`;

function localFile(value, label, suffix) {
  if (
    typeof value !== "string" ||
    !value.trim() ||
    /[\x00-\x1f]/.test(value) ||
    value.includes("://") ||
    (suffix && path.extname(value).toLowerCase() !== suffix)
  )
    throw new Error(`${label} must be a local ${suffix ?? "file"} path.`);
  return path.resolve(root, value);
}

async function readJson(file, label) {
  let data;
  try {
    const info = await stat(file);
    if (!info.isFile() || info.size > 65536) throw new Error();
    data = JSON.parse(await readFile(file, "utf8"));
  } catch {
    throw new Error(
      `${label} must be an existing JSON file smaller than 64 KiB.`,
    );
  }
  if (!data || typeof data !== "object" || Array.isArray(data))
    throw new Error(`${label} must contain a JSON object.`);
  return data;
}

export async function loadOptions(argv) {
  const cli = {};
  const seen = new Set();
  const flags = {
    "--manifest": "manifest",
    "--provider": "provider",
    "--minimum-score": "minimumScore",
    "--config": "config",
    "--recognizer": "recognizer",
  };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (flag === "--help" && argv.length === 1) return { help: true };
    const key = flags[flag];
    if (!key || seen.has(key))
      throw new Error(
        "Unknown or repeated option. Use --help for supported options.",
      );
    seen.add(key);
    const value = argv[++i];
    if (value === undefined || value.startsWith("--"))
      throw new Error("Every option requires a value. Use --help.");
    cli[key] = value;
  }
  const configFile = localFile(
    cli.config ?? "data/local-studio.json",
    "Configuration",
    ".json",
  );
  let configuration = {};
  let hasConfig = true;
  try {
    await access(configFile);
  } catch (error) {
    if (!cli.config && error.code === "ENOENT") hasConfig = false;
    else throw new Error("The requested configuration file is not readable.");
  }
  if (hasConfig) {
    configuration = await readJson(configFile, "Configuration");
    if (Object.keys(configuration).some((key) => !keys.has(key)))
      throw new Error(
        "Configuration has an unsupported key. Use --help for its schema.",
      );
  }
  delete cli.config;
  const options = {
    manifest: "ml/models.example.json",
    provider: "coreml",
    minimumScore: 0.55,
    ...configuration,
    ...cli,
  };
  options.manifest = localFile(options.manifest, "Manifest", ".json");
  if (options.recognizer !== undefined)
    options.recognizer = localFile(options.recognizer, "Recognizer", ".json");
  if (!["cpu", "coreml"].includes(options.provider))
    throw new Error("Provider must be coreml or cpu.");
  if (Object.hasOwn(cli, "minimumScore"))
    options.minimumScore = Number(cli.minimumScore);
  if (
    typeof options.minimumScore !== "number" ||
    !Number.isFinite(options.minimumScore) ||
    options.minimumScore <= 0 ||
    options.minimumScore >= 1
  )
    throw new Error(
      "minimumScore must be a number strictly between zero and one.",
    );
  return options;
}

export async function validateRuntime(options) {
  const manifest = await readJson(options.manifest, "Manifest");
  if (options.recognizer)
    await readJson(options.recognizer, "Recognizer manifest");
  for (const key of ["detector", "pose"]) {
    const name = manifest[key]?.path;
    localFile(name, `${key} model`, ".onnx");
    try {
      const info = await stat(
        path.resolve(path.dirname(options.manifest), name),
      );
      if (!info.isFile()) throw new Error();
    } catch {
      throw new Error(
        `The ${key} ONNX file in the manifest is missing. Set --manifest to your existing local models.`,
      );
    }
  }
  const python = localFile(
    process.env.LOCAL_POSE_PYTHON ?? ".venv/bin/python",
    "Python executable",
  );
  try {
    if (!(await stat(python)).isFile()) throw new Error();
    await access(python, constants.X_OK);
    await access(path.join(root, "node_modules/vite/bin/vite.js"));
  } catch {
    throw new Error(
      "An executable local Python environment and installed Vite are required. Nothing has been installed or downloaded.",
    );
  }
}

/** Probe only; actual child binds remain strict if another process wins a race. */
export async function assertPortsAvailable(ports = [5173, 8765]) {
  const held = [];
  try {
    for (const port of ports) {
      const server = net.createServer();
      await new Promise((resolve, reject) => {
        server.once("error", () =>
          reject(
            new Error(
              `Port ${port} is unavailable. Stop its existing process yourself, then retry.`,
            ),
          ),
        );
        server.listen({ host: "127.0.0.1", port, exclusive: true }, resolve);
      });
      held.push(server);
    }
  } finally {
    await Promise.all(
      held.map((server) => new Promise((resolve) => server.close(resolve))),
    );
  }
}

export function studioCommands(options) {
  return [
    {
      name: "local pose service",
      command: process.execPath,
      args: [
        path.join(root, "scripts/start-local-pose.mjs"),
        "--manifest",
        options.manifest,
        "--provider",
        options.provider,
        "--minimum-score",
        String(options.minimumScore),
        "--port",
        "8765",
        ...(options.recognizer ? ["--recognizer", options.recognizer] : []),
      ],
    },
    {
      name: "Vite",
      command: process.execPath,
      args: [
        path.join(root, "node_modules/vite/bin/vite.js"),
        "--host",
        "127.0.0.1",
        "--port",
        "5173",
        "--strictPort",
      ],
    },
  ];
}

/** Separate owned process groups let shutdown reach Python below its Node wrapper.
 * Children stay attached to this terminal and are always awaited: no daemon/unref.
 */
export async function runForeground(
  commands,
  {
    signalSource = process,
    spawnChild = spawn,
    signalChild = (child, signal) => process.kill(-child.pid, signal),
    groupAlive = (child) => {
      if (!child.pid) return false;
      try {
        process.kill(-child.pid, 0);
        return true;
      } catch (error) {
        return error.code !== "ESRCH";
      }
    },
    log = (message) => console.error(message),
    shutdownMs = 5000,
  } = {},
) {
  const children = [];
  let stopping = false;
  let exitCode = 0;
  let deadline;
  let forced = false;
  const signal = (child, value) => {
    if (!child.pid) return;
    try {
      signalChild(child, value);
    } catch (error) {
      if (error.code !== "ESRCH")
        log("Could not signal an owned child process.");
    }
  };
  const stop = (code, value = "SIGTERM") => {
    if (stopping) return;
    stopping = true;
    exitCode = code;
    for (const child of children) signal(child, value);
    deadline = setTimeout(() => {
      forced = true;
      for (const child of children) signal(child, "SIGKILL");
    }, shutdownMs);
  };
  const onInterrupt = () => stop(130, "SIGINT");
  const onTerminate = () => stop(143);
  signalSource.on("SIGINT", onInterrupt);
  signalSource.on("SIGTERM", onTerminate);
  const completions = [];
  try {
    for (const item of commands) {
      if (stopping) break;
      let child;
      try {
        child = spawnChild(item.command, item.args, {
          cwd: root,
          stdio: "inherit",
          detached: true,
        });
      } catch {
        log(
          `Could not start ${item.name}; stopping this launcher’s processes.`,
        );
        stop(2);
        break;
      }
      children.push(child);
      completions.push(
        new Promise((resolve) => {
          child.once("error", () => {
            log(
              `Could not start ${item.name}; stopping this launcher’s processes.`,
            );
            stop(2);
            resolve();
          });
          child.once("exit", (code) => {
            if (!stopping) {
              log(`${item.name} exited; stopping this launcher’s processes.`);
              stop(Number.isInteger(code) && code !== 0 ? code : 1);
            }
            resolve();
          });
        }),
      );
    }
    await Promise.all(completions);
    // A wrapper may exit before a stubborn grandchild. Keep the owned-group
    // deadline alive until descendants exit; never signal any pre-existing PID.
    while (!forced && children.some(groupAlive))
      await new Promise((resolve) => setTimeout(resolve, 25));
    return exitCode;
  } finally {
    clearTimeout(deadline);
    signalSource.off("SIGINT", onInterrupt);
    signalSource.off("SIGTERM", onTerminate);
  }
}

export async function main(argv = process.argv.slice(2)) {
  try {
    const options = await loadOptions(argv);
    if (options.help) {
      console.log(HELP);
      return 0;
    }
    await validateRuntime(options);
    await assertPortsAvailable();
    console.log(
      "Starting local pose service and Corner at http://127.0.0.1:5173. Keep this terminal open; Ctrl+C stops both.",
    );
    return await runForeground(studioCommands(options));
  } catch (error) {
    console.error(
      error instanceof Error ? error.message : "Local studio could not start.",
    );
    return 2;
  }
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  process.exitCode = await main();
