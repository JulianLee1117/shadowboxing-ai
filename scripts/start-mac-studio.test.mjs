import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { tmpdir } from "node:os";
import {
  assertPortsAvailable,
  loadOptions,
  runForeground,
  studioCommands,
  validateRuntime,
} from "./start-mac-studio.mjs";

async function fixture(run) {
  const directory = await mkdtemp(path.join(tmpdir(), "corner-studio-"));
  try {
    return await run(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

test("local config and explicit overrides produce shell-free, fixed-port commands", () =>
  fixture(async (directory) => {
    const config = path.join(directory, "config.json");
    const manifest = path.join(directory, "models with spaces.json");
    const recognizer = path.join(directory, "personal model.json");
    await writeFile(
      config,
      JSON.stringify({
        manifest,
        recognizer,
        provider: "cpu",
        minimumScore: 0.6,
      }),
    );
    const options = await loadOptions([
      "--config",
      config,
      "--provider",
      "coreml",
      "--minimum-score",
      "0.55",
    ]);
    assert.deepEqual(options, {
      manifest,
      recognizer,
      provider: "coreml",
      minimumScore: 0.55,
    });
    const commands = studioCommands(options);
    assert.equal(commands.length, 2);
    assert.equal(
      commands[0].args.filter((value) => value === "--recognizer").length,
      1,
    );
    assert.equal(commands[0].args.at(-1), recognizer);
    assert.ok(commands[0].args.includes(manifest));
    assert.deepEqual(commands[1].args.slice(-5), [
      "--host",
      "127.0.0.1",
      "--port",
      "5173",
      "--strictPort",
    ]);
    assert.equal(commands[0].command, process.execPath);
    assert.ok(
      !studioCommands({ ...options, recognizer: undefined })[0].args.includes(
        "--recognizer",
      ),
    );
  }));

test("configuration rejects unknown keys, remote filenames and malformed score policies without echoing values", () =>
  fixture(async (directory) => {
    const config = path.join(directory, "config.json");
    for (const invalid of [
      { token: "secret-marker" },
      { manifest: "https://secret-marker/model.json" },
      { manifest: "a\nsecret-marker.json" },
      { manifest: "secret-marker.txt" },
      { provider: "secret-marker" },
      { minimumScore: "secret-marker" },
      { minimumScore: 0 },
      { minimumScore: 1 },
      [],
    ]) {
      await writeFile(config, JSON.stringify(invalid));
      await assert.rejects(
        loadOptions(["--config", config]),
        (error) => !error.message.includes("secret-marker"),
      );
    }
    for (const args of [
      ["--unknown", "secret-marker"],
      ["--provider"],
      ["--provider", "cpu", "--provider", "coreml"],
      ["--minimum-score", "NaN"],
    ])
      await assert.rejects(loadOptions(args));
    await assert.rejects(
      loadOptions(["--config", path.join(directory, "missing.json")]),
    );
    assert.deepEqual(await loadOptions(["--help"]), { help: true });
  }));

test("missing model files fail before launching either service", () =>
  fixture(async (directory) => {
    const manifest = path.join(directory, "manifest.json");
    await writeFile(
      manifest,
      JSON.stringify({
        detector: { path: "missing.onnx" },
        pose: { path: "missing.onnx" },
      }),
    );
    await assert.rejects(
      validateRuntime({ manifest }),
      /detector ONNX file.*missing/,
    );
  }));

test("port collision is reported without touching the existing listener or leaking probes", async () => {
  const existing = net.createServer();
  await new Promise((resolve) => existing.listen(0, "127.0.0.1", resolve));
  const port = existing.address().port;
  try {
    await assert.rejects(
      assertPortsAvailable([0, port]),
      new RegExp(`Port ${port} is unavailable`),
    );
    assert.equal(existing.listening, true);
    const client = net.connect(port, "127.0.0.1");
    await new Promise((resolve, reject) => {
      client.once("connect", resolve);
      client.once("error", reject);
    });
    client.destroy();
  } finally {
    await new Promise((resolve) => existing.close(resolve));
  }
});

test("an unexpected child exit stops its sibling and returns failure even for exit zero", async () => {
  const signals = new EventEmitter();
  const children = [];
  const sent = [];
  const job = runForeground(
    [
      { name: "pose", command: "fake", args: [] },
      { name: "Vite", command: "fake", args: [] },
    ],
    {
      signalSource: signals,
      spawnChild: (_command, _args, options) => {
        assert.equal(options.detached, true);
        assert.equal(options.stdio, "inherit");
        assert.equal(options.shell, undefined);
        const child = Object.assign(new EventEmitter(), {
          pid: children.length + 1,
        });
        children.push(child);
        return child;
      },
      signalChild: (child, signal) => {
        sent.push([child.pid, signal]);
        queueMicrotask(() => child.emit("exit", null));
      },
      groupAlive: () => false,
      log: () => {},
    },
  );
  children[0].emit("exit", 0);
  assert.equal(await job, 1);
  assert.deepEqual(sent, [
    [1, "SIGTERM"],
    [2, "SIGTERM"],
  ]);
  assert.equal(signals.listenerCount("SIGINT"), 0);
  assert.equal(signals.listenerCount("SIGTERM"), 0);
});

test("spawn failure stops the already-started child and never starts later commands", async () => {
  let calls = 0;
  let first;
  const sent = [];
  const code = await runForeground(
    ["first", "failure", "never"].map((name) => ({
      name,
      command: "fake",
      args: [],
    })),
    {
      signalSource: new EventEmitter(),
      spawnChild: () => {
        if (++calls === 2) throw new Error("sensitive executable path");
        return (first = Object.assign(new EventEmitter(), { pid: 1 }));
      },
      signalChild: (child, signal) => {
        sent.push(signal);
        queueMicrotask(() => child.emit("exit", null));
      },
      groupAlive: () => false,
      log: (message) => assert.ok(!message.includes("sensitive")),
    },
  );
  assert.equal(code, 2);
  assert.equal(calls, 2);
  assert.deepEqual(sent, ["SIGTERM"]);
});

async function until(fn, timeout = 3000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    try {
      if (await fn()) return;
    } catch {
      /* startup is asynchronous */
    }
    await new Promise((resolve) => setTimeout(resolve, 15));
  }
  throw new Error("Timed out waiting for owned test process");
}

test(
  "interrupt cleans the actual wrapper and its signal-resistant descendant",
  { skip: process.platform === "win32" },
  () =>
    fixture(async (directory) => {
      const marker = path.join(directory, "descendant.pid");
      const descendant = `process.on('SIGINT',()=>{});process.on('SIGTERM',()=>{});require('fs').writeFileSync(${JSON.stringify(marker)},String(process.pid));setInterval(()=>{},1000);`;
      const wrapper = `require('child_process').spawn(process.execPath,['-e',${JSON.stringify(descendant)}],{stdio:'ignore'});process.on('SIGINT',()=>process.exit(0));process.on('SIGTERM',()=>process.exit(0));setInterval(()=>{},1000);`;
      const signals = new EventEmitter();
      const job = runForeground(
        [
          {
            name: "test wrapper",
            command: process.execPath,
            args: ["-e", wrapper],
          },
        ],
        { signalSource: signals, shutdownMs: 150, log: () => {} },
      );
      try {
        await until(async () => Number(await readFile(marker, "utf8")) > 0);
        const pid = Number(await readFile(marker, "utf8"));
        signals.emit("SIGINT");
        assert.equal(await job, 130);
        await until(() => {
          try {
            process.kill(pid, 0);
            return false;
          } catch (error) {
            return error.code === "ESRCH";
          }
        });
        assert.equal(signals.listenerCount("SIGINT"), 0);
      } finally {
        signals.emit("SIGTERM");
        await job;
      }
    }),
);
