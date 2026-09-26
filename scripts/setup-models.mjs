import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import {
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  copyFile,
  writeFile,
} from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
if (args.some((arg) => arg !== "--all")) {
  console.error("Usage: node scripts/setup-models.mjs [--all]");
  process.exit(1);
}
const all = args.includes("--all");
const modelDir = path.join(root, "public", "models");
const wasmDir = path.join(root, "public", "wasm");
const manifestPath = path.join(modelDir, "manifest.json");

async function hashFile(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

async function downloadModel(model, destination) {
  const temporary = `${destination}.download-${process.pid}`;
  try {
    const response = await fetch(model.source, {
      signal: AbortSignal.timeout(120_000),
    });
    if (!response.ok || !response.body)
      throw new Error(
        `Download failed with HTTP ${response.status}: ${model.source}`,
      );
    const expectedLength =
      Number(response.headers.get("content-length")) || null;
    let bytes = 0;
    const hash = createHash("sha256");
    const check = new Transform({
      transform(chunk, _encoding, callback) {
        bytes += chunk.length;
        if (bytes > 150_000_000)
          return callback(new Error("Unexpectedly large model download."));
        hash.update(chunk);
        callback(null, chunk);
      },
    });
    await pipeline(
      Readable.fromWeb(response.body),
      check,
      createWriteStream(temporary, { flags: "wx" }),
    );
    if (expectedLength !== null && bytes !== expectedLength)
      throw new Error(
        "Model download length does not match its HTTP response.",
      );
    if (bytes < 1_000_000)
      throw new Error(
        "Model response is too small to be a valid Pose Landmarker bundle.",
      );
    const sha256 = hash.digest("hex");
    if (model.sha256 && sha256 !== model.sha256)
      throw new Error(
        `SHA-256 mismatch for ${model.variant}; refusing to replace a pinned artifact.`,
      );
    const magic = (await readFile(temporary)).subarray(0, 8);
    const zipOffset = magic[0] === 0x50 ? 0 : 2;
    if (
      magic[zipOffset] !== 0x50 ||
      magic[zipOffset + 1] !== 0x4b ||
      magic[zipOffset + 2] !== 0x03 ||
      magic[zipOffset + 3] !== 0x04
    ) {
      throw new Error("Downloaded response is not a .task ZIP bundle.");
    }
    await rename(temporary, destination);
    return { bytes, sha256 };
  } finally {
    await rm(temporary, { force: true });
  }
}

async function main() {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const packageDir = path.dirname(require.resolve("@mediapipe/tasks-vision"));
  const runtimePackage = JSON.parse(
    await readFile(path.join(packageDir, "package.json"), "utf8"),
  );
  if (runtimePackage.version !== "1.0.1") {
    throw new Error(
      `Expected @mediapipe/tasks-vision 1.0.1, found ${runtimePackage.version}. Reinstall pinned dependencies.`,
    );
  }
  await mkdir(modelDir, { recursive: true });
  await mkdir(wasmDir, { recursive: true });
  const wasmSource = path.join(packageDir, "wasm");
  const runtimeFiles = [];
  for (const name of (await readdir(wasmSource))
    .filter((name) => /\.(js|wasm)$/.test(name))
    .sort()) {
    const destination = path.join(wasmDir, name);
    await copyFile(path.join(wasmSource, name), destination);
    runtimeFiles.push({
      path: `/wasm/${name}`,
      source: `npm:@mediapipe/tasks-vision@${runtimePackage.version}/wasm/${name}`,
      bytes: (await stat(destination)).size,
      sha256: await hashFile(destination),
    });
  }
  if (
    !runtimeFiles.some((file) =>
      file.path.endsWith("vision_wasm_module_internal.js"),
    )
  ) {
    throw new Error(
      "The installed runtime lacks the required module-worker WASM loader.",
    );
  }
  manifest.runtime = {
    package: runtimePackage.name,
    version: runtimePackage.version,
    license: runtimePackage.license,
    files: runtimeFiles,
  };
  console.log(
    `Copied ${runtimeFiles.length} local WASM/runtime files (${formatBytes(runtimeFiles.reduce((sum, file) => sum + file.bytes, 0))}).`,
  );

  for (const model of manifest.models) {
    if (!all && model.variant !== "full") continue;
    const destination = path.join(
      modelDir,
      `pose_landmarker_${model.variant}.task`,
    );
    let info;
    try {
      info = await stat(destination);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    if (info) {
      const sha256 = await hashFile(destination);
      if (model.sha256 && sha256 !== model.sha256) {
        throw new Error(
          `Existing ${model.variant} model failed SHA-256 verification. Remove that .task file and rerun setup to download the pinned artifact.`,
        );
      }
      if (info.size < 1_000_000)
        throw new Error(
          `Existing ${model.variant} model is invalid; remove it and rerun setup.`,
        );
      if (!model.sha256) {
        // An unpinned existing file is not sufficient provenance for a first run.
        console.log(
          `Verifying ${model.variant} against a fresh official download…`,
        );
        Object.assign(model, await downloadModel(model, destination));
      } else {
        model.bytes = info.size;
      }
    } else {
      console.log(`Downloading official ${model.variant} pose bundle…`);
      Object.assign(model, await downloadModel(model, destination));
    }
    console.log(
      `${model.variant}: ${formatBytes(model.bytes)} · SHA-256 ${model.sha256}`,
    );
  }
  const temporaryManifest = `${manifestPath}.download-${process.pid}`;
  await writeFile(temporaryManifest, `${JSON.stringify(manifest, null, 2)}\n`);
  await rename(temporaryManifest, manifestPath);
  console.log(
    `Manifest saved. ${all ? "All variants are" : "The full variant is"} ready for same-origin local inference.`,
  );
  if (!all)
    console.log(
      "Run npm run models:setup -- --all to install Lite and Heavy for comparisons.",
    );
}

function formatBytes(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
}

main().catch((error) => {
  console.error(`Model setup failed: ${error.message}`);
  process.exitCode = 1;
});
