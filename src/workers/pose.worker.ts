import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { Landmark, ModelVariant, PoseFrame } from "../lib/types";

// The pinned SDK contains optional usage logging via fetch. Enforce the local
// worker's network boundary rather than assuming that self-hosted assets alone
// disable telemetry. This does not change the application's main-thread fetch.
const localFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = (
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> => {
  const url = new URL(
    input instanceof Request ? input.url : String(input),
    globalThis.location.href,
  );
  if (url.origin !== globalThis.location.origin) {
    return Promise.reject(
      new TypeError(
        "External network requests are disabled in the local pose worker.",
      ),
    );
  }
  return localFetch(input, init);
};

export type PoseWorkerRequest =
  | {
      type: "init";
      id: number;
      variant: ModelVariant;
      delegate: "GPU" | "CPU";
      modelUrl: string;
      wasmUrl: string;
    }
  | {
      type: "detect";
      id: number;
      bitmap: ImageBitmap;
      t: number;
      width: number;
      height: number;
    };
export type PoseWorkerReply =
  | { type: "ready"; id: number; delegate: "GPU" | "CPU" }
  | { type: "result"; id: number; frame: PoseFrame }
  | { type: "error"; id: number; error: string };

// Keep the app's DOM type environment while giving this module a worker surface.
const scope = globalThis as unknown as {
  onmessage: ((event: MessageEvent<PoseWorkerRequest>) => void) | null;
  postMessage: (message: PoseWorkerReply) => void;
};
let landmarker: PoseLandmarker | null = null;
let working = false;
let lastInferenceTimestamp = -1;

scope.onmessage = (event) => {
  void handleMessage(event.data);
};

async function handleMessage(message: PoseWorkerRequest): Promise<void> {
  if (working) {
    if (message.type === "detect") message.bitmap.close();
    scope.postMessage({
      type: "error",
      id: message.id,
      error: "Worker is busy. Frames must not be queued.",
    });
    return;
  }
  working = true;
  try {
    if (message.type === "init") {
      if (landmarker)
        throw new Error("This pose worker is already initialized.");
      if (typeof OffscreenCanvas === "undefined")
        throw new Error(
          "This browser does not support offscreen pose processing.",
        );
      const response = await fetch(message.modelUrl, {
        credentials: "same-origin",
      });
      if (
        !response.ok ||
        response.headers.get("content-type")?.includes("text/html")
      ) {
        throw new Error(
          `The ${message.variant} model is missing locally. Run npm run models:setup${message.variant === "full" ? "" : " -- --all"}.`,
        );
      }
      const modelBytes = new Uint8Array(await response.arrayBuffer());
      // Official .task ZIP containers can have two leading alignment bytes.
      // Reject an SPA HTML fallback without rejecting Google's padded archive.
      const zipOffset = modelBytes[0] === 0x50 ? 0 : 2;
      if (
        modelBytes.length < 1_000_000 ||
        modelBytes[zipOffset] !== 0x50 ||
        modelBytes[zipOffset + 1] !== 0x4b ||
        modelBytes[zipOffset + 2] !== 0x03 ||
        modelBytes[zipOffset + 3] !== 0x04
      ) {
        throw new Error(
          "Invalid local pose model bundle. Run npm run models:setup to verify assets.",
        );
      }
      // SDK 1.0.1 provides an ES module WASM loader. The default classic loader
      // does not publish ModuleFactory when dynamically imported by a module worker.
      const fileset = await FilesetResolver.forVisionTasks(
        message.wasmUrl,
        true,
      );
      landmarker = await PoseLandmarker.createFromOptions(fileset, {
        canvas: new OffscreenCanvas(1, 1),
        baseOptions: {
          modelAssetBuffer: modelBytes,
          delegate: message.delegate,
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: false,
      });
      lastInferenceTimestamp = -1;
      scope.postMessage({
        type: "ready",
        id: message.id,
        delegate: message.delegate,
      });
      return;
    }

    if (!landmarker) throw new Error("Pose model has not initialized.");
    // MediaPipe needs strictly increasing timestamps. Preserve the caller's
    // original media t in the result; a new source/seek should use a new runner.
    const inferenceTimestamp = Math.max(message.t, lastInferenceTimestamp + 1);
    lastInferenceTimestamp = inferenceTimestamp;
    const started = performance.now();
    const result = landmarker.detectForVideo(
      message.bitmap,
      inferenceTimestamp,
    );
    const inferenceMs = performance.now() - started;
    try {
      scope.postMessage({
        type: "result",
        id: message.id,
        frame: {
          t: message.t,
          width: message.width,
          height: message.height,
          landmarks: (result.landmarks[0] ?? []).map(copyLandmark),
          worldLandmarks: result.worldLandmarks[0]?.map(copyLandmark),
          inferenceMs,
        },
      });
    } finally {
      result.close();
    }
  } catch (error) {
    scope.postMessage({
      type: "error",
      id: message.id,
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    if (message.type === "detect") message.bitmap.close();
    working = false;
  }
}

function copyLandmark(point: Landmark): Landmark {
  return {
    x: point.x,
    y: point.y,
    z: point.z,
    visibility: point.visibility,
    presence: point.presence,
  };
}
