export type FrameEncoderReply =
  | { type: "encoded"; id: number; blob: Blob }
  | { type: "error"; id: number; error: string };

type Request = {
  id: number;
  bitmap: ImageBitmap;
  width: number;
  height: number;
};
const scope = globalThis as unknown as {
  onmessage: ((message: MessageEvent<Request>) => void) | null;
  postMessage: (message: FrameEncoderReply) => void;
};
let canvas: OffscreenCanvas | null = null;
let working = false;

scope.onmessage = ({ data }) => {
  if (working) {
    data.bitmap.close();
    scope.postMessage({
      type: "error",
      id: data.id,
      error: "Frame encoder is busy.",
    });
    return;
  }
  working = true;
  void encode(data);
};

async function encode({ id, bitmap, width, height }: Request): Promise<void> {
  try {
    if (
      !Number.isInteger(width) ||
      !Number.isInteger(height) ||
      width <= 0 ||
      height <= 0 ||
      width * height > 921_600 ||
      Math.max(width, height) > 1280
    )
      throw new Error("Invalid local image dimensions.");
    if (!canvas || canvas.width !== width || canvas.height !== height)
      canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Local frame drawing is unavailable.");
    context.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    const blob = await canvas.convertToBlob({
      type: "image/jpeg",
      quality: 0.95,
    });
    if (blob.type !== "image/jpeg" || blob.size > 3_000_000)
      throw new Error("Encoded frame exceeds the local image limit.");
    scope.postMessage({ type: "encoded", id, blob });
  } catch (error) {
    scope.postMessage({
      type: "error",
      id,
      error: error instanceof Error ? error.message : "Frame encoding failed.",
    });
  } finally {
    bitmap.close();
    working = false;
  }
}
