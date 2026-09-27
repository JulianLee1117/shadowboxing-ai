import { expect, test } from "@playwright/test";

test("local decoder preserves variable source timestamps, pixels and window bounds", async ({
  page,
}) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const mediaPath = "/node_modules/mediabunny/dist/modules/src/index.js";
    const decoderPath = "/src/lib/videoFrames.ts";
    const { BufferTarget, CanvasSource, Output, WebMOutputFormat } =
      await import(mediaPath);
    const { openVideoFrames } = await import(decoderPath);
    const canvas = new OffscreenCanvas(64, 64);
    const context = canvas.getContext("2d")!;
    const target = new BufferTarget();
    const output = new Output({ format: new WebMOutputFormat(), target });
    const source = new CanvasSource(canvas, { codec: "vp8", bitrate: 500_000 });
    output.addVideoTrack(source);
    await output.start();
    const timestamps = [0, 30, 120, 180, 260, 400];
    for (const [i, t] of timestamps.entries()) {
      const value = 20 + i * 35;
      context.fillStyle = `rgb(${value}, ${value}, ${value})`;
      context.fillRect(0, 0, 64, 64);
      await source.add(t / 1000, ((timestamps[i + 1] ?? 450) - t) / 1000);
    }
    await output.finalize();
    const blob = new Blob([target.buffer!], { type: "video/webm" });
    const controller = new AbortController();
    const decoded = await openVideoFrames(blob, {
      signal: controller.signal,
      offsetMs: 25,
      durationMs: 250,
      maxPixels: 4096,
    });
    const frames: { t: number; value: number }[] = [];
    for await (const frame of decoded.frames) {
      frames.push({
        t: frame.timestampMs,
        value: frame.canvas.getContext("2d").getImageData(32, 32, 1, 1).data[0],
      });
    }
    // Aborting after one frame must reject the next read and release the decoder.
    const abortController = new AbortController();
    const cancelSource = await openVideoFrames(blob, {
      signal: abortController.signal,
      offsetMs: 0,
      durationMs: 450,
      maxPixels: 4096,
    });
    await cancelSource.frames.next();
    abortController.abort();
    let abortName = "";
    try {
      await cancelSource.frames.next();
    } catch (error) {
      abortName = (error as Error).name;
    }
    cancelSource.dispose();
    return { frames, metadata: decoded.metadata, abortName };
  });
  expect(result.frames.map((frame) => frame.t)).toEqual([30, 120, 180, 260]);
  result.frames.forEach((frame, i) =>
    expect(Math.abs(frame.value - (55 + i * 35))).toBeLessThanOrEqual(3),
  );
  expect(result.metadata.width).toBe(64);
  expect(result.metadata.height).toBe(64);
  expect(result.metadata.mirrored).toBe(false);
  expect(result.abortName).toBe("AbortError");
});
