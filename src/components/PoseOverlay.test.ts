import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PoseOverlay } from "./PoseOverlay";
import { demoFrame } from "../lib/demo";

describe("visible tracking evidence", () => {
  it("marks weak MediaPipe presence as uncertain even when visibility is high", () => {
    const frame = demoFrame(0);
    frame.landmarks[15].presence = 0.35;
    const html = renderToStaticMarkup(
      createElement(PoseOverlay, { frame, mirror: false }),
    );
    expect(html).toContain('class="pose-joint hand-left uncertain"');
    expect(html).toContain('class="pose-line hand-left uncertain"');
    expect(html).toContain(">L</text>");
    expect(html).toContain('class="pose-line hand-right "');
  });
  it("shows a weak native wrist as uncertain without hiding or weakening the confident upper arm", () => {
    const frame = demoFrame(0);
    frame.estimator = {
      id: "rtmpose-m",
      scoreType: "simcc",
      minimumScore: 0.55,
    };
    frame.landmarks = frame.landmarks.map((p) => ({
      ...p,
      visibility: undefined,
      score: 0.9,
    }));
    frame.landmarks[15].score = 0.4;
    const html = renderToStaticMarkup(
      createElement(PoseOverlay, { frame, mirror: false }),
    );
    expect(html).toContain('class="pose-line hand-left "');
    expect(html).toContain('class="pose-line hand-left uncertain"');
    expect(html).toContain('class="pose-joint hand-left uncertain"');
    expect(html).toContain('class="pose-line hand-right "');
    expect(html).toContain(">L</text>");
  });
  it("omits unobserved and out-of-frame wrists without inventing replacement positions", () => {
    const frame = demoFrame(0);
    frame.landmarks[15].visibility = 0.1;
    frame.landmarks[16].x = 1.1;
    const html = renderToStaticMarkup(
      createElement(PoseOverlay, { frame, mirror: false }),
    );
    expect(html).not.toContain(">L</text>");
    expect(html).not.toContain(">R</text>");
    expect(html).toContain('class="pose-line hand-left "');
    expect(html).not.toContain("NaN");
  });
  it("mirrors presentation without changing physical-side labels or colors", () => {
    const frame = demoFrame(0);
    const plain = renderToStaticMarkup(
      createElement(PoseOverlay, { frame, mirror: false }),
    );
    const mirrored = renderToStaticMarkup(
      createElement(PoseOverlay, { frame, mirror: true }),
    );
    expect(mirrored).toContain("scaleX(-1)");
    expect(mirrored.match(/hand-left/g)?.length).toBe(
      plain.match(/hand-left/g)?.length,
    );
    expect(mirrored.match(/hand-right/g)?.length).toBe(
      plain.match(/hand-right/g)?.length,
    );
    expect(mirrored).toContain(">L</text>");
    expect(mirrored).toContain(">R</text>");
  });
});
