import { describe, expect, it } from "vitest";
import { MotionEngine } from "./motion";
import {
  NativeEventAdapter,
  nativeDetectorVersion,
  relativePoseFrame,
} from "./nativeRecognition";
import { recheckDetections } from "./recheck";
import type { NativeActionEvent, PoseFrame } from "./types";

const event = (patch: Partial<NativeActionEvent> = {}): NativeActionEvent => ({
  id: "stroke-1",
  hand: "right",
  family: "straight",
  startMs: 100,
  peakMs: 230,
  endMs: 400,
  detectedAtMs: 850,
  score: 0.9,
  ...patch,
});
const frame = (events = [event()], t = 900): PoseFrame => ({
  t,
  width: 1280,
  height: 720,
  landmarks: [],
  inferenceMs: 28,
  estimator: { id: "rtmpose-m", scoreType: "simcc", minimumScore: 0.55 },
  recognition: {
    protocolVersion: "shadowbox-recognition-v1",
    recognizerId: "personal-hybrid-v1",
    fingerprint: "a".repeat(64),
    state: "active",
    events,
  },
});

describe("fingerprinted local model decisions", () => {
  it("maps anatomical hand to stance while leaving extension and guard unassessed", () => {
    const orthodox = new NativeEventAdapter().update(frame(), "orthodox", true);
    expect(orthodox[0]).toMatchObject({
      hand: "right",
      role: "rear",
      label: "cross",
      extension: null,
      guardReturn: "unassessable",
    });
    expect(
      new NativeEventAdapter().update(frame(), "southpaw", true)[0],
    ).toMatchObject({ hand: "right", role: "lead", label: "jab" });
    expect(
      new NativeEventAdapter().update(
        frame([event({ family: "hook" })]),
        "southpaw",
        true,
      )[0].label,
    ).toBe("hook");
  });
  it("keeps source and round clocks consistent without crediting pre-round movements", () => {
    const source = frame(
      [
        event({ id: "before" }),
        event({
          id: "inside",
          startMs: 1100,
          peakMs: 1250,
          endMs: 1400,
          detectedAtMs: 1850,
        }),
      ],
      1900,
    );
    const original = structuredClone(source);
    const relative = relativePoseFrame(source, 1000);
    expect(source).toEqual(original);
    expect(relative.t).toBe(900);
    expect(relative.recognition!.events).toEqual([
      event({ id: "inside", peakMs: 250 }),
    ]);
    expect(
      new NativeEventAdapter().update(relative, "orthodox", true)[0]
        .detectedAtMs,
    ).toBe(850);
  });
  it("ignores duplicate deliveries and respects explicit round resets/calibration", () => {
    const adapter = new NativeEventAdapter();
    expect(adapter.update(frame(), "orthodox", false)).toEqual([]);
    expect(adapter.update(frame(undefined, 1000), "orthodox", true)).toEqual(
      [],
    );
    adapter.reset();
    expect(adapter.update(frame(), "orthodox", true)).toHaveLength(1);
    expect(
      adapter.update(frame(undefined, 1000), "orthodox", true),
    ).toHaveLength(0);
  });
  it("rejects malformed and future decisions instead of changing recognizers silently", () => {
    for (const patch of [
      { startMs: -1 },
      { peakMs: 99 },
      { endMs: 100 },
      { detectedAtMs: 901 },
      { score: NaN },
      { score: 2 },
      { family: "other" },
    ]) {
      expect(() =>
        new NativeEventAdapter().update(
          frame([event(patch as Partial<NativeActionEvent>)]),
          "orthodox",
          true,
        ),
      ).toThrow();
    }
    const adapter = new NativeEventAdapter();
    adapter.update(frame(), "orthodox", true);
    const changed = frame([], 1000);
    changed.recognition!.fingerprint = "b".repeat(64);
    expect(() => adapter.update(changed, "orthodox", true)).toThrow(/changed/);
  });
  it("accepts finalized past decisions separately from current visibility, without heuristic duplicates", () => {
    const engine = new MotionEngine({ stance: "orthodox", calibrated: true });
    const first = engine.update(frame());
    expect(first.quality.assessable).toBe(false);
    expect(first.events).toHaveLength(1);
    expect(engine.update(frame([], 1000)).events).toEqual([]);
    const missing = frame([], 1100);
    delete missing.recognition;
    expect(() => engine.update(missing)).toThrow(/disappeared/);
    engine.reset();
    expect(engine.update(missing).events).toEqual([]);
  });
  it("preserves recognizer fingerprints and requires video to recompute learned decisions", () => {
    const f = frame();
    expect(nativeDetectorVersion({ recognizer: f.recognition })).toBe(
      `personal-hybrid-v1:${"a".repeat(64)}`,
    );
    expect(
      nativeDetectorVersion({
        recognizer: { ...f.recognition, fingerprint: "bad" },
      }),
    ).toBeUndefined();
    expect(() =>
      recheckDetections({ id: "round", stance: "orthodox", frames: [f] }),
    ).toThrow(/Analyze the video/);
  });
});
