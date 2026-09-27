import { describe, expect, it } from "vitest";
import { CombinationRecognizer, groupCombinations } from "./combinations";
import type { PunchEvent } from "./types";
const punch = (
  id: string,
  label: "jab" | "cross",
  peakMs: number,
  patch: Partial<PunchEvent> = {},
): PunchEvent => ({
  id,
  label,
  hand: label === "jab" ? "left" : "right",
  role: label === "jab" ? "lead" : "rear",
  startMs: peakMs - 100,
  peakMs,
  endMs: peakMs + 150,
  detectedAtMs: peakMs + 200,
  score: 0.8,
  extension: 0.7,
  guardReturn: "returned",
  experimental: true,
  ...patch,
});
const options = { stance: "orthodox" as const, sessionId: "round-1" };
describe("observed combination grouping", () => {
  it("groups complete delayed model evidence while an intervening late hook still breaks the combo", () => {
    const jab = punch("a", "jab", 200, {
      detectedAtMs: 2200,
      extension: null,
      guardReturn: "unassessable",
    });
    const cross = punch("b", "cross", 500);
    const grouped = groupCombinations([cross, jab], options);
    expect(grouped[0].notation).toBe("1-2");
    expect(grouped[0].provenance.timing.maximumArrivalDelayMs).toBe(2000);
    const hook = punch("h", "jab", 350, { label: "hook", detectedAtMs: 2400 });
    expect(groupCombinations([cross, hook, jab], options)).toEqual([]);
  });
  it.each([
    [["jab", "cross"], "1-2"],
    [["jab", "jab"], "1-1"],
    [["jab", "jab", "cross"], "1-1-2"],
    [["jab", "cross", "jab"], "1-2-1"],
  ] as const)(
    "recognizes %s with longest nonoverlapping grouping",
    (labels, notation) => {
      const input = labels.map((label, i) =>
        punch(String(i), label, 200 + i * 300),
      );
      const result = groupCombinations(input, options);
      expect(result).toHaveLength(1);
      expect(result[0].notation).toBe(notation);
      expect(result[0].eventIds).toEqual(input.map((e) => e.id));
      expect(result[0].confidenceSemantics).toBe(
        "minimum-uncalibrated-punch-score",
      );
      expect(result[0].finalizedAtMs).toBeGreaterThanOrEqual(
        Math.max(...input.map((e) => e.detectedAtMs!)),
      );
    },
  );
  it("allows overlapping action intervals and sorts by actual peaks", () => {
    const jab = punch("a", "jab", 200, { endMs: 600, detectedAtMs: 700 });
    const cross = punch("b", "cross", 450, {
      startMs: 350,
      endMs: 550,
      detectedAtMs: 600,
    });
    expect(groupCombinations([cross, jab], options)[0]).toMatchObject({
      notation: "1-2",
      eventIds: ["a", "b"],
      endMs: 600,
    });
  });
  it("does not reuse the last punch as the beginning of another combination", () => {
    const labels = ["jab", "cross", "jab", "cross"] as const;
    const result = groupCombinations(
      labels.map((label, i) => punch(String(i), label, 200 + i * 300)),
      options,
    );
    expect(result.map((r) => r.notation)).toEqual(["1-2-1"]);
    expect(new Set(result.flatMap((r) => r.eventIds)).size).toBe(3);
  });
  it("requires label, role, anatomical hand and stance consistency", () => {
    expect(
      groupCombinations(
        [punch("a", "jab", 200, { role: "rear" }), punch("b", "cross", 500)],
        options,
      ),
    ).toEqual([]);
    expect(
      groupCombinations([punch("a", "jab", 200), punch("b", "cross", 500)], {
        stance: "southpaw",
      }),
    ).toEqual([]);
    expect(
      groupCombinations(
        [
          punch("a", "jab", 200, { hand: "right" }),
          punch("b", "cross", 500, { hand: "left" }),
        ],
        { stance: "southpaw" },
      ),
    ).toHaveLength(1);
  });
  it("does not treat spatial guard return as a technique or sequence grade", () => {
    const input = [
      punch("a", "jab", 200, { guardReturn: "not-observed", score: 0.6 }),
      punch("b", "cross", 500),
    ];
    expect(groupCombinations(input, options)[0].confidence).toBe(0.6);
    expect(
      groupCombinations(input, { ...options, requireObservedReturn: true }),
    ).toEqual([]);
  });
  it("honors peak gap limits even if action intervals overlap", () => {
    for (const peak of [230, 851])
      expect(
        groupCombinations(
          [punch("a", "jab", 200), punch("b", "cross", peak)],
          options,
        ),
      ).toEqual([]);
    expect(
      groupCombinations(
        [punch("a", "jab", 200), punch("b", "cross", 850)],
        options,
      ),
    ).toHaveLength(1);
  });
  it("blocks uncertainty in the gap or either participating arm, without suppressing a different arm double jab", () => {
    const interval = {
      hand: "right" as const,
      startMs: 360,
      endMs: 390,
      reasons: ["missing"],
    };
    expect(
      groupCombinations([punch("a", "jab", 200), punch("b", "cross", 550)], {
        ...options,
        uncertaintyIntervals: [interval],
      }),
    ).toEqual([]);
    expect(
      groupCombinations([punch("a", "jab", 200), punch("b", "jab", 550)], {
        ...options,
        uncertaintyIntervals: [interval],
      }),
    ).toHaveLength(1);
  });
  it("does not skip an inconsistent intervening event to manufacture a combination", () => {
    expect(
      groupCombinations(
        [
          punch("a", "jab", 200),
          punch("x", "jab", 350, { hand: "right" }),
          punch("b", "cross", 550),
        ],
        options,
      ),
    ).toEqual([]);
  });
});
describe("causal combination stream", () => {
  it("holds a pair open for a longer pattern and matches offline output IDs", () => {
    const input = [
      punch("a", "jab", 200),
      punch("b", "jab", 500),
      punch("c", "cross", 800),
    ];
    const stream = new CombinationRecognizer(options);
    for (const event of input)
      expect(stream.consume([event], event.detectedAtMs!)).toEqual([]);
    const output = stream.consume([], 2600);
    expect(output.map((e) => e.id)).toEqual(
      groupCombinations(input, options).map((e) => e.id),
    );
    expect(output[0].finalizedAtMs).toBe(2600);
    expect(stream.flush(2700)).toEqual([]);
    expect(stream.flush(2700)).toEqual([]);
  });
  it("accepts delayed finalization in reverse peak order, and rejects arrivals beyond its bound", () => {
    const stream = new CombinationRecognizer(options);
    stream.consume([punch("b", "cross", 500)], 700);
    stream.consume([punch("a", "jab", 200, { detectedAtMs: 800 })], 800);
    expect(stream.flush(800)[0].eventIds).toEqual(["a", "b"]);
    const late = new CombinationRecognizer(options);
    expect(() => late.consume([punch("a", "jab", 200)], 2001)).toThrow(
      /ordering window/,
    );
  });
  it("is idempotent for repeated input IDs, rejects altered duplicates, and resets", () => {
    const input = [punch("a", "jab", 200), punch("b", "cross", 500)];
    const stream = new CombinationRecognizer(options);
    stream.consume(input, 700);
    stream.consume(input, 800);
    expect(() => stream.consume([{ ...input[0], score: 0.5 }], 800)).toThrow(
      /Conflicting/,
    );
    const result = stream.flush(900);
    expect(result).toHaveLength(1);
    expect(() => stream.consume([], 900)).toThrow(/closed/);
    stream.reset();
    stream.consume(input, 700);
    expect(stream.flush(900)[0].id).toBe(result[0].id);
  });
  it("does not revise finalized output when uncertainty arrives too late", () => {
    const stream = new CombinationRecognizer(options);
    stream.consume([punch("a", "jab", 200), punch("b", "cross", 500)], 700);
    expect(stream.consume([], 3301)).toHaveLength(1);
    expect(() =>
      stream.consume([], 3400, [
        { hand: "left", startMs: 250, endMs: 270, reasons: ["late"] },
      ]),
    ).toThrow(/Late uncertainty/);
  });
  it("ignores late uncertainty on a nonparticipating arm and clears old intervals on reset", () => {
    const stream = new CombinationRecognizer(options);
    const input = [punch("a", "jab", 200), punch("b", "jab", 500)];
    stream.consume(input, 700);
    const output = stream.consume([], 3301);
    expect(output).toHaveLength(1);
    output[0].provenance.sourceHands[0] = "right"; // caller mutation cannot alter internal history
    expect(() =>
      stream.consume([], 3400, [
        { hand: "right", startMs: 250, endMs: 270, reasons: ["unrelated"] },
      ]),
    ).not.toThrow();
    stream.reset([
      { hand: "left", startMs: 250, endMs: 270, reasons: ["missing"] },
    ]);
    stream.consume(input, 700);
    expect(stream.flush(700)).toEqual([]);
    stream.reset();
    stream.consume(input, 700);
    expect(stream.flush(700)).toHaveLength(1);
  });
  it("rejects malformed evidence and invalid clocks", () => {
    const stream = new CombinationRecognizer(options);
    expect(() =>
      stream.consume([punch("a", "jab", 200, { endMs: 100 })], 400),
    ).toThrow(/Invalid finalized/);
    stream.consume([], 500);
    expect(() => stream.consume([], 400)).toThrow(/clock/);
    expect(
      () => new CombinationRecognizer({ ...options, minimumPeakGapMs: 0 }),
    ).toThrow(/timing/);
  });
});
