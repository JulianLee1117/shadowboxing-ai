import { describe, expect, it } from "vitest";
import { punchName, punchNotation } from "./punches";
import type { PunchEvent } from "./types";

describe("six-punch display identities", () => {
  it.each([
    ["jab", "lead", "Jab", "1"],
    ["cross", "rear", "Cross", "2"],
    ["hook", "lead", "Lead hook", "3"],
    ["hook", "rear", "Rear hook", "4"],
    ["uppercut", "lead", "Lead uppercut", "5"],
    ["uppercut", "rear", "Rear uppercut", "6"],
  ] as const)(
    "names %s %s independently of physical hand",
    (label, role, name, number) => {
      for (const hand of ["left", "right"] as const) {
        const event: Pick<PunchEvent, "label" | "role" | "hand"> = {
          label,
          role,
          hand,
        };
        expect(punchName(event)).toBe(name);
        expect(punchNotation(event)).toBe(number);
      }
    },
  );
});
