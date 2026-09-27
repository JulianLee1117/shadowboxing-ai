/** Practice prompts, not observed technique judgements. */
export const DRILLS = {
  open: {
    label: "Free practice",
    prompt: "Make space. Take your stance. Start a round.",
  },
  "jab-cross": {
    label: "Jab–cross",
    prompt:
      "Practice a jab followed by a cross. Reset comfortably between pairs.",
  },
  "double-jab": {
    label: "Double jab",
    prompt: "Practice two jabs. Reset comfortably before the next pair.",
  },
  "double-jab-cross": {
    label: "Double jab–cross",
    prompt:
      "Practice two jabs followed by a cross. Start at a comfortable pace.",
  },
} as const;
export type DrillId = keyof typeof DRILLS;
export function combinationLabel(name: string): string {
  if (name === "jab-cross-jab") return "Jab–cross–jab";
  return name in DRILLS ? DRILLS[name as DrillId].label : name;
}
