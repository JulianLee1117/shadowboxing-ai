import type { PunchEvent } from "./types";

type PunchIdentity = Pick<PunchEvent, "label" | "role" | "hand">;

/** Names use anatomical role, independently of a mirrored camera preview. */
export function punchName(event: PunchIdentity): string {
  const label: string = event.label;
  if (label === "jab") return "Jab";
  if (label === "cross") return "Cross";
  if (label === "hook")
    return `${event.role === "lead" ? "Lead" : "Rear"} hook`;
  if (label === "uppercut")
    return `${event.role === "lead" ? "Lead" : "Rear"} uppercut`;
  return "Punch";
}

export function punchNotation(event: PunchIdentity): string {
  const label: string = event.label;
  if (label === "jab") return "1";
  if (label === "cross") return "2";
  if (label === "hook") return event.role === "lead" ? "3" : "4";
  if (label === "uppercut") return event.role === "lead" ? "5" : "6";
  return "–";
}
