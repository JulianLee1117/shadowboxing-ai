import { percentile } from "./storage";

export interface SourceCadenceSummary {
  schemaVersion: "source-cadence-1";
  observations: number;
  duplicates: number;
  regressions: number;
  invalid: number;
  firstMediaMs: number | null;
  lastMediaMs: number | null;
  mediaSpanMs: number | null;
  callbackSpanMs: number | null;
  mediaFps: number | null;
  callbackFps: number | null;
  gapMs: {
    samples: number;
    complete: boolean;
    p50: number | null;
    p95: number | null;
    maximum: number | null;
  };
}

/** Observes source callbacks before the pose pump can skip them. No pixels stored.
 * Media cadence and wall-clock callback delivery are distinct, especially for
 * imported playback. Neither establishes sensor exposures or video/pose sync.
 */
export class SourceCadence {
  private observations = 0;
  private duplicates = 0;
  private regressions = 0;
  private invalid = 0;
  private first: { media: number; callback: number } | null = null;
  private last: { media: number; callback: number } | null = null;
  private gaps: number[] = [];
  private maximumGap: number | null = null;

  constructor(private readonly maximumGapSamples = 20_000) {
    if (
      !Number.isInteger(maximumGapSamples) ||
      maximumGapSamples < 1 ||
      maximumGapSamples > 20_000
    )
      throw new Error("Source cadence sample limit must be 1–20000.");
  }

  observe(mediaMs: number, callbackMs: number): void {
    if (![mediaMs, callbackMs].every((n) => Number.isFinite(n) && n >= 0)) {
      this.invalid++;
      return;
    }
    if (
      this.last &&
      (mediaMs < this.last.media || callbackMs < this.last.callback)
    ) {
      this.regressions++;
      return;
    }
    if (this.last?.media === mediaMs) {
      this.duplicates++;
      return;
    }
    if (this.last) {
      const gap = mediaMs - this.last.media;
      this.maximumGap = Math.max(this.maximumGap ?? 0, gap);
      if (this.gaps.length < this.maximumGapSamples) this.gaps.push(gap);
    }
    this.last = { media: mediaMs, callback: callbackMs };
    this.first ??= this.last;
    this.observations++;
  }

  summary(): SourceCadenceSummary {
    const mediaSpanMs =
      this.first && this.last ? this.last.media - this.first.media : null;
    const callbackSpanMs =
      this.first && this.last ? this.last.callback - this.first.callback : null;
    const rate = (span: number | null) =>
      span && this.observations > 1 && !this.regressions
        ? ((this.observations - 1) * 1000) / span
        : null;
    return {
      schemaVersion: "source-cadence-1",
      observations: this.observations,
      duplicates: this.duplicates,
      regressions: this.regressions,
      invalid: this.invalid,
      firstMediaMs: this.first?.media ?? null,
      lastMediaMs: this.last?.media ?? null,
      mediaSpanMs,
      callbackSpanMs,
      mediaFps: rate(mediaSpanMs),
      callbackFps: rate(callbackSpanMs),
      gapMs: {
        samples: this.gaps.length,
        complete: this.gaps.length === Math.max(0, this.observations - 1),
        p50: this.gaps.length ? percentile(this.gaps, 0.5) : null,
        p95: this.gaps.length ? percentile(this.gaps, 0.95) : null,
        maximum: this.maximumGap,
      },
    };
  }
}
