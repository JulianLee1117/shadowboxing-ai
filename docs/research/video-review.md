# Video-language models and the review layer

Checked September 26, 2026. This is a current candidate survey and proposed evaluation, not a measured model ranking. The user's preference is local live analysis with optional cloud review of selected clips.

## Recommendation

Build the live coach around timestamped motion evidence. Start its wording with reviewed templates. Introduce a review model once we have an evidence contract and coach-labeled critique examples. Compare providers on the same events, allowing for their actual supported input modes.

A model has two possible roles: explain accepted findings, or suggest additional visual hypotheses for human review. These roles need separate evaluation and UI status. A free-form interpretation must not silently become a validated measurement.

## Current verified candidates

| Candidate | Verified capability | Proposed use | What is not established |
|---|---|---|---|
| Gemini 3.8 Flash | Current official video guide uses this model; direct video input, static configurable frame sampling, and agentic video inspection | Optional short-clip review; compare deterministic sampling against agentic inspection | Boxing-critique accuracy, full capture-rate processing, or latency on our clips |
| GPT-6 Sol; Astra as a higher-compute comparison | Current official model catalog lists text/image input; Sol's modality page explicitly marks native video unsupported | Structured evidence explanation and ordered timestamped image review | Native video ingestion for Sol or accurate motion inference from sparse images |
| Qwen3.8-27B | Official August 2026 release; model card provides image/video inference examples | Optional offline open-weight review experiment | Useful quantized Mac latency or quality within this application's memory/thermal budget |
| Qwen3-VL smaller variants | Official repository documents video inputs and sampling controls | Smaller open-weight baseline where deployment tooling is compatible | That model size or available RAM predicts boxing accuracy |

Sources: [Gemini video guide](https://ai.google.dev/gemini-api/docs/video-understanding), [OpenAI model catalog](https://developers.openai.com/api/docs/models), [GPT-6 Sol modalities](https://developers.openai.com/api/docs/models/gpt-6-sol), [Qwen releases](https://github.com/QwenLM/Qwen3.8), [Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-27B), [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL).

These names are a dated shortlist. Pin the deployed model/version where possible and revalidate updates. Account access, rate limits, pricing, and any required terms must be checked at integration time. No API inference, model download, or purchase was performed during planning.

## Sampling is part of the model

Google documents a default of one video frame per second and warns that fast actions can lose detail. It also documents configurable static sampling and agentic exploration. For a punch lasting only a fraction of a second, sparse sampling can omit key phases. Inspect actual preprocessing and timestamp offsets rather than assuming that uploading a 30 fps file gives the model all 30 frames. [Official sampling documentation](https://ai.google.dev/gemini-api/docs/video-understanding#technical-details-about-videos).

Qwen's current model example similarly documents default sampling at 2 fps in its serving example. Sampling, token limits, resizing, and decoder choice are experimental variables. [Qwen3.8 video input](https://huggingface.co/Qwen/Qwen3.8-27B#video-input).

Proposed comparison: use short original clips with enough pre/post context, sweep supported sampling rates and image resolutions, and compare against explicit ordered frames at locally detected event phases. Keep frame timestamps and the clip's original-session offset. If a critical interval is missing or hidden, the output must say it is not assessable. Do not interpolate frames and treat synthetic intermediate poses as observed evidence.

## Evidence packet

Supply:

- Original clip or ordered frames; annotations are an additional view, not the only visual input.
- Canonical stance/anatomical-side mapping and display transform.
- The observed event stream, with unknowns, phase uncertainty and gaps.
- The selected drill and versioned coach rubric; the command given to the boxer is distinct from the observed action.
- Accepted measurements with units and evidence IDs; explicitly list unassessable criteria.
- A small allowed cue vocabulary and a requirement to identify supporting evidence.

Request structured output with `finding_id`, `evidence_ids`, `criterion_id`, `observation`, `suggested_cue`, `status`, and `abstain_reason`. Review the exact output schema against the provider API during implementation.

Do not ask the model to manufacture numeric confidence, power estimates, unseen joint angles, or causal explanations from sparse frames. The official OpenAI vision guide itself lists spatial-localization and incorrect-description limitations. That is a reason to validate this use case, not a proof that every current model fails it. [OpenAI vision limitations](https://developers.openai.com/api/docs/guides/images-vision#limitations).

## Evaluation and cost

Test the explanation task separately from unconstrained visual assessment. Include true positive, true negative, ambiguous, cropped, occluded, wrong-action, wrong-stance, and deliberate style-exception clips. Blind coaches to provider identity. Score factual support, contradiction, timing, action identity, appropriateness, and usefulness; record abstention and elapsed time. A model saying something convincing about every clip is not the goal.

Compare these conditions: templates from validated events; model from events only; model from frames/clips only; model from both. This reveals whether the model adds useful reasoning or merely repeats supplied labels. Audit any newly proposed fault against independent evidence before promoting it.

Track measured input tokens/frames, output tokens, requests, retry rate, processing latency, and actual billed cost per reviewed round. Freeze a maximum spend per experiment and per session. Current pricing is an integration-time variable, so no per-round cost is promised here. Keep live operation usable with review disabled.

For this 36 GiB Mac, local quantization is an experiment with runtime, memory, and quality tradeoffs. Do not run a large review model concurrently with live pose until sustained tests show it does not degrade capture. Optional cloud processing should receive selected clips only and visibly report success, failure, and deletion/retention options supported by the chosen provider.
