## Context

MotionGPT3 already contains a motion-to-text metric module (`M2TMetrics`) and several motion-text retrieval evaluators (`TM2TMetrics`, `TMRMetrics`), but those paths were designed around single-motion supervision. For MotionFix `m2t_diff`, the generated caption describes the relationship between a source motion and a target motion, so the most defensible first quantitative evaluation layer is text-to-text comparison against the reference caption plus simple decoding diagnostics.

The current repo already has useful building blocks in `M2TMetrics`: `ROUGE_L`, `CIDEr`, BLEU, and `Bert_F1`. Those metrics operate on predicted and reference text and do not require a pair-aware motion embedding. By contrast, the retrieval-style `Matching_score` and `R_precision_top_k` calculations rely on a single-motion embedding and are not semantically correct for paired-motion difference captions.

## Goals / Non-Goals

**Goals:**
- Provide a minimal, credible quantitative metric path for `m2t_diff` using text-generation metrics.
- Add cheap decoding diagnostics that catch empty-output regressions early.
- Keep the implementation incremental and compatible with the existing config-driven metric and logging flow.
- Preserve a clean boundary between the first rollout and a future pair-aware retrieval metric.

**Non-Goals:**
- Do not redesign TM2T or TMR encoders to consume motion pairs in this change.
- Do not claim that single-motion retrieval metrics are valid for paired-motion captioning.
- Do not introduce a new external evaluation model unless existing repo dependencies are insufficient.

## Decisions

### 1. Reuse text-generation metrics, not single-motion retrieval metrics

We will treat `Bert_F1`, `ROUGE_L`, and `CIDEr` as the primary `m2t_diff` metrics, with optional BLEU scores as secondary signals. These metrics compare predicted text against reference text directly and are valid regardless of whether the source condition was one motion or two motions.

Why this over alternatives:
- Better than reusing `Matching_score` and `R_precision_top_k`, because those require a single-motion embedding target and do not encode the relation between source and target motions.
- Better than delaying all quantitative evaluation, because the repo already has NLG-oriented metric pieces that can be reused.

### 2. Add explicit decoding diagnostics for empty and short outputs

We will report at least empty-output rate and average generated caption length for `m2t_diff`. These are cheap to compute, easy to interpret, and directly useful for catching regressions like prompt-only or EOS-collapse behavior.

Why this over alternatives:
- Better than relying only on aggregate overlap metrics, because collapsed decoding can sometimes hide behind noisy scores, especially with one reference caption per sample.
- Better than manual inspection alone, because these diagnostics are available every validation/test run.

### 3. Keep the first `m2t_diff` metric path independent from pair-aware motion-text retrieval

We will explicitly document that a future pair-aware evaluator is a separate capability. The first implementation focuses on metrics the repo can defend today.

Why this over alternatives:
- Better than stretching TM2T/TMR semantics to fit paired-motion captions.
- Keeps implementation small and avoids introducing an evaluator whose assumptions are not yet validated.

## Implementation Sketch

- Extend the current motion-to-text metric path or add a dedicated paired-motion text metric class under `motGPT/metrics/`.
- Limit the `m2t_diff` metric outputs to text-generation and diagnostic values.
- Wire the metric through `motGPT/models/motgpt.py` so `m2t_diff` validation/test can update it without enabling single-motion retrieval metrics.
- Add config guidance in `configs/` for recommended MotionFix metric settings.

## Risks / Trade-offs

- One-reference-caption data will make BLEU/ROUGE/CIDEr noisier than in multi-reference settings. Mitigation: treat `Bert_F1` and the diagnostics as the strongest first-line signals.
- Reusing parts of `M2TMetrics` may require small refactoring if retrieval calculations are tightly coupled to the text metrics. Mitigation: split the text-only path rather than forcing all existing outputs to stay bundled.
- Users may expect a motion-aware semantic score. Mitigation: document clearly that pair-aware retrieval is future work and keep a separate research note for it.
