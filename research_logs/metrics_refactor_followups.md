# Metrics Refactor Follow-Ups

## Completed In This Round

- Moved the TorchMetrics BERTScore device-compatibility patch into `motGPT/metrics/bert_compat.py`.
- Updated `M2TDiffMetrics` to use best-over-references for `Bert_F1` instead of only the first reference.

## Remaining Follow-Ups

### 1. Unify Bert implementation across `m2t` and `m2t_diff`

`M2TMetrics` still uses raw `bert_score.score(...)`, while `M2TDiffMetrics` now uses the TorchMetrics compatibility wrapper.

Why this matters:
- same metric name currently comes from two different backends
- the raw `bert_score` path previously showed problematic runtime behavior inside Lightning
- cross-task metric behavior is harder to reason about when implementations differ

Suggested next step:
- switch `M2TMetrics` to the shared `FixedDeviceBERTScore` path

### 2. Standardize text-generation metric backend

`M2TMetrics` uses `NLGMetricverse` for BLEU/ROUGE/CIDEr, while `M2TDiffMetrics` uses local implementations.

Why this matters:
- identical metric names may not reflect identical implementations
- maintenance gets harder when fixes have to be applied in two places

Suggested next step:
- choose one backend for BLEU/ROUGE/CIDEr across both metric classes
- prefer the smallest option that supports the multi-reference behavior we want

### 3. Clarify and document multi-reference policy per metric

`M2TDiffMetrics` now uses:
- BLEU/ROUGE/CIDEr: all references
- Bert_F1: best over references

Suggested next step:
- document this explicitly in README or spec notes so metric interpretation is stable

### 4. Clean up metric construction and logging

`BaseMetrics` still contains a direct debug print:
- `print("in BaseMetrics, data_name:", data_name)`

Suggested next step:
- remove it or replace it with structured logging

### 5. Consider a small registry for metric construction

`BaseMetrics` is still a sequence of string checks and task-specific branches.

Suggested next step:
- only revisit if more metric classes are added
- not urgent yet
