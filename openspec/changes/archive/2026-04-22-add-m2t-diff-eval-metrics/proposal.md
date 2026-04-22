## Why

The current MotionFix `m2t_diff` workflow can train and qualitatively validate paired-motion difference captions, but it does not yet define a recommended quantitative evaluation path. The repo's existing metric stack is centered on single-motion text-to-motion and motion-to-text evaluation, so it mixes metrics that are directly useful for paired-motion captioning with retrieval metrics that assume a single motion input. We need a clear, config-driven metric path for `m2t_diff` that measures text quality and catches degenerate decoding such as empty outputs, without pretending that single-motion retrieval metrics are pair-aware.

## What Changes

- Add a recommended evaluation path for `m2t_diff` based on text-generation metrics rather than single-motion retrieval metrics.
- Add explicit diagnostic metrics for blank-output and caption-length failures that are especially relevant to paired-motion difference captioning.
- Define how these metrics are enabled and reported in config-driven MotionFix evaluation runs.
- Document the boundary that pair-aware retrieval metrics are a future capability and are not part of the first `m2t_diff` metric rollout.

## Capabilities

### New Capabilities
- `two-motion-difference-eval-metrics`: Evaluate `m2t_diff` outputs with text-generation and decoding-diagnostic metrics that are compatible with paired-motion difference captions.

### Modified Capabilities
- `two-motion-difference-to-text`: Clarify which quantitative metrics are valid for the existing paired-motion task and which existing single-motion metrics must not be reused unchanged.

## Impact

- Affected code:
  - `motGPT/metrics/` for a dedicated `m2t_diff`-appropriate metric path or an extension of the current motion-to-text metric flow
  - `motGPT/models/` for metric dispatch during paired-motion validation/test
  - `configs/` for recommended MotionFix evaluation metric settings
  - `test.py` and validation logging paths for metric reporting
- Affected artifacts:
  - `experiments/` and `results/` may gain metric logs for `m2t_diff` runs
  - `research_logs/` may continue to track future pair-aware metric ideas, but the first implementation stays focused on text and diagnostic metrics
- Non-goals for this change:
  - no pair-aware motion-text retrieval model yet
  - no rewrite of TM2T or TMR evaluators to accept motion pairs in the first metric rollout
