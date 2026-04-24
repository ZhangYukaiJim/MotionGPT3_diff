## Why

The current MotionFix `m2t_diff` workflow can render paired-motion qualitative samples during validation, but `test.py` does not provide an equally direct way to materialize all test samples as source/target videos, side-by-side comparisons, predicted text, and ground-truth text. This makes full test-set qualitative review awkward even though the repo already has the model outputs, render-cache helpers, and MotionFix-specific visualization logic needed to support it.

## What Changes

- Add a config-driven test visualization path for `m2t_diff` that can save qualitative artifacts for test samples through `test.py`.
- Reuse the existing MotionFix render-cache helpers so test visualization can materialize source, target, and side-by-side videos without duplicating rendering logic.
- Extend the `m2t_diff` test output path so it preserves the paired-motion fields needed to save predicted text, ground-truth text, and rendered motion assets together.
- Keep raw prediction saving and visualization controls separate so users can request all-sample qualitative outputs without changing metric computation behavior.

## Capabilities

### New Capabilities
- `two-motion-difference-test-visualization`: Save config-driven `m2t_diff` test artifacts, including paired motion renders and caption outputs, for qualitative review of MotionFix test samples.

### Modified Capabilities
- `two-motion-difference-to-text`: Extend the non-web evaluation workflow so test runs can emit paired-motion qualitative artifacts in addition to quantitative metrics.

## Impact

- Affected code:
  - `motGPT/models/` for `m2t_diff` test outputs and artifact-saving hooks
  - `motGPT/utils/render_utils.py` reuse for cached MotionFix renders
  - `test.py` and/or shared test-save paths for config-driven visualization output
  - `configs/` for new test visualization controls
- Affected artifacts:
  - `results/` test runs may now include rendered videos and paired text files for `m2t_diff` when visualization is enabled
  - MotionFix render cache under the dataset root may be reused more heavily during full-test qualitative export
- Non-goals for this change:
  - no new web UI flow
  - no redesign of validation visualization output
  - no attempt to visualize every other task type through the same MotionFix-specific paired-motion path in the first rollout
