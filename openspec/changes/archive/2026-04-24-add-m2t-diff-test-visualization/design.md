## Context

MotionFix `m2t_diff` evaluation already has most of the pieces needed for qualitative test export, but they are split across two different paths. The validation path in `motGPT/models/motgpt.py` can already materialize MotionFix source, target, and side-by-side videos plus predicted and ground-truth text for a small fixed subset of samples. The test path in `test.py` and `motGPT/models/base.py` is broader in coverage, but it is centered on metric computation and raw prediction saving, and it does not preserve enough paired-motion data for `m2t_diff` to reproduce the full validation-style visualization bundle.

This change needs to bridge those paths without introducing a second rendering workflow. The existing MotionFix helpers in `motGPT/utils/render_utils.py` already provide cached single-motion and side-by-side rendering. The cleanest implementation is therefore to extend the current `m2t_diff` test output path so the shared test-save code can materialize the same paired artifacts for all requested test samples.

## Goals / Non-Goals

**Goals:**
- Add a config-driven `m2t_diff` test visualization mode through `test.py`.
- Reuse the current MotionFix rendering and cache helpers rather than adding a new export pipeline.
- Preserve the paired fields needed for source motion, target motion, source/target lengths, predicted text, and ground-truth text during `m2t_diff` test runs.
- Keep raw prediction saving and qualitative visualization as separate toggles so metric-only evaluation remains lightweight.
- Keep output layout compatible with the existing `results/<model>/<NAME>/samples_<TIME>/` convention.

**Non-Goals:**
- Add web UI support for full-test qualitative export.
- Generalize the first implementation to every task type beyond `m2t_diff`.
- Redesign validation visualization output or move it to a new shared abstraction unless required for a minimal implementation.
- Change MotionFix dataset preprocessing, metric definitions, or checkpoint loading behavior.

## Decisions

### 1. Extend the existing test return payload for `m2t_diff`

The `split == "test"` return in `motGPT/models/motgpt.py` will be extended for `m2t_diff` so the downstream save path receives the full paired-motion context: target features, source features, target lengths, source lengths, predicted text, reference text, and sample ids.

Why this over alternatives:
- Better than reloading the dataset inside `save_npy()`, because the model already has the paired tensors prepared and normalized consistently with the generation pass.
- Better than building a separate postprocessing script, because `test.py` already owns output directory creation and per-run artifact saving.
- Better than overloading the current `m2t` test tuple shape, because `m2t_diff` needs extra paired fields that single-motion tasks do not use.

### 2. Keep visualization in the shared test-save path, but gate it with a dedicated config flag

The existing `BaseModel.save_npy()` path will stay responsible for writing test artifacts, but it will gain an `m2t_diff`-specific visualization branch that only runs when a dedicated config flag is enabled. Raw text and `.npy` prediction saving will remain controlled by `TEST.SAVE_PREDICTIONS`, while video materialization will use a new flag such as `TEST.VISUALIZE`.

Why this over alternatives:
- Better than always rendering during test, because full-test MotionFix visualization is much more expensive than metric-only evaluation.
- Better than tying rendering to `SAVE_PREDICTIONS`, because users may want raw outputs without videos or vice versa.
- Better than moving test visualization into callbacks, because the required per-sample payload already flows through the saved test outputs.

### 3. Reuse MotionFix render-cache helpers exactly as validation does

For `m2t_diff`, test visualization will call `materialize_cached_motion_render(...)` for source and target motions and `materialize_cached_motion_side_by_side_render(...)` for the paired comparison video. This mirrors the current validation path and keeps the render cache rooted under the MotionFix dataset cache.

Why this over alternatives:
- Better than calling `render_motion(...)` unconditionally for every sample, because cached materialization avoids redundant rendering cost across repeated test runs.
- Better than introducing a second test-only cache format, because the current cached-video layout already matches the required source/target/paired artifacts.

### 4. Keep output layout flat and consistent with current sample export

The first implementation will keep writing artifacts into the existing `samples_<TIME>/` output directory using stable filenames such as `<id>_source.mp4`, `<id>_target.mp4`, `<id>_source_target.mp4`, `<id>.txt`, and `<id>_gt.txt`.

Why this over alternatives:
- Better than switching to nested per-sample directories in the first pass, because the repo already uses flat sample export layouts and downstream inspection scripts can continue to glob by filename.
- Better than changing validation layout too, because the change is scoped to test export only.

### 5. Document the new behavior in README and config defaults, but keep it opt-in

The change will add config fields in `configs/default.yaml` and document a sample `uv run python -m test ...` workflow in `README.md`. Visualization will remain disabled by default so current test jobs do not unexpectedly spend time rendering videos.

Why this over alternatives:
- Better than hidden behavior behind task-specific code paths, because rendering cost and output volume are user-visible concerns.
- Better than enabling visualization globally for MotionFix tests, because large test exports can produce many videos and symlinks.

## Risks / Trade-offs

- [Full-test rendering is slow and storage-heavy] → Mitigation: keep visualization opt-in, reuse cached renders, and allow future limit knobs without requiring them in the first rollout.
- [Test payload shape divergence across tasks] → Mitigation: confine the richer output tuple to `m2t_diff` and keep existing task branches unchanged.
- [Output layout becomes harder to parse if filenames drift] → Mitigation: specify stable suffixes for source, target, side-by-side, prediction text, and ground-truth text in the spec.
- [Render cache assumptions may fail if MotionFix cache root is missing] → Mitigation: reuse the same MotionFix-root assumptions already required for validation and fail in the same way rather than introducing a second dataset-root contract.
- [README/examples drift from behavior] → Mitigation: update the documented `test.py` workflow at the same time as the config and save-path changes.
