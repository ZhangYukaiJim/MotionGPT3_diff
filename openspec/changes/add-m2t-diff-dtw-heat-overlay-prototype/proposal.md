## Why

MotionFix `m2t_diff` currently exports source, target, and side-by-side videos, but those views do not localize where the user motion differs from the target motion. For paired motions that are not perfectly time-synced, we need a self-contained prototype that validates whether constrained-DTW alignment plus a vertex-level mesh heat overlay can make those differences easier to inspect before changing the main evaluation or browser workflows.

## What Changes

- Add a standalone prototype script that loads a MotionFix paired-motion sample from existing cached features or direct `.npy` inputs.
- Compute constrained DTW over root-normalized 22-joint motion features to align the source and target motions for qualitative comparison.
- Export temporal-alignment validation artifacts, including the frame-cost matrix with warping path, path plots, scalar trace plots, and an aligned side-by-side preview video.
- Export a vertex-level mesh heat-overlay video that renders the source motion with per-vertex heat and the target motion as a translucent ghost reference.
- Document the prototype workflow, prerequisites, and output layout in `README.md`.

## Capabilities

### New Capabilities
- `two-motion-difference-heat-overlay-prototype`: standalone qualitative tooling for DTW-aligned temporal validation and vertex-level heat-overlay rendering for MotionFix `m2t_diff` motion pairs.

### Modified Capabilities

## Impact

- Affected code: `scripts/prototype_m2t_diff_heat_overlay.py`, `README.md`
- Affected components: MotionFix dataset cache lookup, render/SMPL dependencies, qualitative `m2t_diff` inspection workflows
- Affected outputs: new prototype artifact directories under an explicit output folder or under an exported `results/.../samples_<TIME>/` sample directory when that workflow is used
- Non-goals: no changes to `train.py`, `test.py`, `app.py`, `app_m2t_diff_browser.py`, `motGPT/metrics/`, or the standard `m2t_diff` sample export schema
