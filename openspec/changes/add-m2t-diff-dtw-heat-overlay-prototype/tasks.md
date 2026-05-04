## 1. OpenSpec And Documentation

- [ ] 1.1 Add the OpenSpec proposal, design, and prototype spec under `openspec/changes/add-m2t-diff-dtw-heat-overlay-prototype/`.
- [ ] 1.2 Update `README.md` with prototype prerequisites, CLI usage, and expected output artifacts.

## 2. Prototype Script

- [ ] 2.1 Add `scripts/prototype_m2t_diff_heat_overlay.py` with config loading, MotionFix sample resolution, root-centered joint recovery, and constrained-DTW alignment.
- [ ] 2.2 Export alignment validation artifacts from the prototype script, including plots, serialized path data, and an aligned side-by-side preview video.

## 3. Mesh Heat Rendering

- [ ] 3.1 Add prototype-local SMPL mesh conversion and ghost-target plus source-heat rendering in `scripts/prototype_m2t_diff_heat_overlay.py`.
- [ ] 3.2 Export vertex-level heat-overlay artifacts, including the main overlay video and at least one static validation image.

## 4. Validation

- [ ] 4.1 Run `uv run python -m py_compile scripts/prototype_m2t_diff_heat_overlay.py`.
- [ ] 4.2 Run `uv run python scripts/prototype_m2t_diff_heat_overlay.py --help` to validate the CLI entry point without executing a full render.
