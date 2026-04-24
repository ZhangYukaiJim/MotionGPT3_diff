## 1. Config And Test Output Plumbing

- [x] 1.1 Add opt-in test visualization config fields in `configs/default.yaml` for qualitative export without changing existing metric-only test behavior.
- [x] 1.2 Extend the `m2t_diff` test return path in `motGPT/models/motgpt.py` so test outputs preserve source motion, target motion, source/target lengths, predicted text, reference text, and sample ids.

## 2. Paired-Motion Test Artifact Saving

- [x] 2.1 Update `motGPT/models/base.py` so the shared test save path recognizes `m2t_diff` outputs and writes stable text artifacts for predictions and ground truth.
- [x] 2.2 Reuse `motGPT/utils/render_utils.py` MotionFix cache helpers from the test save path to materialize `<id>_source.mp4`, `<id>_target.mp4`, and `<id>_source_target.mp4` when test visualization is enabled.
- [x] 2.3 Keep non-`m2t_diff` test save behavior unchanged while confining the richer visualization branch to the `m2t_diff` task.

## 3. Documentation And Validation

- [x] 3.1 Update `README.md` with the new `m2t_diff` test visualization workflow, including the new config flag and an example `uv run python -m test ...` command.
- [x] 3.2 Validate the implementation with a config-driven MotionFix test run that enables `m2t_diff` visualization and confirms the expected text files and paired videos are written under `results/<model>/<NAME>/samples_<TIME>/`.
- [x] 3.3 Run `uv run python -m py_compile motGPT/models/base.py motGPT/models/motgpt.py test.py` after the code changes to catch syntax or import regressions in the touched test path.
