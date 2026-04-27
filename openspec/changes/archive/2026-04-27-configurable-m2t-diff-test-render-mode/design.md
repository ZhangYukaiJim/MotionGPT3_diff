## Context

`m2t_diff` test export already writes predicted text, ground-truth text, and three qualitative videos through the shared save path in `motGPT/models/base.py`. The rendering helpers in `motGPT/utils/render_utils.py` support both fast and slow rendering for single motions, but the save path always uses the default fast method and the side-by-side helper only materializes a fast visualization. Current cache layout is only partially method-aware, so a slow option must also isolate cache directories to avoid collisions with the existing fast renders.

## Goals / Non-Goals

**Goals:**
- Add a config-driven render method for MotionFix `m2t_diff` test visualization.
- Make source, target, and side-by-side videos all use the configured method.
- Keep fast and slow render caches separate for every exported video role.
- Preserve the existing output filenames in `results/.../samples_<TIME>/`.

**Non-Goals:**
- Change non-`m2t_diff` test visualization behavior.
- Change the browser app or sample export naming.
- Introduce new render modes beyond the existing fast and slow pipelines.

## Decisions

### Add a dedicated `TEST.VISUALIZE_RENDER_METHOD` config

The smallest compatible change is a new test config key with default `fast`. This keeps current runs unchanged while making render cost an explicit choice in experiment configs.

- Better than overloading `TEST.VISUALIZE`, because enablement and render fidelity are separate concerns.
- Better than task-specific hidden logic, because rendering time and cache usage are user-visible.

### Thread render method through the shared `m2t_diff` test artifact save path

`BaseModel._save_m2t_diff_test_artifacts()` already centralizes test-time export, so it should read the configured method once and pass it into `materialize_m2t_diff_motion_artifacts(...)`.

- Better than branching in `test.py`, because the model save path already owns the task-specific artifact logic.

### Make all MotionFix render cache directories method-aware

The existing helper already includes `method` in single-motion cache directories, but not in the `m2t_diff` side-by-side cache path. The helper that returns per-role cache directories should therefore accept `method` and use it for source, target, and side-by-side cache roots.

- Better than renaming output files in `results/...`, because the sample directory layout is already consumed by existing review workflows.
- Better than sharing one cache namespace, because fast and slow outputs are not interchangeable and must never overwrite each other.

### Implement a slow side-by-side render by composing slow-rendered frame sequences

The repository already has a slow SMPL-based single-motion render path. The minimal consistent extension is to factor that path so it can produce frames for one motion, then render the left and right motions with the same pipeline and concatenate the frames horizontally for the side-by-side video.

- Better than keeping side-by-side fast-only, because the user explicitly wants slow mode to affect all three videos.
- Better than inventing a new paired-scene renderer, because it reuses the current slow rendering style and minimizes architectural churn.

## Risks / Trade-offs

- [Slow rendering is much more expensive during full-test export] → Mitigation: keep `fast` as the default and document the cost trade-off.
- [Slow side-by-side rendering duplicates per-frame work for source and target] → Mitigation: reuse one shared slow-frame helper and preserve cache reuse so repeated runs do not rerender existing assets.
- [Method-aware cache changes could break reuse if output paths change unexpectedly] → Mitigation: keep sample output filenames unchanged and only scope the internal render-cache directories by method.
- [Slow rendering depends on optional SMPL/render assets] → Mitigation: rely on the same prerequisites already required by the existing slow single-motion renderer.
