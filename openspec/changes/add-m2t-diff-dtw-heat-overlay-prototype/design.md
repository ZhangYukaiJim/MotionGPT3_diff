## Context

The current MotionFix `m2t_diff` flow already produces paired source/target videos and caption files under `results/.../samples_<TIME>/`, and the repository already contains everything needed to recover MotionFix joints and render SMPL meshes. What it does not have is a temporal alignment layer or any localized difference visualization beyond raw side-by-side playback. The user wants a quick prototype that stays isolated from the training, evaluation, and browser paths, and that produces visual validation artifacts for both temporal alignment and the final mesh heat overlay.

## Goals / Non-Goals

**Goals:**
- Add a standalone prototype script that can resolve a MotionFix motion pair from either cached feature files or an exported sample id.
- Compute constrained DTW on root-normalized joint features using only existing repository dependencies.
- Export visual validation artifacts that make the DTW path easy to inspect before trusting the overlay.
- Render a source-focused vertex heat overlay with the target mesh shown as a translucent ghost.
- Keep the implementation self-contained and avoid changing standard config-driven training, evaluation, or browsing workflows.

**Non-Goals:**
- Integrate the prototype into `test.py`, `demo.py`, `app.py`, or `app_m2t_diff_browser.py`.
- Introduce new model checkpoints, metrics, or training-time supervision.
- Build a body-part segmentation system or a production-ready coaching UI.
- Preserve global trajectory differences inside the first heat-overlay prototype; the prototype focuses on root-centered pose comparison.

## Decisions

### Use a standalone script instead of modifying `test.py` or the browser

The prototype should live in `scripts/` so it can reuse existing MotionFix caches and render assets without changing standard `m2t_diff` test export or result-browsing behavior.

- Better than wiring the prototype into `test.py`, because the user explicitly wants small, self-contained steps with minimal codebase impact.
- Better than changing `app_m2t_diff_browser.py` immediately, because we first need to validate whether the alignment and heat overlay are actually informative.

### Resolve motions through the existing MotionFix cache and datamodule conventions

The script will load config values through the same OmegaConf merge path used elsewhere, instantiate the MotionFix datamodule, and reuse its normalization helpers when loading cached feature arrays or searching manifests by sample id.

- Better than hardcoding dataset paths or normalization assets inside the prototype, because it stays compatible with the current MotionFix cache layout and config overrides.
- Better than depending on model checkpoints, because the qualitative prototype only needs feature-to-joint recovery and SMPL rendering.

### Align motions with constrained DTW on root-centered joint positions plus velocities

The alignment feature will be a flattened per-frame representation of root-centered 22-joint positions plus a lower-weight velocity term. DTW will use a Sakoe-Chiba-style band centered on the scaled diagonal plus a mild penalty for horizontal and vertical moves.

- Better than frame-to-frame comparison, because MotionFix pairs are not perfectly synchronized.
- Better than aligning directly on mesh vertices, because joint features are cheaper to inspect and easier to validate.
- Better than importing a new DTW dependency, because the repository already has `numpy` and `scipy` and the user wants a small isolated prototype.

### Validate the alignment explicitly before rendering the heat overlay

The script will always write alignment artifacts before or alongside the final overlay: cost matrix with warping path, frame-index mapping plot, scalar trace plots, a serialized path file, and an aligned side-by-side preview video.

- Better than trusting the DTW path implicitly, because over-alignment is the main failure mode for this prototype.
- Better than a text-only summary, because the user asked for visual validation of the temporal alignment step.

### Render root-centered vertex heat on the source mesh and show the target as a ghost mesh

After DTW alignment, the script will convert root-centered joint sequences to SMPL vertices, compute per-vertex displacement magnitudes on aligned frame pairs, smooth and clip the heat values, and render the source mesh with per-vertex colors while the target mesh is shown as a translucent ghost.

- Better than coloring both meshes, because the user chose a source-focused coaching view.
- Better than a body-part-only heat map for this prototype, because the user wants the more expressive vertex-level view.
- Better than a full trajectory-preserving overlay for the first iteration, because root-centered rendering isolates pose difference and matches the chosen DTW feature space.

## Risks / Trade-offs

- [DTW can over-align mismatched phases and make the overlay look cleaner than the motions really are] → Mitigation: always export path diagnostics and the aligned side-by-side preview.
- [Root-centered comparison hides global translation or path differences] → Mitigation: document that limitation clearly in the README and keep the standard source/target videos available as complementary context.
- [Vertex-level heat can be noisy frame to frame] → Mitigation: smooth heat values temporally and clip them to a robust percentile before color mapping.
- [The overlay depends on optional render assets and `smplx`/`pyrender`] → Mitigation: keep those imports local to the prototype path and document the `uv sync --extra render` prerequisite.
- [A standalone script duplicates small pieces of config-loading logic] → Mitigation: reuse the repository's existing OmegaConf conventions and keep the script focused on prototype-only behavior instead of introducing shared abstractions prematurely.
