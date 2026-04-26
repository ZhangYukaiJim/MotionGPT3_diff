## 1. Browser entrypoint and sample loading

- [x] 1.1 Add a new lightweight Gradio browser entrypoint outside `app.py` for browsing one MotionFix `m2t_diff` `samples_<TIME>` directory.
- [x] 1.2 Implement sample-directory scanning that infers sample ids from predicted text files and resolves optional `_gt.txt`, `_source.mp4`, `_target.mp4`, and `_source_target.mp4` siblings.
- [x] 1.3 Add sample-record parsing that preserves available file paths and text content while marking missing artifacts for partial or interrupted exports.

## 2. Gallery and detail interactions

- [x] 2.1 Implement the gallery-first main view with paginated sample cards showing sample id, side-by-side preview when available, generated text, ground-truth text, and favorite state.
- [x] 2.2 Implement synchronized selection state so choosing a gallery sample updates a detail panel with larger source, target, and side-by-side videos plus full generated and ground-truth text.
- [x] 2.3 Add previous or next page controls and previous or next sample controls that work across page boundaries.

## 3. Favorites persistence and keyboard controls

- [x] 3.1 Implement favorite toggle behavior for gallery cards or the selected sample.
- [x] 3.2 Persist favorites in a deterministic sidecar metadata file associated with the browsed sample directory and reload them on startup.
- [x] 3.3 Add keyboard shortcuts for page navigation, sample navigation, and favorite toggling using minimal custom JS where needed.

## 4. Validation and documentation

- [x] 4.1 Validate the browser against an existing `results/motgpt/.../samples_<TIME>/` directory, including a partial export directory with missing artifacts.
- [x] 4.2 Run targeted validation commands with `uv`, such as a syntax check for the new browser script and a local launch command for manual verification.
- [x] 4.3 Update `README.md` with the new browser workflow, required `uv` invocation, sample directory argument, and favorites sidecar behavior.
