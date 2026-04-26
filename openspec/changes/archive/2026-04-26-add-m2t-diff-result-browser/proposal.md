## Why

MotionFix `m2t_diff` test visualization now produces a useful flat sample directory, but reviewing hundreds of samples through filesystem navigation is slow and makes comparison, paging, and annotation awkward. A lightweight browser for those exported sample folders will make qualitative evaluation faster and more repeatable without coupling static result review to the model inference UI.

## What Changes

- Add a separate browser workflow for MotionFix `m2t_diff` sample directories that reads existing exported artifacts without rerunning inference.
- Provide a gallery-first view that pages through sample cards showing the side-by-side video, sample id, generated text, and ground-truth text.
- Provide a detail view for the selected sample with larger source, target, and side-by-side videos plus full generated and ground-truth text.
- Add browser navigation controls for page changes and sample selection, including keyboard shortcuts for common review actions.
- Add a bookmark or favorites feature that lets users mark selected samples and persist those marks outside the sample artifacts themselves.
- Keep the browser tolerant of partial or interrupted sample exports so users can inspect incomplete test runs.

## Capabilities

### New Capabilities
- `two-motion-difference-result-browser`: Browse exported MotionFix `m2t_diff` sample directories through a gallery-first local web UI with detail view, keyboard navigation, and persisted favorites.

### Modified Capabilities

## Impact

- Affected code:
  - a new lightweight browser entrypoint outside `app.py`, likely under the repo root or `scripts/`
  - helper code to scan `results/<model>/<NAME>/samples_<TIME>/` directories and assemble sample records from existing files
  - browser-side state and persistence for page selection, sample selection, and favorites
  - `README.md` usage documentation for launching the browser with `uv`
- Affected artifacts:
  - exported sample directories remain read-only inputs to the browser
  - the browser may create a small metadata file for favorites alongside or near the browsed sample directory
- Non-goals for this change:
  - no integration of static result browsing into the existing inference-heavy `app.py`
  - no changes to how `test.py` renders or names `m2t_diff` qualitative artifacts
  - no server-backed multi-user review workflow, authentication, or database storage in the first rollout
