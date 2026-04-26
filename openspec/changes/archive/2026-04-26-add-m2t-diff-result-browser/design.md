## Context

The recent `m2t_diff` test visualization work standardized qualitative artifacts into a flat sample directory containing predicted text, ground-truth text, and source, target, and side-by-side videos keyed by sample id. That layout is already sufficient for a browser, but today review still happens through the filesystem or ad hoc video opening. The existing `app.py` uses Gradio and already demonstrates local video serving patterns, yet it is structured around model loading and inference pipelines rather than browsing static result folders.

This change should introduce a separate lightweight browser that accepts a `samples_<TIME>` directory as input, scans the flat artifacts into sample records, and renders a gallery-first review experience. The browser also needs lightweight persistence for favorites so users can mark notable samples without modifying the generated videos or text files.

## Goals / Non-Goals

**Goals:**
- Add a separate local browser entrypoint dedicated to MotionFix `m2t_diff` sample directories.
- Make the main screen gallery-first, with pagination over sample cards that use side-by-side video as the preview.
- Provide a detail panel for the selected sample showing source, target, and side-by-side videos and full generated and ground-truth text.
- Support mouse and keyboard navigation for page changes and sample navigation.
- Persist favorites across browser restarts in a small local metadata file rather than in browser-only session state.
- Tolerate partial exports by showing whatever artifacts are present and marking missing assets clearly.

**Non-Goals:**
- Merge static sample browsing into `app.py` or the online inference workflow.
- Introduce a backend database, user accounts, or collaborative annotations.
- Change the `m2t_diff` sample export layout, filenames, or rendering behavior produced by `test.py`.
- Generalize the first browser implementation to all tasks beyond MotionFix `m2t_diff`.

## Decisions

### 1. Build a separate Gradio app rather than extending `app.py`

The browser will be implemented as a distinct lightweight Gradio entrypoint that only reads existing sample folders. It will not import model-loading or inference setup unless required by shared utility code.

Why this over alternatives:
- Better than extending `app.py`, because the current app is inference-heavy and would mix static review concerns with runtime model setup.
- Better than a FastAPI or Flask service for the first rollout, because the repo already uses Gradio and the browser is intended for local interactive inspection rather than a deployed multi-user service.
- Better than generating static HTML first, because favorites, page selection, selected-sample state, and future filters are easier to evolve in an interactive app.

### 2. Treat the sample directory as the source of truth and derive records by sample id

The browser will scan one selected `samples_<TIME>` directory, infer sample ids from predicted text files such as `<id>.txt` while excluding `<id>_gt.txt`, and then resolve optional siblings for `_gt.txt`, `_source.mp4`, `_target.mp4`, and `_source_target.mp4`.

Why this over alternatives:
- Better than requiring a manifest file, because existing exports are already flat and self-describing by filename.
- Better than scanning videos first, because the predicted text file provides a stable primary key and remains useful even when rendering is incomplete.
- Better than assuming every artifact exists, because interrupted or timed-out test exports can leave partial folders that still need inspection.

### 3. Use gallery-first paging with a synchronized detail panel

The main view will render a page of sample cards, each showing the side-by-side preview, sample id, generated text, ground-truth text, and favorite state. Selecting a card updates a detail panel below or beside the gallery, where users can inspect the larger source, target, and side-by-side videos plus full texts for the selected sample.

Why this over alternatives:
- Better than a single-sample-first UI, because users need fast scanning across many qualitative outputs before drilling into details.
- Better than showing all three videos in every card, because that makes the gallery visually noisy and heavier to load.
- Better than a separate detail page route in the first pass, because a synchronized panel keeps browsing state simple and avoids extra navigation complexity.

### 4. Persist favorites in a sidecar metadata file, not in the sample artifacts

Favorites will be stored in a small local sidecar file associated with the browsed sample directory, such as JSON containing the folder path, sample ids, and timestamps if needed. The browser will read that file on startup and update it whenever the user toggles a favorite.

Why this over alternatives:
- Better than ephemeral in-memory state, because favorites need to survive browser restarts.
- Better than mutating the exported sample files or filenames, because the qualitative outputs should remain stable products of `test.py`.
- Better than a database, because single-user local persistence is enough for the current workflow.

### 5. Keyboard support should cover page-level and sample-level review actions

The browser will support page navigation shortcuts for moving between gallery pages and sample navigation shortcuts for moving the selected sample within or across pages. It will also include a shortcut to toggle favorites and may include a ground-truth hide or reveal shortcut if the implementation remains incremental.

Why this over alternatives:
- Better than mouse-only navigation, because qualitative review is faster when users can keep focus on the content.
- Better than implementing a larger shortcut set immediately, because a focused set of keys keeps the first rollout easier to document and test.

## Risks / Trade-offs

- [Large sample folders may make the gallery slow to initialize] → Mitigation: scan lightweight text and path metadata first, paginate cards, and avoid eagerly loading every full-size asset into the initial page.
- [Partial exports may produce confusing empty cards] → Mitigation: define clear missing-asset behavior and only require enough files to establish a sample id.
- [Gradio keyboard customization may be awkward] → Mitigation: keep keyboard support to a small set of shortcuts and use minimal custom JS only where needed for page or selection actions.
- [Favorites sidecar placement may be ambiguous] → Mitigation: define one deterministic location and filename for the sidecar relative to the browsed sample directory.
- [Future support for multiple sample folders could complicate persistence] → Mitigation: scope the initial design to one browsed directory at a time and key favorites to that directory.
