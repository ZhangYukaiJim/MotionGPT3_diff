from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from urllib.parse import quote

import gradio as gr


FAVORITES_SIDECAR = ".m2t_diff_browser_favorites.json"
DEFAULT_PAGE_SIZE = 8
MIN_PAGE_SIZE = 4
MAX_PAGE_SIZE = 24
RESULTS_ROOT = Path("results/motgpt")


CUSTOM_CSS = """
body {
    background: #0f1117;
}

.app-shell {
    max-width: 1500px;
    margin: 0 auto;
}

.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
}

.sample-card {
    background: #171b24;
    border: 1px solid #2a3140;
    border-radius: 16px;
    overflow: hidden;
    color: #edf2f7;
    cursor: pointer;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.sample-card:hover {
    transform: translateY(-2px);
    border-color: #5b6c8a;
}

.sample-card.selected {
    border-color: #6ea8fe;
    box-shadow: 0 0 0 1px rgba(110, 168, 254, 0.35);
}

.sample-card.favorite {
    border-color: #f6c85f;
}

.sample-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 14px 8px;
    font-weight: 600;
}

.sample-card-id {
    font-family: ui-monospace, SFMono-Regular, SFMono-Regular, Menlo, monospace;
    font-size: 0.95rem;
}

.sample-card-star {
    color: #f6c85f;
    font-size: 1.1rem;
}

.sample-card-star-button {
    appearance: none;
    border: 0;
    background: transparent;
    color: #f6c85f;
    cursor: pointer;
    font-size: 1.2rem;
    line-height: 1;
    padding: 0;
}

.sample-card-star-button:hover {
    transform: scale(1.08);
}

.sample-card-preview video {
    display: block;
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: cover;
    background: #06080d;
}

.sample-card-preview-missing {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: linear-gradient(135deg, #1a2030, #0f1117);
    color: #9fb3c8;
    font-size: 0.95rem;
}

.sample-card-body {
    padding: 12px 14px 16px;
}

.sample-card-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8ea2bd;
    margin-bottom: 4px;
}

.sample-card-text {
    font-size: 0.92rem;
    line-height: 1.45;
    min-height: 2.8em;
    color: #edf2f7;
    margin-bottom: 10px;
}

.sample-card-text.missing {
    color: #9fb3c8;
    font-style: italic;
}

.sample-card-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}

.sample-card-badge {
    background: #202737;
    border: 1px solid #313c52;
    border-radius: 999px;
    color: #c7d3e1;
    font-size: 0.78rem;
    padding: 4px 10px;
}

.sample-card-badge.warn {
    border-color: #72552b;
    color: #f6c85f;
}

.empty-gallery {
    padding: 40px 24px;
    border: 1px dashed #3a455d;
    border-radius: 16px;
    text-align: center;
    background: #171b24;
    color: #c7d3e1;
}

.hotkey-strip {
    margin-bottom: 12px;
    border: 1px solid #273146;
    background: #171b24;
    border-radius: 14px;
    padding: 10px 14px;
    color: #d9e2ec;
    font-size: 0.9rem;
}

.hotkey-strip code {
    background: #0f1117;
    padding: 2px 6px;
    border-radius: 6px;
}

.detail-video-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin: 8px 0 4px;
}

.detail-video-card {
    background: #171b24;
    border: 1px solid #2a3140;
    border-radius: 16px;
    overflow: hidden;
}

.detail-video-card-title {
    padding: 10px 14px;
    border-bottom: 1px solid #2a3140;
    color: #d9e2ec;
    font-weight: 600;
}

.detail-video-card video {
    display: block;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: #06080d;
}

.detail-video-missing {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: linear-gradient(135deg, #1a2030, #0f1117);
    color: #9fb3c8;
    padding: 16px;
    text-align: center;
}

"""


HOTKEYS_HTML = """
<div class="hotkey-strip">
  Use the native controls below the gallery to pick a sample on the current page, then click the bookmark button. The stars in the gallery and selector are indicators only in this Gradio build.
</div>
"""


APP_HEAD = """
<script>
(() => {
  if (window.__m2tDiffBrowserInstalled) {
    return;
  }
  window.__m2tDiffBrowserInstalled = true;

  const clickButton = (elemId) => {
    const button = document.querySelector(`#${elemId} button`);
    if (button) {
      button.click();
      return true;
    }
    return false;
  };

  const startAutoplayVideos = (root) => {
    root.querySelectorAll("video.autoplay-video").forEach((video) => {
      video.muted = true;
      video.defaultMuted = true;
      video.loop = true;
      const tryPlay = () => {
        const playPromise = video.play();
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch(() => {});
        }
      };
      tryPlay();
      if (!video.dataset.autoplayBound) {
        video.dataset.autoplayBound = "1";
        video.addEventListener("loadeddata", tryPlay);
      }
    });
  };

  document.addEventListener(
    "keydown",
    (event) => {
      const active = document.activeElement;
      const tagName = active ? active.tagName : "";
      if (
        tagName === "INPUT" ||
        tagName === "TEXTAREA" ||
        (active && active.isContentEditable)
      ) {
        return;
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        clickButton("prev-page-btn");
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        clickButton("next-page-btn");
      } else if (event.key === "j") {
        event.preventDefault();
        clickButton("prev-sample-btn");
      } else if (event.key === "k") {
        event.preventDefault();
        clickButton("next-sample-btn");
      } else if (event.key === "f") {
        event.preventDefault();
        clickButton("favorite-toggle-btn");
      }
    },
    true,
  );

  const observer = new MutationObserver(() => startAutoplayVideos(document));
  document.addEventListener("DOMContentLoaded", () => startAutoplayVideos(document));
  if (document.body) {
    observer.observe(document.body, { childList: true, subtree: true });
    startAutoplayVideos(document);
  } else {
    window.addEventListener("load", () => {
      observer.observe(document.body, { childList: true, subtree: true });
      startAutoplayVideos(document);
    });
  }
})();
</script>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse MotionFix m2t_diff sample directories in a local Gradio UI."
    )
    parser.add_argument(
        "--sample-dir",
        default="",
        help="Path to a results/.../samples_<TIME> directory. Defaults to the newest one under results/motgpt if omitted.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Number of gallery cards per page (clamped to {MIN_PAGE_SIZE}-{MAX_PAGE_SIZE}).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for Gradio.")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share mode.",
    )
    return parser.parse_args()


def normalize_page_size(page_size: int | float | None) -> int:
    if page_size is None:
        return DEFAULT_PAGE_SIZE
    return max(MIN_PAGE_SIZE, min(MAX_PAGE_SIZE, int(page_size)))


def discover_latest_sample_dir(search_root: Path) -> str:
    if not search_root.exists():
        return ""

    sample_dirs = [path for path in search_root.rglob("samples_*") if path.is_dir()]
    if not sample_dirs:
        return ""

    newest = max(sample_dirs, key=lambda path: path.stat().st_mtime)
    return str(newest.resolve())


def collect_allowed_paths(sample_dir_value: str) -> list[str]:
    allowed_paths = {str(Path.cwd().resolve())}
    if not sample_dir_value:
        return sorted(allowed_paths)

    sample_dir = Path(sample_dir_value).expanduser().resolve()
    allowed_paths.add(str(sample_dir))

    try:
        browser_state = build_browser_state(str(sample_dir), DEFAULT_PAGE_SIZE)
    except ValueError:
        return sorted(allowed_paths)

    video_keys = ("source_video_path", "target_video_path", "preview_video_path")
    for sample_id in browser_state["sample_ids"]:
        record = browser_state["records"][sample_id]
        for key in video_keys:
            video_path_value = str(record.get(key, ""))
            if not video_path_value:
                continue

            video_path = Path(video_path_value)
            allowed_paths.add(str(video_path.parent))
            try:
                allowed_paths.add(str(video_path.resolve().parent))
            except OSError:
                pass

    return sorted(allowed_paths)


def file_url(path: str) -> str:
    return f"/gradio_api/file={quote(path, safe='/')}"


def favorites_sidecar(sample_dir: Path) -> Path:
    return sample_dir / FAVORITES_SIDECAR


def absolute_path(path: Path) -> str:
    return str(path.expanduser().absolute())


def sample_sort_key(sample_id: str) -> tuple[int, int | str, str]:
    if sample_id.isdigit():
        return (0, int(sample_id), sample_id)
    return (1, sample_id, sample_id)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_favorites(sample_dir: Path) -> list[str]:
    sidecar_path = favorites_sidecar(sample_dir)
    if not sidecar_path.exists():
        return []

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    favorites = payload.get("favorites", [])
    normalized: list[str] = []
    for sample_id in favorites:
        if isinstance(sample_id, (int, str)):
            normalized.append(str(sample_id))
    return normalized


def save_favorites(sample_dir: Path, favorites: list[str]) -> str:
    sidecar_path = favorites_sidecar(sample_dir)
    payload = {
        "version": 1,
        "sample_dir": str(sample_dir),
        "favorites": sorted(set(favorites), key=sample_sort_key),
    }
    sidecar_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(sidecar_path)


def build_sample_record(prediction_path: Path) -> dict[str, object]:
    sample_id = prediction_path.stem
    ground_truth_path = prediction_path.with_name(f"{sample_id}_gt.txt")
    source_video_path = prediction_path.with_name(f"{sample_id}_source.mp4")
    target_video_path = prediction_path.with_name(f"{sample_id}_target.mp4")
    preview_video_path = prediction_path.with_name(f"{sample_id}_source_target.mp4")

    missing_assets: list[str] = []
    if not ground_truth_path.exists():
        missing_assets.append("ground-truth text")
    if not source_video_path.exists():
        missing_assets.append("source video")
    if not target_video_path.exists():
        missing_assets.append("target video")
    if not preview_video_path.exists():
        missing_assets.append("side-by-side video")

    return {
        "id": sample_id,
        "prediction_path": absolute_path(prediction_path),
        "prediction_text": read_text(prediction_path),
        "ground_truth_path": absolute_path(ground_truth_path)
        if ground_truth_path.exists()
        else "",
        "ground_truth_text": read_text(ground_truth_path)
        if ground_truth_path.exists()
        else "",
        "source_video_path": absolute_path(source_video_path)
        if source_video_path.exists()
        else "",
        "target_video_path": absolute_path(target_video_path)
        if target_video_path.exists()
        else "",
        "preview_video_path": absolute_path(preview_video_path)
        if preview_video_path.exists()
        else "",
        "missing_assets": missing_assets,
    }


def empty_state(
    sample_dir: str = "",
    page_size: int = DEFAULT_PAGE_SIZE,
    bookmark_only: bool = False,
) -> dict[str, object]:
    normalized_page_size = normalize_page_size(page_size)
    return {
        "sample_dir": sample_dir,
        "page_size": normalized_page_size,
        "bookmark_only": bool(bookmark_only),
        "sample_ids": [],
        "records": {},
        "favorites": [],
        "favorites_path": str(favorites_sidecar(Path(sample_dir)))
        if sample_dir
        else "",
        "current_page": 1,
        "selected_id": "",
    }


def build_browser_state(
    sample_dir_value: str,
    page_size: int | float | None,
    bookmark_only: bool = False,
) -> dict[str, object]:
    if not sample_dir_value:
        raise ValueError("Sample directory is required.")

    sample_dir = Path(sample_dir_value).expanduser().resolve()
    if not sample_dir.exists() or not sample_dir.is_dir():
        raise ValueError(f"Sample directory does not exist: {sample_dir}")

    prediction_paths = [
        path
        for path in sample_dir.glob("*.txt")
        if path.is_file() and not path.name.endswith("_gt.txt")
    ]
    prediction_paths.sort(key=lambda path: sample_sort_key(path.stem))
    if not prediction_paths:
        raise ValueError(
            f"No prediction text files were found in {sample_dir}. Expected files like <id>.txt."
        )

    records = {path.stem: build_sample_record(path) for path in prediction_paths}
    sample_ids = sorted(records, key=sample_sort_key)
    favorites = [
        sample_id for sample_id in load_favorites(sample_dir) if sample_id in records
    ]

    return {
        "sample_dir": str(sample_dir),
        "page_size": normalize_page_size(page_size),
        "bookmark_only": bool(bookmark_only),
        "sample_ids": sample_ids,
        "records": records,
        "favorites": favorites,
        "favorites_path": str(favorites_sidecar(sample_dir)),
        "current_page": 1,
        "selected_id": sample_ids[0],
    }


def visible_sample_ids(state: dict[str, object]) -> list[str]:
    sample_ids = list(state.get("sample_ids", []))
    if not state.get("bookmark_only", False):
        return sample_ids

    favorites = set(str(sample_id) for sample_id in state.get("favorites", []))
    return [sample_id for sample_id in sample_ids if sample_id in favorites]


def page_count(state: dict[str, object]) -> int:
    sample_ids = visible_sample_ids(state)
    page_size = max(1, int(state.get("page_size", DEFAULT_PAGE_SIZE)))
    if not sample_ids:
        return 1
    return max(1, math.ceil(len(sample_ids) / page_size))


def page_sample_ids(
    state: dict[str, object], page_number: int | None = None
) -> list[str]:
    sample_ids = visible_sample_ids(state)
    if not sample_ids:
        return []

    page_size = int(state.get("page_size", DEFAULT_PAGE_SIZE))
    page_number = page_number or int(state.get("current_page", 1))
    start = max(0, (page_number - 1) * page_size)
    end = start + page_size
    return sample_ids[start:end]


def normalize_state(state: dict[str, object] | None) -> dict[str, object]:
    if not state:
        return empty_state()

    sample_ids = list(state.get("sample_ids", []))
    state["page_size"] = normalize_page_size(state.get("page_size", DEFAULT_PAGE_SIZE))
    state["bookmark_only"] = bool(state.get("bookmark_only", False))

    if not sample_ids:
        state["current_page"] = 1
        state["selected_id"] = ""
        state["favorites"] = []
        return state

    favorites = [
        sample_id for sample_id in state.get("favorites", []) if sample_id in sample_ids
    ]
    state["favorites"] = sorted(set(favorites), key=sample_sort_key)

    visible_ids = visible_sample_ids(state)
    if not visible_ids:
        state["current_page"] = 1
        state["selected_id"] = ""
        return state

    state["current_page"] = max(
        1, min(page_count(state), int(state.get("current_page", 1)))
    )
    selected_id = str(state.get("selected_id", ""))
    if selected_id not in visible_ids:
        state["selected_id"] = visible_ids[0]
    return state


def sample_index(state: dict[str, object], sample_id: str) -> int:
    return visible_sample_ids(state).index(sample_id)


def go_to_page(state: dict[str, object], delta: int) -> dict[str, object]:
    state = normalize_state(state)
    if not visible_sample_ids(state):
        return state

    current_page = int(state["current_page"])
    next_page = max(1, min(page_count(state), current_page + delta))
    state["current_page"] = next_page
    current_page_ids = page_sample_ids(state, next_page)
    if current_page_ids:
        state["selected_id"] = current_page_ids[0]
    return state


def select_sample(state: dict[str, object], sample_id: str) -> dict[str, object]:
    state = normalize_state(state)
    sample_id = str(sample_id).strip()
    visible_ids = visible_sample_ids(state)
    if sample_id not in visible_ids:
        return state

    state["selected_id"] = sample_id
    state["current_page"] = visible_ids.index(sample_id) // int(state["page_size"]) + 1
    return state


def go_to_sample(state: dict[str, object], delta: int) -> dict[str, object]:
    state = normalize_state(state)
    visible_ids = visible_sample_ids(state)
    if not visible_ids:
        return state

    selected_id = str(state["selected_id"])
    current_index = visible_ids.index(selected_id)
    next_index = max(0, min(len(visible_ids) - 1, current_index + delta))
    next_id = visible_ids[next_index]
    state["selected_id"] = next_id
    state["current_page"] = next_index // int(state["page_size"]) + 1
    return state


def toggle_favorite(state: dict[str, object]) -> tuple[dict[str, object], str]:
    state = normalize_state(state)
    selected_id = str(state.get("selected_id", ""))
    if not selected_id:
        return state, "No sample is selected."

    favorites = set(str(sample_id) for sample_id in state.get("favorites", []))
    if selected_id in favorites:
        favorites.remove(selected_id)
        status = f"Removed `{selected_id}` from favorites."
    else:
        favorites.add(selected_id)
        status = f"Saved `{selected_id}` to favorites."

    state["favorites"] = sorted(favorites, key=sample_sort_key)
    try:
        save_favorites(Path(str(state["sample_dir"])), list(state["favorites"]))
    except OSError as exc:
        return state, f"Favorite state changed in memory, but saving failed: `{exc}`"
    return state, status


def truncate_text(text: str, limit: int = 140) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "..."


def format_text_html(text: str, fallback: str) -> str:
    if not text:
        return f'<div class="sample-card-text missing">{html.escape(fallback)}</div>'
    snippet = html.escape(truncate_text(text))
    return f'<div class="sample-card-text">{snippet}</div>'


def render_detail_video_html(title: str, path: str, missing_label: str) -> str:
    if path:
        return (
            '<div class="detail-video-card">'
            f'<div class="detail-video-card-title">{html.escape(title)}</div>'
            f'<video class="autoplay-video" controls autoplay muted loop playsinline preload="metadata">'
            f'<source src="{file_url(path)}" type="video/mp4">'
            "</video>"
            "</div>"
        )

    return (
        '<div class="detail-video-card">'
        f'<div class="detail-video-card-title">{html.escape(title)}</div>'
        f'<div class="detail-video-missing">{html.escape(missing_label)}</div>'
        "</div>"
    )


def render_gallery_html(state: dict[str, object]) -> str:
    state = normalize_state(state)
    current_page_ids = page_sample_ids(state)
    if not current_page_ids:
        return '<div class="empty-gallery">Load a sample directory to browse exported MotionFix `m2t_diff` results.</div>'

    favorites = set(str(sample_id) for sample_id in state.get("favorites", []))
    selected_id = str(state.get("selected_id", ""))
    cards: list[str] = []

    for sample_id in current_page_ids:
        record = state["records"][sample_id]
        classes = ["sample-card"]
        if sample_id == selected_id:
            classes.append("selected")
        if sample_id in favorites:
            classes.append("favorite")

        preview_video_path = str(record["preview_video_path"])
        if preview_video_path:
            preview_html = (
                '<div class="sample-card-preview">'
                f'<video class="autoplay-video" autoplay muted loop playsinline preload="metadata">'
                f'<source src="{file_url(preview_video_path)}" type="video/mp4">'
                "</video></div>"
            )
        else:
            preview_html = (
                '<div class="sample-card-preview">'
                '<div class="sample-card-preview-missing">Side-by-side preview unavailable</div>'
                "</div>"
            )

        missing_assets = list(record["missing_assets"])
        badges = [f'<span class="sample-card-badge">{html.escape(sample_id)}</span>']
        for missing_asset in missing_assets:
            badges.append(
                f'<span class="sample-card-badge warn">Missing {html.escape(missing_asset)}</span>'
            )

        cards.append(
            """
            <div class="{classes}" data-sample-id="{sample_id}">
              <div class="sample-card-header">
                <span class="sample-card-id">{sample_id}</span>
                <span class="sample-card-star">{star}</span>
              </div>
              {preview_html}
              <div class="sample-card-body">
                <div class="sample-card-label">Generated</div>
                {prediction_html}
                <div class="sample-card-label">Ground Truth</div>
                {ground_truth_html}
                <div class="sample-card-badges">{badges}</div>
              </div>
            </div>
            """.format(
                classes=" ".join(classes),
                sample_id=html.escape(sample_id),
                star="★" if sample_id in favorites else "☆",
                preview_html=preview_html,
                prediction_html=format_text_html(
                    str(record["prediction_text"]), "Generated text is empty."
                ),
                ground_truth_html=format_text_html(
                    str(record["ground_truth_text"]),
                    "Ground-truth text is unavailable.",
                ),
                badges="".join(badges),
            )
        )

    return '<div class="gallery-grid">' + "".join(cards) + "</div>"


def build_summary_markdown(state: dict[str, object]) -> str:
    sample_ids = list(state.get("sample_ids", []))
    visible_ids = visible_sample_ids(state)
    if not sample_ids:
        return "No sample directory is loaded."

    records = state["records"]
    preview_count = sum(
        1 for sample_id in sample_ids if records[sample_id]["preview_video_path"]
    )
    source_count = sum(
        1 for sample_id in sample_ids if records[sample_id]["source_video_path"]
    )
    target_count = sum(
        1 for sample_id in sample_ids if records[sample_id]["target_video_path"]
    )
    ground_truth_count = sum(
        1 for sample_id in sample_ids if records[sample_id]["ground_truth_path"]
    )

    return (
        f"Loaded `{len(sample_ids)}` samples from `{state['sample_dir']}`. "
        f"Visible in current filter: `{len(visible_ids)}`. "
        f"Preview videos: `{preview_count}`, source videos: `{source_count}`, target videos: `{target_count}`, "
        f"ground-truth texts: `{ground_truth_count}`, favorites: `{len(state['favorites'])}`. "
        f"Favorites sidecar: `{state['favorites_path']}`"
    )


def build_page_selector_update(state: dict[str, object]) -> object:
    state = normalize_state(state)
    current_page_ids = page_sample_ids(state)
    if not current_page_ids:
        return gr.update(choices=[], value=None, interactive=False)

    favorites = set(str(sample_id) for sample_id in state.get("favorites", []))
    choices: list[tuple[str, str]] = []
    for sample_id in current_page_ids:
        record = state["records"][sample_id]
        prefix = "★" if sample_id in favorites else "☆"
        label = f"{prefix} {sample_id} · {truncate_text(str(record['prediction_text']), 60) or '[empty prediction]'}"
        choices.append((label, sample_id))

    selected_id = str(state.get("selected_id", ""))
    if selected_id not in current_page_ids:
        selected_id = current_page_ids[0]
    return gr.update(choices=choices, value=selected_id, interactive=True)


def build_page_markdown(state: dict[str, object]) -> str:
    sample_ids = visible_sample_ids(state)
    if not sample_ids:
        if state.get("bookmark_only", False):
            return "Page `0/0` with bookmarks-only filter enabled and no bookmarked samples yet"
        return "Page `0/0`"

    current_page = int(state["current_page"])
    current_page_ids = page_sample_ids(state)
    start = sample_index(state, current_page_ids[0]) + 1 if current_page_ids else 0
    end = sample_index(state, current_page_ids[-1]) + 1 if current_page_ids else 0
    scope = "bookmarked samples" if state.get("bookmark_only", False) else "samples"
    return f"Page `{current_page}/{page_count(state)}` showing {scope} `{start}-{end}` of `{len(sample_ids)}`"


def build_detail_outputs(
    state: dict[str, object],
) -> tuple[str, str, str, str, str, object]:
    sample_ids = list(state.get("sample_ids", []))
    if not sample_ids:
        return (
            "### No sample selected",
            "Load a sample directory to populate the detail panel.",
            '<div class="detail-video-grid"></div>',
            "",
            "",
            gr.update(value="Favorite ☆", interactive=False),
        )

    selected_id = str(state["selected_id"])
    record = state["records"][selected_id]
    selected_index = sample_index(state, selected_id) + 1
    is_favorite = selected_id in set(state.get("favorites", []))
    missing_assets = list(record["missing_assets"])

    detail_header = f"### Sample `{selected_id}` · `{selected_index}/{len(sample_ids)}` {'· Favorite' if is_favorite else ''}"
    detail_status = (
        "Missing assets: " + ", ".join(f"`{asset}`" for asset in missing_assets)
        if missing_assets
        else "All expected artifacts for this sample are present."
    )

    favorite_label = (
        "Remove Bookmark ★" if is_favorite else "Bookmark Selected Sample ☆"
    )
    ground_truth_text = (
        str(record["ground_truth_text"]) or "[ground-truth text unavailable]"
    )
    videos_html = (
        '<div class="detail-video-grid">'
        + render_detail_video_html(
            "Source",
            str(record["source_video_path"]),
            "Source video unavailable for this sample.",
        )
        + render_detail_video_html(
            "Target",
            str(record["target_video_path"]),
            "Target video unavailable for this sample.",
        )
        + render_detail_video_html(
            "Source / Target",
            str(record["preview_video_path"]),
            "Side-by-side preview unavailable for this sample.",
        )
        + "</div>"
    )

    return (
        detail_header,
        detail_status,
        videos_html,
        str(record["prediction_text"]),
        ground_truth_text,
        gr.update(value=favorite_label, interactive=True),
    )


def build_ui_outputs(
    state: dict[str, object], status_message: str
) -> tuple[object, ...]:
    state = normalize_state(state)
    detail_outputs = build_detail_outputs(state)
    return (
        state,
        status_message,
        build_summary_markdown(state),
        build_page_markdown(state),
        render_gallery_html(state),
        build_page_selector_update(state),
        *detail_outputs,
    )


def load_directory(
    sample_dir_value: str,
    page_size: int | float | None,
    bookmark_only: bool,
) -> tuple[object, ...]:
    try:
        state = build_browser_state(sample_dir_value.strip(), page_size, bookmark_only)
    except ValueError as exc:
        state = empty_state(
            sample_dir_value.strip(),
            normalize_page_size(page_size),
            bookmark_only,
        )
        return build_ui_outputs(state, f"{exc}")

    status_message = (
        f"Loaded `{len(state['sample_ids'])}` samples from `{state['sample_dir']}`."
    )
    return build_ui_outputs(state, status_message)


def move_page(state: dict[str, object], delta: int) -> tuple[object, ...]:
    next_state = go_to_page(state, delta)
    return build_ui_outputs(next_state, build_page_markdown(next_state))


def choose_sample(state: dict[str, object], selected_id: str) -> tuple[object, ...]:
    next_state = select_sample(state, selected_id)
    status_message = (
        f"Selected `{next_state['selected_id']}`."
        if next_state["selected_id"]
        else "No sample selected."
    )
    return build_ui_outputs(next_state, status_message)


def move_sample(state: dict[str, object], delta: int) -> tuple[object, ...]:
    next_state = go_to_sample(state, delta)
    status_message = (
        f"Selected `{next_state['selected_id']}`."
        if next_state["selected_id"]
        else "No sample selected."
    )
    return build_ui_outputs(next_state, status_message)


def toggle_favorite_and_refresh(state: dict[str, object]) -> tuple[object, ...]:
    next_state, status_message = toggle_favorite(state)
    return build_ui_outputs(next_state, status_message)


def build_demo(initial_sample_dir: str, initial_page_size: int) -> gr.Blocks:
    initial_state = empty_state(initial_sample_dir, initial_page_size, False)
    initial_outputs = build_ui_outputs(
        initial_state, "Load a sample directory to start browsing."
    )

    with gr.Blocks(
        css=CUSTOM_CSS,
        head=APP_HEAD,
        title="MotionFix m2t_diff Result Browser",
    ) as demo:
        browser_state = gr.State(initial_outputs[0])

        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown("# MotionFix `m2t_diff` Result Browser")
            gr.Markdown(
                "Browse exported `results/.../samples_<TIME>/` folders without loading checkpoints or rerunning inference. Use the sample selector under the gallery to choose the current sample, then use the favorite button to bookmark it."
            )
            gr.HTML(HOTKEYS_HTML)

            with gr.Row():
                sample_dir_input = gr.Textbox(
                    value=initial_sample_dir,
                    label="Sample directory",
                    placeholder="results/motgpt/.../samples_<TIME>",
                    scale=6,
                )
                page_size_slider = gr.Slider(
                    minimum=MIN_PAGE_SIZE,
                    maximum=MAX_PAGE_SIZE,
                    step=1,
                    value=normalize_page_size(initial_page_size),
                    label="Cards per page",
                    scale=2,
                )
                load_button = gr.Button("Load directory", variant="primary", scale=1)

            bookmark_only_checkbox = gr.Checkbox(
                label="Bookmarks only",
                value=bool(initial_state.get("bookmark_only", False)),
            )

            status_markdown = gr.Markdown(value=initial_outputs[1])
            summary_markdown = gr.Markdown(value=initial_outputs[2])

            with gr.Row():
                prev_page_button = gr.Button("Prev page", elem_id="prev-page-btn")
                page_markdown = gr.Markdown(value=initial_outputs[3])
                next_page_button = gr.Button("Next page", elem_id="next-page-btn")

            gallery_html = gr.HTML(value=initial_outputs[4], elem_id="sample-gallery")

            page_sample_selector = gr.Radio(
                label="Selected sample on current page",
                choices=[],
                value=None,
                interactive=False,
            )

            with gr.Row():
                prev_sample_button = gr.Button("Prev sample", elem_id="prev-sample-btn")
                favorite_toggle_button = gr.Button(
                    value=initial_outputs[11].get(
                        "value", "Bookmark Selected Sample ☆"
                    ),
                    elem_id="favorite-toggle-btn",
                    interactive=initial_outputs[11].get("interactive", False),
                )
                next_sample_button = gr.Button("Next sample", elem_id="next-sample-btn")

            detail_header = gr.Markdown(value=initial_outputs[6])
            detail_status = gr.Markdown(value=initial_outputs[7])
            detail_videos_html = gr.HTML(value=initial_outputs[8])

            with gr.Row():
                prediction_text = gr.Textbox(
                    label="Generated text",
                    lines=5,
                    interactive=False,
                    value=initial_outputs[9],
                )
                ground_truth_text = gr.Textbox(
                    label="Ground-truth text",
                    lines=5,
                    interactive=False,
                    value=initial_outputs[10],
                )

        common_outputs = [
            browser_state,
            status_markdown,
            summary_markdown,
            page_markdown,
            gallery_html,
            page_sample_selector,
            detail_header,
            detail_status,
            detail_videos_html,
            prediction_text,
            ground_truth_text,
            favorite_toggle_button,
        ]

        load_inputs = [sample_dir_input, page_size_slider, bookmark_only_checkbox]
        load_button.click(
            load_directory, inputs=load_inputs, outputs=common_outputs, queue=False
        )
        sample_dir_input.submit(
            load_directory, inputs=load_inputs, outputs=common_outputs, queue=False
        )
        page_size_slider.release(
            load_directory, inputs=load_inputs, outputs=common_outputs, queue=False
        )
        bookmark_only_checkbox.change(
            load_directory, inputs=load_inputs, outputs=common_outputs, queue=False
        )
        page_sample_selector.change(
            choose_sample,
            inputs=[browser_state, page_sample_selector],
            outputs=common_outputs,
            queue=False,
        )

        prev_page_button.click(
            lambda state: move_page(state, -1),
            inputs=[browser_state],
            outputs=common_outputs,
            queue=False,
        )
        next_page_button.click(
            lambda state: move_page(state, 1),
            inputs=[browser_state],
            outputs=common_outputs,
            queue=False,
        )
        prev_sample_button.click(
            lambda state: move_sample(state, -1),
            inputs=[browser_state],
            outputs=common_outputs,
            queue=False,
        )
        next_sample_button.click(
            lambda state: move_sample(state, 1),
            inputs=[browser_state],
            outputs=common_outputs,
            queue=False,
        )
        favorite_toggle_button.click(
            toggle_favorite_and_refresh,
            inputs=[browser_state],
            outputs=common_outputs,
            queue=False,
        )

        if initial_sample_dir:
            demo.load(
                load_directory,
                inputs=[sample_dir_input, page_size_slider, bookmark_only_checkbox],
                outputs=common_outputs,
                queue=False,
            )

    return demo


def main() -> None:
    args = parse_args()
    initial_sample_dir = args.sample_dir or discover_latest_sample_dir(RESULTS_ROOT)
    demo = build_demo(initial_sample_dir, args.page_size)

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=collect_allowed_paths(initial_sample_dir),
    )


if __name__ == "__main__":
    main()
