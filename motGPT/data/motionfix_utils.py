import json
from pathlib import Path


MOTIONFIX_ARCHIVE = "motionfix.pth.tar"
MOTIONFIX_SPLITS = "splits.json"
DEFAULT_CACHE_DIR = "mgpt_humanml_cache"


def validate_motionfix_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MotionFix root does not exist: {root}")

    archive = root / MOTIONFIX_ARCHIVE
    splits = root / MOTIONFIX_SPLITS
    missing = [str(path.name) for path in [archive, splits] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"MotionFix dataset root is incomplete: missing {', '.join(missing)} in {root}"
        )
    return root


def load_motionfix_splits(root: str | Path) -> dict:
    root = validate_motionfix_root(root)
    with open(root / MOTIONFIX_SPLITS, "r", encoding="utf-8") as handle:
        splits = json.load(handle)
    required = ["train", "val", "test"]
    missing = [name for name in required if name not in splits]
    if missing:
        raise ValueError(
            f"MotionFix splits.json is missing required keys: {', '.join(missing)}"
        )
    return splits


def load_motionfix_splits_file(split_path: str | Path) -> dict:
    split_path = Path(split_path).expanduser().resolve()
    if not split_path.exists():
        raise FileNotFoundError(f"MotionFix split file does not exist: {split_path}")

    with open(split_path, "r", encoding="utf-8") as handle:
        splits = json.load(handle)

    required = ["train", "val", "test"]
    missing = [name for name in required if name not in splits]
    if missing:
        raise ValueError(
            f"MotionFix split file is missing required keys: {', '.join(missing)}"
        )
    return splits


def validate_motionfix_split_ids(
    splits: dict, dataset_ids, require_full_coverage: bool = True
) -> None:
    dataset_ids = set(dataset_ids)
    split_ids = set()
    duplicates = []

    for split_name, ids in splits.items():
        current_ids = set(ids)
        if len(current_ids) != len(ids):
            seen = set()
            dup_ids = []
            for sample_id in ids:
                if sample_id in seen:
                    dup_ids.append(sample_id)
                else:
                    seen.add(sample_id)
            duplicates.extend(
                f"{split_name}:{sample_id}" for sample_id in sorted(set(dup_ids))
            )
        split_ids.update(current_ids)

    unknown_ids = sorted(split_ids - dataset_ids)
    missing_ids = sorted(dataset_ids - split_ids)

    problems = []
    if duplicates:
        problems.append("duplicate sample ids in splits: " + ", ".join(duplicates[:10]))
    if unknown_ids:
        problems.append(
            "sample ids in splits but not archive: " + ", ".join(unknown_ids[:10])
        )
    if require_full_coverage and missing_ids:
        problems.append(
            "sample ids in archive but not splits: " + ", ".join(missing_ids[:10])
        )

    if problems:
        raise ValueError("MotionFix split validation failed: " + "; ".join(problems))


def get_motionfix_cache_root(
    root: str | Path, cache_root: str | Path | None = None
) -> Path:
    root = Path(root).expanduser().resolve()
    if cache_root is None:
        return root / DEFAULT_CACHE_DIR
    return Path(cache_root).expanduser().resolve()
