import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motGPT.data.humanml.scripts.motion_process import process_humanml_joints
from motGPT.data.motionfix_utils import (
    MOTIONFIX_ARCHIVE,
    get_motionfix_cache_root,
    load_motionfix_splits,
    load_motionfix_splits_file,
    validate_motionfix_split_ids,
    validate_motionfix_root,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MotionFix paired samples into HumanML-compatible MotionGPT3 features"
    )
    parser.add_argument(
        "--motionfix-root",
        required=True,
        help="Directory containing motionfix.pth.tar and splits.json",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Output cache directory (defaults to <motionfix-root>/mgpt_humanml_cache)",
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help="Optional split JSON file to use instead of <motionfix-root>/splits.json",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="Original MotionFix frame rate",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=20.0,
        help="Target frame rate for MotionGPT3-compatible features",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples to preprocess",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cached features even if files already exist",
    )
    return parser.parse_args()


def resample_joints(
    joints: np.ndarray, source_fps: float, target_fps: float
) -> np.ndarray:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if np.isclose(source_fps, target_fps):
        return joints.astype(np.float32, copy=False)

    step = source_fps / target_fps
    indices = np.round(np.arange(0, len(joints), step)).astype(int)
    indices = np.clip(indices, 0, len(joints) - 1)
    indices = np.unique(indices)
    if indices[-1] != len(joints) - 1:
        indices = np.append(indices, len(joints) - 1)
    return joints[indices].astype(np.float32, copy=False)


def motionfix_zup_to_humanml_yup(joints: np.ndarray) -> np.ndarray:
    """Rotate MotionFix z-up joints into the y-up convention used by HumanML."""
    joints = np.asarray(joints, dtype=np.float32)
    remapped = np.empty_like(joints)
    remapped[..., 0] = joints[..., 0]
    remapped[..., 1] = joints[..., 2]
    remapped[..., 2] = -joints[..., 1]
    return remapped


def ensure_layout(cache_root: Path):
    for subdir in ["source", "target", "manifests"]:
        (cache_root / subdir).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    root = validate_motionfix_root(args.motionfix_root)
    split_source = root / "splits.json"
    if args.splits_file is None:
        splits = load_motionfix_splits(root)
    else:
        split_source = Path(args.splits_file).expanduser().resolve()
        splits = load_motionfix_splits_file(split_source)
    cache_root = get_motionfix_cache_root(root, args.cache_root)
    ensure_layout(cache_root)

    dataset = joblib.load(root / MOTIONFIX_ARCHIVE)
    validate_motionfix_split_ids(
        splits,
        dataset.keys(),
        require_full_coverage=args.splits_file is None,
    )
    sample_ids = sorted(set().union(*[set(ids) for ids in splits.values()]))
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    split_membership = {split_name: set(ids) for split_name, ids in splits.items()}
    manifests = {split_name: [] for split_name in splits}
    processed = 0
    processed_ids = []

    for sample_id in sample_ids:
        item = dataset[sample_id]
        source_joints = item["motion_source"]["joint_positions"]
        target_joints = item["motion_target"]["joint_positions"]
        source_joints = (
            source_joints.detach().cpu().numpy()
            if hasattr(source_joints, "detach")
            else np.asarray(source_joints)
        )
        target_joints = (
            target_joints.detach().cpu().numpy()
            if hasattr(target_joints, "detach")
            else np.asarray(target_joints)
        )

        source_joints = motionfix_zup_to_humanml_yup(source_joints)
        target_joints = motionfix_zup_to_humanml_yup(target_joints)

        source_joints = resample_joints(source_joints, args.source_fps, args.target_fps)
        target_joints = resample_joints(target_joints, args.source_fps, args.target_fps)

        source_path = cache_root / "source" / f"{sample_id}.npy"
        target_path = cache_root / "target" / f"{sample_id}.npy"

        if args.overwrite or not source_path.exists() or not target_path.exists():
            source_feats = process_humanml_joints(source_joints)
            target_feats = process_humanml_joints(target_joints)
            np.save(source_path, source_feats.astype(np.float32))
            np.save(target_path, target_feats.astype(np.float32))
        else:
            source_feats = np.load(source_path)
            target_feats = np.load(target_path)

        record = {
            "id": sample_id,
            "text": item["text"],
            "all_captions": [item["text"]],
            "source_path": str(source_path.relative_to(cache_root)),
            "target_path": str(target_path.relative_to(cache_root)),
            "length_source": int(source_feats.shape[0]),
            "length_target": int(target_feats.shape[0]),
            "source_fps": float(args.source_fps),
            "target_fps": float(args.target_fps),
        }
        for split_name, ids in split_membership.items():
            if sample_id in ids:
                split_record = dict(record)
                split_record["split"] = split_name
                manifests[split_name].append(split_record)
        processed += 1
        processed_ids.append(sample_id)

    for split_name, records in manifests.items():
        write_json(cache_root / "manifests" / f"{split_name}.json", records)

    summary = {
        "motionfix_root": str(root),
        "cache_root": str(cache_root),
        "processed": processed,
        "split_file": str(split_source),
        "source_fps": float(args.source_fps),
        "target_fps": float(args.target_fps),
        "manifests": {name: len(records) for name, records in manifests.items()},
        "normalization": {
            "mean": "assets/meta/mean.npy",
            "std": "assets/meta/std.npy",
            "mean_eval": "assets/meta/mean_eval.npy",
            "std_eval": "assets/meta/std_eval.npy",
        },
        "coordinate_transform": "motionfix_z_up_to_humanml_y_up",
    }
    write_json(cache_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
