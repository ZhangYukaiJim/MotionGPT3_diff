import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as RRR

from motGPT.config import get_module_config
from motGPT.data.MotionFix import MotionFixDataModule


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Prototype MotionFix m2t_diff visualization with constrained DTW "
            "alignment diagnostics and a source-focused vertex heat overlay."
        )
    )
    parser.add_argument(
        "--cfg_assets",
        type=str,
        default="./configs/assets.yaml",
        help="Asset config used to resolve the config folder and MotionFix paths.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="./configs/MoT_vae_stage4_motionfix_yzh.yaml",
        help="Experiment config used to instantiate the MotionFix datamodule.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="MotionFix sample id to resolve from the cache manifests.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Manifest split used when resolving --sample-id.",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help=(
            "Existing results/.../samples_<TIME> directory. If provided with --sample-id, "
            "artifacts are written under <sample-dir>/<sample-id>_heat_overlay/."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Direct source feature .npy path. Overrides sample-id resolution.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Direct target feature .npy path. Overrides sample-id resolution.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for prototype artifacts.",
    )
    parser.add_argument(
        "--dtw-band-ratio",
        type=float,
        default=0.15,
        help="Half-band width for constrained DTW as a fraction of the longer sequence.",
    )
    parser.add_argument(
        "--hv-penalty",
        type=float,
        default=0.05,
        help="Extra cost added to horizontal and vertical DTW steps.",
    )
    parser.add_argument(
        "--velocity-weight",
        type=float,
        default=0.35,
        help="Weight applied to the root-centered joint velocity features.",
    )
    parser.add_argument(
        "--heat-percentile",
        type=float,
        default=95.0,
        help="Upper percentile used to clip per-vertex heat before color mapping.",
    )
    parser.add_argument(
        "--heat-smooth-window",
        type=int,
        default=5,
        help="Odd temporal smoothing window for per-vertex heat values.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS for the preview and heat-overlay videos.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(768, 768),
        metavar=("WIDTH", "HEIGHT"),
        help="Output video resolution for the heat-overlay render.",
    )
    parser.add_argument(
        "--max-preview-frames",
        type=int,
        default=120,
        help="Maximum number of matched DTW frames kept for exported preview/overlay videos.",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        default=None,
        help="Optional override for the SMPL model directory used by the mesh renderer.",
    )
    return parser


def load_cfg(cfg_path: str, cfg_assets_path: str):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_assets = OmegaConf.load(cfg_assets_path)
    cfg_base = OmegaConf.load(Path(cfg_assets.CONFIG_FOLDER) / "default.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    return OmegaConf.merge(cfg_exp, cfg_assets)


def load_datamodule(cfg):
    datamodule = MotionFixDataModule(cfg=cfg, phase="test")
    datamodule.setup("test")
    return datamodule


def get_manifest_path(datamodule: MotionFixDataModule, split: str) -> Path:
    manifest_map = {
        "train": datamodule._train_manifest,
        "val": datamodule._val_manifest,
        "test": datamodule._test_manifest,
    }
    return Path(manifest_map[split])


def resolve_pair_from_manifest(datamodule, sample_id: str, split: str):
    manifest_path = get_manifest_path(datamodule, split)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    match = None
    for record in records:
        if str(record["id"]) == str(sample_id):
            match = record
            break
    if match is None:
        raise ValueError(f"Sample id {sample_id!r} not found in {manifest_path}")
    cache_root = manifest_path.parent.parent
    source_path = cache_root / match["source_path"]
    target_path = cache_root / match["target_path"]
    return {
        "sample_id": str(match["id"]),
        "source_path": source_path,
        "target_path": target_path,
        "source_feat": np.load(source_path).astype(np.float32),
        "target_feat": np.load(target_path).astype(np.float32),
        "source_length": int(match["length_source"]),
        "target_length": int(match["length_target"]),
        "text": match.get("text", ""),
        "all_captions": match.get("all_captions") or [match.get("text", "")],
    }


def resolve_pair_from_paths(source_path: str, target_path: str):
    src_path = Path(source_path)
    tgt_path = Path(target_path)
    return {
        "sample_id": src_path.stem,
        "source_path": src_path,
        "target_path": tgt_path,
        "source_feat": np.load(src_path).astype(np.float32),
        "target_feat": np.load(tgt_path).astype(np.float32),
        "source_length": int(np.load(src_path).shape[0]),
        "target_length": int(np.load(tgt_path).shape[0]),
        "text": "",
        "all_captions": [],
    }


def determine_output_dir(args, sample_id: str):
    if args.out_dir:
        return Path(args.out_dir).expanduser().resolve()
    if args.sample_dir:
        return (
            Path(args.sample_dir).expanduser().resolve() / f"{sample_id}_heat_overlay"
        )
    return Path("results") / "prototype_m2t_diff_heat_overlay" / sample_id


def to_tensor(features: np.ndarray):
    return torch.from_numpy(features).float()


def recover_joints(datamodule, source_feat: np.ndarray, target_feat: np.ndarray):
    source_tensor = datamodule.normalize(to_tensor(source_feat))
    target_tensor = datamodule.normalize(to_tensor(target_feat))
    source_joints = datamodule.feats2joints(source_tensor).detach().cpu().numpy()
    target_joints = datamodule.feats2joints(target_tensor).detach().cpu().numpy()
    return source_joints, target_joints


def root_center_joints(joints: np.ndarray):
    return joints - joints[:, :1, :]


def compute_alignment_features(joints: np.ndarray, velocity_weight: float):
    centered = root_center_joints(joints)
    velocity = np.zeros_like(centered)
    velocity[1:] = centered[1:] - centered[:-1]
    pose_features = centered.reshape(centered.shape[0], -1)
    velocity_features = velocity.reshape(velocity.shape[0], -1)
    return np.concatenate([pose_features, velocity_weight * velocity_features], axis=1)


def compute_local_cost_matrix(src_features: np.ndarray, tgt_features: np.ndarray):
    src_sq = np.sum(src_features * src_features, axis=1, keepdims=True)
    tgt_sq = np.sum(tgt_features * tgt_features, axis=1, keepdims=True).T
    sq = np.maximum(src_sq + tgt_sq - 2.0 * (src_features @ tgt_features.T), 0.0)
    return np.sqrt(sq).astype(np.float32)


def constrained_dtw(cost_matrix: np.ndarray, band_ratio: float, hv_penalty: float):
    src_len, tgt_len = cost_matrix.shape
    inf = np.float32(np.inf)
    accumulated = np.full((src_len, tgt_len), inf, dtype=np.float32)
    backpointer = np.full((src_len, tgt_len, 2), -1, dtype=np.int32)
    band = max(1, int(np.ceil(max(src_len, tgt_len) * band_ratio)))

    accumulated[0, 0] = cost_matrix[0, 0]
    for i in range(src_len):
        scaled_center = int(round(i * (tgt_len - 1) / max(src_len - 1, 1)))
        j_start = max(0, scaled_center - band)
        j_end = min(tgt_len, scaled_center + band + 1)
        for j in range(j_start, j_end):
            if i == 0 and j == 0:
                continue
            best_cost = inf
            best_prev = (-1, -1)
            if i > 0 and accumulated[i - 1, j] < inf:
                candidate = accumulated[i - 1, j] + hv_penalty
                if candidate < best_cost:
                    best_cost = candidate
                    best_prev = (i - 1, j)
            if j > 0 and accumulated[i, j - 1] < inf:
                candidate = accumulated[i, j - 1] + hv_penalty
                if candidate < best_cost:
                    best_cost = candidate
                    best_prev = (i, j - 1)
            if i > 0 and j > 0 and accumulated[i - 1, j - 1] < inf:
                candidate = accumulated[i - 1, j - 1]
                if candidate < best_cost:
                    best_cost = candidate
                    best_prev = (i - 1, j - 1)
            if best_prev[0] >= 0:
                accumulated[i, j] = cost_matrix[i, j] + best_cost
                backpointer[i, j] = best_prev

    if not np.isfinite(accumulated[-1, -1]):
        raise RuntimeError("Constrained DTW failed to find an endpoint-aligned path")

    path = []
    i, j = src_len - 1, tgt_len - 1
    while i >= 0 and j >= 0:
        path.append((int(i), int(j)))
        if i == 0 and j == 0:
            break
        i, j = backpointer[i, j]
        if i < 0 or j < 0:
            raise RuntimeError("DTW backtracking encountered an invalid predecessor")
    path.reverse()
    return np.asarray(path, dtype=np.int32), accumulated


def sample_path(path: np.ndarray, max_frames: int):
    if len(path) <= max_frames:
        return path
    indices = np.linspace(0, len(path) - 1, num=max_frames, dtype=np.int32)
    return path[indices]


def rolling_mean(values: np.ndarray, window: int):
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(values)
    for dim in range(values.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def scalar_signals(joints: np.ndarray):
    centered = root_center_joints(joints)
    speed = np.zeros((joints.shape[0],), dtype=np.float32)
    speed[1:] = np.linalg.norm(centered[1:] - centered[:-1], axis=2).mean(axis=1)
    return {
        "root_height": joints[:, 0, 1],
        "left_wrist_height": joints[:, 20, 1],
        "right_wrist_height": joints[:, 21, 1],
        "mean_joint_speed": speed,
    }


def plot_cost_matrix(cost_matrix: np.ndarray, path: np.ndarray, output_path: Path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cost_matrix.T, origin="lower", aspect="auto", cmap="viridis")
    plt.plot(path[:, 0], path[:, 1], color="white", linewidth=1.5)
    plt.xlabel("Source frame")
    plt.ylabel("Target frame")
    plt.title("Constrained DTW local cost and warping path")
    plt.colorbar(label="Local cost")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_alignment_path(path: np.ndarray, output_path: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(path[:, 0], path[:, 1], linewidth=1.5)
    plt.xlabel("Source frame index")
    plt.ylabel("Matched target frame index")
    plt.title("DTW frame-index mapping")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_scalar_traces(
    source_joints: np.ndarray,
    target_joints: np.ndarray,
    path: np.ndarray,
    output_path: Path,
):
    src_signals = scalar_signals(source_joints)
    tgt_signals = scalar_signals(target_joints)
    src_idx = path[:, 0]
    tgt_idx = path[:, 1]
    fig, axes = plt.subplots(len(src_signals), 1, figsize=(10, 9), sharex=True)
    for axis, key in zip(np.atleast_1d(axes), src_signals.keys()):
        axis.plot(src_signals[key], label="source (raw)", alpha=0.35)
        axis.plot(tgt_signals[key], label="target (raw)", alpha=0.35)
        axis.plot(
            np.arange(len(path)), src_signals[key][src_idx], label="source (aligned)"
        )
        axis.plot(
            np.arange(len(path)), tgt_signals[key][tgt_idx], label="target (aligned)"
        )
        axis.set_ylabel(key)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Aligned path step")
    fig.suptitle("Scalar traces before and after DTW alignment")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def render_skeleton_frame(joints: np.ndarray, title: str = ""):
    mins = joints.min(axis=0)
    maxs = joints.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float(np.max(maxs - mins))
    radius = max(extent * 0.75, 0.8)
    chains = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]
    colors = ["#4c78a8", "#f58518", "#bab0ab", "#e45756", "#72b7b2"]
    fig = plt.figure(figsize=(4.5, 4.5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    for chain, color in zip(chains, colors):
        ax.plot(
            joints[chain, 0],
            joints[chain, 2],
            joints[chain, 1],
            color=color,
            linewidth=3,
        )
    ax.view_init(elev=18, azim=-90)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(max(0.0, mins[1] - 0.05), maxs[1] + 0.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return frame


def write_video(frames, output_path: Path, fps: int):
    if not frames:
        raise ValueError(f"No frames to write for {output_path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def render_alignment_preview(
    source_joints: np.ndarray,
    target_joints: np.ndarray,
    sampled_path: np.ndarray,
    output_path: Path,
    fps: int,
):
    frames = []
    for src_idx, tgt_idx in sampled_path:
        src_frame = render_skeleton_frame(
            source_joints[src_idx], title=f"source {src_idx}"
        )
        tgt_frame = render_skeleton_frame(
            target_joints[tgt_idx], title=f"target {tgt_idx}"
        )
        separator = np.full((src_frame.shape[0], 10, 3), 255, dtype=np.uint8)
        frames.append(np.concatenate([src_frame, separator, tgt_frame], axis=1))
    write_video(frames, output_path, fps=fps)


def joints_to_smpl_params(joints: np.ndarray):
    from motGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat

    data = joints.astype(np.float32).copy()
    data = data - data[0, 0]
    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(data)
    pose = np.concatenate(
        [pose, np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)],
        axis=1,
    )
    # Match the existing slow SMPL render orientation used elsewhere in the repo.
    root_flip = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0], dtype=np.float32))
    pose[:, 0] = np.matmul(root_flip.as_matrix().reshape(1, 3, 3), pose[:, 0])
    root = data[:, 0].copy()
    root[:, 1] = -root[:, 1]
    root[:, 2] = -root[:, 2]
    return {
        "pred_shape": np.zeros((1, 10), dtype=np.float32),
        "pred_root": root.astype(np.float32),
        "pred_pose": pose.astype(np.float32),
    }


def build_vertices_and_camera(render, joints: np.ndarray):
    params = joints_to_smpl_params(joints)
    render.init_renderer([768, 768, 3], params)
    camera = render.pred_camera_t
    if isinstance(camera, torch.Tensor):
        camera = camera.detach().cpu().numpy()
    return (
        render.vertices.copy(),
        np.asarray(camera).copy(),
        render.smpl.faces.copy(),
    )


def smooth_heat(heat_values: np.ndarray, window: int):
    if window <= 1:
        return heat_values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(heat_values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(heat_values)
    for vertex_idx in range(heat_values.shape[1]):
        smoothed[:, vertex_idx] = np.convolve(
            padded[:, vertex_idx], kernel, mode="valid"
        )
    return smoothed


def compute_vertex_heat(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    heat_percentile: float,
    smooth_window: int,
):
    heat_values = np.linalg.norm(source_vertices - target_vertices, axis=2)
    heat_values = smooth_heat(heat_values, smooth_window)
    clip_value = np.percentile(heat_values, heat_percentile)
    clip_value = max(float(clip_value), 1e-6)
    normalized = np.clip(heat_values / clip_value, 0.0, 1.0)
    return heat_values, normalized, clip_value


def render_overlay_frame(
    renderer, source_vertices, target_vertices, camera_translation, face_colors
):
    import pyrender
    import trimesh

    floor_render = pyrender.Mesh.from_trimesh(renderer.floor, smooth=False)
    scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 0.8), ambient_light=(0.4, 0.4, 0.4))
    scene.add(floor_render, pose=renderer.floor_pose)

    source_mesh = trimesh.Trimesh(source_vertices, renderer.faces, process=False)
    source_mesh.visual.face_colors = face_colors
    source_mesh.apply_transform(renderer.rot)
    scene.add(pyrender.Mesh.from_trimesh(source_mesh, smooth=False), "source_mesh")

    ghost_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.05,
        alphaMode="BLEND",
        baseColorFactor=(0.22, 0.22, 0.26, 0.18),
    )
    target_mesh = trimesh.Trimesh(target_vertices, renderer.faces, process=False)
    target_mesh.apply_transform(renderer.rot)
    scene.add(
        pyrender.Mesh.from_trimesh(target_mesh, material=ghost_material, smooth=True),
        "target_mesh",
    )

    camera = pyrender.PerspectiveCamera(yfov=(np.pi / 4.0))
    cam_pose = np.array(renderer.camera_pose)
    cam_pose[:3, 3] += camera_translation
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=350)
    for position in ([0, -1, 1], [0, 1, 1], [1, 1, 2]):
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array(position)
        scene.add(light, pose=light_pose)

    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    color, _ = renderer.renderer.render(scene, flags=flags)
    return color[:, :, :3]


def save_heat_contact_sheet(frames, output_path: Path):
    if not frames:
        return
    sample_count = min(6, len(frames))
    indices = np.linspace(0, len(frames) - 1, sample_count, dtype=np.int32)
    chosen = [frames[idx] for idx in indices]
    rows = []
    for start in range(0, len(chosen), 3):
        row_frames = chosen[start : start + 3]
        if len(row_frames) < 3:
            row_frames.extend([np.full_like(chosen[0], 255)] * (3 - len(row_frames)))
        rows.append(np.concatenate(row_frames, axis=1))
    image = np.concatenate(rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def build_face_colors(source_vertices: np.ndarray, normalized_heat: np.ndarray):
    cmap = cm.get_cmap("inferno")
    vertex_rgba = cmap(normalized_heat)
    vertex_rgba[:, 3] = 1.0
    vertex_rgba = (vertex_rgba * 255).astype(np.uint8)
    return vertex_rgba[source_vertices].mean(axis=1).astype(np.uint8)


def write_metadata(output_dir: Path, metadata: dict):
    (output_dir / "prototype_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    args = build_parser().parse_args()
    if (args.source is None) != (args.target is None):
        raise ValueError("--source and --target must be provided together")
    if args.sample_id is None and args.source is None:
        raise ValueError("Provide either --sample-id or both --source and --target")

    cfg = load_cfg(args.cfg, args.cfg_assets)
    datamodule = load_datamodule(cfg)

    if args.source and args.target:
        pair = resolve_pair_from_paths(args.source, args.target)
    else:
        pair = resolve_pair_from_manifest(datamodule, args.sample_id, args.split)

    sample_id = pair["sample_id"]
    output_dir = determine_output_dir(args, sample_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_feat = pair["source_feat"][: pair["source_length"]]
    target_feat = pair["target_feat"][: pair["target_length"]]
    source_joints, target_joints = recover_joints(datamodule, source_feat, target_feat)

    source_features = compute_alignment_features(source_joints, args.velocity_weight)
    target_features = compute_alignment_features(target_joints, args.velocity_weight)
    cost_matrix = compute_local_cost_matrix(source_features, target_features)
    path, accumulated = constrained_dtw(
        cost_matrix, band_ratio=args.dtw_band_ratio, hv_penalty=args.hv_penalty
    )
    sampled_path = sample_path(path, args.max_preview_frames)

    plot_cost_matrix(cost_matrix, path, output_dir / "alignment_cost_matrix.png")
    plot_alignment_path(path, output_dir / "alignment_path_plot.png")
    plot_scalar_traces(
        source_joints, target_joints, path, output_dir / "alignment_scalar_traces.png"
    )
    render_alignment_preview(
        root_center_joints(source_joints),
        root_center_joints(target_joints),
        sampled_path,
        output_dir / "alignment_preview.mp4",
        fps=args.fps,
    )

    np.savez(
        output_dir / "alignment_pairs.npz",
        path=path,
        sampled_path=sampled_path,
        accumulated_cost=accumulated[-1, -1],
    )

    source_aligned = root_center_joints(source_joints)[sampled_path[:, 0]]
    target_aligned = root_center_joints(target_joints)[sampled_path[:, 1]]

    smpl_model_path = args.smpl_model_path or cfg.RENDER.SMPL_MODEL_PATH
    from motGPT.render.pyrender.smpl_render import SMPLRender, Renderer

    render = SMPLRender(smpl_model_path)
    source_vertices, source_camera, faces = build_vertices_and_camera(
        render, source_aligned
    )
    target_vertices, _, _ = build_vertices_and_camera(render, target_aligned)
    overlay_renderer = Renderer(
        vertices=np.concatenate([source_vertices, target_vertices], axis=0),
        focal_length=render.focal_length,
        img_res=(args.resolution[0], args.resolution[1]),
        faces=faces,
    )

    raw_heat, normalized_heat, clip_value = compute_vertex_heat(
        source_vertices,
        target_vertices,
        heat_percentile=args.heat_percentile,
        smooth_window=args.heat_smooth_window,
    )

    overlay_frames = []
    for frame_idx in range(len(sampled_path)):
        face_colors = build_face_colors(faces, normalized_heat[frame_idx])
        overlay_frames.append(
            render_overlay_frame(
                overlay_renderer,
                source_vertices[frame_idx],
                target_vertices[frame_idx],
                source_camera[frame_idx],
                face_colors,
            )
        )

    write_video(overlay_frames, output_dir / "vertex_heat_overlay.mp4", fps=args.fps)
    save_heat_contact_sheet(
        overlay_frames, output_dir / "vertex_heat_contact_sheet.png"
    )
    np.savez(
        output_dir / "vertex_heat_values.npz",
        raw_heat=raw_heat,
        normalized_heat=normalized_heat,
        sampled_path=sampled_path,
    )

    metadata = {
        "sample_id": sample_id,
        "source_path": str(pair["source_path"]),
        "target_path": str(pair["target_path"]),
        "source_length": int(source_feat.shape[0]),
        "target_length": int(target_feat.shape[0]),
        "aligned_frame_count": int(len(sampled_path)),
        "dtw_band_ratio": float(args.dtw_band_ratio),
        "hv_penalty": float(args.hv_penalty),
        "velocity_weight": float(args.velocity_weight),
        "heat_percentile": float(args.heat_percentile),
        "heat_clip_value": float(clip_value),
        "fps": int(args.fps),
    }
    write_metadata(output_dir, metadata)
    print(f"Wrote prototype artifacts to {output_dir}")


if __name__ == "__main__":
    main()
