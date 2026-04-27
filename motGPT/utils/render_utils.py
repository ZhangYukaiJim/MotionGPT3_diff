import os
import time
from pathlib import Path

import imageio
import moviepy.editor as mp
import numpy as np
import torch
from scipy.spatial.transform import Rotation as RRR
import motGPT.render.matplot.plot_3d_global as plot_3d
from motGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from motGPT.render.pyrender.smpl_render import SMPLRender

SMPL_MODEL_PATH = "deps/smpl_models/smpl"
STATIC_RENDER_CACHE_VERSION = "v1"
M2T_DIFF_SINGLE_RENDER_TASK = "m2t_diff_single"
M2T_DIFF_SIDE_BY_SIDE_RENDER_TASK = "m2t_diff_side_by_side"
M2T_DIFF_SOURCE_SUFFIX = "source"
M2T_DIFF_TARGET_SUFFIX = "target"
M2T_DIFF_SIDE_BY_SIDE_SUFFIX = "source_target"


def _normalize_render_method(method):
    method = str(method).lower()
    if method not in {"fast", "slow"}:
        raise ValueError(f"Unsupported render method: {method}")
    return method


def _prepare_motion_array(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if len(data.shape) == 4:
        data = data[0]
    return data


def _prepare_fast_render_frames(data, fps=20):
    data = _prepare_motion_array(data)
    if len(data.shape) == 3:
        data = data[None]
    return plot_3d.draw_to_batch(data, [""], None, fps=fps)[0].cpu().numpy()


def _prepare_slow_render_frames(data, smpl_model_path=SMPL_MODEL_PATH):
    data = _prepare_motion_array(data)
    data = data - data[0, 0]
    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(data)
    pose = np.concatenate(
        [pose, np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)], 1
    )
    render = SMPLRender(smpl_model_path)

    r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
    aroot = data[:, 0].copy()
    aroot[:, 1] = -aroot[:, 1]
    aroot[:, 2] = -aroot[:, 2]
    params = dict(pred_shape=np.zeros([1, 10]), pred_root=aroot, pred_pose=pose)
    render.init_renderer([768, 768, 3], params)

    frames = []
    for i in range(data.shape[0]):
        frames.append(render.render(i))

    del render
    return np.stack(frames, axis=0)


def render_motion(
    data,
    feats,
    output_dir,
    fname=None,
    method="fast",
    smpl_model_path=SMPL_MODEL_PATH,
    fps=20,
):
    if fname is None:
        fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) + str(
            np.random.randint(10000, 99999)
        )
    method = _normalize_render_method(method)
    video_fname = fname + ".mp4"
    feats_fname = fname + ".npy"
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    # np.save(output_npy_path, feats)

    if method == "slow":
        slow_frames = _prepare_slow_render_frames(data, smpl_model_path=smpl_model_path)
        out_video = mp.ImageSequenceClip(list(slow_frames), fps=fps)
        out_video.write_videofile(output_mp4_path, fps=fps)
        del slow_frames

    elif method == "fast":
        pose_vis = _prepare_fast_render_frames(data, fps=fps)

        out_video = mp.ImageSequenceClip(list(pose_vis), fps=fps)
        out_video.write_videofile(output_mp4_path, fps=fps)
        # out_video = mp.VideoClip(make_frame=lambda t:pose_vis[int(t*fps)], duration=len(pose_vis)/fps)
        # out_video.write_videofile(output_mp4_path,fps=fps)
        del pose_vis


def render_motion_side_by_side(
    data_left,
    data_right,
    output_dir,
    fname=None,
    method="fast",
    smpl_model_path=SMPL_MODEL_PATH,
    fps=20,
):
    if fname is None:
        fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) + str(
            np.random.randint(10000, 99999)
        )
    method = _normalize_render_method(method)
    output_mp4_path = os.path.join(output_dir, fname + ".mp4")

    if method == "slow":
        left_frames = _prepare_slow_render_frames(
            data_left, smpl_model_path=smpl_model_path
        )
        right_frames = _prepare_slow_render_frames(
            data_right, smpl_model_path=smpl_model_path
        )
    else:
        left_frames = _prepare_fast_render_frames(data_left, fps=fps)
        right_frames = _prepare_fast_render_frames(data_right, fps=fps)
    frame_count = min(len(left_frames), len(right_frames))
    stacked_frames = np.concatenate(
        [left_frames[:frame_count], right_frames[:frame_count]], axis=2
    )

    out_video = mp.ImageSequenceClip(list(stacked_frames), fps=fps)
    out_video.write_videofile(output_mp4_path, fps=fps)

    del left_frames
    del right_frames
    del stacked_frames


def get_render_cache_dir(
    dataset_root,
    task,
    fps=20,
    method="fast",
    version=STATIC_RENDER_CACHE_VERSION,
):
    return Path(dataset_root) / "render_cache" / task / f"{method}_fps{fps}_{version}"


def _symlink_cached_video(cache_path, output_path):
    cache_path = Path(cache_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_symlink():
        if os.path.realpath(output_path) == str(cache_path):
            return str(output_path)
        output_path.unlink()
    elif output_path.exists():
        output_path.unlink()

    output_path.symlink_to(cache_path)
    return str(output_path)


def materialize_cached_motion_render(
    data,
    cache_dir,
    cache_key,
    output_dir,
    output_name=None,
    method="fast",
    fps=20,
):
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{cache_key}.mp4"
    if not cache_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        render_motion(
            data,
            None,
            output_dir=cache_dir,
            fname=cache_key,
            method=method,
            fps=fps,
        )

    output_stem = output_name or cache_key
    return _symlink_cached_video(cache_path, Path(output_dir) / f"{output_stem}.mp4")


def materialize_cached_motion_side_by_side_render(
    data_left,
    data_right,
    cache_dir,
    cache_key,
    output_dir,
    output_name=None,
    method="fast",
    fps=20,
):
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{cache_key}.mp4"
    if not cache_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        render_motion_side_by_side(
            data_left,
            data_right,
            output_dir=cache_dir,
            fname=cache_key,
            method=method,
            fps=fps,
        )

    output_stem = output_name or cache_key
    return _symlink_cached_video(cache_path, Path(output_dir) / f"{output_stem}.mp4")


def get_m2t_diff_render_cache_dirs(dataset_root, fps=20, method="fast"):
    method = _normalize_render_method(method)
    single_render_dir = get_render_cache_dir(
        dataset_root,
        task=M2T_DIFF_SINGLE_RENDER_TASK,
        fps=fps,
        method=method,
    )
    return {
        M2T_DIFF_SOURCE_SUFFIX: single_render_dir / M2T_DIFF_SOURCE_SUFFIX,
        M2T_DIFF_TARGET_SUFFIX: single_render_dir / M2T_DIFF_TARGET_SUFFIX,
        M2T_DIFF_SIDE_BY_SIDE_SUFFIX: get_render_cache_dir(
            dataset_root,
            task=M2T_DIFF_SIDE_BY_SIDE_RENDER_TASK,
            fps=fps,
            method=method,
        ),
    }


def materialize_m2t_diff_motion_artifacts(
    source_joints,
    target_joints,
    dataset_root,
    output_dir,
    sample_id,
    fps=20,
    method="fast",
):
    method = _normalize_render_method(method)
    cache_dirs = get_m2t_diff_render_cache_dirs(dataset_root, fps=fps, method=method)
    materialize_cached_motion_render(
        source_joints,
        cache_dir=cache_dirs[M2T_DIFF_SOURCE_SUFFIX],
        cache_key=sample_id,
        output_dir=output_dir,
        output_name=f"{sample_id}_{M2T_DIFF_SOURCE_SUFFIX}",
        method=method,
        fps=fps,
    )
    materialize_cached_motion_render(
        target_joints,
        cache_dir=cache_dirs[M2T_DIFF_TARGET_SUFFIX],
        cache_key=sample_id,
        output_dir=output_dir,
        output_name=f"{sample_id}_{M2T_DIFF_TARGET_SUFFIX}",
        method=method,
        fps=fps,
    )
    materialize_cached_motion_side_by_side_render(
        source_joints,
        target_joints,
        cache_dir=cache_dirs[M2T_DIFF_SIDE_BY_SIDE_SUFFIX],
        cache_key=sample_id,
        output_dir=output_dir,
        output_name=f"{sample_id}_{M2T_DIFF_SIDE_BY_SIDE_SUFFIX}",
        method=method,
        fps=fps,
    )
