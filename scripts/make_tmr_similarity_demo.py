import argparse
import itertools
import os
from pathlib import Path
import sys
from textwrap import wrap

import moviepy.editor as mp
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts").resolve()))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from compare_two_motions_prompt import load_model_and_cfg
from compare_two_motions_tmr import build_tmr_components, get_motion_embedding
from motGPT.utils.render_utils import render_motion


DEFAULT_PROMPTS = [
    "a person walks forward confidently",
    "a person jogs in a small circle",
    "a person raises one arm and waves",
    "a person jumps in place twice",
    "a person crouches and stands up",
    "a person turns around slowly",
    "a person kicks forward with one leg",
    "a person side steps to the right",
    "a person punches forward with both hands",
    "a person leans left and recovers",
    "a person bows politely",
    "a person reaches overhead with both arms",
    "a person hops backward twice",
    "a person swings both arms while stepping forward",
    "a person spins quickly and stops",
    "a person marches in place",
    "a person lunges forward and returns",
    "a person shuffles sideways to the left",
    "a person stretches one arm across the chest",
    "a person ducks and steps back",
    "a person makes a wide celebratory gesture",
    "a person balances on one leg briefly",
    "a person reaches down toward the floor",
    "a person steps forward and points ahead",
]


def generate_motion_from_text(model, prompt: str):
    batch = {
        "text": [prompt],
        "length": [196],
        "motion_tokens_input": None,
    }
    outputs = model(batch, task="t2m")
    out_feats = outputs["feats"][0].detach().cpu().numpy()
    out_length = outputs["length"][0]
    out_joints = outputs["joints"][:out_length].detach().cpu().numpy()
    return out_feats[:out_length], out_joints


def score_motion_pairs(items, datamodule, motionencoder, device):
    for item in items:
        feats = torch.tensor(item["feats"], dtype=torch.float32, device=device)
        item["embedding"] = get_motion_embedding(
            motionencoder, datamodule.normalize(feats), len(item["feats"])
        )

    pairs = []
    for a, b in itertools.combinations(items, 2):
        score = torch.nn.functional.cosine_similarity(
            a["embedding"].unsqueeze(0), b["embedding"].unsqueeze(0)
        ).item()
        pairs.append({"a": a, "b": b, "score": score})
    pairs.sort(key=lambda item: item["score"], reverse=True)
    return pairs


def select_non_reused_pairs(sorted_pairs, num_pairs):
    selected = []
    used = set()
    for pair in sorted_pairs:
        ids = {pair["a"]["id"], pair["b"]["id"]}
        if used & ids:
            continue
        selected.append(pair)
        used |= ids
        if len(selected) == num_pairs:
            break
    return selected


def rerender_slow(item, output_dir: Path, fps: int):
    stem = item["id"]
    output_mp4_path = output_dir / f"{stem}_slow.mp4"
    render_motion(
        item["joints"],
        item["feats"],
        output_dir=str(output_dir),
        fname=stem + "_slow",
        method="slow",
        fps=fps,
    )
    return output_mp4_path


def load_font(size: int):
    for font_name in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_label_panel(width: int, height: int, title: str, subtitle: str):
    image = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(image)
    score_font = load_font(max(24, height // 3))
    prompt_font = load_font(max(14, height // 7))
    draw.text((16, 8), title, fill=(255, 255, 255), font=score_font)
    wrapped = "\n".join(wrap(subtitle, width=max(22, width // 14)))
    title_bbox = draw.textbbox((16, 8), title, font=score_font)
    prompt_y = title_bbox[3] + 6
    draw.multiline_text(
        (16, prompt_y),
        wrapped,
        fill=(210, 210, 210),
        font=prompt_font,
        spacing=4,
    )
    return np.array(image)


def make_pair_tile(pair, rendered_paths, tile_width, tile_height, duration):
    clip_a = mp.VideoFileClip(str(rendered_paths[pair["a"]["id"]]), audio=False)
    clip_b = mp.VideoFileClip(str(rendered_paths[pair["b"]["id"]]), audio=False)
    clip_a = clip_a.subclip(0, duration)
    clip_b = clip_b.subclip(0, duration)

    label_height = max(88, tile_height // 3)
    padding = 12
    available_height = tile_height - label_height - (padding * 2)
    available_width = tile_width - (padding * 2)
    gap = max(6, tile_width // 60)
    target_side = int(min(available_height, (available_width - gap) / 2))
    clip_a = clip_a.resize(width=target_side)
    clip_b = clip_b.resize(width=target_side)

    title = f"TMR cosine {pair['score']:.4f}"
    subtitle = f"A: {pair['a']['prompt']} | B: {pair['b']['prompt']}"
    label_frame = build_label_panel(tile_width, label_height, title, subtitle)
    label_clip = mp.ImageClip(label_frame).set_duration(duration)
    background = mp.ColorClip((tile_width, tile_height), color=(8, 8, 8)).set_duration(
        duration
    )
    pair_width = target_side * 2 + gap
    x1 = max(padding, int((tile_width - pair_width) / 2))
    x2 = x1 + target_side + gap
    y = label_height + padding + max(0, int((available_height - target_side) / 2))
    return mp.CompositeVideoClip(
        [
            background,
            label_clip.set_position((0, 0)),
            clip_a.set_position((x1, y)),
            clip_b.set_position((x2, y)),
        ],
        size=(tile_width, tile_height),
    ).set_duration(duration)


def main():
    parser = argparse.ArgumentParser(
        description="Create a one-page ranked TMR similarity demo from fresh prompts"
    )
    parser.add_argument(
        "--cfg", default="./configs/webui.yaml", help="Config file to load"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=12, help="How many fresh prompts to generate"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=10,
        help="How many ranked non-reused pairs to show",
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Render fps for slow SMPL videos"
    )
    parser.add_argument(
        "--duration-cap", type=float, default=4.0, help="Max seconds per row"
    )
    parser.add_argument(
        "--output",
        default="outputs/tmr_similarity_demo_page.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of columns in the final page layout",
    )
    parser.add_argument(
        "--page-width",
        type=int,
        default=1920,
        help="Final page video width",
    )
    parser.add_argument(
        "--page-height",
        type=int,
        default=1080,
        help="Final page video height",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for generation")
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS[: args.num_prompts]
    if len(prompts) < args.num_pairs * 2:
        raise ValueError(
            "Need at least two fresh prompts per requested non-reused pair"
        )
    print(
        f"Using {len(prompts)} fresh prompts to generate {args.num_pairs} scored motion pairs"
    )

    cfg, datamodule, model, device = load_model_and_cfg(args.cfg)
    pl.seed_everything(args.seed)
    motionencoder, _, _, _ = build_tmr_components(cfg, device)

    items = []
    for idx, prompt in enumerate(prompts):
        feats, joints = generate_motion_from_text(model, prompt)
        items.append(
            {
                "id": f"prompt_{idx:02d}",
                "prompt": prompt,
                "feats": feats,
                "joints": joints,
            }
        )

    scored_pairs = score_motion_pairs(items, datamodule, motionencoder, device)
    selected_pairs = select_non_reused_pairs(scored_pairs, args.num_pairs)
    if len(selected_pairs) < args.num_pairs:
        raise ValueError(
            f"Could only find {len(selected_pairs)} non-reused pairs from {len(items)} motions"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output_path.parent / (output_path.stem + "_assets")
    asset_dir.mkdir(parents=True, exist_ok=True)

    rendered_paths = {}
    for pair in selected_pairs:
        for item in [pair["a"], pair["b"]]:
            if item["id"] not in rendered_paths:
                rendered_paths[item["id"]] = rerender_slow(item, asset_dir, args.fps)

    pair_durations = [
        min(
            len(pair["a"]["feats"]) / args.fps,
            len(pair["b"]["feats"]) / args.fps,
            args.duration_cap,
        )
        for pair in selected_pairs
    ]
    common_duration = min(pair_durations)

    columns = max(1, args.columns)
    rows = int(np.ceil(len(selected_pairs) / columns))
    tile_width = args.page_width // columns
    tile_height = args.page_height // rows
    tiles = [
        make_pair_tile(pair, rendered_paths, tile_width, tile_height, common_duration)
        for pair in selected_pairs
    ]
    while len(tiles) < rows * columns:
        tiles.append(
            mp.ColorClip((tile_width, tile_height), color=(8, 8, 8)).set_duration(
                common_duration
            )
        )

    grid = [tiles[i * columns : (i + 1) * columns] for i in range(rows)]
    final = mp.clips_array(grid)
    final.write_videofile(str(output_path), fps=args.fps, codec="libx264", audio=False)

    print(f"Saved page demo video to {output_path}")
    print(f"Saved slow-rendered motion assets under {asset_dir}")
    print("Selected pairs:")
    for rank, pair in enumerate(selected_pairs, start=1):
        print(
            f"{rank}. {pair['score']:.4f} | {pair['a']['prompt']} | {pair['b']['prompt']}"
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
