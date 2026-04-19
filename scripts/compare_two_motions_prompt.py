import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pytorch_lightning as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model


def load_model_and_cfg(cfg_path: str):
    import sys

    argv_backup = sys.argv[:]
    sys.argv = [argv_backup[0], "--cfg", cfg_path]
    try:
        cfg = parse_args(phase="webui")
    finally:
        sys.argv = argv_backup

    pl.seed_everything(cfg.SEED_VALUE)
    if cfg.ACCELERATOR == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    datamodule = build_data(cfg)
    model = build_model(cfg, datamodule).eval()
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return cfg, datamodule, model, device


def load_motion_features(path: str, model):
    feats = np.load(path)
    feats = torch.tensor(feats, dtype=torch.float32, device=model.device)
    if feats.ndim == 3:
        if feats.shape[0] != 1:
            raise ValueError(
                f"Expected a single motion array in {path}, got shape {tuple(feats.shape)}"
            )
        feats = feats[0]
    if feats.ndim != 2:
        raise ValueError(
            f"Expected motion features shaped [T, C] in {path}, got shape {tuple(feats.shape)}"
        )
    norm_feats = model.datamodule.normalize(feats)
    return norm_feats, feats.shape[0]


def encode_motion_tokens(model, feats: torch.Tensor, length: int):
    batched = feats.unsqueeze(0)
    tokens = model.lm.motion_feats_to_tokens(
        model.vae, batched, [length], modes="motion"
    )
    return tokens[0]


def build_two_motion_prompt(prompt: str, model, length1: int, length2: int):
    seconds1 = length1 / model.lm.framerate
    seconds2 = length2 / model.lm.framerate
    return (
        prompt.replace("<Motion_Placeholder_s1>", model.lm.input_motion_holder_seq)
        .replace("<Motion_Placeholder_s2>", model.lm.input_motion_holder_seq)
        .replace("<Motion_Placeholder>", model.lm.input_motion_holder_seq)
        .replace("<Caption_Placeholder>", "")
        .replace("<Frame_Placeholder_s1>", str(length1))
        .replace("<Frame_Placeholder_s2>", str(length2))
        .replace("<Frame_Placeholder>", str(length1))
        .replace("<Second_Placeholder_s1>", f"{seconds1:.1f}")
        .replace("<Second_Placeholder_s2>", f"{seconds2:.1f}")
        .replace("<Second_Placeholder>", f"{seconds1:.1f}")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prompt-only two-motion comparison experiment"
    )
    parser.add_argument(
        "--motion-a", required=True, help="Path to first motion .npy features file"
    )
    parser.add_argument(
        "--motion-b", required=True, help="Path to second motion .npy features file"
    )
    parser.add_argument(
        "--prompt",
        default="Describe the difference between <Motion_Placeholder_s1> and <Motion_Placeholder_s2> in detail.",
        help="Prompt to send to the released checkpoint",
    )
    parser.add_argument(
        "--cfg", default="./configs/webui.yaml", help="Config file to load"
    )
    parser.add_argument(
        "--max-length", type=int, default=80, help="Max generated text length"
    )
    args = parser.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    _, _, model, _ = load_model_and_cfg(args.cfg)

    feats_a, length_a = load_motion_features(args.motion_a, model)
    feats_b, length_b = load_motion_features(args.motion_b, model)

    motion_tokens = torch.cat(
        [
            encode_motion_tokens(model, feats_a, length_a),
            encode_motion_tokens(model, feats_b, length_b),
        ],
        dim=0,
    )
    prompt = build_two_motion_prompt(args.prompt, model, length_a, length_b)

    _, cleaned_text = model.lm.generate_direct(
        [prompt],
        motion_tokens=[motion_tokens],
        max_length=args.max_length,
        num_beams=1,
        do_sample=True,
        gen_mode="text",
    )

    print("Prompt:")
    print(prompt)
    print("\nResponse:")
    print(cleaned_text[0])


if __name__ == "__main__":
    with torch.no_grad():
        main()
