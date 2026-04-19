import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motGPT.config import instantiate_from_config, parse_args
from motGPT.data.build_data import build_data
from motGPT.metrics.tmr import get_sim_matrix
from motGPT.metrics.tmr_utils import length_to_mask


def load_cfg(cfg_path: str):
    import sys

    argv_backup = sys.argv[:]
    sys.argv = [argv_backup[0], "--cfg", cfg_path]
    try:
        cfg = parse_args(phase="webui")
    finally:
        sys.argv = argv_backup
    return cfg


def load_motion_features(path: str, datamodule, device):
    feats = np.load(path)
    feats = torch.tensor(feats, dtype=torch.float32, device=device)
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
    return datamodule.normalize(feats), feats.shape[0]


def build_tmr_components(cfg, device):
    token_cfg = cfg.METRIC.TMR.text_to_token_emb
    sent_cfg = cfg.METRIC.TMR.text_to_sent_emb
    annotation_root = Path(token_cfg.params.path)
    if not annotation_root.exists():
        token_cfg = OmegaConf.create(OmegaConf.to_container(token_cfg, resolve=True))
        sent_cfg = OmegaConf.create(OmegaConf.to_container(sent_cfg, resolve=True))
        token_cfg.params.preload = False
        sent_cfg.params.preload = False

    text_to_token_emb = instantiate_from_config(token_cfg)
    text_to_sent_emb = instantiate_from_config(sent_cfg)
    motionencoder = (
        instantiate_from_config(cfg.METRIC.TMR.tmr_motionencoder).to(device).eval()
    )
    textencoder = (
        instantiate_from_config(cfg.METRIC.TMR.tmr_textencoder).to(device).eval()
    )

    motion_state = torch.load(
        Path(cfg.METRIC.TMR.tmr_path) / "last_weights" / "motion_encoder.pt",
        map_location="cpu",
    )
    text_state = torch.load(
        Path(cfg.METRIC.TMR.tmr_path) / "last_weights" / "text_encoder.pt",
        map_location="cpu",
    )
    motionencoder.load_state_dict(motion_state)
    textencoder.load_state_dict(text_state)

    for module in [motionencoder, textencoder]:
        for param in module.parameters():
            param.requires_grad = False

    return motionencoder, textencoder, text_to_token_emb, text_to_sent_emb


def process_encoded_into_latent(encoded, sample_mean=True, fact=1.0):
    dists = encoded.unbind(1)
    mu, logvar = dists
    if sample_mean:
        return mu.detach()
    std = logvar.exp().pow(0.5)
    eps = std.data.new(std.size()).normal_()
    return (mu + fact * eps * std).detach()


def get_motion_embedding(motionencoder, feats: torch.Tensor, length: int):
    batch = feats.unsqueeze(0)
    mask = length_to_mask([length], device=batch.device)
    x_dict = {"x": batch, "length": [length], "mask": mask}
    encoded = motionencoder(x_dict)
    return process_encoded_into_latent(encoded, sample_mean=True)[0]


def get_text_embedding(textencoder, text_to_token_emb, text: str, device):
    from motGPT.metrics.tmr_utils import collate_x_dict

    text_x_dict = text_to_token_emb(text)
    text_x_dict = collate_x_dict(text_x_dict, device=device)
    encoded = textencoder(text_x_dict)
    return process_encoded_into_latent(encoded, sample_mean=True)[0]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return torch.sum(a * b).item()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two motions with TMR embeddings"
    )
    parser.add_argument(
        "--motion-a", required=True, help="Path to first motion .npy features file"
    )
    parser.add_argument(
        "--motion-b", required=True, help="Path to second motion .npy features file"
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Optional text prompt to compare against both motions",
    )
    parser.add_argument(
        "--cfg", default="./configs/webui.yaml", help="Config file to load"
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    pl.seed_everything(cfg.SEED_VALUE)
    device = torch.device(
        "cuda" if cfg.ACCELERATOR == "gpu" and torch.cuda.is_available() else "cpu"
    )
    datamodule = build_data(cfg)
    motionencoder, textencoder, text_to_token_emb, text_to_sent_emb = (
        build_tmr_components(cfg, device)
    )
    _ = text_to_sent_emb  # loaded to match repo config/cache behavior

    feats_a, length_a = load_motion_features(args.motion_a, datamodule, device)
    feats_b, length_b = load_motion_features(args.motion_b, datamodule, device)

    emb_a = get_motion_embedding(motionencoder, feats_a, length_a)
    emb_b = get_motion_embedding(motionencoder, feats_b, length_b)
    motion_motion_cosine = cosine_similarity(emb_a, emb_b)
    motion_motion_matrix = get_sim_matrix(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()

    result = {
        "motion_a": args.motion_a,
        "motion_b": args.motion_b,
        "length_a": length_a,
        "length_b": length_b,
        "motion_motion_cosine": motion_motion_cosine,
        "motion_motion_tmr_score": motion_motion_matrix,
    }

    if args.text:
        text_emb = get_text_embedding(textencoder, text_to_token_emb, args.text, device)
        result["text"] = args.text
        result["text_motion_a_cosine"] = cosine_similarity(text_emb, emb_a)
        result["text_motion_b_cosine"] = cosine_similarity(text_emb, emb_b)
        result["text_motion_a_tmr_score"] = get_sim_matrix(
            text_emb.unsqueeze(0), emb_a.unsqueeze(0)
        ).item()
        result["text_motion_b_tmr_score"] = get_sim_matrix(
            text_emb.unsqueeze(0), emb_b.unsqueeze(0)
        ).item()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
