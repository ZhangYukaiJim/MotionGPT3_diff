import json
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

from . import BASEDataModule
from .humanml.scripts.motion_process import recover_from_ric
from .motionfix_utils import get_motionfix_cache_root, validate_motionfix_root
from .utils import motionfix_collate
from .humanml.utils.word_vectorizer import WordVectorizer


class MotionFixPairedDataset(data.Dataset):
    def __init__(self, manifest_path, mean, std, all_captions_as_text=True, **kwargs):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            self.records = json.load(handle)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.all_captions_as_text = all_captions_as_text

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        cache_root = self.manifest_path.parent.parent
        source = np.load(cache_root / record["source_path"]).astype(np.float32)
        target = np.load(cache_root / record["target_path"]).astype(np.float32)
        source = (source - self.mean) / self.std
        target = (target - self.mean) / self.std
        all_captions = record.get("all_captions") or [record["text"]]
        task = {
            "class": "m2t_diff",
            "input": [
                "Describe the difference between <Motion_Placeholder_s1> and <Motion_Placeholder_s2> in detail."
            ],
            "output": [""],
        }
        return {
            "id": record["id"],
            "text": record["text"],
            "all_captions": all_captions,
            "motion_source": source,
            "motion_target": target,
            "length_source": int(record["length_source"]),
            "length_target": int(record["length_target"]),
            "fname": record["id"],
            "tasks": task,
        }


class MotionFixDataModule(BASEDataModule):
    def __init__(self, cfg, phase="train", **kwargs):
        super().__init__(collate_fn=motionfix_collate)
        self.cfg = cfg
        self.phase = phase
        self.save_hyperparameters(logger=False)

        self.name = "motionfix"
        self.njoints = 22
        self.fps = cfg.DATASET.MOTIONFIX.FPS
        self.hparams.fps = cfg.DATASET.MOTIONFIX.FPS

        root = validate_motionfix_root(cfg.DATASET.MOTIONFIX.ROOT)
        cache_root_override = getattr(cfg.DATASET, "MOTIONFIX_CACHE_ROOT", None)
        cache_root = get_motionfix_cache_root(
            root,
            cache_root_override or cfg.DATASET.MOTIONFIX.CACHE_ROOT,
        )
        self.hparams.motionfix_root = str(root)
        self.hparams.cache_root = str(cache_root)
        self.hparams.mean = np.load("assets/meta/mean.npy")
        self.hparams.std = np.load("assets/meta/std.npy")
        self.hparams.mean_eval = np.load("assets/meta/mean_eval.npy")
        self.hparams.std_eval = np.load("assets/meta/std_eval.npy")
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab"
        )
        self.hparams.stage = cfg.TRAIN.STAGE
        self.nfeats = cfg.DATASET.NFEATS

        self._train_manifest = cache_root / "manifests" / "train.json"
        self._val_manifest = cache_root / "manifests" / "val.json"
        self._test_manifest = cache_root / "manifests" / "test.json"
        for manifest in [self._train_manifest, self._val_manifest, self._test_manifest]:
            if not manifest.exists():
                raise FileNotFoundError(
                    f"MotionFix cache manifest missing: {manifest}. Run scripts/preprocess_motionfix.py first."
                )

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = MotionFixPairedDataset(
                self._train_manifest,
                self.hparams.mean,
                self.hparams.std,
            )
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = MotionFixPairedDataset(
                self._val_manifest,
                self.hparams.mean,
                self.hparams.std,
            )
        return self._val_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            self._test_dataset = MotionFixPairedDataset(
                self._test_manifest,
                self.hparams.mean,
                self.hparams.std,
            )
        return self._test_dataset

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        return (features - mean) / std

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        return features * std + mean

    def renorm4t2m(self, features):
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def renorm4m(self, features):
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * eval_std + eval_mean
        features = (features - ori_mean) / ori_std
        return features

    def denormalizefromt2m(self, features):
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        features = features * eval_std + eval_mean
        features = (features - ori_mean) / ori_std
        return features
