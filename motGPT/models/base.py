import os
import numpy as np
import torch
import logging
from pathlib import Path
from pytorch_lightning import LightningModule
from os.path import join as pjoin
from collections import OrderedDict
from motGPT.metrics import BaseMetrics
from motGPT.config import get_obj_from_str
from motGPT.utils.render_utils import (
    materialize_m2t_diff_motion_artifacts,
)
import gc


class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.configure_metrics()
        # self.train_epoch = 0

        # Ablation
        self.test_step_outputs = []
        self.times = []
        self.rep_i = 0
        self.cnt_val = 0

        # self.val_step_outputs = []

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        # if self.cnt_val % 5 == 0:
        #     loss0 = self.allsplit_step("val", batch, batch_idx, 'm2t')
        loss = self.allsplit_step("val", batch, batch_idx)
        self.cnt_val += 1
        # self.val_step_outputs.append(outputs)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.allsplit_step("test", batch, batch_idx)
        self.test_step_outputs.append(outputs)
        return outputs

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def on_train_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict("train"))
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
        # self.train_epoch +=1
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict("train"))
        dico.update(self.loss_log_dict("val"))
        # Log metrics
        dico.update(self.metrics_log_dict())
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_test_epoch_end(self):
        # Log metrics
        dico = self.metrics_log_dict()
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
        self.save_npy(self.test_step_outputs)
        self.rep_i = self.rep_i + 1
        # Free up the memory
        self.test_step_outputs.clear()

    def preprocess_state_dict(self, state_dict):
        new_state_dict = OrderedDict()

        metric_state_dict = self.metrics.state_dict()
        loss_state_dict = self._losses.state_dict()

        for k, v in metric_state_dict.items():
            new_state_dict["metrics." + k] = v

        for k, v in loss_state_dict.items():
            new_state_dict["_losses." + k] = v

        for k, v in state_dict.items():
            if "_losses" not in k and "Metrics" not in k:
                new_state_dict[k] = v

        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = self.preprocess_state_dict(state_dict)
        super().load_state_dict(new_state_dict, strict)

    def step_log_dict(self):
        return {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.current_epoch)
            * len(self.datamodule.train_dataset),
            # 'lr':self.optimizers()[0].param_groups[0]['lr']
        }

    def loss_log_dict(self, split: str):
        losses = self._losses["losses_" + split]
        loss_dict = losses.compute(split)
        return loss_dict

    def metrics_log_dict(self):
        # For TM2TMetrics MM
        if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.hparams.metrics_dict:
            metrics_dicts = ["MMMetrics"]
        else:
            metrics_dicts = self.hparams.metrics_dict

        # Compute all metrics
        metrics_log_dict = {}
        for metric in metrics_dicts:
            metrics_dict = getattr(self.metrics, metric).compute(
                sanity_flag=self.trainer.sanity_checking
            )
            metrics_log_dict.update(
                {
                    f"Metrics/{metric}": value.item()
                    if isinstance(value, (torch.Tensor, np.ndarray))
                    else value
                    for metric, value in metrics_dict.items()
                }
            )

        return metrics_log_dict

    def configure_optimizers(self):
        # Optimizer
        optim_target = self.hparams.cfg.TRAIN.OPTIM.target
        if len(optim_target.split(".")) == 1:
            optim_target = "torch.optim." + optim_target

        optimizers_0 = self.parameters()
        # optimizers_1 = self.language_model.diffloss.parameters()
        optimizer = get_obj_from_str(optim_target)(
            params=optimizers_0, **self.hparams.cfg.TRAIN.OPTIM.params
        )

        # Scheduler
        scheduler_target = self.hparams.cfg.TRAIN.LR_SCHEDULER.target
        if len(scheduler_target.split(".")) == 1:
            scheduler_target = "torch.optim.lr_scheduler." + scheduler_target
        lr_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer, **self.hparams.cfg.TRAIN.LR_SCHEDULER.params
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_metrics(self):
        self.metrics = BaseMetrics(datamodule=self.datamodule, **self.hparams)

    def _normalize_text_list(self, text):
        if isinstance(text, str) or text is None:
            return [text or ""]
        return [str(item) for item in text]

    def _should_save_test_outputs(self):
        cfg = self.hparams.cfg
        return cfg.TEST.SAVE_PREDICTIONS or getattr(cfg.TEST, "VISUALIZE", False)

    def _get_test_output_dir(self):
        cfg = self.hparams.cfg
        return Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.target.split(".")[-2].lower()),
                str(cfg.NAME),
                "samples_" + cfg.TIME,
            )
        )

    def _save_text_file(self, path, lines):
        np.savetxt(path, np.array(lines), fmt="%s")

    def _save_m2t_diff_test_artifacts(self, output_dir, outputs):
        from tqdm import tqdm

        visualize = getattr(self.hparams.cfg.TEST, "VISUALIZE", False)
        render_cache_root = self.datamodule.hparams.motionfix_root

        for batch_outputs in tqdm(outputs):
            pred_texts = batch_outputs["pred_texts"]
            target_lengths = batch_outputs["target_lengths"]
            target_feats = batch_outputs["target_feats"]
            source_lengths = batch_outputs["source_lengths"]
            source_feats = batch_outputs["source_feats"]
            gt_texts = batch_outputs["gt_texts"]
            fnames = batch_outputs["fnames"]

            for bid in range(len(pred_texts)):
                fname = fnames[bid].split("/")[-1]
                pred_text = str(pred_texts[bid])
                gt_text = self._normalize_text_list(gt_texts[bid])

                self._save_text_file(output_dir / f"{fname}.txt", [pred_text])
                self._save_text_file(output_dir / f"{fname}_gt.txt", gt_text)

                if not visualize:
                    continue

                target_feat = target_feats[bid][: target_lengths[bid]]
                source_feat = source_feats[bid][: source_lengths[bid]]
                target_joints = self.feats2joints(self.datamodule.renorm4m(target_feat))
                source_joints = self.feats2joints(self.datamodule.renorm4m(source_feat))

                materialize_m2t_diff_motion_artifacts(
                    source_joints,
                    target_joints,
                    dataset_root=render_cache_root,
                    output_dir=output_dir,
                    sample_id=fname,
                    fps=self.datamodule.fps,
                )

    def _save_legacy_test_artifacts(self, output_dir, outputs):
        lengths = [i[1] for i in outputs]
        gt_feats = [i[2] for i in outputs]
        texts = [i[3] for i in outputs]
        fnames = [i[4] for i in outputs]
        outputs = [i[0] for i in outputs]
        test_type = "m2t" if isinstance(outputs[0][0], str) else "t2m"

        if self.datamodule.name.lower() in ["humanml3d", "kit", "motionx"]:
            from tqdm import tqdm

            for i in tqdm(range(len(outputs))):
                for bid in range(
                    min(self.hparams.cfg.TEST.BATCH_SIZE, len(outputs[i]))
                ):
                    text = texts[i][bid]
                    fname = fnames[i][bid].split("/")[-1]
                    if test_type == "m2t":
                        pred_text = outputs[i][bid]
                        text.append(pred_text)
                        text_list = text
                    else:
                        text_list = [text]

                    self._save_text_file(output_dir / f"{fname}.txt", text_list)

                    if test_type == "t2m":
                        gen_feats = outputs[i][bid][: lengths[i][bid]]
                        gen_feats = (
                            self.datamodule.denormalizefromt2m(gen_feats).cpu().numpy()
                        )
                        np.save(output_dir / f"{fname}.npy", gen_feats)

                    gt_feat = gt_feats[i][bid][: lengths[i][bid]]
                    gt_feat = self.datamodule.denormalizefromt2m(gt_feat).cpu().numpy()
                    np.save(output_dir / f"{fname}_gt.npy", gt_feat)

        elif self.hparams.cfg.TEST.DATASETS[0].lower() in ["humanact12", "uestc"]:
            assert False
            keyids = range(len(self.trainer.datamodule.test_dataset))
            for i in range(len(outputs)):
                for bid in range(
                    min(self.hparams.cfg.TEST.BATCH_SIZE, outputs[i].shape[0])
                ):
                    keyid = keyids[i * self.hparams.cfg.TEST.BATCH_SIZE + bid]
                    gen_joints = outputs[i][bid].cpu()
                    gen_joints = gen_joints.permute(2, 0, 1)[
                        : lengths[i][bid], ...
                    ].numpy()
                    if self.hparams.cfg.TEST.REPLICATION_TIMES > 1:
                        name = f"{keyid}_{self.rep_i}"
                    else:
                        name = f"{keyid}.npy"
                    np.save(output_dir / name, gen_joints)

    def save_npy(self, outputs):
        if not self._should_save_test_outputs():
            return

        output_dir = self._get_test_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        if (
            outputs
            and isinstance(outputs[0], dict)
            and outputs[0].get("task") == "m2t_diff"
        ):
            self._save_m2t_diff_test_artifacts(output_dir, outputs)
            return

        self._save_legacy_test_artifacts(output_dir, outputs)
