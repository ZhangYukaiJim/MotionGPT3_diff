import argparse
import logging

import pytorch_lightning as pl
import torch
from bert_score import score as score_bert
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric


class TinyTextDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "pred_text": "lift the left arm",
            "ref_text": "lift the left arm",
        }


def collate_fn(batch):
    return {
        "pred_text": [item["pred_text"] for item in batch],
        "ref_text": [item["ref_text"] for item in batch],
    }


class BertMetric(Metric):
    def __init__(self, bert_device=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.bert_device = bert_device
        self.add_state("num_predictions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pred_texts = []
        self.ref_texts = []

    def update(self, pred_texts, ref_texts):
        self.pred_texts.extend(pred_texts)
        self.ref_texts.extend(ref_texts)
        self.num_predictions += len(pred_texts)

    def compute(self):
        device = self.num_predictions.device
        bert_device = self.bert_device or str(device)
        logging.info(
            "BertMetric.compute: starting Bert_F1 on bert_device=%s num_predictions=%s",
            bert_device,
            int(self.num_predictions.item()),
        )
        _, _, f1 = score_bert(
            self.pred_texts,
            self.ref_texts,
            lang="en",
            rescale_with_baseline=True,
            idf=True,
            device=bert_device,
            verbose=False,
        )
        logging.info("BertMetric.compute: finished Bert_F1")
        return {"bert_f1": f1.mean()}

    def reset(self):
        super().reset()
        self.pred_texts = []
        self.ref_texts = []


class ReproModule(pl.LightningModule):
    def __init__(self, bert_device=None):
        super().__init__()
        self.metric = BertMetric(bert_device=bert_device, dist_sync_on_step=False)

    def test_step(self, batch, batch_idx):
        logging.info("ReproModule.test_step: batch_idx=%s", batch_idx)
        self.metric.update(batch["pred_text"], batch["ref_text"])
        return {"ok": torch.tensor(1.0)}

    def on_test_epoch_end(self):
        logging.info("ReproModule.on_test_epoch_end: calling metric.compute()")
        metrics = self.metric.compute()
        logging.info(
            "ReproModule.on_test_epoch_end: metric keys=%s", sorted(metrics.keys())
        )
        self.log_dict(metrics, rank_zero_only=True)
        self.metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            [torch.nn.Parameter(torch.zeros(1, requires_grad=True))], lr=1e-3
        )


def run_direct(bert_device):
    logging.info("run_direct: starting bert_score")
    _, _, f1 = score_bert(
        ["lift the left arm"],
        ["lift the left arm"],
        lang="en",
        rescale_with_baseline=True,
        idf=True,
        device=bert_device,
        verbose=False,
    )
    logging.info("run_direct: finished bert_score f1=%s", float(f1.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "lightning"], default="lightning")
    parser.add_argument("--bert-device", default="cpu")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("args=%s", vars(args))

    if args.mode == "direct":
        run_direct(args.bert_device)
        return

    dataset = TinyTextDataset()
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    model = ReproModule(bert_device=args.bert_device)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=[args.device],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.test(model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
