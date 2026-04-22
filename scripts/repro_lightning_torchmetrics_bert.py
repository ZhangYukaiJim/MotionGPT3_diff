import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.text.bert import BERTScore


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


class ReproModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.metric = BERTScore(
            device="cpu",
            idf=True,
            rescale_with_baseline=True,
            verbose=False,
        )

    def test_step(self, batch, batch_idx):
        logging.info("TorchMetricsBERT test_step: batch_idx=%s", batch_idx)
        self.metric.update(batch["pred_text"], batch["ref_text"])
        return {"ok": torch.tensor(1.0)}

    def on_test_epoch_end(self):
        logging.info("TorchMetricsBERT compute start")
        metrics = self.metric.compute()
        logging.info("TorchMetricsBERT compute done keys=%s", sorted(metrics.keys()))
        self.log("bert_f1", metrics["f1"].mean(), rank_zero_only=True)
        self.metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            [torch.nn.Parameter(torch.zeros(1, requires_grad=True))], lr=1e-3
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dataset = TinyTextDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.test(ReproModule(), dataloaders=dataloader)


if __name__ == "__main__":
    main()
