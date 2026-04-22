import torch
from torchmetrics.functional.text import bert as torchmetrics_bert
from torchmetrics.text import bert as torchmetrics_text_bert
from torchmetrics.text.bert import BERTScore


def _get_embeddings_and_idf_scale_fixed(
    dataloader,
    target_len,
    model,
    device=None,
    num_layers=None,
    all_layers=False,
    idf=False,
    verbose=False,
    user_forward_fn=None,
):
    embeddings_list = []
    idf_scale_list = []
    for batch in torchmetrics_bert._get_progress_bar(dataloader, verbose):
        with torch.no_grad():
            batch = torchmetrics_bert._input_data_collator(batch, device)
            if not all_layers:
                if not user_forward_fn:
                    out = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        output_hidden_states=True,
                    )
                    out = out.hidden_states[
                        num_layers if num_layers is not None else -1
                    ]
                else:
                    out = user_forward_fn(model, batch)
                    torchmetrics_bert._check_shape_of_model_output(
                        out, batch["input_ids"]
                    )
                out = out.unsqueeze(1)
            else:
                if user_forward_fn:
                    raise ValueError(
                        "The option `all_layers=True` can be used only with default `transformers` models."
                    )
                out = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    output_hidden_states=True,
                )
                out = torch.cat([o.unsqueeze(1) for o in out.hidden_states], dim=1)

        out /= out.norm(dim=-1).unsqueeze(-1)
        out, attention_mask = torchmetrics_bert._output_data_collator(
            out, batch["attention_mask"], target_len
        )
        processed_attention_mask = (
            torchmetrics_bert._process_attention_mask_for_special_tokens(attention_mask)
        )
        out = torch.einsum("blsd, bs -> blsd", out, processed_attention_mask)
        embeddings_list.append(out.cpu())

        input_ids_idf = (
            batch["input_ids_idf"].to(processed_attention_mask.device)
            * processed_attention_mask
            if idf
            else processed_attention_mask.type(out.dtype)
        )
        input_ids_idf /= input_ids_idf.sum(-1, keepdim=True)
        idf_scale_list.append(input_ids_idf.cpu())

    embeddings = torch.cat(embeddings_list)
    idf_scale = torch.cat(idf_scale_list)
    return embeddings, idf_scale


class FixedDeviceBERTScore(BERTScore):
    """TorchMetrics BERTScore wrapper with consistent tensor placement on GPU."""

    def compute(self):
        preds = torchmetrics_text_bert._get_input_dict(
            self.preds_input_ids, self.preds_attention_mask
        )
        target = torchmetrics_text_bert._get_input_dict(
            self.target_input_ids, self.target_attention_mask
        )

        model = self.model
        if model is None:
            if not torchmetrics_bert._TRANSFORMERS_AUTO_AVAILABLE:
                raise ModuleNotFoundError(
                    "`BERTScore` metric with default models requires `transformers` package be installed."
                )
            model = torchmetrics_bert.AutoModel.from_pretrained(self.model_name_or_path)

        model.eval()
        model.to(self.embedding_device)

        if self.num_layers and self.num_layers > model.config.num_hidden_layers:
            raise ValueError(
                f"num_layers={self.num_layers} is forbidden for {self.model_name_or_path}. "
                f"Please use num_layers <= {model.config.num_hidden_layers}"
            )

        baseline = (
            torchmetrics_bert._load_baseline(
                self.lang,
                self.model_name_or_path,
                self.baseline_path,
                self.baseline_url,
            )
            if self.rescale_with_baseline
            else None
        )

        target_dataset = torchmetrics_bert.TokenizedDataset(**target, idf=self.idf)
        preds_dataset = torchmetrics_bert.TokenizedDataset(
            **preds,
            idf=self.idf,
            tokens_idf=target_dataset.tokens_idf,
        )

        target_loader = torchmetrics_bert.DataLoader(
            target_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_threads,
        )
        preds_loader = torchmetrics_bert.DataLoader(
            preds_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_threads,
        )

        target_embeddings, target_idf_scale = _get_embeddings_and_idf_scale_fixed(
            target_loader,
            target_dataset.max_length,
            model,
            self.embedding_device,
            self.num_layers,
            self.all_layers,
            self.idf,
            self.verbose,
            self.user_forward_fn,
        )
        preds_embeddings, preds_idf_scale = _get_embeddings_and_idf_scale_fixed(
            preds_loader,
            preds_dataset.max_length,
            model,
            self.embedding_device,
            self.num_layers,
            self.all_layers,
            self.idf,
            self.verbose,
            self.user_forward_fn,
        )

        precision, recall, f1_score = torchmetrics_bert._get_precision_recall_f1(
            preds_embeddings,
            target_embeddings,
            preds_idf_scale,
            target_idf_scale,
        )

        if baseline is not None:
            precision, recall, f1_score = (
                torchmetrics_bert._rescale_metrics_with_baseline(
                    precision,
                    recall,
                    f1_score,
                    baseline,
                    self.num_layers,
                    self.all_layers,
                )
            )

        output_dict = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1_score.tolist(),
        }
        if self.return_hash:
            output_dict["hash"] = torchmetrics_bert._get_hash(
                self.model_name_or_path,
                self.num_layers,
                self.idf,
            )
        return output_dict
