import math
import logging
import re
from collections import Counter, defaultdict
from typing import Iterable, List

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from torchmetrics import Metric
from .bert_compat import FixedDeviceBERTScore


class M2TDiffMetrics(Metric):
    def __init__(
        self,
        cfg,
        dist_sync_on_step=True,
        include_bleu=True,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        metric_cfg = getattr(cfg.METRIC, "M2T_DIFF", None)
        self.include_bleu = getattr(metric_cfg, "INCLUDE_BLEU", include_bleu)
        self.include_bert_f1 = getattr(metric_cfg, "INCLUDE_BERT_F1", True)
        self.bert_device = getattr(metric_cfg, "BERT_DEVICE", None)
        self.bert_idf = getattr(metric_cfg, "BERT_IDF", True)
        self.metrics = ["ROUGE_L", "CIDEr", "Empty_output_rate", "Avg_generated_length"]
        if self.include_bert_f1:
            self.metrics.insert(0, "Bert_F1")
        if self.include_bleu:
            self.metrics.extend(["Bleu_1", "Bleu_4"])

        self.add_state("num_predictions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "empty_predictions", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "total_generated_length", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

        self.pred_texts = []
        self.gt_texts = []
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.bleu_smoother = SmoothingFunction().method1
        self.bert_metric = None
        if self.include_bert_f1:
            self.bert_metric = FixedDeviceBERTScore(
                device=self.bert_device,
                idf=self.bert_idf,
                rescale_with_baseline=True,
                verbose=False,
            )

    def _normalize_prediction(self, prediction: str) -> str:
        if prediction is None:
            return ""
        return str(prediction).strip()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _extract_ngrams(self, tokens: List[str], n: int) -> Counter:
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def _compute_bleu(
        self, predictions: List[str], references: List[List[str]], max_order: int
    ):
        weights = [0.0, 0.0, 0.0, 0.0]
        for idx in range(max_order):
            weights[idx] = 1.0 / max_order

        scores = []
        for prediction, sample_references in zip(predictions, references):
            pred_tokens = self._tokenize(prediction)
            ref_tokens = [self._tokenize(reference) for reference in sample_references]
            if not pred_tokens or not any(ref_tokens):
                scores.append(0.0)
                continue
            scores.append(
                sentence_bleu(
                    ref_tokens,
                    pred_tokens,
                    weights=tuple(weights),
                    smoothing_function=self.bleu_smoother,
                )
            )
        return sum(scores) / max(len(scores), 1)

    def _compute_rouge_l(self, predictions: List[str], references: List[List[str]]):
        scores = []
        for prediction, sample_references in zip(predictions, references):
            best_fmeasure = 0.0
            for reference in sample_references:
                score = self.rouge_scorer.score(reference, prediction)[
                    "rougeL"
                ].fmeasure
                best_fmeasure = max(best_fmeasure, score)
            scores.append(best_fmeasure)
        return sum(scores) / max(len(scores), 1)

    def _compute_cider(self, predictions: List[str], references: List[List[str]]):
        document_frequency = [defaultdict(int) for _ in range(4)]

        for sample_references in references:
            for n in range(1, 5):
                sample_ngrams = set()
                for reference in sample_references:
                    sample_ngrams.update(
                        self._extract_ngrams(self._tokenize(reference), n).keys()
                    )
                for ngram in sample_ngrams:
                    document_frequency[n - 1][ngram] += 1

        num_documents = max(len(references), 1)
        cider_scores = []
        for prediction, sample_references in zip(predictions, references):
            pred_tokens = self._tokenize(prediction)
            sample_score = 0.0
            for n in range(1, 5):
                pred_counts = self._extract_ngrams(pred_tokens, n)
                pred_total = sum(pred_counts.values())
                if pred_total == 0:
                    continue

                pred_vector = {}
                pred_norm = 0.0
                for ngram, count in pred_counts.items():
                    idf = math.log(
                        (num_documents + 1.0) / (document_frequency[n - 1][ngram] + 1.0)
                    )
                    weight = (count / pred_total) * idf
                    pred_vector[ngram] = weight
                    pred_norm += weight * weight
                pred_norm = math.sqrt(pred_norm)
                if pred_norm == 0.0:
                    continue

                reference_score = 0.0
                for reference in sample_references:
                    ref_tokens = self._tokenize(reference)
                    ref_counts = self._extract_ngrams(ref_tokens, n)
                    ref_total = sum(ref_counts.values())
                    if ref_total == 0:
                        continue

                    ref_vector = {}
                    ref_norm = 0.0
                    for ngram, count in ref_counts.items():
                        idf = math.log(
                            (num_documents + 1.0)
                            / (document_frequency[n - 1][ngram] + 1.0)
                        )
                        weight = (count / ref_total) * idf
                        ref_vector[ngram] = weight
                        ref_norm += weight * weight
                    ref_norm = math.sqrt(ref_norm)
                    if ref_norm == 0.0:
                        continue

                    dot = sum(
                        pred_vector.get(ngram, 0.0) * ref_vector.get(ngram, 0.0)
                        for ngram in pred_vector
                    )
                    reference_score += dot / (pred_norm * ref_norm)

                sample_score += reference_score / max(len(sample_references), 1)

            cider_scores.append((sample_score / 4.0) * 10.0)

        return sum(cider_scores) / max(len(cider_scores), 1)

    def _normalize_references(self, references: Iterable[str]) -> List[str]:
        if isinstance(references, str) or references is None:
            references = [references or ""]

        normalized = [
            str(reference).strip() for reference in references if reference is not None
        ]
        if not normalized:
            normalized = [""]
        return normalized

    def _compute_best_bert_f1(
        self, predictions: List[str], references: List[List[str]]
    ):
        if not predictions:
            return 0.0

        flat_predictions = []
        flat_references = []
        sample_indices = []

        for sample_idx, (prediction, sample_references) in enumerate(
            zip(predictions, references)
        ):
            for reference in sample_references:
                flat_predictions.append(prediction)
                flat_references.append(reference)
                sample_indices.append(sample_idx)

        self.bert_metric.update(flat_predictions, flat_references)
        bert_scores = self.bert_metric.compute()
        self.bert_metric.reset()

        best_f1_by_sample = [float("-inf")] * len(predictions)
        for sample_idx, f1 in zip(sample_indices, bert_scores["f1"]):
            best_f1_by_sample[sample_idx] = max(best_f1_by_sample[sample_idx], f1)

        return sum(best_f1_by_sample) / max(len(best_f1_by_sample), 1)

    @torch.no_grad()
    def update(self, pred_texts: List[str], gt_texts: List[Iterable[str]]):
        for pred_text, gt_text in zip(pred_texts, gt_texts):
            normalized_pred = self._normalize_prediction(pred_text)
            normalized_refs = self._normalize_references(gt_text)

            self.pred_texts.append(normalized_pred)
            self.gt_texts.append(normalized_refs)
            self.num_predictions += 1
            self.empty_predictions += int(normalized_pred == "")
            self.total_generated_length += float(len(normalized_pred.split()))

    @torch.no_grad()
    def compute(self, sanity_flag):
        device = self.num_predictions.device
        metrics = {metric: torch.tensor(0.0, device=device) for metric in self.metrics}

        if sanity_flag or self.num_predictions.item() == 0:
            self.reset()
            return metrics

        metrics["ROUGE_L"] = torch.tensor(
            self._compute_rouge_l(self.pred_texts, self.gt_texts), device=device
        )
        metrics["CIDEr"] = torch.tensor(
            self._compute_cider(self.pred_texts, self.gt_texts), device=device
        )
        if self.include_bleu:
            metrics["Bleu_1"] = torch.tensor(
                self._compute_bleu(self.pred_texts, self.gt_texts, max_order=1),
                device=device,
            )
            metrics["Bleu_4"] = torch.tensor(
                self._compute_bleu(self.pred_texts, self.gt_texts, max_order=4),
                device=device,
            )

        if self.include_bert_f1:
            bert_device = self.bert_device or str(device)
            logging.getLogger(__name__).info(
                "M2TDiffMetrics: starting Bert_F1 on device=%s idf=%s for %s predictions",
                bert_device,
                self.bert_idf,
                len(self.pred_texts),
            )
            metrics["Bert_F1"] = torch.tensor(
                self._compute_best_bert_f1(self.pred_texts, self.gt_texts),
                device=device,
            )
            logging.getLogger(__name__).info("M2TDiffMetrics: finished Bert_F1")

        count = self.num_predictions.float().clamp_min(1.0)
        metrics["Empty_output_rate"] = self.empty_predictions.float() / count
        metrics["Avg_generated_length"] = self.total_generated_length / count

        self.reset()
        return metrics

    def reset(self):
        super().reset()
        self.pred_texts = []
        self.gt_texts = []
