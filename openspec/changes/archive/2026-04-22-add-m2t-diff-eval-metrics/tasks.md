## 1. Metric Definition

- [x] 1.1 Add or adapt a metric module under `motGPT/metrics/` for `m2t_diff` that reports `Bert_F1`, `ROUGE_L`, and `CIDEr`, with optional BLEU scores as secondary outputs.
- [x] 1.2 Add decoding diagnostics for `m2t_diff`, including empty-output rate and average generated caption length.
- [x] 1.3 Ensure the `m2t_diff` metric path does not compute or headline single-motion `Matching_score` or `R_precision_top_k` values.

## 2. Runtime Wiring

- [x] 2.1 Update `motGPT/models/motgpt.py` so `m2t_diff` validation/test updates the new text-and-diagnostics metric path.
- [x] 2.2 Update metric instantiation in `motGPT/metrics/base.py` so the paired-motion metric can be enabled from config without depending on single-motion dataset assumptions.
- [x] 2.3 Verify that `test.py` reports the new `m2t_diff` metrics in the saved metrics JSON and console summary.

## 3. Config and Validation

- [x] 3.1 Add recommended `METRIC.TYPE` guidance for MotionFix `m2t_diff` configs under `configs/`.
- [x] 3.2 Run a config-driven validation or test command such as `uv run python test.py --cfg configs/MoT_vae_stage4_motionfix_yzh.yaml --nodebug` after enabling the new metrics and confirm the expected values are logged.
- [x] 3.3 Confirm that blank or whitespace-only predictions increase the empty-output diagnostic metric in a controlled check.

## 4. Documentation

- [x] 4.1 Update nearby documentation or README guidance to explain the recommended `m2t_diff` metric set and why single-motion retrieval metrics are excluded.
- [x] 4.2 Document pair-aware motion-text retrieval as future work rather than part of the first `m2t_diff` metric rollout.
