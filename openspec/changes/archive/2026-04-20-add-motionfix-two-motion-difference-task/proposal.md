## Why

MotionGPT3 currently supports single-motion understanding and generation tasks such as `m2t`, `pred`, and `inbetween`, but it does not provide a trainable pathway for describing how one motion differs from another. We want to finetune the released `motiongpt3.ckpt` on MotionFix so the model can generate natural-language difference descriptions or edit-style descriptions from paired source and target motions.

## What Changes

- Add a MotionFix-backed paired-motion training path that can load source motion, target motion, and paired text supervision from a local MotionFix dataset export.
- Add a new two-motion-to-text task that reuses the existing motion feature and motion-token encoding pipeline, but accepts two motion sequences per sample and generates text from their combined representation.
- Add configuration support for stage-4 finetuning from the released `checkpoints/motiongpt3.ckpt`, including guidance settings that remain compatible with the released checkpoint structure.
- Add evaluation and debugging entry points for paired-motion prompting so the new task can be tested outside the Gradio web UI before full integration.
- Document dataset prerequisites, local file layout, and expected setup steps for MotionFix-based finetuning.

## Capabilities

### New Capabilities
- `motionfix-paired-motion-data`: Load MotionFix-style paired source/target motions plus supervision text into MotionGPT3 training and evaluation workflows.
- `two-motion-difference-to-text`: Generate text from two encoded motion sequences using MotionGPT3 finetuning and prompt/evaluation utilities.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `motGPT/data/` for a new paired-motion dataset/datamodule and collate path
  - `motGPT/models/` and/or `motGPT/archs/` for two-motion text generation task handling
  - `scripts/` for dataset inspection, prompt-only comparison, and evaluation helpers
  - `configs/` for new finetuning configs and dataset asset wiring
  - `README.md` for setup and training guidance
- Affected local systems and artifacts:
  - `deps/` unchanged for core model assets, but training will depend on a local MotionFix dataset root outside the repo
  - `experiments/` will gain stage-4 finetune outputs for the new paired-motion task
  - `results/` may gain paired-motion comparison outputs for evaluation/debugging
- New external dependency on a complete local MotionFix dataset export, including split metadata and any preprocessing needed to map MotionFix motions into MotionGPT3-compatible motion features.
