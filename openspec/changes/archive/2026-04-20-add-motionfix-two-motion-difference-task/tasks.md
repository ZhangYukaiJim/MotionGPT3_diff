## 1. MotionFix Dataset Preparation

- [x] 1.1 Add a MotionFix dataset root validator that requires `motionfix.pth.tar` and `splits.json` before preprocessing or datamodule setup.
- [x] 1.2 Implement a preprocessing script under `scripts/` that reads MotionFix paired samples and converts `motion_source` and `motion_target` joint sequences into HumanML-compatible MotionGPT3 feature arrays.
- [x] 1.3 Reuse HumanML normalization for the converted MotionFix features and write cached paired feature outputs plus sample metadata to a deterministic MotionFix cache layout.
- [x] 1.4 Add a smoke-test command using `uv run` to validate that a small MotionFix subset preprocesses successfully and reports missing split metadata clearly.

## 2. MotionFix Datamodule Integration

- [x] 2.1 Add a MotionFix dataset/datamodule in `motGPT/data/` that loads paired source and target feature caches, text supervision, and source/target lengths for train, val, and test splits.
- [x] 2.2 Add a paired-motion collate path that preserves source tensors, target tensors, text, and both length fields in each batch.
- [x] 2.3 Add config wiring in `configs/` and any required asset config entries so `train.py` and `test.py` can build the MotionFix datamodule through the existing OmegaConf path.
- [x] 2.4 Validate datamodule construction with a runnable command such as `uv run python -m train --cfg <motionfix-config> --nodebug` up to the first batch load.

## 3. Two-Motion Difference-to-Text Task

- [x] 3.1 Add a new paired-motion text task path in `motGPT/models/` and/or `motGPT/archs/` that encodes source and target motions separately and preserves their order in the LM input.
- [x] 3.2 Add two-motion prompt template handling that supports source and target motion placeholders for finetuning and prompt-only debugging.
- [x] 3.3 Update or extend helper scripts in `scripts/` so two local `.npy` motion feature files can be used for qualitative comparison generation outside the web UI.
- [x] 3.4 Validate the new task path with a runnable `uv run python ...` command that generates text from two cached motion feature files.

## 4. Stage-4 Finetuning and Checkpoint Compatibility

- [x] 4.1 Add a MotionFix stage-4 finetuning config that loads `checkpoints/motiongpt3.ckpt` with guidance settings compatible with `lm.fake_latent`.
- [x] 4.2 Verify that the new finetuning config can instantiate the model and load the released checkpoint without strict state-dict failures.
- [x] 4.3 Run a short finetuning smoke test on MotionFix with the new config to confirm the paired-motion task reaches a training step on the target hardware.

## 5. Documentation and Developer Workflow

- [x] 5.1 Update `README.md` with MotionFix prerequisites, required local file layout, preprocessing steps, and stage-4 finetuning commands.
- [x] 5.2 Document the recommended HumanML-normalized MotionFix workflow and note that the first implementation does not require Gradio integration.
- [x] 5.3 Add concise usage examples for preprocessing, smoke testing, and prompt-only two-motion generation using `uv run` commands.
