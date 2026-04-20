## Context

MotionGPT3 already contains the main building blocks needed for a two-motion comparison finetuning workflow, but they are not connected into a trainable task. The current codebase supports single-motion tasks such as `m2t`, `pred`, and `inbetween`, and it already contains a prototype prompt-only script (`scripts/compare_two_motions_prompt.py`) that encodes two motions independently, concatenates their motion-token blocks, and asks the released checkpoint to describe the difference. That prototype demonstrates that the current VAE and LM stack can ingest two motions without introducing a new backbone architecture.

The main missing pieces are data plumbing, task wiring, and configuration. MotionFix provides paired `motion_source`, `motion_target`, and text supervision, but MotionGPT3's built-in datamodules are centered on HumanML3D-style single-motion samples. MotionFix is also stored in a different local format (`joblib` archive plus split metadata) and uses AMASS/SMPL-derived fields rather than the current HumanML3D feature files.

We also need to preserve compatibility with the released `checkpoints/motiongpt3.ckpt`. That checkpoint was saved from a configuration with classifier-free guidance enabled, which introduces `lm.fake_latent`. Any new finetuning path that continues from the released checkpoint must instantiate the same CFG-related model structure.

## Goals / Non-Goals

**Goals:**
- Add a MotionFix-backed paired-motion dataset path that can be used from `train.py`, `test.py`, and targeted helper scripts.
- Add a new two-motion-to-text finetuning task that reuses the existing motion encoder, VAE latent normalization, and LM text generation stack.
- Keep the implementation incremental by following current config-driven patterns in `configs/`, `motGPT/config.py`, `motGPT/data/`, and `motGPT/models/`.
- Support finetuning from the released `motiongpt3.ckpt` without checkpoint-structure mismatches.
- Provide a clear setup path for local MotionFix data, including split metadata requirements and dataset root configuration.

**Non-Goals:**
- Rebuild MotionFix's original model, losses, or preprocessing pipeline inside MotionGPT3.
- Add a polished Gradio web UI workflow for two-motion comparison in the first implementation.
- Replace MotionGPT3's single-motion task paths or refactor the general LM architecture beyond what is required for paired-motion text generation.
- Solve every possible text-style variant up front; the first implementation may target edit-style or comparison-style supervision as long as the prompt and data are internally consistent.

## Decisions

### 1. Reuse the existing motion feature and motion-token encoding path instead of introducing a new motion backbone

We will encode MotionFix source and target motions separately using the same MotionGPT3 motion feature pipeline and `motion_feats_to_tokens(..., modes='motion')`, then concatenate the resulting token blocks for text generation. This follows the pattern already prototyped in `scripts/compare_two_motions_prompt.py` and avoids introducing a second motion encoder path.

Why this over alternatives:
- Better than reusing `inbetween`, because `inbetween` assumes a single coherent sequence with masked middle frames, not two independent motions.
- Better than introducing a new paired-motion encoder module, because the released checkpoint and current LM already accept concatenated motion placeholders and can be extended incrementally.
- Better than operating directly on MotionFix raw SMPL fields inside the LM, because the rest of MotionGPT3 already assumes motion features are normalized before VAE encoding.

### 2. Introduce a dedicated MotionFix datamodule instead of overloading HumanML3DDataModule

We will add a MotionFix-specific datamodule under `motGPT/data/` that loads `motionfix.pth.tar` plus `splits.json`, extracts paired source/target motions and text, and converts each motion into MotionGPT3-compatible features. The new datamodule will expose paired tensors and lengths rather than trying to coerce MotionFix into the single-motion HumanML3D sample format.

Why this over alternatives:
- Better than patching `HumanML3DDataModule`, because MotionFix samples have two motions, different on-disk structure, and different normalization/statistics needs.
- Better than pre-converting MotionFix into a fake HumanML3D directory tree, because that would hide important paired-sample semantics and make debugging harder.
- It keeps MotionFix-specific setup, validation, and documentation local to one dataset path.

### 3. Add a new two-motion-to-text task path instead of pretending the task is standard `m2t`

We will add a new task class and prompt template for paired-motion-to-text generation. The minimal version will support prompts like `Describe the difference between <Motion_Placeholder_s1> and <Motion_Placeholder_s2>.` and will map source and target motion tokens into a single LM input with stable ordering.

Why this over alternatives:
- Better than overloading `m2t`, because `m2t` currently assumes one motion placeholder and one motion sequence length.
- Better than pure prompt engineering with no code changes, because finetuning requires the training path to understand how paired motion tokens align with paired text supervision.
- It creates an explicit contract for future evaluation, dataset handling, and prompt templates.

### 4. Preserve released-checkpoint compatibility by matching CFG structure during stage-4 finetuning

The new finetuning config will continue using `guidance_scale > 1.0`, `model_guidance_scale > 1.0`, and `fake_latent_mode: learnable_zero` when loading `checkpoints/motiongpt3.ckpt`. This ensures `lm.fake_latent` exists and the model can load strictly from the released checkpoint.

Why this over alternatives:
- Better than switching to `strict=False`, because silent dropping of checkpoint parameters makes continued finetuning harder to reason about.
- Better than reducing guidance to `1.0` at load time, because that changes the instantiated model structure and causes the exact mismatch already observed.

### 5. Require a complete local MotionFix export and validate it explicitly

The dataset setup will require `motionfix.pth.tar` and `splits.json` under a configured MotionFix root. We will treat missing split metadata as a setup error rather than silently guessing train/val/test splits. If motion feature preprocessing needs cached statistics or converted features, those artifacts will be generated into a dedicated MotionFix cache location instead of reusing HumanML3D statistics.

Why this over alternatives:
- Better than inferring splits from sample IDs, because reproducible evaluation requires the original MotionFix split definitions.
- Better than reusing HumanML3D stats, because MotionFix motions are distributed differently and contain paired source/target sequences.

### 6. Stage the work so prompt-only evaluation lands before web UI support

The first implementation will support training/evaluation through `train.py`, `test.py`, and dedicated scripts. Prompt-only or batch evaluation scripts will be the primary debugging surface for paired-motion generation. Gradio integration can be deferred until the data path and task quality stabilize.

Why this over alternatives:
- Better than adding web UI support first, because remote file serving, rendering, and Gradio compatibility are already complex and unrelated to the core learning problem.
- Keeps iteration tight on the task itself and reduces scope for the first change.

## Risks / Trade-offs

- [MotionFix feature mismatch] → MotionFix stores AMASS/SMPL representations rather than existing HumanML3D feature files. Mitigation: define a deterministic conversion step into MotionGPT3-compatible features and cache those outputs with dataset-local statistics.
- [Prompt ambiguity between edit-style and comparison-style text] → MotionFix text often uses imperative edit language rather than neutral comparison language. Mitigation: choose one supervision style explicitly for the first version and document it in config and dataset preparation.
- [Checkpoint compatibility drift] → Changing CFG settings or placeholder layout can break loading from `motiongpt3.ckpt`. Mitigation: keep stage-4 configs aligned with released checkpoint guidance settings and add lightweight load-time verification scripts.
- [GPU memory growth from paired motions] → Encoding two motions increases sequence length and LM context pressure. Mitigation: start with the proven batch size envelope on the current hardware, allow gradient accumulation, and keep the first design based on concatenated tokens rather than larger architectural additions.
- [Dataset incompleteness] → MotionFix local exports may be missing `splits.json` or companion files. Mitigation: validate dataset root contents before training and document the expected layout in README/setup notes.
- [Evaluation uncertainty] → The repo does not yet have a native metric for paired-motion difference text. Mitigation: begin with qualitative scripts and reuse existing text-generation logging, then add task-specific evaluation once the task path stabilizes.
