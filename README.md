<div align= "center">
    <h1> Official repo for MotionGPT3 </h1>

</div>

<div align="center">
    <h2> <a href="https://motiongpt3.github.io">MotionGPT3: Human Motion as a Second Modality</a></h2>

<p align="center">
  <a href="https://motiongpt3.github.io">Project Page</a> •
  <a href="https://arxiv.org/abs/2506.24086 ">Arxiv Paper</a> •
  <a href="#-citation">Citation
</p>

</div>

<div align="center">

</div>


## 🏃 Intro MotionGPT3

MotionGPT3 is a **bimodal** motion-language framework using MoT architecture designed to address the challenges of **unified** motion understanding and generation.

<details open>
    <summary><b>Technical details</b></summary>

<p style="margin-bottom: 4px;">
Inspired by the mixture of experts, we propose MotionGPT3, a bimodal motion-language model that treats human motion as a second modality, decoupling motion modeling via separate model parameters and enabling both effective cross-modal interaction and efficient multimodal scaling training. 
</p>
<p style="margin-bottom: 4px;">
To preserve language intelligence, the text branch remains the same with the pretrained language model, while a motion branch is integrated via shared attention, enabling bidirectional information flow between two modalities. We employ a motion VAE to encode raw human motion into latent representations, while motion branch predicts motion latents directly from intermediate hidden states using a diffusion head, bypassing discrete tokenization. 
</p>
<p>
Extensive experiments show that our approach achieves competitive performance on both motion understanding and generation tasks while preserving strong language capabilities, establishing a unified bimodal motion diffusion framework within an autoregressive manner.
</p>

<img width="1194" alt="pipeline" src="./assets/images/training.png">
</details>

## 🚩 News

- [2025/10/17] 🔥🔥 Release **MotionGPT3 models** on [huggingface](https://huggingface.co/OpenMotionLab/motiongpt3)
- [2025/06/30] Upload and init project

## ⚡ Quick Start

<details open>
  <summary><b>Setup and download</b></summary>

### 1. Python environment with `uv`

We test our code on Python 3.11 and PyTorch 2.0.0.

```
uv sync
uv run python -m spacy download en_core_web_sm
```

If you want to run the web UI or rendering tools, install the optional extras too:

```
uv sync --extra webui --extra render
```

The web UI currently uses `gradio==5.49.1` for compatibility with this repo.
The `render` extra adds the legacy SMPL rendering dependencies used by the slow visualization path.

### 2. Dependencies

Run the script to download dependency materials:

```
bash prepare/download_smpl_model.sh
bash prepare/prepare_gpt2.sh
```

`prepare/prepare_gpt2.sh` now downloads GPT-2 directly through `transformers`, so `git-lfs` is not required.

For Text to Motion Evaluation

```
bash prepare/download_t2m_evaluators.sh
```

This archive extracts under `deps/t2m/` and provides the evaluator checkpoints and glove assets used by the HumanML3D/KIT pipelines.

For pre-trained MotionVAE:

```
bash prepare/download_mld_pretrained_models.sh
```

Then run the following script to process checkpoints:
```
uv run python -m scripts.gen_mot_gpt
```

### 3. Pre-trained model

Run the script to download the pre-trained model

```
bash prepare/download_pretrained_motiongpt3_model.sh
```

This script saves the web UI checkpoint to `checkpoints/motiongpt3.ckpt`.

### 4. (Optional) Download manually

Visit [the Google Driver](https://drive.google.com/drive/folders/1NMDuI2F0UO2Opl778C37DWCZdHcy5DOh?usp=drive_link) to download the previous dependencies.

Visit [the Hugging Face](https://huggingface.co/OpenMotionLab/motiongpt3) to download the pretrained models.

</details>

## ▶️ Demo

<details open>
  <summary><b>Webui</b></summary>

Run the following script to launch webui, then visit [0.0.0.0:8888](http://0.0.0.0:8888)

```
uv run --extra webui python app.py
```

If you want to use the slow SMPL renderer in the web UI, launch it with both extras:

```bash
uv run --extra webui --extra render python app.py
```

Before launching the web UI, make sure you have completed all of the following:

```bash
uv sync --extra webui
bash prepare/download_smpl_model.sh
bash prepare/prepare_gpt2.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/download_pretrained_motiongpt3_model.sh
uv run python -m scripts.gen_mot_gpt
```

For slow SMPL visualization, install the render extra too:

```bash
uv sync --extra webui --extra render
```

The default Whisper backend is `openai/whisper-large-v2`. On the first run, the model may be downloaded from Hugging Face and startup can take a while.

On remote Linux servers without X11, the app uses headless EGL for the slow renderer by default.

If `uv run` prints a `VIRTUAL_ENV` mismatch warning, it means another virtual environment is currently active. `uv` will still use this project's `.venv`, but you can run `deactivate` first to avoid the warning.

### MotionFix Finetuning

The repository now includes an experimental MotionFix preprocessing and paired-motion finetuning path for a two-motion difference-to-text task.

Expected local MotionFix layout:

```bash
/opt/data/motionfix-dataset/
  motionfix.pth.tar
  splits.json
```

The preprocessing step converts MotionFix source/target motions into the same HumanML-compatible feature representation used by the pretrained MotionGPT3 VAE, and reuses the bundled HumanML normalization assets in `assets/meta/`.

Build the default MotionFix cache:

```bash
uv run python scripts/preprocess_motionfix.py \
  --motionfix-root /opt/data/motionfix-dataset
```

This writes a cache under:

```bash
/opt/data/motionfix-dataset/mgpt_humanml_cache/
  manifests/{train,val,test}.json
  source/*.npy
  target/*.npy
  summary.json
```

If `splits.json` is missing, preprocessing fails fast with a setup error instead of guessing splits.

Run a MotionFix finetuning smoke test:

```bash
uv run python -m train \
  --cfg configs/MoT_vae_stage4_motionfix_yzh.yaml \
  --cfg_assets configs/assets_humanml3d_yzh.yaml \
  --nodebug \
  --batch_size 1
```

Resume a previous MotionFix finetuning run:

```bash
uv run python -m train \
  --cfg configs/MoT_vae_stage4_motionfix_yzh.yaml \
  --cfg_assets configs/assets_humanml3d_yzh.yaml \
  --nodebug \
  --wandb on
```

Then set `TRAIN.RESUME` in the config to either the experiment directory or a checkpoint path, for example:

```yaml
TRAIN:
  RESUME: experiments/motgpt/debug--MoT_vae_stage4_motionfix_yzh
```

Prompt-only paired-motion generation is available without Gradio:

```bash
uv run python scripts/compare_two_motions_prompt.py \
  --motion-a /opt/data/motionfix-dataset/mgpt_humanml_cache/source/000000.npy \
  --motion-b /opt/data/motionfix-dataset/mgpt_humanml_cache/target/000000.npy \
  --max-length 40
```

Export qualitative `m2t_diff` test artifacts through `test.py`:

```yaml
TEST:
  SAVE_PREDICTIONS: true
  VISUALIZE: true
```

```bash
uv run python -m test \
  --cfg configs/MoT_vae_stage4_motionfix_yzh_resume_debug.yaml \
  --cfg_assets configs/assets_humanml3d_yzh.yaml \
  --batch_size 8
```

This writes test outputs under `results/<model>/<NAME>/samples_<TIME>/`, including:
- `<id>.txt` for the predicted caption
- `<id>_gt.txt` for ground-truth caption text
- `<id>_source.mp4`, `<id>_target.mp4`, and `<id>_source_target.mp4` when `TEST.VISUALIZE: true`

Notes:

- The current MotionFix path is script and training oriented; it does not add a Gradio web UI workflow.
- The paired-motion task uses `m2t_diff` and continues from `checkpoints/motiongpt3.ckpt` with guidance settings compatible with `lm.fake_latent`.
- The first implementation uses HumanML normalization intentionally, so the pretrained HumanML VAE can be reused without changing its expected input distribution.
- Validation visualizations now default to the first `VIS_NUM` items from the first validation batch. Set `VAL_VIS_IDS` in config to a list of dataset ids if you want a fixed curated subset instead.
- `TEST.SAVE_PREDICTIONS` controls raw test artifact export, while `TEST.VISUALIZE` controls MotionFix `m2t_diff` video materialization during `test.py`.
- Recommended `m2t_diff` evaluation uses `METRIC.TYPE: [M2TDiffMetrics]`, which reports `Bert_F1`, `ROUGE_L`, `CIDEr`, `Bleu_1`, `Bleu_4`, `Empty_output_rate`, and `Avg_generated_length`.
- These `Metrics/*` values are logged through Lightning and will appear in TensorBoard and WandB when those loggers are enabled.
- `Bert_F1` uses a TorchMetrics-based BERTScore path with local device-placement fixes for MotionFix evaluation. Validated configs can use `METRIC.M2T_DIFF.BERT_DEVICE: cpu` or `cuda:0`; `METRIC.M2T_DIFF.BERT_IDF` defaults to `true` and remains configurable alongside `INCLUDE_BERT_F1` and `INCLUDE_BLEU`.
- `Matching_score` and `R_precision_top_k` remain excluded from `m2t_diff` because they assume a single motion-text pairing rather than a caption about the relation between two motions.
- Pair-aware motion-text retrieval is still future work; the first rollout only ships text-generation metrics plus decoding diagnostics.

Logger controls:

- `DEBUG: true` now mainly changes the run name prefix and validation cadence; it does not disable metrics.
- `LOGGER.ENABLE_TENSORBOARD` and `LOGGER.ENABLE_WANDB` control whether each logger is attached.
- `--wandb on|off|offline` can override WandB mode from the CLI.

</details>

<details open>
  <summary><b>Batch demo</b></summary>

We support txt file input, the output motions are npy files and output texts are txt files. Please check the `configs/assets.yaml` for path config, TEST.FOLDER as output folder.

Then, run the following script:

```
uv run python demo.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt
```

Some parameters:

- `--example=./demo/t2m.txt`: input file as text prompts
- `--task=t2m`: evaluation tasks including t2m, m2t, pred, inbetween

The outputs:

- `npy file`: the generated motions with the shape of (nframe, 22, 3)
- `txt file`: the input text prompt or text output
</details>

## 💻 Train your own models

<details open>
  <summary><b>Training guidance</b></summary>

### 1. Prepare the datasets

1. Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup.

2. Put the instructions data in `prepare/instructions` to the same folder of HumanML3D dataset.

4. (Optional) Refer to [MotionGPT-Training guidance](https://github.com/OpenMotionLab/MotionGPT/tree/main#22-ready-to-pretrain-motiongpt-model) to generate motion code for VQ-based training.
    ```
    bash prepare/download_motiongpt_pretrained_models.sh
    uv run python -m scripts.get_motion_code --cfg configs/config_motiongpt.yaml
    ```


### 2.1. Ready to train MotionGPT3 model

Please first check the parameters in `configs/MoT_vae_stage1_t2m.yaml`, e.g. `NAME`, `instruction_type`, `lm_ablation`, `DEBUG`.

Then, run the following command:

```
uv run python -m scripts.gen_mot_gpt
uv run python -m train --cfg configs/MoT_vae_stage1_t2m.yaml --nodebug
```

### 2.2. Ready to pretrain MotionGPT3 model

Please update the parameters in `configs/MoT_vae_stage2_instruct.yaml` and `configs/MoT_vae_stage2_all.yaml`, e.g. `NAME`, `instruction_type`, `lm_ablation`, `DEBUG`, `PRETRAINED_VAE`(change to your `latest ckpt model path` in previous step)


Then, run the following command:
```
uv run python -m train --cfg configs/MoT_vae_stage2_all.yaml --nodebug
uv run python -m train --cfg configs/MoT_vae_stage2_instruct.yaml --nodebug
```

### 2.3. Ready to instruct-tuning MotionGPT3 model

Please update the parameters in `configs/MoT_vae_stage3.yaml`, e.g. `NAME`, `instruction_type`, `lm_ablation`, `DEBUG`, `PRETRAINED` (change to your `latest ckpt model path` in previous step)

Then, run the following command:

```
uv run python -m train --cfg configs/MoT_vae_stage3.yaml --nodebug
```

### 3. Evaluate the model

Please first put the tained model checkpoint path to `TEST.CHECKPOINT` in config files, e.g. `configs/MoT_vae_stage3.yaml`.

Then, run the following command:

```
uv run python -m test --cfg configs/MoT_vae_stage3.yaml --task t2m
```

Some parameters:

- `--task`: evaluation tasks including t2m(Text-to-Motion), m2t(Motion translation), pred(Motion prediction), inbetween(Motion inbetween)

<!-- Due to the python package conflit, the released implement of linguistic metrics in motion translation task is by [nlg-metricverse](https://github.com/disi-unibo-nlp/nlg-metricverse), which may not be consistent to the results implemented by [nlg-eval](https://github.com/Maluuba/nlg-eval). We will fix this in the future. -->

</details>

## 👀 Visualization

<details open>
  <summary><b>Render SMPL</b></summary>

### 1. Set up blender - WIP

Refer to [TEMOS-Rendering motions](https://github.com/Mathux/TEMOS) for blender setup, then install the following dependencies.

```
YOUR_BLENDER_PYTHON_PATH/python -m pip install -r prepare/requirements_render.txt
```

### 2. (Optional) Render rigged cylinders

Run the following command using blender:

```
YOUR_BLENDER_PATH/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=YOUR_NPY_FOLDER --mode=video
```

### 2. Create SMPL meshes with:

```
uv run python -m fit --dir YOUR_NPY_FOLDER --save_folder TEMP_PLY_FOLDER --cuda
```

This outputs:

- `mesh npy file`: the generate SMPL vertices with the shape of (nframe, 6893, 3)
- `ply files`: the ply mesh file for blender or meshlab

### 3. Render SMPL meshes

Run the following command to render SMPL using blender:

```
YOUR_BLENDER_PATH/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=YOUR_NPY_FOLDER --mode=video
```

optional parameters:

- `--mode=video`: render mp4 video
- `--mode=sequence`: render the whole motion in a png image.
</details>


## 📖 Citation

If you find our code or paper helps, please consider citing:

```bibtex
  @misc{zhu2025motiongpt3humanmotionsecond,
    title={MotionGPT3: Human Motion as a Second Modality}, 
    author={Bingfan Zhu and Biao Jiang and Sunyi Wang and Shixiang Tang and Tao Chen and Linjie Luo and Youyi Zheng and Xin Chen},
    year={2025},
    eprint={2506.24086},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.24086}, 
  }
```

## Acknowledgments

Thanks to [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), [Motion-latent-diffusion](https://github.com/ChenFengYe/motion-latent-diffusion), [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and [MAR](https://github.com/LTH14/mar), our code is partially borrowing from them.

## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.
