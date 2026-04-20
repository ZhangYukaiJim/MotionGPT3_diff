# MotionFix Overfit Blank Text Diagnosis

## Context

Experiment:
- `experiments/motgpt/debug--MoT_vae_stage4_motionfix_overfit1batch`

Config:
- `configs/MoT_vae_stage4_motionfix_overfit1batch.yaml`

Task:
- `m2t_diff`

Question investigated:
- Why does the model appear to stop outputting text during overfit training on the MotionFix motion-comparison task?

## Short Answer

The strongest current explanation is a paired-motion supervision bug, not a motion-input encoding bug.

The current `m2t_diff` training path appears to supervise the model with an empty output string after the paired-motion prompt. That would train the LM to emit EOS / no text after the prompt, which matches the observed blank validation predictions.

## Main Evidence

### 1. Saved validation predictions are blank early, not only late

Checked files under:
- `experiments/motgpt/debug--MoT_vae_stage4_motionfix_overfit1batch/validate_motion/`

Examples:
- `epoch_19/000362.txt`: blank
- `epoch_19/001515.txt`: blank
- `epoch_44/000362.txt`: blank
- `epoch_103/001515.txt`: blank
- `epoch_286/000362.txt`: blank
- `epoch_286/001515.txt`: blank

Ground-truth text files are non-empty, for example:
- `epoch_19/000362_gt.txt`: `raise your left hand higher and wave wider in the end`
- `epoch_271/001515_gt.txt`: `do the circular motion slower and only once`

This suggests the model was already producing blank outputs fairly early in the run, rather than gradually collapsing only late in training.

### 2. The MotionFix dataset task template uses an empty output

In `motGPT/data/MotionFix.py` the dataset returns:

```python
task = {
    "class": "m2t_diff",
    "input": [
        "Describe the difference between <Motion_Placeholder_s1> and <Motion_Placeholder_s2> in detail."
    ],
    "output": [""],
}
```

This differs from the standard single-motion `m2t` setup, where the output template is the caption placeholder.

Relevant references:
- `prepare/instructions/template_m2t_pretrain.json`
- `prepare/instructions/template_witht2t_instructions.json`

Those use:

```json
"output": ["<Caption_Placeholder>"]
```

### 3. The paired-motion decoder path appears to train directly on that empty output

In `motGPT/archs/motgpt_lm.py`, `forward_dec_two_motion()` builds labels as:

```python
labels.append(
    inputs[i] + " \n " + outputs[i] + self.tokenizer.eos_token
)
```

Since `outputs[i]` comes from the MotionFix task definition and is currently `""`, the supervised text target is effectively:

- paired-motion prompt
- newline
- immediate EOS

That is fully consistent with the model learning to output no text after the prompt.

## Assessment Of Input Encoding

The paired-motion input encoding does not currently look like the primary failure.

Relevant code:
- `motGPT/archs/motgpt_lm.py:392-412` `encode_two_motion_tokens()`
- `motGPT/archs/motgpt_lm.py:380-390` `build_two_motion_prompt()`
- `motGPT/archs/motgpt_lm.py:1406-1481` `forward_dec_two_motion()`

What it does:
- encodes source motion and target motion separately
- concatenates the two token blocks in stable order
- inserts two input motion placeholders into the prompt
- replaces those placeholder positions with the motion embeddings

That path may still deserve a sanity check, but it is less likely to explain consistent blank outputs than the empty-output supervision bug.

## Additional Observation

The current `overfit1batch` setup is not actually a 2-sample overfit setup anymore.

The current file:
- `/opt/data/motionfix-dataset/overfit_splits.json`

contains 16 ids in each of `train`, `val`, and `test`.

The cached overfit manifest also reflects those 16 samples:
- `/opt/data/motionfix-dataset/mgpt_humanml_cache_overfit/manifests/train.json`

So this run is:
- one fixed training batch per epoch
- batch size 16
- not the earlier 2-sample setup

This is probably not the root cause of blank text, but it matters for interpreting the experiment.

## Most Likely Explanations

1. `m2t_diff` training supervision is wrong.
2. The decoder is learning immediate EOS / empty response because that is what the labels imply.
3. Deterministic generation in validation (`do_sample=False`) then makes the blank behavior stable and obvious.
4. Motion-pair input encoding issues are possible but currently less likely than the supervision bug.

## Recommended Probes

### Highest priority

1. Inspect one batch of `labels` strings inside `forward_dec_two_motion()`.
2. Count how many non-ignored target tokens remain after masking in `forward_dec_two_motion()`.
3. Check first-step generation logits for EOS during validation.

### Good controls

1. Compare overfit behavior against standard single-motion `m2t` on a tiny batch.
2. Run an even smaller sanity test on one sample only.
3. Confirm the fully expanded paired-motion prompt string looks as expected.

## Likely Fix Direction

If the intention is to train paired-motion-to-text generation analogously to `m2t`, then the likely fix is:

- keep the paired-motion input prompt
- change the `m2t_diff` output template from `""` to `"<Caption_Placeholder>"`

That would make the decoder explicitly learn the textual description of the difference, rather than learn to stop after the prompt.

## Relevant Files

- `configs/MoT_vae_stage4_motionfix_overfit1batch.yaml`
- `motGPT/data/MotionFix.py`
- `motGPT/archs/motgpt_lm.py`
- `prepare/instructions/template_m2t_pretrain.json`
- `prepare/instructions/template_witht2t_instructions.json`
- `experiments/motgpt/debug--MoT_vae_stage4_motionfix_overfit1batch/validate_motion/`
- `/opt/data/motionfix-dataset/overfit_splits.json`
- `/opt/data/motionfix-dataset/mgpt_humanml_cache_overfit/manifests/train.json`
