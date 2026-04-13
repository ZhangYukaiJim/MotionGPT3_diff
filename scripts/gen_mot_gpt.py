import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

MODEL_ID = "openai-community/gpt2"
MODEL_DIR = Path("deps/gpt2")
TARGET_DIR = Path("deps/mot-gpt2")


def load_gpt2():
    try:
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        return model, tokenizer
    except Exception as exc:
        print(
            f"Local GPT-2 files in {MODEL_DIR} are unusable ({exc}). Downloading {MODEL_ID} instead."
        )
        model = GPT2LMHeadModel.from_pretrained(MODEL_ID).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        return model, tokenizer


model, tokenizer = load_gpt2()

state_dict = model.state_dict()
new_state_dict = dict()
for name, param in state_dict.items():
    name = name.replace("attn.c_attn", "c_attn.fn.0")
    name = name.replace("attn.c_proj", "c_proj.fn.0")
    name = name.replace("mlp", "mlp.fn.0")
    name = name.replace("ln_f", "ln_f.fn.0")
    name = name.replace("ln_1", "ln_1.fn.0")
    name = name.replace("ln_2", "ln_2.fn.0")
    new_state_dict[name] = param

os.makedirs(TARGET_DIR, exist_ok=True)
torch.save(new_state_dict, TARGET_DIR / "model_state_dict.pth")
model.config.to_json_file(TARGET_DIR / "config.json")
model.generation_config.to_json_file(TARGET_DIR / "generation_config.json")
tokenizer.save_pretrained(TARGET_DIR)
