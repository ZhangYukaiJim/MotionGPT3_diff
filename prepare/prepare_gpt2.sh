#!/usr/bin/env bash
set -euo pipefail

mkdir -p deps
uv run python - <<'PY'
from pathlib import Path

from transformers import AutoTokenizer, GPT2LMHeadModel

model_id = "openai-community/gpt2"
target_dir = Path("deps/gpt2")

model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

target_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(target_dir)
tokenizer.save_pretrained(target_dir)

print(f"Saved GPT-2 files to {target_dir}")
PY
