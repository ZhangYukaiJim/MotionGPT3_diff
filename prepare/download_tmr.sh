#!/usr/bin/env bash
set -euo pipefail

mkdir -p deps/tmr

echo "The TMR pretrained model will be stored in './deps/tmr'"

tmp_archive="tmr_models.tgz"

echo "Downloading"
uv run gdown --no-cookies --fuzzy --output "$tmp_archive" "https://drive.google.com/file/d/1n6kRb-d2gKsk8EXfFULFIpaUKYcnaYmm/view?usp=sharing"

echo "Extracting"
tar xfzv "$tmp_archive"

echo "Installing"
rm -rf deps/tmr/tmr_humanml3d_guoh3dfeats
mv models/tmr_humanml3d_guoh3dfeats deps/tmr/

echo "Cleaning"
rm -f "$tmp_archive"
rm -rf models

echo "Downloading done!"
