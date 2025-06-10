#!/bin/bash

set -x

MODELS=("meta-llama/Llama-3.2-3B" "CohereLabs/aya-expanse-8b")  # "CohereLabs/aya-23-8B"

for MODEL in "${MODELS[@]}"; do
  python3 save_embeds.py --model "$MODEL"
  python3 calc_scores.py --model "$MODEL" --score "cosine" "anc" "dist" "ratio" "nn_abs"
done
