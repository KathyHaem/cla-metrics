#!/bin/bash

set -x
export HF_HOME="/home/alali/personal_work_troja/hf"

MODELS=("meta-llama/Llama-3.2-3B") #"CohereLabs/aya-expanse-8b")  # "CohereLabs/aya-23-8B"

for MODEL in "${MODELS[@]}"; do
  /lnet/troja/work/people/alali/miniconda/envs/torch/bin/python3 save_embeds.py --model "$MODEL" --sent-rep "fewshot"
  /lnet/troja/work/people/alali/miniconda/envs/torch/bin/python3 calc_scores.py --model "$MODEL" --sent-rep "fewshot" --score "cosine" "anc" "dist" "ratio" "nn-abs"
done
