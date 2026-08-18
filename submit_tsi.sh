#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -J tsi_scores
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -o xtrace
set -e

# --- Configuration ---

TSI_MODELS=("Qwen/Qwen3-14B-Base" "google/gemma-3-12b-pt")
SENT_REPS=("mean" "weighted-mean" "prompt" "last-token" "fewshot")
TSI_SCORES=("tsi" "tsi-cosine")

# --- Step 2: save_embeds ---
# This runs for every model and every sentence representation
for full_model in "${TSI_MODELS[@]}"; do
    for rep in "${SENT_REPS[@]}"; do
        echo "Generating embeds for $full_model ($rep)..."
        uv run python3 save_embeds.py \
            --model "$full_model" \
            --sent-rep "$rep" \
            --dataset "flores" \
            --batch-size 16
    done
done

# --- Step 3: calc_scores ---
# This runs for every model & representation
for full_model in "${TSI_MODELS[@]}"; do
    for rep in "${SENT_REPS[@]}"; do
          echo "Calculating scores for $full_model ($rep)..."
            uv run python3 calc_scores.py \
                --model "$full_model" \
                --sent-rep "$rep" \
                --score "${TSI_SCORES[@]}" \
                --dataset "flores"
        done
    done
done

echo "TSI workflow complete."
