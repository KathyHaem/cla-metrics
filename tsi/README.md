# TSI

## 1. Compute scores

```bash
sbatch snakemake_tsi.sh
```

Targets the `tsi` rule (`Snakefile`), producing `scores/flores/{Qwen3-14B-Base,gemma-3-12b-pt}_{sent_rep}_tsi.json`
for all `SENT_REPS`. Add models by extending `TSI_MODELS` in `Snakefile`.

## 2. Correlate against downstream tasks

Run from `evaluation/`:

```bash
cd evaluation

# sib-200 / belebele
python3 monolingual_task_corr.py --model "Qwen/Qwen3-14B-Base" --task sib-200  --tgt-lang en
python3 monolingual_task_corr.py --model "Qwen/Qwen3-14B-Base" --task belebele --tgt-lang en
python3 monolingual_task_corr.py --model "google/gemma-3-12b-pt" --task sib-200  --tgt-lang en
python3 monolingual_task_corr.py --model "google/gemma-3-12b-pt" --task belebele --tgt-lang en

# translation - chrF
python3 chrf_per_target_corr.py --model "Qwen/Qwen3-14B-Base"
python3 chrf_per_target_corr.py --model "google/gemma-3-12b-pt"

# translation - PMI
python3 pmi_corr.py --model "Qwen/Qwen3-14B-Base"
python3 pmi_corr.py --model "google/gemma-3-12b-pt"
```

TSI shows up automatically as an extra column (`cla_utils.METRICS` now includes `"tsi"`).
