# Predicting Multilingual Classification and Translation Performance of LLMs with Cross-Lingual Alignment – Is English Enough?

This is the repository for the article *Predicting Multilingual Classification and Translation Performance of LLMs with Cross-Lingual Alignment – Is English Enough?*

# Running

This project is managed with [Snakemake](https://snakemake.github.io/), which uses [Conda](https://www.anaconda.com/docs/main) to manage environments. The repository is designed to be run on a [slurm](https://slurm.schedmd.com/overview.html) GPU cluser. To run all the experiments:

1. Inspect the [`Snakefile`](Snakefile) and change lines 12–15 to suit your cluster configuration.
1. Install [miniconda](https://www.anaconda.com/docs/getting-started/installation).
1. Run:

```
conda create -n snakemake python=3.12
conda activate snakemake
pip install -r requirements.txt
./snakemake.sh
```

Note that generating tables and figures is not fully automated using Snakemake. Use the scripts in [`evaluation/`](evaluation/). Manual edits in the tables may be required.