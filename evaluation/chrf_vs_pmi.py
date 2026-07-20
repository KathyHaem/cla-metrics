from cla_utils import ALL_LANGUAGES, get_translation_scores
from scipy.stats import pearsonr
import numpy as np
import os

def main(model: str, dataset: str):
    scores_chrf = get_translation_scores(model.split('/')[1], "chrf", dataset)
    scores_pmi = get_translation_scores(model.split('/')[1], "mutinf", dataset)

    correlations = []
    for tgt in ALL_LANGUAGES:
        chrfs = [scores_chrf[f"{src}-{tgt}"] for src in ALL_LANGUAGES if src != tgt]
        pmis = [scores_pmi[f"{src}-{tgt}"] for src in ALL_LANGUAGES if src != tgt]
        corr = pearsonr(chrfs, pmis).correlation
        correlations.append(corr)

    os.makedirs("evaluation/text", exist_ok=True)

    with open(f"evaluation/text/chrf_pmi_corr_{dataset}_{model.split('/')[1]}.txt", "w") as f:
        f.write(f"Mean correlation: {np.mean(correlations)}\n")
        f.write(f"Standard deviation: {np.std(correlations)}\n")

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model, snakemake.params.dataset)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Calculate the correlation between chrF and PMI translation scores.")
        parser.add_argument("--model", type=str, help="Model ID.", default="google/gemma-3-12b-pt")
        parser.add_argument("--dataset", type=str, help="Dataset to use for evaluation.", choices=["flores", "bouquet"], default="flores")

        args = parser.parse_args()
        main(args.model, args.dataset)