from cla_utils import ALL_LANGUAGES, FULL_MODELS, get_translation_scores, df_to_tex
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os

def get_model_dataset_corr(model: str, dataset: str):
    scores_chrf = get_translation_scores(model.split('/')[1], "chrf", dataset)
    scores_pmi = get_translation_scores(model.split('/')[1], "mutinf", dataset)

    correlations = []
    for tgt in ALL_LANGUAGES:
        chrfs = [scores_chrf[f"{src}-{tgt}"] for src in ALL_LANGUAGES if src != tgt]
        pmis = [scores_pmi[f"{src}-{tgt}"] for src in ALL_LANGUAGES if src != tgt]
        corr = pearsonr(chrfs, pmis).correlation
        correlations.append(corr)


    return np.mean(correlations), np.std(correlations)
    os.makedirs("evaluation/text", exist_ok=True)

    with open(f"evaluation/text/chrf_pmi_corr_{dataset}_{model.split('/')[1]}.txt", "w") as f:
        f.write(f"Mean correlation: {np.mean(correlations)}\n")
        f.write(f"Standard deviation: {np.std(correlations)}\n")

def main():
    df = pd.DataFrame(index=["flores", "bouquet"], 
                columns=range(len(FULL_MODELS)))
    for m, model in enumerate(FULL_MODELS):
        for dataset in ["flores", "bouquet"]:
            df.loc[dataset, m] = get_model_dataset_corr(model, dataset)[0]
    os.makedirs("evaluation/tables", exist_ok=True)
    with open("evaluation/tables/chrf_pmi_corr.tex", "w") as f:
        f.write(
            df_to_tex(df, "", "", grad_command="\\percentGrad", highlight_max=True, cells_only=True, eflomal_in_last_col=False)
        )


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        #main(snakemake.params.model, snakemake.params.dataset)
        main()
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Calculate the correlation between chrF and PMI translation scores.")
        # parser.add_argument("--model", type=str, help="Model ID.", default="ALL")
        # parser.add_argument("--dataset", type=str, help="Dataset to use for evaluation.", choices=["flores", "bouquet"], default="ALL")

        args = parser.parse_args()
        # main(args.model, args.dataset)
        main()