from pmi_corr import calculate_table
from cla_utils import FULL_MODELS, SHORT_MODELS, METRICS, SENT_REPS, REPS_SHORT_NAMES, METRICS_SHORT_NAMES, df_to_tex
import pandas as pd
import numpy as np
import argparse
import os

def main(layer_pooling: str = "HIGHEST", metric: str = "pmi"):
    all_tables = []
    for model in FULL_MODELS:
        corr_table, gammas_table = calculate_table(model, layer_pooling=layer_pooling, no_print=True, metric=metric)
        all_tables.append((corr_table, gammas_table))

    df = pd.DataFrame(index=["corr. combined", "\\texttt{src-tgt} weight", "representation", "CLA metric"], columns=SHORT_MODELS)
    for model_idx, model in enumerate(SHORT_MODELS):
        corr_table, gammas_table = all_tables[model_idx]
        best_idx = np.unravel_index(np.nanargmax(corr_table), corr_table.shape)
        df.loc["corr. combined", model] = corr_table[best_idx]
        df.loc["\\texttt{src-tgt} weight", model] = gammas_table[best_idx]
        df.loc["representation", model] = REPS_SHORT_NAMES.get(SENT_REPS[best_idx[0]], SENT_REPS[best_idx[0]])
        df.loc["CLA metric", model] = METRICS_SHORT_NAMES.get(METRICS[best_idx[1]], METRICS[best_idx[1]])
    
    os.makedirs("evaluation/tables", exist_ok=True)
    with open(f"evaluation/tables/corr_pmi_overview_{layer_pooling}_{metric}.tex", "w") as f:
        f.write(df_to_tex(
            df, 
            f"Overview of the best correlations between the optimal convex combination of \\texttt{{src-en}}, \\texttt{{en-tgt}}, and \\texttt{{src-tgt}} and the translation {'PMI' if metric == 'pmi' else '\\textsc{chrF}'} scores. Values are shown as per cent.", f"corrPmiOverview{layer_pooling}_{metric}", 
            highlight_max=False, 
            eflomal_in_last_col=False, 
            heatmap=True, 
            grad_command="\\percentGrad"))

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.layer_pooling)
    else:
        parser = argparse.ArgumentParser(description="Generate a LaTeX table comparing the best correlations between the optimal convex combination of \\texttt{src-en}, \\texttt{en-tgt}, and \\texttt{src-tgt} and the translation PMI for different models.")
        parser.add_argument("--layer_pooling", type=str, help="Layer pooling method to use for the alignments.", choices=["MEAN", "HIGHEST", "best"], default="HIGHEST")
        parser.add_argument("--metric", type=str, help="Task metric to use for the PMI scores.", choices=["translation", "pmi"], default="pmi")

        args = parser.parse_args()
        main(args.layer_pooling, args.metric)