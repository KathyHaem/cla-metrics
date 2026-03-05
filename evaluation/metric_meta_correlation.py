from cla_utils import ALL_LANGUAGES, METRICS, SENT_REPS, REPS_SHORT_NAMES, METRICS_SHORT_NAMES, get_cla_all, get_alignment, df_to_tex
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
import os


def main(model: str):
    ALL_LANGUAGES_NO_EN = ALL_LANGUAGES.copy()
    ALL_LANGUAGES_NO_EN.remove("en")
    corr_table_layer_mean = np.ones((len(SENT_REPS), len(METRICS))) * np.nan

    cla_all = get_cla_all(model.split("/")[1])

    for m_i, metric in enumerate(METRICS):
        for s_i, sent_rep in enumerate(SENT_REPS):
            try:
                current_cla = cla_all[metric][sent_rep]
            except KeyError:
                print("Key error for", metric, sent_rep, file=sys.stderr)
                continue

            cla_en = [get_alignment(current_cla, lang, "en") for lang in ALL_LANGUAGES_NO_EN]
            cla_rest = [get_alignment(current_cla, lang, "MEAN", exclude_targets=["en"]) for lang in ALL_LANGUAGES_NO_EN]
            corr_table_layer_mean[s_i, m_i] = pearsonr(cla_en, cla_rest).correlation

            if np.isnan(pearsonr(cla_en, cla_rest).correlation):
                print("Something is wrong")
            

    df = pd.DataFrame(corr_table_layer_mean, 
                      index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS], 
                      columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS])

    os.makedirs("evaluation/tables", exist_ok=True)
    with open(f"evaluation/tables/corr_with_cla_en_{model.split('/')[1]}.tex", "w") as f:
        f.write(df_to_tex(df, 
                          caption=f"Pearson correlation measured across the \\texttt{{src}} languages between the \\texttt{{src-en}} and \\texttt{{src-[\\textasciitilde{{}}en]}} alignment scores for {model.split('/')[1]}. Values are displayed as per cent.",
                          label=f"corr-{model.split('/')[1]}", 
                          grad_command="\\percentGrad"))

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model)
    else:
        parser = argparse.ArgumentParser(description="Calculate the correlations between src-en and src-~en.")
        parser.add_argument("--model", type=str, help="Model ID.", default="Qwen/Qwen3-14B")

        args = parser.parse_args()
        main(args.model)