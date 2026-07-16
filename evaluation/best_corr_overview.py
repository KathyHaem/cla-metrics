import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
import os
from typing import Literal
from cla_utils import ALL_LANGUAGES, METRICS, SENT_REPS, REPS_SHORT_NAMES, METRICS_SHORT_NAMES, FULL_MODELS, get_cla_all, get_alignment, df_to_tex
from monolingual_task_corr import load_task_scores
from task_results import SHORT_MODELS_DICT, SHORT_MODELS
import argparse

ROWS = ["correlation", "representation", "CLA metric"]

def main(tgt_lang: Literal["en", "MEAN"],
         task: Literal["sib-200", "belebele"] = "sib-200",
         layer_pooling: Literal["MEAN", "HIGHEST", "best"] = "HIGHEST"):
    ALL_LANGUAGES_NO_EN = ALL_LANGUAGES.copy()
    ALL_LANGUAGES_NO_EN.remove("en")

    best_corrs = pd.DataFrame(index=ROWS, columns=SHORT_MODELS)
    for model_idx, model in enumerate(FULL_MODELS):
        cla_all = get_cla_all(model.split("/")[1])

        task_scores = load_task_scores(model, task)

        best_corr = -2
        for m_i, metric in enumerate(METRICS):
            for s_i, sent_rep in enumerate(SENT_REPS):
                try:
                    current_cla = cla_all[metric][sent_rep]
                except KeyError:
                    print("Key error for", metric, sent_rep, file=sys.stderr)
                    continue

                cla = [get_alignment(current_cla,
                                     lang, tgt_lang,
                                     exclude_targets=["en"] if tgt_lang == "MEAN" else [],
                                     layer=layer_pooling)
                       for lang in ALL_LANGUAGES_NO_EN]

                task_scores_ordered = [task_scores[f"{lang}"] for lang in ALL_LANGUAGES_NO_EN]
                current_corr = pearsonr(cla, task_scores_ordered).correlation
                if current_corr > best_corr:
                    best_corr = current_corr
                    best_corrs[SHORT_MODELS_DICT[model]] = [best_corr, REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)]

    os.makedirs("evaluation/tables", exist_ok=True)
    with open(f"evaluation/tables/overview_corr_with_{task}_{tgt_lang}.tex", "w") as f:
        f.write(df_to_tex(best_corrs,
                          caption=f"Overview of the best Pearson correlation measured across the \\texttt{{src}} languages of \\texttt{{src-{'en' if tgt_lang == 'en' else '[\\textasciitilde{{}}en]'}}} alignment score and the {'\\textbf{{SIB-200}} \\(F_1\\) score' if task == 'sib-200' else '\\textbf{{Belebele}} accuracy'} in the \\texttt{{src}} language for all models. Values are displayed as per cent.",
                          label=f"overview-corr-{tgt_lang}-task-{task}",
                          highlight_max=False,
                          heatmap=False,
                          eflomal_in_last_col=False))


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake

        main(snakemake.params.tgt_lang, snakemake.params.task)
    else:
        parser = argparse.ArgumentParser(description="Generate a LaTeX table comparing the best correlation between alignment scores and task performance across different models for a given task and target language.")
        parser.add_argument("--tgt-lang", type=str, help="Target language.", default="en")
        parser.add_argument("--task", type=str, help="Task name.", choices=["sib-200", "belebele"], default="sib-200")

        args = parser.parse_args()
        main(args.tgt_lang, args.task)