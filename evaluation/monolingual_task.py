from cla_utils import ALL_LANGUAGES, METRICS, SENT_REPS, REPS_SHORT_NAMES, METRICS_SHORT_NAMES, get_cla_all, get_alignment, df_to_tex
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
import os
from typing import Literal
import json

def load_task_scores(model: str, task: Literal["sib200", "belebele"]) -> dict:
    match task:
        case "sib200":
            with open(os.path.join("scores", "sib200", f"{model.split('/')[1]}.json")) as f:
                return json.load(f)
        case "belebele":
            with open(os.path.join("scores", "belebele", f"{model.split('/')[1]}_loglik.json")) as f:
                whole_dict = json.load(f)
                return {k: v["accuracy"] for k, v in whole_dict.items()}

def main(model: str, tgt_lang: Literal["en", "MEAN"], task: Literal["sib200", "belebele"] = "sib200"):
    ALL_LANGUAGES_NO_EN = ALL_LANGUAGES.copy()
    ALL_LANGUAGES_NO_EN.remove("en")
    corr_table_layer_mean = np.ones((len(SENT_REPS), len(METRICS))) * np.nan

    cla_all = get_cla_all(model.split("/")[1])

    task_scores = load_task_scores(model, task)

    for m_i, metric in enumerate(METRICS):
        for s_i, sent_rep in enumerate(SENT_REPS):
            try:
                current_cla = cla_all[metric][sent_rep]
            except KeyError:
                print("Key error for", metric, sent_rep, file=sys.stderr)
                continue
            
            best_layer = 0
            best_corr = -2
            for layer in range(len(next(current_cla.values().__iter__()))):
                cla = [get_alignment(current_cla, 
                                     lang, tgt_lang, 
                                     exclude_targets=["en"] if tgt_lang == "MEAN" else [], 
                                     layer=layer) 
                                     for lang in ALL_LANGUAGES_NO_EN]
                
                task_scores_ordered = [task_scores[f"{lang}"] for lang in ALL_LANGUAGES_NO_EN]
                corr = pearsonr(cla, task_scores_ordered).correlation
                if corr > best_corr:
                    best_corr = corr
                    best_layer = layer
            corr_table_layer_mean[s_i, m_i] = best_corr
            

    df = pd.DataFrame(corr_table_layer_mean, 
                      index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS], 
                      columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS])

    os.makedirs("evaluation/tables", exist_ok=True)
    with open(f"evaluation/tables/corr_with_{task}_{model.split('/')[1]}_{tgt_lang}.tex", "w") as f:
        f.write(df_to_tex(df, 
                          caption=f"Pearson correlation measured across the \\texttt{{src}} languages of \\texttt{{src-{'en' if tgt_lang == 'en' else '[\\textasciitilde{{}}en]'}}} alignment score and the {'SIB-200 \\(F_1\\) score' if task == 'sib200' else 'Belebele accuracy'} in the \\texttt{{src}} language for {model}. The layer with the highest correlation is selected for each metric and sentence representation.", 
                          label=f"task-{model.split('/')[1]}-{tgt_lang}", 
                          grad_command="\\percentGrad", highlight_max=True))

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model, snakemake.params.tgt_lang, snakemake.params.task)
    else:
        parser = argparse.ArgumentParser(description="Calculate the correlations between src-en and src-~en.")
        parser.add_argument("--model", type=str, help="Model ID.", default="Qwen/Qwen3-14B")
        parser.add_argument("--tgt-lang", type=str, help="Target language.", default="en")
        parser.add_argument("--task", type=str, help="Task name.", default="sib200")

        args = parser.parse_args()
        main(args.model, args.tgt_lang, args.task)