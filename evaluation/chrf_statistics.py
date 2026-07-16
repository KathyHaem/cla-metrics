from monolingual_task_corr import load_task_scores
from cla_utils import ALL_LANGUAGES, FULL_MODELS, SHORT_MODELS, df_to_tex
import pandas as pd
import os
import numpy as np
import argparse

def main(languages: list[str] = ALL_LANGUAGES):
    df = pd.DataFrame(index=languages, columns=SHORT_MODELS)

    for model_long, model_short in zip(FULL_MODELS, SHORT_MODELS):
        task_scores = load_task_scores(model_long, "translation")
        for tgt_lang in languages:
            df.loc[tgt_lang, model_short] = np.mean([task_scores[f"{src_lang}-{tgt_lang}"] for src_lang in ALL_LANGUAGES if src_lang != tgt_lang])
    os.makedirs("evaluation/tables", exist_ok=True)
    with open(f"evaluation/tables/translation_performance_by_language{'-short' if len(languages) < len(ALL_LANGUAGES) else ''}.tex", "w") as f:
        f.write(df_to_tex(df, 
                          caption=f"Average translation performance (\\textsc{{chrF}}) (over all source languages) for the displayed target languages. Values are displayed as per cent.", 
                          label=f"translation-performance-by-language{'-short' if len(languages) < len(ALL_LANGUAGES) else ''}", 
                          highlight_max=True, highlight_max_in_each_row=True, heatmap=True, eflomal_in_last_col=False))

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.languages)
    else:
        parser = argparse.ArgumentParser(description="Generate a LaTeX table comparing the average translation performance (chrF) for different target languages. Values are averaged over all source languages.")
        parser.add_argument("--langs", nargs="+", default=ALL_LANGUAGES, help="List of target languages to include in the table. Default is all languages.")
        args = parser.parse_args()
        main(args.langs)