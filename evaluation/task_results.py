import numpy as np
import pandas as pd
from cla_utils import FULL_MODELS, df_to_tex, SHORT_MODELS, SHORT_MODELS_DICT
from monolingual_task_corr import load_task_scores
from typing import Literal
import os
import json
import re

STATISTICS = ["mean", "std", "max", "min"]

def main(task: Literal["sib-200", "belebele", "translation"]) -> None:
    df = pd.DataFrame(index=STATISTICS, columns=SHORT_MODELS)

    for model_idx, model in enumerate(SHORT_MODELS):
        scores = load_task_scores(FULL_MODELS[model_idx], task)
        scores_array = np.array(list(scores.values()))
        for stat in STATISTICS:
            df.loc[stat, model] = getattr(np, stat)(scores_array)

    os.makedirs("evaluation/tables", exist_ok=True)
    text_table = df_to_tex(df, f"Performance statistics for the task {task}. Values are displayed as per cent.", f"performanceTask{task}", highlight_max=False, eflomal_in_last_col=False, heatmap=False)
    with open(f"evaluation/tables/performance_task_{task}.tex", "w") as f:
        f.write(text_table)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.task)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Generate a LaTeX table comparing the performance of different models on a given task. C")
        parser.add_argument("--task", type=str, help="Task name.", choices=["sib-200", "belebele", "translation"], default="translation")

        args = parser.parse_args()
        main(args.task)