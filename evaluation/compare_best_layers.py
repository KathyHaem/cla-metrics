import monolingual_task
from typing import Literal
import numpy as np
import pandas as pd
from task_results import SHORT_MODELS_DICT
from cla_utils import df_to_tex, REPS_SHORT_NAMES, METRICS_SHORT_NAMES, SENT_REPS
import argparse

models_to_compare = [
    "Qwen/Qwen3-14B",
    "mistralai/Ministral-3-14B-Base-2512",
]

def join_into_tex_table(best_layer_tablesper_model: dict, task: Literal["sib-200", "belebele"] = "sib-200") -> str:
    joined_table = np.zeros((best_layer_tablesper_model[models_to_compare[0]].shape[0], best_layer_tablesper_model[models_to_compare[0]].shape[1]*len(models_to_compare)))
    for i, model in enumerate(models_to_compare):
        column_index = np.arange(0, best_layer_tablesper_model[models_to_compare[0]].shape[1]*len(models_to_compare), len(models_to_compare)) + i
        joined_table[:, column_index] = best_layer_tablesper_model[model]

    joined_table = pd.DataFrame(joined_table, index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS])

    colspec = "{ X |" + " | ".join(["c"*len(models_to_compare) for _ in range(best_layer_tablesper_model[models_to_compare[0]].shape[1])]) + " }"
    header = " & ".join([f"\\multicolumn{{{len(models_to_compare)}}}{{c}}{{{metric}}}" for metric in METRICS_SHORT_NAMES]) + " \\\\ \n"

    print(best_layer_tablesper_model[models_to_compare[0]])
    print(best_layer_tablesper_model[models_to_compare[1]])
    print(joined_table)
    print(header)
    print(colspec)

def main(task: Literal["sib-200", "belebele"] = "sib-200"):
    best_layer_tables_per_model = {
        model: monolingual_task.main(model, tgt_lang="en", task=task, layer_pooling="best", no_print=True)[1]
        for model in models_to_compare
    }

    join_into_tex_table(best_layer_tables_per_model, task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare best layers of different models.")
    parser.add_argument("--task", type=str, choices=["sib-200", "belebele"], default="sib-200", help="Task to evaluate on.")
    args = parser.parse_args()

    main(args.task)