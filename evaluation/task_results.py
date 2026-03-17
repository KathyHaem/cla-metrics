import numpy as np
import pandas as pd
from cla_utils import FULL_MODELS, SHORT_MODELS
from monolingual_task_corr import load_task_scores
from typing import Literal
import os
from matplotlib import pyplot as plt

def main(task: Literal["sib-200", "belebele", "translation"]) -> None:
    records = []
    for model in FULL_MODELS:
        scores = load_task_scores(model, task)
        scores_array = np.array(list(scores.values()))
        records.append(scores_array*100)

    plt.style.use("seaborn-v0_8-dark")
    plt.style.use("seaborn-v0_8-talk")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid(axis="y")
    ax.boxplot(records, tick_labels=SHORT_MODELS, patch_artist=True, boxprops=dict(facecolor="#1f77b4aa"), flierprops=dict(alpha=0.5))
    ylim = 60 if task == "sib-200" else 40
    ax.set_ylim(bottom=ylim)

    for model_idx, model in enumerate(SHORT_MODELS):
        model_scores = records[model_idx]
        min_score = model_scores.min()
        
        if min_score < ylim:
            x_pos = model_idx + 1  # 1-indexed position in boxplot
            
            # Position triangle and text just inside the plot at y=61
            ax.scatter(
                x_pos,
                ylim + 1.5,
                marker="v",
                s=60,
                color="black",
                zorder=5,
            )
            ax.text(
                x_pos,
                ylim + 2.5,
                f"{min_score:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # ax.set_title(f"{task} results by model")
    # ax.set_xlabel("Model")
    ax.set_ylabel(("$\\text{F}_1$ score" if task == "sib-200" else "accuracy")  + " (%)")
    # ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("")
    # fig.subplots_adjust(bottom=0.22)
    plt.tight_layout()

    os.makedirs("evaluation/plots", exist_ok=True)
    plt.savefig(f"evaluation/plots/performance_task_{task}.pdf")
    plt.close(fig)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.task)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Generate a box plot comparing model performance on a given task.")
        parser.add_argument("--task", type=str, help="Task name.", choices=["sib-200", "belebele", "translation"], default="sib-200")

        args = parser.parse_args()
        main(args.task)