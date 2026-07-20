import numpy as np
import pandas as pd
from cla_utils import FULL_MODELS, SHORT_MODELS, SHORT_MODELS_DICT
from monolingual_task_corr import load_task_scores
from typing import Literal
import os
from matplotlib import pyplot as plt

def format_family_label(model: str) -> str:
    model_name = model.split("/")[1]
    parts = model_name.split("-")

    if parts[-1] in {"Base", "Instruct", "pt", "it"}:
        parts = parts[:-1]
    elif len(parts) >= 2 and parts[-2] in {"Base", "Instruct"}:
        parts = parts[:-2]

    return "-".join(parts)

def main(task: Literal["sib-200", "belebele", "translation"]) -> None:
    records = []
    for model in FULL_MODELS:
        scores = load_task_scores(model, task)
        scores_array = np.array(list(scores.values()))
        records.append(scores_array*100)

    positions = []
    current_pos = 1.0
    within_pair_gap = 0.55
    between_pair_gap = 1.25
    for model_idx, model in enumerate(FULL_MODELS):
        if model_idx > 0:
            current_pos += between_pair_gap if model_idx % 2 == 0 else within_pair_gap
        positions.append(current_pos)

    plt.style.use("seaborn-v0_8-dark")
    plt.style.use("seaborn-v0_8-talk")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid(axis="y")
    ax.boxplot(records, positions=positions, widths=0.28, patch_artist=True, boxprops=dict(facecolor="#1f77b4aa"), flierprops=dict(alpha=0.5))
    ylim = 65 if task == "sib-200" else 40
    ax.set_ylim(bottom=ylim)
    ax.set_xticks(positions)
    ax.set_xticklabels([""] * len(positions))
    ax.tick_params(axis="x", length=0)

    for pair_start in range(0, len(FULL_MODELS), 2):
        left_model = FULL_MODELS[pair_start]
        right_model = FULL_MODELS[pair_start + 1]
        family_label = format_family_label(left_model)
        left_x = positions[pair_start]
        right_x = positions[pair_start + 1]
        center_x = (left_x + right_x) / 2

        ax.text(
            left_x,
            -0.07,
            "B",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
        )
        ax.text(
            right_x,
            -0.07,
            "I",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
        )
        ax.text(
            center_x,
            -0.18,
            family_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
        )

    for model_idx, model in enumerate(SHORT_MODELS):
        model_scores = records[model_idx]
        min_score = model_scores.min()
        
        if min_score < ylim:
            x_pos = positions[model_idx]
            
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
    fig.subplots_adjust(bottom=0.36)
    plt.tight_layout()

    os.makedirs("evaluation/plots", exist_ok=True)
    plt.savefig(f"evaluation/plots/performance_task_{task}.pdf")
    plt.close(fig)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.task)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Generate a box plot comparing model performance on a given task.")
        parser.add_argument("--task", type=str, help="Task name.", choices=["sib-200", "belebele", "translation"], default="sib-200")

        args = parser.parse_args()
        main(args.task)