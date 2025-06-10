import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt


def save_plot(plot_name, fig):
    plot_filename = f"{plot_name}.pdf"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    fig.savefig(plot_filename, bbox_inches="tight", format="pdf")
    plot_filename = f"{plot_name}.png"
    fig.savefig(plot_filename, bbox_inches="tight")
    plt.close(fig)


def plot_show_pairs(scores):
    """ Scores: {(src_lang, tgt_lang): [layer_0_score, layer_1_score, ...]} """
    fig, ax = plt.subplots(figsize=(10, 6))
    # TODO step into debug
    scores_np = np.array(list(scores.values()))
    means_over_pairs = np.mean(scores_np, axis=0)

    # plot a lighter line for each language pair
    for i, lang_pair in enumerate(scores.keys()):
        src_lang = lang_pair.split("-")[0]
        tgt_lang = lang_pair.split("-")[1]
        ax.plot(range(len(scores_np[i])), scores_np[i], label=f"{src_lang} - {tgt_lang}", alpha=0.5)  # probably don't want the labels
    # plot the mean line
    ax.plot(range(len(means_over_pairs)), means_over_pairs, label="Mean", color="darkblue", linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    return fig


def plot_only_mean(scores):
    """ Scores: {(src_lang, tgt_lang): [layer_0_score, layer_1_score, ...]} """
    fig, ax = plt.subplots(figsize=(10, 6))

    # TODO step into debug
    scores_np = np.array(list(scores.values()))
    means_over_pairs = np.mean(scores_np, axis=0)

    # plot the mean line
    ax.plot(range(len(means_over_pairs)), means_over_pairs, label="Mean", color="darkblue", linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Score")
    return fig


def main(model, dataset, show_pairs, score):
    with open(f"../scores/{dataset}/{model}_mean_{score}.json", "r") as fin:
        scores = json.load(fin)

    if show_pairs:
        fig = plot_show_pairs(scores)
        plot_name = f"../figures/{dataset}/{model}_{score}_show_pairs"
    else:
        fig = plot_only_mean(scores)
        plot_name = f"../figures/{dataset}/{model}_{score}"

    save_plot(plot_name, fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot score per layer")
    parser.add_argument("--model", type=str, required=True, help="Model name. Assuming an HF decoder")
    parser.add_argument("--dataset", type=str, default="flores")
    parser.add_argument("--score", type=str, default="cosine")  # anc, dist, ratio,...
    parser.add_argument("--show_pairs", action="store_true", help="Differentiate by language pairs."
                                                                  "Default: Just Mean")

    args = parser.parse_args()
    main(args.model, args.dataset, args.show_pairs, args.score)
