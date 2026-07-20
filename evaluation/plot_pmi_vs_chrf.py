from monolingual_task_corr import load_task_scores
from cla_utils import ALL_LANGUAGES, df_to_tex, METRICS, SENT_REPS, REPS_SHORT_NAMES, METRICS_SHORT_NAMES, get_cla_all, get_alignment, get_flores_code
from scipy.stats import pearsonr
import pandas as pd
import os
import argparse
import numpy as np
import sys
from typing import Literal
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
def calculate_table(model: str, 
         layer_pooling: Literal["MEAN", "HIGHEST", "best"] = "HIGHEST",
         ):
    
    best_rep_per_model = {
        "Qwen/Qwen3-14B-Base": {
            "pmi": "weighted-mean",
            "chrf": "mean"
        },
        "google/gemma-3-12b-it": {
            "pmi": "weighted-mean",
            "chrf": "prompt"
        },
        "mistralai/Ministral-3-14B-Base-2512": {
            "pmi": "weighted-mean",
            "chrf": "last-token"
        }
    }
    best_metric_per_model = {
        "Qwen/Qwen3-14B-Base": {
            "pmi": "cosine",
            "chrf": "anc"
        },
        "google/gemma-3-12b-it": {
            "pmi": "cosine",
            "chrf": "anc"
        },
        "mistralai/Ministral-3-14B-Base-2512": {
            "pmi": "ratio",
            "chrf": "ratio"
        }
    }


    ALL_LANGUAGES_NO_EN = ALL_LANGUAGES.copy()
    ALL_LANGUAGES_NO_EN.remove("en")


    cla_all = get_cla_all(model.split("/")[1])
    task_scores_pmi = load_task_scores(model, "pmi")
    task_scores_chrf = load_task_scores(model, "translation")

    lang_pairs = [f"{src}-{tgt}" for src in ALL_LANGUAGES_NO_EN for tgt in ALL_LANGUAGES_NO_EN if src != tgt]
    translation_pmis = np.array([task_scores_pmi[lang_pair] for lang_pair in lang_pairs])
    translation_chrf = np.array([task_scores_chrf[lang_pair] for lang_pair in lang_pairs])

    plt.style.use("seaborn-v0_8-dark")
    plt.style.use("seaborn-v0_8-talk")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)
    ax[0].grid()
    ax[1].grid()
    ax[0].tick_params(axis='both', which='major', labelsize=9)
    ax[1].tick_params(axis='both', which='major', labelsize=9)

    # PMI
    alignments_src_en = np.array([get_alignment(cla_all[best_metric_per_model[model]["pmi"]][best_rep_per_model[model]["pmi"]], lang_pair.split('-')[0], "en", layer=layer_pooling) for lang_pair in lang_pairs])
    alignments_en_tgt = np.array([get_alignment(cla_all[best_metric_per_model[model]["pmi"]][best_rep_per_model[model]["pmi"]], "en", lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs])
    alignments_src_tgt = np.array([get_alignment(cla_all[best_metric_per_model[model]["pmi"]][best_rep_per_model[model]["pmi"]], lang_pair.split('-')[0], lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs])

    STEPS = 26
    corrs = np.ones((STEPS, STEPS))* np.nan

    for alpha_step, alpha in enumerate(np.linspace(0, 1, STEPS)):
        for beta_step, beta in enumerate(np.linspace(0, 1 - alpha, STEPS - alpha_step)):
            combined_alignments = alpha * alignments_src_en + beta * alignments_en_tgt + (1-alpha-beta) * alignments_src_tgt
            corr = pearsonr(combined_alignments, translation_pmis).correlation
            corrs[int(alpha*(STEPS-1)), int(beta*(STEPS-1))] = corr

    best_alpha, best_beta = np.unravel_index(np.nanargmax(corrs.flatten()), corrs.shape)
    best_alpha /= (STEPS-1)
    best_beta /= (STEPS-1)

    for tgt_lang in ALL_LANGUAGES_NO_EN:
        _, script = get_flores_code(tgt_lang).split("_")
        alignments_src_en = np.array([get_alignment(cla_all[best_metric_per_model[model]["pmi"]][best_rep_per_model[model]["pmi"]], lang_pair.split('-')[0], "en", layer=layer_pooling) for lang_pair in lang_pairs if lang_pair.split('-')[1] == tgt_lang])
        alignments_en_tgt = np.array([get_alignment(cla_all[best_metric_per_model[model]["pmi"]][best_rep_per_model[model]["pmi"]], "en", lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs if lang_pair.split('-')[1] == tgt_lang])
        alignments_src_tgt = np.array([get_alignment(cla_all[best_metric_per_model[model]["pmi"]][best_rep_per_model[model]["pmi"]], lang_pair.split('-')[0], lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs if lang_pair.split('-')[1] == tgt_lang])
        best_alignments_tgt = best_alpha * alignments_src_en + best_beta * alignments_en_tgt + (1-best_alpha-best_beta) * alignments_src_tgt
        mi_scores_tgt = [task_scores_pmi[f"{src_lang}-{tgt_lang}"] for src_lang in ALL_LANGUAGES_NO_EN if src_lang != tgt_lang]
        ax[0].plot(best_alignments_tgt, mi_scores_tgt, "o", markersize=3, color="tab:blue" if script=="Latn" else "tab:orange", alpha=0.7)
        centroid = (np.mean(best_alignments_tgt), np.mean(mi_scores_tgt))
        ax[0].text(*centroid, tgt_lang, fontsize=11, zorder=10)

    # clip y axis to -0.1+ if it's smaller than that
    if np.nanmin(translation_pmis) < -0.1:
        ax[0].set_ylim(bottom=-0.1)
    ax[0].set_title(f"$\\rho={round(np.nanmax(corrs)*100, 1)}\\%$")
    ax[0].set_ylabel("PMI of translations")
    ax[0].set_xlabel(f"Combined alignments")
    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # chrF
    alignments_src_en = np.array([get_alignment(cla_all[best_metric_per_model[model]["chrf"]][best_rep_per_model[model]["chrf"]], lang_pair.split('-')[0], "en", layer=layer_pooling) for lang_pair in lang_pairs])
    alignments_en_tgt = np.array([get_alignment(cla_all[best_metric_per_model[model]["chrf"]][best_rep_per_model[model]["chrf"]], "en", lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs])
    alignments_src_tgt = np.array([get_alignment(cla_all[best_metric_per_model[model]["chrf"]][best_rep_per_model[model]["chrf"]], lang_pair.split('-')[0], lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs])

    corrs = np.ones((STEPS, STEPS))* np.nan
    for alpha_step, alpha in enumerate(np.linspace(0, 1, STEPS)):
        for beta_step, beta in enumerate(np.linspace(0, 1 - alpha, STEPS - alpha_step)):
            combined_alignments = alpha * alignments_src_en + beta * alignments_en_tgt + (1-alpha-beta) * alignments_src_tgt
            corr = pearsonr(combined_alignments, translation_chrf).correlation
            corrs[int(alpha*(STEPS-1)), int(beta*(STEPS-1))] = corr
    best_alpha, best_beta = np.unravel_index(np.nanargmax(corrs.flatten()), corrs.shape)
    best_alpha /= (STEPS-1)
    best_beta /= (STEPS-1)

    for tgt_lang in ALL_LANGUAGES_NO_EN:
        _, script = get_flores_code(tgt_lang).split("_")
        alignments_src_en = np.array([get_alignment(cla_all[best_metric_per_model[model]["chrf"]][best_rep_per_model[model]["chrf"]], lang_pair.split('-')[0], "en", layer=layer_pooling) for lang_pair in lang_pairs if lang_pair.split('-')[1] == tgt_lang])
        alignments_en_tgt = np.array([get_alignment(cla_all[best_metric_per_model[model]["chrf"]][best_rep_per_model[model]["chrf"]], "en", lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs if lang_pair.split('-')[1] == tgt_lang])
        alignments_src_tgt = np.array([get_alignment(cla_all[best_metric_per_model[model]["chrf"]][best_rep_per_model[model]["chrf"]], lang_pair.split('-')[0], lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs if lang_pair.split('-')[1] == tgt_lang])
        best_alignments_tgt = best_alpha * alignments_src_en + best_beta * alignments_en_tgt + (1-best_alpha-best_beta) * alignments_src_tgt
        chrf_scores_tgt = [task_scores_chrf[f"{src_lang}-{tgt_lang}"] for src_lang in ALL_LANGUAGES_NO_EN if src_lang != tgt_lang]
        ax[1].plot(best_alignments_tgt, chrf_scores_tgt, "o", markersize=3, color="tab:blue" if script=="Latn" else "tab:orange", alpha=0.7)
        centroid = (np.mean(best_alignments_tgt), np.mean(chrf_scores_tgt))
        ax[1].text(*centroid, tgt_lang, fontsize=11, zorder=10)
    ax[1].set_title(f"$\\rho={round(np.nanmax(corrs)*100, 1)}\\%$")
    ax[1].set_ylabel("chrF of translations")
    ax[1].set_xlabel(f"Combined alignments")
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.tight_layout()
    os.makedirs("evaluation/plots", exist_ok=True)
    plt.savefig(f"evaluation/plots/pmi_vs_chrf_{model.split('/')[1]}_{layer_pooling}.pdf", pad_inches=0, bbox_inches=0)
    plt.close()

    
if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        calculate_table(snakemake.params.model)
    else:
        parser = argparse.ArgumentParser(description="Calculate the correlation table for PMI.")
        parser.add_argument("--model", type=str, default="google/gemma-3-12b-it", help="Model name on Hugging Face.")
        args = parser.parse_args()
        calculate_table(args.model)