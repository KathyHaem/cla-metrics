from monolingual_task_corr import load_task_scores
from cla_utils import ALL_LANGUAGES, df_to_tex, METRICS, SENT_REPS, REPS_SHORT_NAMES, METRICS_SHORT_NAMES, FULL_MODELS, get_cla_all, get_alignment
from scipy.stats import pearsonr
import pandas as pd
import os
import argparse
import numpy as np
import sys
from typing import Literal
import matplotlib.pyplot as plt

def plot_alpha_beta(corrs, steps, best_alpha_idx, best_beta_idx, model, layer_pooling, metric, sent_rep, best_alpha_idx_bouquet=None, best_beta_idx_bouquet=None):
    fig, ax = plt.subplots(figsize=(3.8, 3))
    im = ax.matshow(corrs, cmap="BuPu", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_xlabel("$\\beta$ (en-tgt weight)")
    ax.set_ylabel("$\\alpha$ (src-en weight)")
    ax.set_xticks(np.linspace(0, steps - 1, 6))
    ax.set_xticklabels(np.linspace(0, steps - 1, 6)/(steps - 1))
    ax.set_yticks(np.linspace(0, steps - 1, 6))
    ax.set_yticklabels(np.linspace(0, steps - 1, 6)/(steps - 1))

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.plot(best_beta_idx, best_alpha_idx, "o", color="red", markersize=5, markeredgecolor="white", markeredgewidth=0.5)

    if best_alpha_idx_bouquet is not None and best_beta_idx_bouquet is not None:
        ax.plot(best_beta_idx_bouquet, best_alpha_idx_bouquet, "o", color="gray", markersize=5, markeredgecolor="white", markeredgewidth=0.5)

    fig.tight_layout()
    os.makedirs("evaluation/plots", exist_ok=True)
    fig.savefig(f"evaluation/plots/alpha_beta_{model.split('/')[1]}_{layer_pooling}_{metric}_{sent_rep}.pdf")

def calculate_table(model: str, 
         layer_pooling: Literal["MEAN", "HIGHEST", "best"] = "HIGHEST",
         no_print: bool = False,
         metric: Literal["translation", "pmi"] = "pmi"
         ):
    ALL_LANGUAGES_NO_EN = ALL_LANGUAGES.copy()
    ALL_LANGUAGES_NO_EN.remove("en")

    corr_table = np.ones((len(SENT_REPS), len(METRICS))) * np.nan
    corresponding_bouquet_corrs = np.ones((len(SENT_REPS), len(METRICS))) * np.nan
    gammas_table = np.ones((len(SENT_REPS), len(METRICS))) * np.nan

    cla_all = get_cla_all(model.split("/")[1])
    task_scores = load_task_scores(model, metric, dataset="flores")
    taslk_scores_bouquet = load_task_scores(model, metric, dataset="bouquet")

    lang_pairs = [f"{src}-{tgt}" for src in ALL_LANGUAGES_NO_EN for tgt in ALL_LANGUAGES_NO_EN if src != tgt]
    translation_pmis = np.array([task_scores[lang_pair] for lang_pair in lang_pairs])
    translation_pmis_bouquet = np.array([taslk_scores_bouquet[lang_pair] for lang_pair in lang_pairs])
    for m_i, metric in enumerate(METRICS):
        for s_i, sent_rep in enumerate(SENT_REPS):
            try:
                current_cla = cla_all[metric][sent_rep]
            except KeyError:
                print("Key error for", metric, sent_rep, file=sys.stderr)
                continue
            
            alignments_src_en = np.array([get_alignment(current_cla, lang_pair.split('-')[0], "en", layer=layer_pooling) for lang_pair in lang_pairs])
            alignments_en_tgt = np.array([get_alignment(current_cla, "en", lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs])
            alignments_src_tgt = np.array([get_alignment(current_cla, lang_pair.split('-')[0], lang_pair.split('-')[1], layer=layer_pooling) for lang_pair in lang_pairs])

            STEPS = 26
            corrs = np.ones((STEPS, STEPS))* np.nan
            corrs_bouquet = np.ones((STEPS, STEPS))* np.nan

            for alpha_step, alpha in enumerate(np.linspace(0, 1, STEPS)):
                for beta_step, beta in enumerate(np.linspace(0, 1 - alpha, STEPS - alpha_step)):
                    combined_alignments = alpha * alignments_src_en + beta * alignments_en_tgt + (1-alpha-beta) * alignments_src_tgt
                    corr = pearsonr(combined_alignments, translation_pmis).correlation
                    corr_bouquet = pearsonr(combined_alignments, translation_pmis_bouquet).correlation
                    if alpha_step == 8 and beta_step == 16:
                        pass
                    corrs[alpha_step, beta_step] = corr
                    corrs_bouquet[alpha_step, beta_step] = corr_bouquet

            corr_table[s_i, m_i] = np.nanmax(corrs)
            if np.isnan(corrs).all():
                continue
            best_alpha_idx, best_beta_idx = np.unravel_index(np.nanargmax(corrs.flatten()), corrs.shape)
            best_alpha_idx_bouquet, best_beta_idx_bouquet = np.unravel_index(np.nanargmax(corrs_bouquet.flatten()), corrs_bouquet.shape)
            corresponding_bouquet_corrs[s_i, m_i] = corrs_bouquet[best_alpha_idx, best_beta_idx]
            best_alpha = best_alpha_idx / (STEPS - 1)
            best_beta = best_beta_idx / (STEPS - 1)
            # corrs[corrs == -1] = np.nan

            plot_alpha_beta(corrs, STEPS, best_alpha_idx, best_beta_idx, model, layer_pooling, metric, sent_rep, best_alpha_idx_bouquet, best_beta_idx_bouquet)

            gammas_table[s_i, m_i] = 1 - best_alpha - best_beta

    if no_print:
        return corr_table, gammas_table, corresponding_bouquet_corrs
    else:
        df = pd.DataFrame(corr_table, index=[REPS_SHORT_NAMES.get(sent_rep, sent_rep) for sent_rep in SENT_REPS], columns=[METRICS_SHORT_NAMES.get(metric, metric) for metric in METRICS])
        os.makedirs("evaluation/tables", exist_ok=True)
        with open(f"evaluation/tables/corr_pmi_{model.split('/')[1]}_{layer_pooling}.tex", "w") as f:
            f.write(df_to_tex(
                df, f"Correlation between the best convex combination of alignments \\texttt{{src-en}}, \\texttt{{en-tgt}} and \\texttt{{src-tgt}} and the PMI of translations for {model.split('/')[1]}.", f"corr-pmi-{model.split('/')[1]}_{layer_pooling}", grad_command="\\percentGrad", highlight_max=True))
            
        with open(f"evaluation/tables/gammas_pmi_{model.split('/')[1]}_{layer_pooling}.tex", "w") as f:
            f.write(df_to_tex(
                pd.DataFrame(gammas_table, index=[REPS_SHORT_NAMES.get(sent_rep, sent_rep) for sent_rep in SENT_REPS], columns=[METRICS_SHORT_NAMES.get(metric, metric) for metric in METRICS]),
                f"Optimal \\texttt{{src-en}} weights in the convex combination of alignments \\texttt{{src-en}}, \\texttt{{en-tgt}} and \\texttt{{src-tgt}} that maximize the correlation with the PMI of translations for {model.split('/')[1]}.", f"gammas-pmi-{model.split('/')[1]}_{layer_pooling}", highlight_max=True, grad_command="\\percentGrad"))

def _write_all_models_table(models, layer_pooling: str, metric_name: Literal["translation", "pmi"], table_kind: Literal["corr", "gammas", "validation"]):
    table_parts = [
        r"\begin{table}[ht]",
        r"\centering\footnotesize",
        r"\begin{tabularx}{\columnwidth}%",
        r"{p{4em} | C  C  C  C  C  C | C}"
        r"\toprule",
    ]

    is_first = True
    for model in models:
        corr_table, gammas_table, bouquet_val_table = calculate_table(model, layer_pooling=layer_pooling, no_print=True, metric=metric_name)
        model_name = model.split("/")[1]
        match table_kind:
            case "corr":
                current_table = corr_table
                caption_suffix = "best correlations"
            case "gammas":
                current_table = gammas_table
                caption_suffix = "src-tgt weight"
            case "validation":
                current_table = bouquet_val_table
                caption_suffix = "validation correlations"

        if not is_first:
            table_parts.append(r"\midrule")
        is_first = False

        table_parts.append(r"\multicolumn{8}{c}{\textbf{" + model_name + r"}} \\")
        table_parts.append(r"\midrule")

        df = pd.DataFrame(
            current_table,
            index=[REPS_SHORT_NAMES.get(sent_rep, sent_rep) for sent_rep in SENT_REPS],
            columns=[METRICS_SHORT_NAMES.get(metric, metric) for metric in METRICS],
        )

        table_parts.append(df_to_tex(df, label="", caption="", cells_only=True, highlight_max=True, grad_command="\\percentGrad"))

    table_parts += [
        r"\bottomrule",
        r"\end{tabularx}",
        fr"\caption{{Overview of the {caption_suffix} between alignments \texttt{{src-en}}, \texttt{{en-tgt}}, and \texttt{{src-tgt}} and the translation {'PMI' if metric_name == 'pmi' else '\\textsc{chrF}'} scores for all models. Values are shown as per cent.}}",
        fr"\label{{{'corr' if table_kind == 'corr' else 'gammas'}-pmi-all-{layer_pooling}-{metric_name}}}",
        r"\end{table}",
    ]

    os.makedirs("evaluation/tables", exist_ok=True)
    out_path = f"evaluation/tables/{table_kind}_pmi_all_{layer_pooling}_{metric_name}.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(table_parts))

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        calculate_table(snakemake.params.model, snakemake.params.layer_pooling)
    else:
        parser = argparse.ArgumentParser(description="Calculate the correlation table for PMI.")
        parser.add_argument("--model", type=str, default="ALL", help="Model name on Hugging Face.")
        parser.add_argument("--layer_pooling", type=str, default="HIGHEST", choices=["MEAN", "HIGHEST", "best"], help="Layer pooling strategy.")
        parser.add_argument("--metric", type=str, default="pmi", choices=["translation", "pmi"], help="Metric to use for evaluation.")
        args = parser.parse_args()
        if args.model == "ALL":
            _write_all_models_table(FULL_MODELS, args.layer_pooling, args.metric, "validation")
            _write_all_models_table(FULL_MODELS, args.layer_pooling, args.metric, "corr")
            _write_all_models_table(FULL_MODELS, args.layer_pooling, args.metric, "gammas")
        else:
            calculate_table(args.model, args.layer_pooling)