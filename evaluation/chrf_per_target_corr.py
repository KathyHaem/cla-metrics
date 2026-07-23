from monolingual_task_corr import load_task_scores
from cla_utils import ALL_LANGUAGES, df_to_tex, METRICS, SENT_REPS, REPS_SHORT_NAMES, METRICS_SHORT_NAMES, get_cla_all, get_alignment, FULL_MODELS
from scipy.stats import pearsonr
import pandas as pd
import os
import argparse
import numpy as np
import sys
from typing import Literal

def _plot_alpha_corrs(alphas, corrs, corrs_bouquet, model_name, m_i, s_i):
    import matplotlib.pyplot as plt

    metric_name = METRICS_SHORT_NAMES.get(METRICS[m_i], METRICS[m_i])
    sent_rep_name = REPS_SHORT_NAMES.get(SENT_REPS[s_i], SENT_REPS[s_i])

    plt.style.use("seaborn-v0_8-dark")
    plt.style.use("seaborn-v0_8-talk")
    plt.figure(figsize=(5,3))

    plt.plot(alphas, corrs_bouquet, color="gray", alpha=0.5)
    best_corr_bouq = np.max(corrs_bouquet)
    best_alpha_bouq = alphas[np.argmax(corrs_bouquet)]
    plt.plot(best_alpha_bouq, best_corr_bouq, "o", color="gray", markersize=8, markeredgecolor="white", markeredgewidth=0.5)

    plt.plot(alphas, corrs, color="tab:blue")
    # plt.title(f"Weight for src-en vs. correlation for {model_name}, {metric_name}, {sent_rep_name}")

    best_corr = np.max(corrs)
    best_alpha = alphas[np.argmax(corrs)]
    plt.plot(best_alpha, best_corr, "o", color="tab:blue", markersize=8, markeredgecolor="white", markeredgewidth=0.5)
    plt.xlabel("weight for src-en ($\\alpha$)")
    plt.ylabel("Pearson correlation")
    plt.gca().yaxis.tick_right()

    plt.grid()

    os.makedirs("evaluation/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"evaluation/plots/alpha_corr_{model_name}_{metric_name}_{sent_rep_name}.pdf")

def calculate_table(model: str, 
         layer_pooling: Literal["MEAN", "HIGHEST", "best"] = "HIGHEST",
         tgt_lang_cla: Literal["en", "tgt", "partial_corr", "combination"] = "en",
         no_print: bool = False,
         plot_alphas = True,
         return_bouquet = False):
    ALL_LANGUAGES_NO_EN = ALL_LANGUAGES.copy()
    ALL_LANGUAGES_NO_EN.remove("en")
    corr_table = np.ones((len(SENT_REPS), len(METRICS))) * np.nan
    best_layer_table = np.ones((len(SENT_REPS), len(METRICS))) * np.nan
    alphas_table = np.ones((len(SENT_REPS), len(METRICS))) * np.nan

    cla_all = get_cla_all(model.split("/")[1])

    task_scores = load_task_scores(model, "translation")
    bouquet_task_scores = load_task_scores(model, "translation", dataset="bouquet")

    for m_i, metric in enumerate(METRICS):
        for s_i, sent_rep in enumerate(SENT_REPS):
            correlations = []
            if tgt_lang_cla == "combination":
                STEPS = 26
                local_corrs = []
                local_corrs_bouquet = []
                best_corr = -2
                corresponding_corr_bouquet = -2
                best_corr_bouquet = -2
                best_alpha = 0
                best_alpha_bouquet = 0
                for alpha in np.linspace(0, 1, STEPS):
                    combined_correlations = []
                    combined_correlations_bouquet = []
                    for tgt in ALL_LANGUAGES_NO_EN:
                        try:
                            current_cla = cla_all[metric][sent_rep]
                        except KeyError:
                            print("Key error for", metric, sent_rep, file=sys.stderr)
                            continue
                        
                        cla_en = [get_alignment(current_cla, 
                                            lang, "en", 
                                            layer=layer_pooling) 
                                            for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                        cla_tgt = [get_alignment(current_cla, 
                                            lang, tgt, 
                                            layer=layer_pooling) 
                                            for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                        combined_cla = [alpha * en + (1-alpha) * tgt for en, tgt in zip(cla_en, cla_tgt)]
                        task_scores_ordered = [task_scores[f"{lang}-{tgt}"] for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                        task_scores_bouquet_ordered = [bouquet_task_scores[f"{lang}-{tgt}"] for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                        combined_correlations.append(pearsonr(combined_cla, task_scores_ordered).correlation)
                        combined_correlations_bouquet.append(pearsonr(combined_cla, task_scores_bouquet_ordered).correlation)
                    avg_combined_corr = np.nanmean(combined_correlations)
                    avg_combined_corr_bouquet = np.nanmean(combined_correlations_bouquet)
                    local_corrs.append(avg_combined_corr)
                    local_corrs_bouquet.append(avg_combined_corr_bouquet)
                    if avg_combined_corr > best_corr:
                        best_corr = avg_combined_corr
                        best_alpha = alpha
                        corresponding_corr_bouquet = avg_combined_corr_bouquet # Store the corresponding bouquet correlation for the best alpha for flores

                if plot_alphas:
                    _plot_alpha_corrs(np.linspace(0, 1, STEPS), local_corrs, local_corrs_bouquet, model.split("/")[1], m_i, s_i)
                if best_corr < -1:
                    best_corr = np.nan
                corr_table[s_i, m_i] = best_corr if not return_bouquet else corresponding_corr_bouquet
                alphas_table[s_i, m_i] = best_alpha
            else:
                for tgt in ALL_LANGUAGES_NO_EN:
                    try:
                        current_cla = cla_all[metric][sent_rep]
                    except KeyError:
                        print("Key error for", metric, sent_rep, file=sys.stderr)
                        continue
                    
                    if layer_pooling == "best":
                        if tgt_lang_cla == "partial_corr" or tgt_lang_cla == "combination":
                            raise NotImplementedError("Partial correlation/combination is not implemented for best layer selection.")
                        best_layer = 0
                        best_corr = -2
                        for layer in range(len(next(current_cla.values().__iter__()))):
                            cla = [get_alignment(current_cla, 
                                                lang, tgt if tgt_lang_cla == "tgt" else "en", 
                                                layer=layer) 
                                                for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                            
                            task_scores_ordered = [task_scores[f"{lang}-{tgt}"] for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                            corr = pearsonr(cla, task_scores_ordered).correlation
                            if corr > best_corr:
                                best_corr = corr
                                best_layer = layer
                        corr_table[s_i, m_i] = best_corr
                        best_layer_table[s_i, m_i] = best_layer
                    else:
                        if tgt_lang_cla == "partial_corr":
                            cla_en = [get_alignment(current_cla, 
                                                lang, "en", 
                                                layer=layer_pooling) 
                                                for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                            cla_tgt = [get_alignment(current_cla, 
                                                lang, tgt, 
                                                layer=layer_pooling) 
                                                for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                            task_scores_ordered = [task_scores[f"{lang}-{tgt}"] for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                            corr_xy = pearsonr(cla_tgt, task_scores_ordered).correlation
                            corr_xz = pearsonr(cla_tgt, cla_en).correlation
                            corr_zy = pearsonr(cla_en, task_scores_ordered).correlation
                            corr = (corr_xy - corr_xz * corr_zy) / (np.sqrt(1 - corr_xz**2) * np.sqrt(1 - corr_zy**2))
                            correlations.append(corr)
                        else:
                            cla = [get_alignment(current_cla, 
                                                lang, tgt if tgt_lang_cla == "tgt" else "en", 
                                                layer=layer_pooling) 
                                                for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                            
                            if not return_bouquet:
                                task_scores_ordered = [task_scores[f"{lang}-{tgt}"] for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                            else:
                                task_scores_ordered = [bouquet_task_scores[f"{lang}-{tgt}"] for lang in ALL_LANGUAGES_NO_EN if lang != tgt]
                            correlations.append(pearsonr(cla, task_scores_ordered).correlation)
                corr_table[s_i, m_i] = np.nanmean(correlations)

    if no_print:
        return corr_table, alphas_table

    df = pd.DataFrame(corr_table, 
                      index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS], 
                      columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS])
    
    os.makedirs("evaluation/tables", exist_ok=True)
    with open(f"evaluation/tables/corr_{tgt_lang_cla}_with_chrf_{model.split('/')[1]}{'-HIGHEST' if layer_pooling == 'HIGHEST' else ''}.tex", "w") as f:
        f.write(df_to_tex(df, 
                          caption=("CHANGE THIS CAPTION TO PARTIAL CORRELATION! " if tgt_lang_cla == "partial_corr" else "") + f"Pearson correlation measured first across the \\texttt{{src}} languages with a fixed \\texttt{{tgt}} language and then averaged over the target language. The correlation is measured between the \\texttt{{src-{tgt_lang_cla}}} alignment score and the \\textsc{{chrF}} score in the \\texttt{{tgt}} language for {model.split('/')[1]}. Values are displayed as per cent.", 
                          label=f"corr-{tgt_lang_cla}-with-chrf-{model.split('/')[1]}{'-HIGHEST' if layer_pooling == 'HIGHEST' else ''}",
                          grad_command="\\percentGrad", highlight_max=True))

n_digits = 0
def preprocess_value(value, max, color=True):
    if np.isnan(value):
        return "--"
    else:
        percent = round(value*100, n_digits)
        is_max = int(round(max*100, n_digits)) == percent
        if n_digits == 0:
            percent = int(percent)
        if color:
            return f"\\percentGrad{{{percent}}}{{{int(is_max)}}}"
        elif is_max:
            return f"\\textbf{{{percent}}}"
        else:
            return str(percent)

def main(model: str, 
         layer_pooling: Literal["MEAN", "HIGHEST", "best"] = "HIGHEST",
         tgt_lang_cla: Literal["en", "tgt", "both", "partial_corr", "combination"] = "en",
         return_table: Literal["corr", "improvement", "alpha"] = "corr",
         return_bouquet: bool = False,
         no_print: bool = False):
    if model != "ALL":
        if tgt_lang_cla == "both":
            corr_table_en, _ = calculate_table(model, layer_pooling, "en", no_print=True)
            corr_table_tgt, _ = calculate_table(model, layer_pooling, "tgt", no_print=True)

            max_en = np.nanmax(corr_table_en)
            max_tgt = np.nanmax(corr_table_tgt)

            df_both = pd.DataFrame(
                          index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS], 
                          columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS])
            
            for m_i, metric in enumerate(METRICS):
                for s_i, sent_rep in enumerate(SENT_REPS):
                    val_en = corr_table_en[s_i, m_i]
                    val_tgt = corr_table_tgt[s_i, m_i]
                    if metric == "eflomal":
                        eflomal_score_en = corr_table_en[SENT_REPS.index("mean"), m_i]
                        eflomal_score_tgt = corr_table_tgt[SENT_REPS.index("mean"), m_i]
                        if s_i != len(SENT_REPS)//2:
                            df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"\\percentGrad{{{int(round(eflomal_score_en*100, 0))}}}{{2}}"
                        else:
                            df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"{preprocess_value(eflomal_score_en, max_en)}{'\\hspace{0.6em}' if not np.isnan(eflomal_score_en) else ''}/{preprocess_value(eflomal_score_tgt, max_tgt, color=False)}"
                    else:
                        df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"{preprocess_value(val_en, max_en)}{'\\hspace{0.6em}' if not np.isnan(val_en) else ''}/{preprocess_value(val_tgt, max_tgt, color=False)}"

            os.makedirs("evaluation/tables", exist_ok=True)
            with open(f"evaluation/tables/corr_{tgt_lang_cla}_with_chrf_{model.split('/')[1]}{'-HIGHEST' if layer_pooling == 'HIGHEST' else ''}.tex", "w") as f:
                f.write(df_to_tex(df_both, 
                                caption=f"Pearson correlation measured first across the \\texttt{{src}} languages with a fixed \\texttt{{tgt}} language and then averaged over the target language. The correlation is measured between the \\texttt{{src-en}} (left side of the '/' symbol) or \\texttt{{src-tgt}} (right side of the '/' symbol) alignment score and the \\textsc{{chrF}} score in the \\texttt{{tgt}} language for {model.split('/')[1]}. Values are displayed as per cent. Cell colours are based on the \\texttt{{src-en}} scores.", 
                                label=f"corr-{tgt_lang_cla}-with-chrf-{model.split('/')[1]}{'-HIGHEST' if layer_pooling == 'HIGHEST' else ''}",
                                custom_colspec="{X | c  c  c  c  c  c | c}",
                                eflomal_in_last_col=False,
                                heatmap=False, highlight_max=False))
        elif tgt_lang_cla == "combination":
            corr_table_combination, alphas_table = calculate_table(model, layer_pooling, "combination", no_print=True, return_bouquet=return_bouquet)
            corr_prior, _ = calculate_table(model, layer_pooling, "en", no_print=True)
            print(pd.DataFrame(alphas_table, index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS], columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS]))
            improvement = corr_table_combination - corr_prior
            max_comb = np.nanmax(corr_table_combination)
            max_imp = np.nanmax(improvement)
            
            df_both = pd.DataFrame(
                          index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS], 
                          columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS])
            
            for m_i, metric in enumerate(METRICS):
                for s_i, sent_rep in enumerate(SENT_REPS):
                    val_comb = corr_table_combination[s_i, m_i]
                    val_imp = improvement[s_i, m_i]
                    if metric == "eflomal":
                        eflomal_score_comb = corr_table_combination[SENT_REPS.index("mean"), m_i]
                        eflomal_score_imp = improvement[SENT_REPS.index("mean"), m_i]
                        if s_i != len(SENT_REPS)//2:
                            df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"\\percentGrad{{{int(round(eflomal_score_comb*100, n_digits))}}}{{2}}"
                        else:
                            df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"{preprocess_value(eflomal_score_comb, max_comb)}{'\\hspace{0.6em}' if not np.isnan(eflomal_score_comb) else ''}/{preprocess_value(eflomal_score_imp, max_imp, color=False)}"
                    else:
                        df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"{preprocess_value(val_comb, max_comb)}{'\\hspace{0.6em}' if not np.isnan(val_comb) else ''}/{preprocess_value(val_imp, max_imp, color=False)}"
            
            os.makedirs("evaluation/tables", exist_ok=True)
            with open(f"evaluation/tables/corr_{tgt_lang_cla}_with_chrf_{model.split('/')[1]}{'_validation' if return_bouquet else ''}{'-HIGHEST' if layer_pooling == 'HIGHEST' else ''}.tex", "w") as f:
                f.write(df_to_tex(df_both, 
                                caption=f"Pearson correlation measured first across the \\texttt{{src}} languages with a fixed \\texttt{{tgt}} language and then averaged over the target language. The correlation is measured between the optimal convex combination of \\texttt{{src-en}} and \\texttt{{src-tgt}} alignment scores and the \\textsc{{chrF}} score in the \\texttt{{tgt}} language for {model.split('/')[1]}. Values are displayed as per cent. The improvement of the combination over the \\texttt{{en-tgt}} is shown right to the '/' symbol.", 
                                label=f"corr-{tgt_lang_cla}-with-chrf-{model.split('/')[1]}{'-HIGHEST' if layer_pooling == 'HIGHEST' else ''}",
                                custom_colspec="{X | c  c  c  c  c  c | c}",
                                eflomal_in_last_col=False,
                                heatmap=False, highlight_max=False))
        else:
            corr_table_combination, _ = calculate_table(model, layer_pooling, tgt_lang_cla, no_print)
    else:
        table_parts = [
            r"\setlength{\tabcolsep}{3pt}"
            r"\begin{table}[ht]",
            r"\centering\footnotesize",
            r"\begin{tabularx}{\columnwidth}%",
            r"{X | c  c  c  c  c  c | c}"
            r"\toprule",
        ]
        is_first = True
        for model_id in FULL_MODELS:
            if not is_first:
                table_parts.append(r"\midrule")
            is_first = False
            table_parts.append(r"\multicolumn{8}{c}{\textbf{" + model_id.split("/")[1] + r"}} \\")
            table_parts.append(r"\midrule")
            # Respect tgt_lang_cla modes per-model: 'both', 'combination', or others
            if tgt_lang_cla == "both":
                corr_table_en, _ = calculate_table(model_id, layer_pooling, "en", no_print=True)
                corr_table_tgt, _ = calculate_table(model_id, layer_pooling, "tgt", no_print=True)

                max_en = np.nanmax(corr_table_en)
                max_tgt = np.nanmax(corr_table_tgt)

                df_both = pd.DataFrame(
                              index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS], 
                              columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS])
                
                for m_i, metric in enumerate(METRICS):
                    for s_i, sent_rep in enumerate(SENT_REPS):
                        val_en = corr_table_en[s_i, m_i]
                        val_tgt = corr_table_tgt[s_i, m_i]
                        if metric == "eflomal":
                            eflomal_score_en = corr_table_en[SENT_REPS.index("mean"), m_i]
                            eflomal_score_tgt = corr_table_tgt[SENT_REPS.index("mean"), m_i]
                            if s_i != len(SENT_REPS)//2:
                                df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"\\percentGrad{{{int(round(eflomal_score_en*100, 0))}}}{{2}}"
                            else:
                                df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"{preprocess_value(eflomal_score_en, max_en)}{'\\hspace{0.6em}' if not np.isnan(eflomal_score_en) else ''}/{preprocess_value(eflomal_score_tgt, max_tgt, color=False)}"
                        else:
                            df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"{preprocess_value(val_en, max_en)}{'\\hspace{0.6em}' if not np.isnan(val_en) else ''}/{preprocess_value(val_tgt, max_tgt, color=False)}"

                table_parts.append(df_to_tex(df_both, label="", grad_command="\\percentGrad", highlight_max=True, cells_only=True, caption="", eflomal_in_last_col=False))
            elif tgt_lang_cla == "combination":
                corr_table_combination, alphas_table = calculate_table(model_id, layer_pooling, "combination", no_print=True, return_bouquet=return_bouquet)
                print(model_id)
                print(pd.DataFrame(alphas_table, index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS], columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS]))
                corr_prior, _ = calculate_table(model_id, layer_pooling, "en", no_print=True, return_bouquet=return_bouquet)
                improvement = corr_table_combination - corr_prior
                max_comb = np.nanmax(corr_table_combination)
                max_imp = np.nanmax(improvement)

                df_both = pd.DataFrame(
                              index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS], 
                              columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS])
                
                for m_i, metric in enumerate(METRICS):
                    for s_i, sent_rep in enumerate(SENT_REPS):
                        val_comb = corr_table_combination[s_i, m_i]
                        val_imp = improvement[s_i, m_i]
                        if metric == "eflomal":
                            eflomal_score_comb = corr_table_combination[SENT_REPS.index("mean"), m_i]
                            eflomal_score_imp = improvement[SENT_REPS.index("mean"), m_i]
                            if s_i != len(SENT_REPS)//2:
                                df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"\\percentGrad{{{int(round(eflomal_score_comb*100, n_digits))}}}{{2}}"
                            else:
                                df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"{preprocess_value(eflomal_score_comb, max_comb)}{'\\hspace{0.6em}' if not np.isnan(eflomal_score_comb) else ''}/{preprocess_value(eflomal_score_imp, max_imp, color=False)}"
                        else:
                            df_both.loc[REPS_SHORT_NAMES.get(sent_rep, sent_rep), METRICS_SHORT_NAMES.get(metric, metric)] = f"{preprocess_value(val_comb, max_comb)}{'\\hspace{0.6em}' if not np.isnan(val_comb) else ''}/{preprocess_value(val_imp, max_imp, color=False)}"

                table_parts.append(df_to_tex(df_both, label="", grad_command="\\percentGrad", highlight_max=True, cells_only=True, caption="", eflomal_in_last_col=False))
            else:
                corr_table, _ = calculate_table(model_id, layer_pooling, tgt_lang_cla, no_print=True)
                df = pd.DataFrame(corr_table,
                                  index=[REPS_SHORT_NAMES.get(s, s) for s in SENT_REPS],
                                  columns=[METRICS_SHORT_NAMES.get(m, m) for m in METRICS])
                table_parts.append(df_to_tex(df, label="", grad_command="\\percentGrad", highlight_max=True, cells_only=True, caption="", eflomal_in_last_col=False))
        table_parts += [
            r"\bottomrule",
            r"\end{tabularx}",
            fr"\caption{{Pearson correlation measured first across the \\texttt{{src}} languages with a fixed \\texttt{{tgt}} language and then averaged over the target language. The correlation is measured between the \\texttt{{src-{tgt_lang_cla}}} alignment score and the \\textsc{{chrF}} score in the \\texttt{{tgt}} language for all models. Values are displayed as per cent.}}",
            fr"\label{{corr-{tgt_lang_cla}-with-chrf-all-{layer_pooling}{'-validation' if return_bouquet else ''}}}",
            r"\end{table}"
        ]
        os.makedirs("evaluation/tables", exist_ok=True)
        with open(f"evaluation/tables/corr_{tgt_lang_cla}_with_chrf_all_{'-HIGHEST' if layer_pooling == 'HIGHEST' else ''}{'-validation' if return_bouquet else ''}.tex", "w") as f:
            f.write("\n".join(table_parts))

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model, snakemake.params.layer_pooling, snakemake.params.tgt_lang_cla)
    else:
        parser = argparse.ArgumentParser(description="Calculate the correlations between src-tgt and chrF for translation task.")
        parser.add_argument("--model", type=str, help="Model ID.", default="ALL")
        parser.add_argument("--layer_pooling", type=str, choices=["MEAN", "HIGHEST", "best"], default="HIGHEST", help="Layer pooling strategy.")
        parser.add_argument("--tgt-lang-cla", type=str, choices=["en", "tgt", "both", "partial-corr", "combination"], default="combination", help="Whether to use 'en' or 'tgt' languages for CLA calculation.")
        parser.add_argument("--bouquet_validation", action="store_true", help="Whether to return the bouquet validation correlation instead of the flores correlation.")
        args = parser.parse_args()
        main(args.model, args.layer_pooling, args.tgt_lang_cla, return_bouquet=args.bouquet_validation)