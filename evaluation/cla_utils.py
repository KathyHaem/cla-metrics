import json
import os
from typing import Union, Literal
from collections.abc import Iterable
import numpy as np
import pandas as pd
import re
from langcodes import Language


from constants import ALL_LANGUAGES

METRICS = ["cosine", "anc", "nn-abs", "dist", "ratio", "dali", "eflomal"]
SENT_REPS = ["fewshot", "last-token", "weighted-mean", "mean", "prompt"]

REPS_SHORT_NAMES = {
    "last-token": "last",
    "weighted-mean": "p-wm"
}

METRICS_SHORT_NAMES = {
    "cosine": "cos",
    "anc": "ANC",
    "nn-abs": "abs",
    "dist": "dist",
    "ratio": "ratio",
    "dali": "Dali",
    "eflomal": "Eflo"
}

FULL_MODELS = [
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-14B",
    "google/gemma-3-12b-pt",
    "google/gemma-3-12b-it",
    "mistralai/Ministral-3-14B-Base-2512",
    "mistralai/Ministral-3-14B-Instruct-2512",
]

MODEL_FAMILIES = ["Qwen3", "gemma-3", "Ministral-3"]

# shorten models: "mistralai/Ministral-3-14B-Base-2512" -> "Ministral-3"
SHORT_MODELS = [re.search(r"([A-Za-z]+[^a-zA-Z\d\s:]?[0-9]+).*", model.split("/")[1]).group(1) for model in FULL_MODELS]
SHORT_MODELS_DICT = {model: re.search(r"([A-Za-z]+[^a-zA-Z\d\s:]?[0-9]+).*", model.split("/")[1]).group(1) for model in FULL_MODELS}

def get_cla_all(model: str) -> dict:
    """
    Returns dict containing the alignment scores for each metric and sentence representation.
    
    :param model: Huggingface name of the model (only the part after the '/')
    :type model: str
    :return: Format: `cla_all["<metric>"]["<sent_rep>"]["<src>-<tgt>"]: float`
    :rtype: dict
    """

    cla_all = dict()

    for metric in METRICS:
        cla_all[metric] = dict()

        if metric == "eflomal":
            with open(os.path.join("scores", "eflomal", f"{model}.json")) as f:
                eflomal_dict = json.load(f)
            eflomal_scores = dict()
            for lang1, record in eflomal_dict.items():
                for lang2, value in record.items():
                    eflomal_scores[f"{lang1}-{lang2}"] = [-value]
            cla_all[metric]["mean"] = eflomal_scores

        for sent_rep in SENT_REPS:

            if metric == "dali":
                if sent_rep == "last-token":
                    with open(os.path.join("scores", "dali_belebele", f"{model}.json")) as f:
                        cla_all[metric][sent_rep] = json.load(f)
                continue

            if metric == "eflomal":
                if sent_rep == "mean":
                    with open(os.path.join("scores", "eflomal", f"{model}.json")) as f:
                        eflomal_dict = json.load(f)
                    eflomal_scores = dict()
                    for lang1, record in eflomal_dict.items():
                        for lang2, value in record.items():
                            eflomal_scores[f"{lang1}-{lang2}"] = [-value]
                    cla_all[metric][sent_rep] = eflomal_scores

            if os.path.isfile(os.path.join("scores", "flores", f"{model}_{sent_rep}_{metric}.json")):
                with open(os.path.join("scores", "flores", f"{model}_{sent_rep}_{metric}.json")) as f:
                    cla_all[metric][sent_rep] = json.load(f)
    
    return cla_all

def get_alignment(cla: dict, 
                  source_lang: str, 
                  target_lang: str = "MEAN", 
                  exclude_targets: Iterable[str] = [], 
                  layer: Union[int, Literal["HIGHEST", "MEAN"]] = "MEAN"
                  ) -> float:
    """
    Returns the alinment score.
    
    :param cla: A `dict` with `"<src>-<tgt>"` as keys and list of alignments (one for each layer) as values.
    :type cla: dict
    :param source_lang: Source language: 2 character code.
    :type source_lang: str
    :param target_lang: Target language: 2 character code ot `"MEAN"` to use all the languages except for `exclude_targets`.
    :type target_lang: str
    :param exclude_targets: Target languages to exclude when `target_lang` is set to `"MEAN"`
    :type exclude_targets: Iterable[str]
    :param layer: Layer index or `"HIGHEST"` or `"MEAN"`.
    :type layer: Union[int, Literal["HIGHEST", "MEAN"]]
    """
    if target_lang != "MEAN":
        if target_lang == source_lang:
            return 1
        layer_scores = cla[f"{source_lang}-{target_lang}"]
        if layer == "HIGHEST":
            return max(layer_scores)
        if layer == "MEAN":
            return np.mean(layer_scores)
        return layer_scores[layer]
    
    targ_langs = sorted(set(ALL_LANGUAGES).difference(exclude_targets + [source_lang]))
    aligments_matrix = np.zeros((len(targ_langs), len(next(cla.values().__iter__()))))
    for i, tgt_lang in enumerate(targ_langs):
        if tgt_lang == source_lang:
            continue
        aligments_matrix[i] = cla[f"{source_lang}-{tgt_lang}"]
    mean_scores = aligments_matrix.mean(axis=0)
    if layer == "HIGHEST":
        return max(mean_scores)
    if layer == "MEAN":
        return mean_scores.mean()
    
    return mean_scores[layer]

def best_aligned_layer_all(cla_all: dict) -> dict:
    """
    For each metric and sentence representation, computes the dict containing the layer index that has the highest alignments when averaged per language pair.
    

    :param cla_all: Dict of alignment scores , as obtained from `get_cla_all` .
    :type cla_all: dict
    :return: Format: `best_layers_all["<metric>"]["<sent_rep>"]["<src>-<tgt>"]: int`
    :rtype: dict
    """

    def get_best_aligned_layer(cla):
        alignments = np.array(list(cla.values()))
        mean_align = alignments.mean(axis=0)
        return mean_align.argmax()

    best_layers_all = dict()

    for metric in METRICS:
        best_layers_all[metric] = dict()
        for sent_rep in SENT_REPS:
            try:
                best_layers_all[metric][sent_rep] = get_best_aligned_layer(cla_all[metric][sent_rep])
            except KeyError:
                continue

def df_to_tex(df: pd.DataFrame,
              caption: str,
              label: str,
              use_index_column=True,
              heatmap=True,
              grad_command="",
              highlight_max=False,
              highlight_max_in_each_row=False,
              eflomal_in_last_col=True,
              cells_only=False,
              custom_colspec="",
              custom_header="",
              rounding_digits=1) -> str:
    """
    Converts the dataframe into a formate TeX table.

    :param df: dataframe to convert
    :type df: pd.DataFrame
    :param caption: Table caption
    :type caption: str
    :param label: Table label, letters only. `tab:` will be prepended.
    :type label: str
    :param use_index_column: Print index as the first column?
    :type use_index_column: bool
    :param heatmap: Color the cells based on teir values?
    :type heatmap: bool
    :param grad_command: name of the existing gradientcell command for coloring. If not specified, a new command will be created with min and max values inferred from the table.
    :type grad_command: str 
    :param highlight_max: Highlight the maximum value in the table?
    :type highlight_max: bool
    :param highlight_max_in_each_row: Highlight the maximum value in each row instead of the whole table?
    :type highlight_max_in_each_row: bool
    :param eflomal_in_last_col: Is Eflomal column in the last column?
    :type eflomal_in_last_col: bool
    :param cells_only: Only print the cells, without the tabularx environment and caption?
    :type cells_only: bool
    :param custom_colspec: Custom column specification for the tabularx environment.
    :type custom_colspec: str
    :param custom_header: Custom header for the table.
    :type custom_header: str
    :param rounding_digits: Number of digits to round the values to.
    :type rounding_digits: int

    :return: A TeX table string.
    :rtype: str
    """
    def print_value(val, max_val, background_only=False):
        if isinstance(val, str):
            return val

        if np.isnan(val):
            return "--"

        percents = round(val*100, rounding_digits) + 0.0 # to avoid -0.0

        if heatmap:
            if background_only:
                return f"{grad_command}{{{percents}}}{{2}}"
            if highlight_max and max_val - percents <= 0.1:
                return f"{grad_command}{{{percents}}}{{1}}"
            return f"{grad_command}{{{percents}}}{{0}}"
        if highlight_max and max_val - percents <= 0.1:
            return f"\\textbf{{{percents}}}"
        return f"{percents}"

    def make_alphabetic_command_name(label: str) -> str:
        def process_char(c):
            if c.isalpha():
                return c
            if c.isdigit():
                return chr(ord("A") + int(c))
            else:
                return ""
        
        return "".join(process_char(c) for c in label)
    
    if eflomal_in_last_col:
        # extract the only non-nan value
        eflomal_value = df["Eflo"][df["Eflo"].first_valid_index()]
    
    grad_command_defined = False
    if not grad_command:
        grad_command = f'\\{make_alphabetic_command_name(label)}Grad'
        grad_command_defined = True

    max_value = None
    if (heatmap and grad_command_defined) or highlight_max:
        max_value = df.max(axis=None)*100
    lines = []
    if not cells_only:
        if heatmap and grad_command_defined:
            lines.append(f'\\newcommand{{{grad_command}}}[2]{{\\gradientcell{{#1}}{{{df.min(axis=None)*100}}}{{{max_value}}}{{cyan}}{{yellow}}{{70}}{{#2}}}}')
        lines += [
            r'\setlength{\tabcolsep}{4.5pt}',
            r'\begin{table}[ht]',
            r'    \centering\footnotesize',
            f'    \\begin{{tabularx}}{{\\columnwidth}}%',
            custom_colspec if custom_colspec else f'    {{{"p{4em} | " if use_index_column else ""}{("  ".join(["C"] * (len(df.columns) - 1)) + (" | C" if eflomal_in_last_col else "  C"))}}}',
            r'        \toprule',
        ]
    lines += [
        custom_header if custom_header else f'        {" & " if use_index_column else ""}{" & ".join([f"\\textbf{{{col}}}" for col in df.columns])} \\\\',
        r'        \midrule'
    ]

    for row_num, (i, row) in enumerate(df.iterrows()):
        lines += [
            f'        {"\\textbf{" + str(i) + "} & " if use_index_column else ""}{" & ".join([str(print_value(val, max_val=max(row)*100 if highlight_max_in_each_row else max_value)) for val in row])} \\\\' if not eflomal_in_last_col else f'        {"\\textbf{" + str(i) + "} & " if use_index_column else ""}{" & ".join([str(print_value(val, max_val=max_value)) for val in row[:-1]])} & {print_value(eflomal_value, background_only=row_num != df.shape[0]//2, max_val=max_value)} \\\\'

        ]
    
    if not cells_only:
        lines += [
            r'        \bottomrule',
            r'    \end{tabularx}',
            f'    \\caption{{{caption}}}',
            f'    \\label{{tab:{label}}}',
            r'\end{table}'
        ]

    return "\n".join(lines)


def get_translation_scores(model: str, metric: Literal["chrf", "mutinf"], dataset: str = "flores") -> dict:
    with open(os.path.join("scores", f"translation_{dataset}_{metric}", f"{model}.json")) as f:
        pmi_dict = json.load(f)

    return {k: v["mean"] for k, v in pmi_dict.items()}

def get_flores_code(short_code: str):
    custom_codes = {
        "ar": "arb_Arab",
        "zh": "zho_Hant",
        "az": "azj_Latn",
        "fa": "pes_Arab",
        "jv": "jav_Latn",
        "ko": "kor_Hang",
        "tl": "tgl_Latn",
        "sw": "swh_Latn"
    }

    if short_code in custom_codes:
        return custom_codes[short_code]

    lang = Language.get(short_code)
    script = lang.script or lang.assume_script().script
    return f"{lang.to_alpha3()}_{script}"