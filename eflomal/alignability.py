# modified from https://github.com/KathyHaem/token-alignability/blob/main/alignability.py

import argparse
import json
import math
import os
from collections import defaultdict, OrderedDict, Counter
from tqdm import tqdm

import numpy as np

# from constants import MODEL_VOCAB_SIZES, TOKENIZERS_BY_TYPE

ALL_LANGUAGES = ['af', 'am', 'ar', 'az', 'bg', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'he',
                 'hi', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'km', 'kn', 'ko', 'lt', 'ml', 'mr', 'nl',
                 'pl', 'pt', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'zh']


def count_alignments(alignments_line, tok_line):
    alignments = alignments_line.split(" ")
    src_tok, tgt_tok = tok_line.split(" ||| ")
    src_tok = src_tok.split(" ")
    tgt_tok = tgt_tok.split(" ")

    fwd = defaultdict(list)
    rev = defaultdict(list)
    count_nulls_src = 0
    count_nulls_tgt = 0
    count_onetoone = 0
    count_onetomany = 0
    count_manytoone = 0
    count_manytomany_src = 0
    count_manytomany_tgt = 0

    for pair in alignments:
        src, tgt = pair.split("-")
        src = int(src)
        tgt = int(tgt)
        fwd[src].append(tgt)
        rev[tgt].append(src)

    last_idx_src = len(src_tok)
    last_idx_tgt = len(tgt_tok)

    for i in range(last_idx_src):
        if not fwd[i]:
            count_nulls_src += 1
        elif len(fwd[i]) == 1 and len(rev[fwd[i][0]]) == 1:
            count_onetoone += 1
        elif len(fwd[i]) > 1:
            many = [0 if len(rev[j]) == 1 else 1 for j in fwd[i]]
            if any(many):
                count_manytomany_src += 1
            else:
                count_onetomany += 1

    for i in range(last_idx_tgt):
        if not rev[i]:
            count_nulls_tgt += 1
        # don't double-count onetoone
        elif len(rev[i]) > 1:
            many = [0 if len(fwd[j]) == 1 else 1 for j in rev[i]]
            if any(many):
                count_manytomany_tgt += 1
            else:
                count_manytoone += 1

    return {
        "nulls_src": count_nulls_src,
        "nulls_tgt": count_nulls_tgt,
        "one_to_one": count_onetoone,
        "one_to_many": count_onetomany,
        "many_to_one": count_manytoone,
        "many_to_many_src": count_manytomany_src,
        "many_to_many_tgt": count_manytomany_tgt,
        "len_src": last_idx_src,
        "len_tgt": last_idx_tgt
    }


def aggregate_counts_onetoone_prop(counter):
    prop_onetoone_src = counter["one_to_one"] / counter["len_src"]
    prop_onetoone_tgt = counter["one_to_one"] / counter["len_tgt"]
    return prop_onetoone_src, prop_onetoone_tgt


def aggregate_counts_nulls_prop(counter):
    prop_nulls_src = counter["nulls_src"] / counter["len_src"]
    prop_nulls_tgt = counter["nulls_tgt"] / counter["len_tgt"]
    return prop_nulls_src, prop_nulls_tgt


def read_priors_file(model_type, src_lang, tgt_lang, dataset="opus-100", subset="train", aligner="eflomal"):
    if subset:
        filename = f"{dataset}-{subset}/{model_type}/{src_lang}-{tgt_lang}.{aligner}.prior"
    else:
        filename = f"{dataset}/{model_type}/{src_lang}-{tgt_lang}.{aligner}.prior"

    if not os.path.exists(filename):
        # if the specified file (usually OPUS) doesn't exist, it might be a "fallback" language with a cc-aligned file
        dataset = "cc-aligned"
        filename = f"{dataset}-{subset}/{model_type}/{src_lang}-{tgt_lang}.{aligner}.prior"

    if not os.path.exists(filename):
        # trying one more thing basically:
        dataset = "multi-cc"
        filename = f"{dataset}/{model_type}/{src_lang}-{tgt_lang}.{aligner}.prior"

    if not os.path.exists(filename):
        # if that doesn't work either, tell calling function to move on
        return None

    # will later need to query cases that don't have a matching prior. add a significant penalty for unsupported align.
    priors = defaultdict(lambda: defaultdict(lambda: -20.0))
    total_priors_count = 0
    with open(filename, "r") as f:
        for line in f:
            prior_type = line.split("\t")[0]
            if prior_type != "LEX":
                break
            _, src, tgt, count = line.split("\t")
            total_priors_count += float(count)
            priors[src][tgt] = float(count)
    return priors


def distribution_from_frequencies(frequencies: dict) -> OrderedDict:
    sum_freq = np.sum(list(frequencies.values()))
    # normalises by sum of frequencies
    distribution = OrderedDict([(tok, freq / sum_freq) for tok, freq in frequencies.items()])
    return distribution


def translate_distribution(src_dist: OrderedDict[str, float], tgt_dist, priors):
    """ the 'matrix multiplication' step """
    tgt_transformed = OrderedDict().fromkeys(tgt_dist.keys(), 0.0)
    for token, prob in src_dist.items():
        for tgt_tok, prior in priors[token].items():
            tgt_transformed[tgt_tok] += prior * prob
    tgt_transformed = np.array(list(tgt_transformed.values()))
    norm = np.sum(tgt_transformed)
    return tgt_transformed / norm


def read_distribution(lang, model_type):
    frequencies_dir = f"./tok_eval_out/{model_type}"
    stats_path = os.path.join(frequencies_dir, f"token_freq_{lang}_decoded.json")
    if not os.path.isfile(stats_path):
        return None
    freqs = json.load(open(stats_path, 'r'))
    freqs = OrderedDict([(tok, freq) for tok, freq in sorted(freqs.items(), key=lambda item: item[0])])
    dist = distribution_from_frequencies(freqs)
    return dist


def invert_priors_dict(priors):
    inv_priors = defaultdict(lambda: defaultdict(float))
    for src_tok, tgt_dict in priors.items():
        for tgt_tok, prior in tgt_dict.items():
            inv_priors[tgt_tok][src_tok] += prior
    return inv_priors



def line_prob_by_priors(priors, alignments_line, tok_line):
    alignments = alignments_line.split(" ")
    src_tok, tgt_tok = tok_line.split(" ||| ")
    src_tok = src_tok.split(" ")
    tgt_tok = tgt_tok.split(" ")
    # punishes having few alignments for a long target sequence, and punishes significant over-tokenisation on one side
    # normalises by number of alignments overall
    norm_factor = len(alignments) * (len(alignments) / len(tgt_tok)) * min(
        (len(src_tok) / len(tgt_tok), (len(tgt_tok) / len(src_tok))))
    line_count = 0
    for pair in alignments:
        src, tgt = pair.split("-")
        src = int(src)
        tgt = int(tgt)
        src_token = src_tok[src]
        tgt_token = tgt_tok[tgt]
        line_count += math.log(priors[src_token][tgt_token])

    return line_count / norm_factor


def read_eflomal_scores(model_id, src_lang, tgt_lang):
    filename_fwd = f"eflomal/computations/{model_id.split('/')[-1]}/{src_lang}-{tgt_lang}.eflomal.scores.fwd"
    filename_rev = f"eflomal/computations/{model_id.split('/')[-1]}/{src_lang}-{tgt_lang}.eflomal.scores.rev"
    with open(filename_fwd, "r") as f:
        fwd = f.readlines()
        fwd_scores = [float(line.strip()) for line in fwd]

    with open(filename_rev, "r") as f:
        rev = f.readlines()
        rev_scores = [float(line.strip()) for line in rev]

    fwd_scores = np.array(fwd_scores)
    rev_scores = np.array(rev_scores)

    # drop inf values
    fwd_scores = fwd_scores[np.isfinite(fwd_scores)]
    rev_scores = rev_scores[np.isfinite(rev_scores)]


    fwd_mean = np.mean(fwd_scores)
    rev_mean = np.mean(rev_scores)
    return fwd_mean, rev_mean


def construct_outputs(model_id, src_langs, target_langs):
    outputs_dict = defaultdict(dict)
    for src_lang in tqdm(src_langs, desc="Processing source languages"):
        for tgt_lang in target_langs:
            if src_lang == tgt_lang:
                continue

            # read alignments and actual tokenised corpus
            sym_file = f"eflomal/computations/{model_id.split('/')[-1]}/{src_lang}-{tgt_lang}.eflomal.sym"
            tok_file = f"eflomal/computations/{model_id.split('/')[-1]}/{src_lang}-{tgt_lang}.tok.fast_align"

            if os.path.isfile(sym_file) and os.path.isfile(tok_file):
                with open(sym_file, "r") as afile:
                    sym_alignments = afile.readlines()
                with open(tok_file, "r") as tfile:
                    tok_lines = tfile.readlines()

                if not sym_alignments or not tok_lines:
                    print(f"Warning: Skipping empty file for {src_lang}-{tgt_lang}")
                    continue

                # do all the per-line stuff
                lines = zip(sym_alignments, tok_lines)
                # prior_counts = 0
                counter = Counter()
                for align_line, tok_line in lines:
                    counter.update(count_alignments(align_line, tok_line))
                    # prior_counts += line_prob_by_priors(priors, align_line, tok_line)

                eflomal_fwd, eflomal_rev = read_eflomal_scores(model_id, src_lang, tgt_lang)
                eflomal_mean = (eflomal_fwd + eflomal_rev) / 2
                outputs_dict[tgt_lang][src_lang] = eflomal_mean
                outputs_dict[src_lang][tgt_lang] = eflomal_mean
                continue
            # simplest way of moving on if file doesn't exist, can also backfire next time I mess up my filenames.
            # but basically since we don't have everything in both directions (and it doesn't make sense to do).:
            print(f"Warning: Skipping missing file for {src_lang}-{tgt_lang}")
    return outputs_dict


def main(model_id, langs):
    os.makedirs("scores/eflomal/", exist_ok=True)
    output_file = f"scores/eflomal/{model_id.split('/')[-1]}.json"
    outputs_dict = construct_outputs(model_id, langs, langs)

    with open(output_file, "w+", encoding="utf-8") as out_file:
        json.dump(outputs_dict, out_file, indent=2)


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model, snakemake.params.langs)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", help="Model ID, as in HuggingFace.")
        parser.add_argument("--langs", type=str, nargs="+", default=ALL_LANGUAGES, help="Short language name")
        args = parser.parse_args()
        main(args.model, args.langs)