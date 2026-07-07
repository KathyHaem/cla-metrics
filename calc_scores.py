import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import AutoConfig

from anc.anc_scoring import anc_score
from constants import ALL_LANGUAGES
from xsim.xsim import Margin, calculate_error


def load_embeds(model_name, dataset_name, sent_rep, lang):
    model_short_name = model_name.split("/")[-1]
    embeds_file = f"embeds/{dataset_name}/{model_short_name}-{sent_rep}-{lang}.pickle"
    with open(embeds_file, "rb") as fin:
        embeds = pickle.load(fin)
    return embeds[lang]


def cosine_sim(src_layer: torch.Tensor, tgt_layer: torch.Tensor):
    # TODO you know what I should probably actually do correct-pair sim normalised by any-pair sim
    # ...let's try and implement that efficiently though if we do it
    score_value = cosine_similarity(src_layer, tgt_layer)
    score = score_value.mean().cpu().item()
    return score


def calculate_score(src_layer: torch.Tensor, tgt_layer: torch.Tensor, score: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if score in ("dist", "ratio", "nn-abs"):
        # I tried to make xsim work on GPU but it actually got slower. The best option is to use more CPUs.
        torch.set_float32_matmul_precision('high')
        src_layer = src_layer.to(dtype=torch.float32, device="cpu").numpy()
        tgt_layer = tgt_layer.to(dtype=torch.float32, device="cpu").numpy()
    else:
        src_layer = src_layer.to(device)
        tgt_layer = tgt_layer.to(device)

    match score:
        case "cosine":
            return cosine_sim(src_layer, tgt_layer)
        case "anc":
            return anc_score(src_layer, tgt_layer)
        case "dist":
            # this is an xsim score, based on nearest neighbours -> needs faiss
            error_abs, num, _ = calculate_error(x=src_layer, y=tgt_layer, margin=Margin.DISTANCE.value)
            return 1 - error_abs/num
        case "ratio":
            error_abs, num, _ = calculate_error(x=src_layer, y=tgt_layer, margin=Margin.RATIO.value)
            return 1 - error_abs/num
        case "nn-abs":
            error_abs, num, _ = calculate_error(x=src_layer, y=tgt_layer, margin=Margin.ABSOLUTE)
            return 1 - error_abs/num
        case _:
            raise NotImplementedError()


def collect_score(model, dataset, config, score, sent_rep):
    scores = defaultdict(list)  # keys are lang pairs, lists are one score per layer
    num_layers = config.num_hidden_layers + 1
    for src_lang in ALL_LANGUAGES:
        print("Processing language", src_lang)
        src_embeds = load_embeds(model, dataset, sent_rep, src_lang)

        for tgt_lang in tqdm(ALL_LANGUAGES):
            if src_lang == tgt_lang:
                continue
            tgt_embeds = load_embeds(model, dataset, sent_rep, tgt_lang)

            for layer in range(num_layers):
                src_layer = src_embeds[f"{sent_rep}_{layer}"]
                tgt_layer = tgt_embeds[f"{sent_rep}_{layer}"]

                score_value = calculate_score(src_layer, tgt_layer, score)
                scores[f"{src_lang}-{tgt_lang}"].append(score_value)
    return scores


def main(model, dataset, sent_rep, requested_scores, overwrite=False):
    all_scores = {x: {} for x in requested_scores}
    # actually save scores somewhere
    model_short_name = model.split("/")[-1]
    out_path = f"scores/{dataset}/"
    os.makedirs(out_path, exist_ok=True)

    config = AutoConfig.from_pretrained(model)
    if "num_hidden_layers" not in config:
        config.num_hidden_layers = config.text_config.num_hidden_layers

    for score in requested_scores:
        out_filename = f"{out_path}/{model_short_name}_{sent_rep}_{score}.json"
        if os.path.exists(out_filename) and not overwrite:
            print(f"Already collected {score} scores. Skipping.")
            continue

        scores = collect_score(model, dataset, config, score, sent_rep)
        all_scores[score] = scores

        with open(out_filename, "w") as fout:
            json.dump(scores, fout)


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model, 
             snakemake.params.dataset, 
             snakemake.params.sent_rep, 
             snakemake.params.score, 
             snakemake.params.overwrite)
    else:
        parser = argparse.ArgumentParser(description="Calculate simple-ish CLA scores")
        parser.add_argument("--model", type=str, required=True, help="Model name. Assuming an HF decoder")
        parser.add_argument("--dataset", type=str, default="flores")
        parser.add_argument("--score", type=str, nargs="+", default=["cosine"],
                            choices=["cosine", "anc", "dist", "ratio", "nn-abs"])
        parser.add_argument("--sent-rep", type=str, default="mean", help="How to sentence rep",
                            choices=["mean", "prompt", "fewshot", "last-token", "weighted-mean"])
        parser.add_argument("--overwrite", action="store_true", default=False)

        args = parser.parse_args()
        main(args.model, args.dataset, args.sent_rep, args.score, args.overwrite)
