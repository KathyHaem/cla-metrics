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
from xsim.xsim import x_sim, Margin, get_xsim_correct_rate


def load_embeds(model_name, dataset_name):
    model_short_name = model_name.split("/")[-1]
    embeds_file = f"embeds/{dataset_name}/{model_short_name}.pickle"
    print(f"Loading embeddings from {embeds_file}")
    with open(embeds_file, "rb") as fin:
        embeds = pickle.load(fin)
    return embeds


def cosine_sim(src_layer, tgt_layer):
    # TODO you know what I should probably actually do correct-pair sim normalised by any-pair sim
    # ...let's try and implement that efficiently though if we do it
    src_layer.cuda()
    tgt_layer.cuda()
    score_value = cosine_similarity(src_layer, tgt_layer)
    score = score_value.mean().cpu().item()
    return score


def calculate_score(src_layer: torch.Tensor, tgt_layer: torch.Tensor, score: str):
    # each tensor shape num_examples x hidden_size
    if score == "cosine":
        return cosine_sim(src_layer, tgt_layer)

    src_layer = src_layer.numpy()
    tgt_layer = tgt_layer.numpy()
    if score == "anc":
        return anc_score(src_layer, tgt_layer)
    if score == "dist":
        # this is an xsim score, based on nearest neighbours -> needs faiss
        return get_xsim_correct_rate(x=src_layer, y=tgt_layer, margin=Margin.DISTANCE.value)
    if score == "ratio":
        return get_xsim_correct_rate(x=src_layer, y=tgt_layer, margin=Margin.RATIO.value)
    if score == "nn_abs":
        # this is an xsim score, based on nearest neighbours but with cosine as criterion
        return get_xsim_correct_rate(x=src_layer, y=tgt_layer, margin=Margin.ABSOLUTE.value)


def collect_score(config, embeds, score, sent_rep):
    scores = defaultdict(list)  # keys are lang pairs, lists are one score per layer
    num_layers = config.num_hidden_layers
    for src_lang in tqdm(ALL_LANGUAGES, desc=f"Collecting {score} scores"):
        if src_lang not in embeds:
            continue
        src_embeds = embeds[src_lang]

        for tgt_lang in ALL_LANGUAGES:
            if tgt_lang not in embeds:
                continue
            if src_lang == tgt_lang:
                continue
            tgt_embeds = embeds[tgt_lang]

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

    embeds = load_embeds(model, dataset)
    config = AutoConfig.from_pretrained(model)

    for score in requested_scores:
        out_filename = f"{out_path}/{model_short_name}_{sent_rep}_{score}.json"
        if os.path.exists(out_filename) and not overwrite:
            print(f"Already collected {score} scores. Skipping.")
            continue

        scores = collect_score(config, embeds, score, sent_rep)
        all_scores[score] = scores

        with open(out_filename, "w") as fout:
            json.dump(scores, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate simple-ish CLA scores")
    parser.add_argument("--model", type=str, required=True, help="Model name. Assuming an HF decoder")
    parser.add_argument("--dataset", type=str, default="flores")
    parser.add_argument("--score", type=str, nargs="+", default=["cosine"],
                        choices=["cosine", "anc", "dist", "ratio", "nn_abs"])
    parser.add_argument("--sent-rep", type=str, default="mean", help="How to sentence rep",
                        choices=["mean", "prompt"])
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()
    main(args.model, args.dataset, args.sent_rep, args.score, args.overwrite)
