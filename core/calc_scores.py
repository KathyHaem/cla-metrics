import argparse
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoConfig

from anc.anc_scoring import anc_score
from constants import ALL_LANGUAGES
from xsim.xsim import x_sim, Margin


def load_embeds(model_name, dataset_name):
    model_short_name = model_name.split("/")[-1]
    embeds_file = f"../embeds/{dataset_name}/{model_short_name}.pickle"
    with open(embeds_file, "rb") as fin:
        embeds = pickle.load(fin)
    return embeds


def calculate_score(src_layer: torch.Tensor, tgt_layer: torch.Tensor, score: str):
    if score == "anc":
        return anc_score(src_layer, tgt_layer)
    if score == "cosine":
        score_value = cosine_similarity(src_layer, tgt_layer)
        return score_value.mean().item()

    src_layer = np.ndarray(src_layer)
    tgt_layer = np.ndarray(tgt_layer)
    if score == "dist":
        # this is an xsim score, based on nearest neighbours -> needs faiss
        err, nbex, augmented_report = x_sim(x=src_layer, y=tgt_layer, margin=Margin.DISTANCE.value)
        print(err, nbex, augmented_report)  # need to take a look at this
        return err
    if score == "ratio":
        err, nbex, augmented_report = x_sim(x=src_layer, y=tgt_layer, margin=Margin.RATIO.value)
        print(err, nbex, augmented_report)  # need to take a look at this
        return err
    if score == "nn_abs":
        # this is an xsim score, based on nearest neighbours but with cosine as criterion
        err, nbex, augmented_report = x_sim(x=src_layer, y=tgt_layer, margin=Margin.ABSOLUTE.value)
        print(err, nbex, augmented_report)  # need to take a look at this
        return err


def main(model, dataset, sent_rep, score):
    scores = defaultdict(list)  # keys are lang pairs, lists are layers
    embeds = load_embeds(model, dataset)
    config = AutoConfig.from_pretrained(model)
    num_layers = config.num_hidden_layers

    for src_lang in ALL_LANGUAGES:
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
                scores[(src_lang, tgt_lang)].append(score_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate simple-ish CLA scores")
    parser.add_argument("--model", type=str, required=True, help="Model name. Assuming an HF decoder")
    parser.add_argument("--dataset", type=str, default="flores")
    parser.add_argument("--score", type=str, default="cosine")  # anc, dist, ratio,...
    parser.add_argument("--sent-rep", type=str, default="mean", help="How to sentence rep",
                        choices=["mean", "prompt"])

    args = parser.parse_args()
    main(args.model, args.dataset, args.sent_rep, args.score)
