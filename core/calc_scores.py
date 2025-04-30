import argparse
import pickle
from collections import defaultdict

from transformers import AutoConfig

from constants import ALL_LANGUAGES


def load_embeds(model_name, dataset_name):
    model_short_name = model_name.split("/")[-1]
    embeds_file = f"../embeds/{dataset_name}/{model_short_name}.pickle"
    with open(embeds_file, "rb") as fin:
        embeds = pickle.load(fin)
    return embeds


def calculate_score(src_layer, tgt_layer, score):
    # TODO
    pass


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
            tgt_embeds = embeds[tgt_lang]
            if src_lang == tgt_lang:
                continue

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
