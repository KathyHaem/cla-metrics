import argparse
import os
import pickle

import torch
from datasets import load_dataset
from langcodes import Language
from transformers import AutoModel, AutoTokenizer

from constants import SENT_SUMM_TEMPLATE, ALL_LANGUAGES


def get_flores_code(short_code: str):
    lang = Language.get(short_code)
    script = lang.script or lang.assume_script().script
    return "arb_Arab" if short_code == "ar" else "zho_Hant" if short_code == "zh" \
        else "swh_Latn" if short_code == "sw" else f"{lang.to_alpha3()}_{script}"


def masked_mean(data, mask):
    return (data * mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True)


def encode_batch(batch, model, tokenizer, sent_rep, key="sentence"):
    out_dict = {}
    if sent_rep == "prompt":
        batch[key] = [f"{SENT_SUMM_TEMPLATE.format(sent=sent)} {sent}" for sent in batch[key]]
        inputs = tokenizer(batch[key], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            for layer in range(model.config.num_hidden_layers):
                rep = outputs.hidden_states[layer][:, -1, :]
                out_dict[f"prompt_{layer}"] = rep
    elif sent_rep == "mean":
        inputs = tokenizer(batch[key], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            mask = inputs["attention_mask"]
            for layer in range(model.config.num_hidden_layers):
                hidden_states = outputs.hidden_states[layer]  # is layer 0 embeddings?
                mean_rep = masked_mean(hidden_states, mask)
                out_dict[f"mean_{layer}"] = mean_rep

    return out_dict


def main(dataset_name, model_name, langs, sent_rep):
    if dataset_name != "flores":  # happy path or w/e. bit more effort if I do implement other datasets
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    out_dict = {}
    for lang in langs:
        flores_code = get_flores_code(lang)
        dataset = load_dataset("facebook/flores", flores_code, trust_remote_code=True)["devtest"]  # todo use dev?

        # todo check that shit works
        dataset.map(encode_batch, batched=True,
                    fn_kwargs={"model": model, "tokenizer": tokenizer, "sent_rep": sent_rep},
                    remove_columns=dataset.column_names)
        out_dict[lang] = dataset

    # save out_dict in pickle file
    model_short_name = model_name.split("/")[-1]
    out_path = f"../embeds/{dataset_name}/"
    os.makedirs(out_path, exist_ok=True)
    out_file = f"{out_path}/{model_short_name}.pickle"
    with open(out_file, "wb") as fout:
        pickle.dump(out_dict, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
    parser.add_argument("--dataset", type=str, default="flores", help="Dataset name. Just FLORES atm")
    parser.add_argument("--model", type=str, required=True, help="Model name. Assuming an HF decoder")
    # might do model short names again
    parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                        default=ALL_LANGUAGES)

    parser.add_argument("--sent-rep", type=str, default="mean", help="How to sentence rep",
                        choices=["mean", "prompt"])
    # layers? probably just do all of them?

    args = parser.parse_args()
    main(args.dataset, args.model, args.langs, args.sent_rep)
