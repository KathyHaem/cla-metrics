import argparse
import os
import pickle

import torch
from datasets import load_dataset
from langcodes import Language
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from constants import SENT_SUMM_TEMPLATE, ALL_LANGUAGES


def get_flores_code(short_code: str):
    lang = Language.get(short_code)
    script = lang.script or lang.assume_script().script
    return "arb_Arab" if short_code == "ar" \
        else "zho_Hant" if short_code == "zh" \
        else "azj_Latn" if short_code == "az" \
        else "pes_Arab" if short_code == "fa" \
        else "jav_Latn" if short_code == "jv" \
        else "kor_Hang" if short_code == "ko" \
        else "tgl_Latn" if short_code == "tl" \
        else "swh_Latn" if short_code == "sw" else f"{lang.to_alpha3()}_{script}"


def masked_mean(data, mask):
    return (data * mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True)


def encode_batch(batch, model, tokenizer, sent_rep, key="sentence"):
    out_dict = {}
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if sent_rep == "prompt":
        batch[key] = [f"{SENT_SUMM_TEMPLATE.format(sent=sent)} {sent}" for sent in batch[key]]
        inputs = tokenizer(batch[key], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            for layer in range(model.config.num_hidden_layers):
                rep = outputs.hidden_states[layer][:, -1, :]
                out_dict[f"prompt_{layer}"] = rep
    elif sent_rep == "mean":
        inputs = tokenizer(batch[key], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            mask = inputs["attention_mask"]
            for layer in range(model.config.num_hidden_layers):
                hidden_states = outputs.hidden_states[layer]  # is layer 0 embeddings?
                mean_rep = masked_mean(hidden_states, mask).cpu()
                out_dict[f"mean_{layer}"] = mean_rep

    return out_dict


def main(dataset_name, model_name, langs, sent_rep, overwrite=False):
    if dataset_name != "flores":  # happy path or w/e. bit more effort if I do implement other datasets
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda()

    model_short_name = model_name.split("/")[-1]
    out_path = f"embeds/{dataset_name}"
    os.makedirs(out_path, exist_ok=True)

    # We may have saved a previous version where some languages were already processed.
    out_file = f"{out_path}/{model_short_name}.pickle"
    if os.path.exists(out_file) and not overwrite:
        print(f"Loading existing pickle file {out_file}")
        with open(out_file, "rb") as fin:
            out_dict = pickle.load(fin)
    else:
        out_dict = {}

    for lang in langs:
        if lang in out_dict:
            print(f"Already processed {lang}. Skipping.")
            continue

        flores_code = get_flores_code(lang)
        print(f"Loading {lang}. Trying to use flores code {flores_code}")
        try:
            dataset = load_dataset("facebook/flores", flores_code, trust_remote_code=True)["devtest"]  # todo use dev?
        except Exception as e:
            print(f"Couldn't find flores code {flores_code}. Will skip.")
            continue

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        results = {f"{sent_rep}_{layer}": [] for layer in range(model.config.num_hidden_layers)}

        for batch in tqdm(dataloader, desc=f"Processing {lang} / {flores_code}"):
            batch_results = (encode_batch(batch, model, tokenizer, sent_rep))
            for key, value in batch_results.items():
                results[key].append(value)
        out_dict[lang] = {k: torch.cat(v, dim=0) for k, v in results.items()}

        # save out_dict in pickle file ... does it work if we just simply update it like this after every language??

        print(f"Updating pickle file to save {lang} / {flores_code}")
        with open(out_file, "wb") as fout:
            pickle.dump(out_dict, fout)

    missing_langs = set(langs) - set(out_dict.keys())
    print(f"Still missing: {missing_langs}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
    parser.add_argument("--dataset", type=str, default="flores", help="Dataset name. Just FLORES atm")
    parser.add_argument("--model", type=str, required=True, help="Model name. Assuming an HF decoder")
    # might do model short names again
    parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                        default=ALL_LANGUAGES)

    parser.add_argument("--sent-rep", type=str, default="mean", help="How to sentence rep",
                        choices=["mean", "prompt"])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    # layers? probably just do all of them?

    args = parser.parse_args()
    main(args.dataset, args.model, args.langs, args.sent_rep, args.overwrite)
