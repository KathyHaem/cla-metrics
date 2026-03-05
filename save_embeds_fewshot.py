import argparse
import os
import pickle

import torch
from datasets import load_dataset, Dataset
from langcodes import Language
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, cache_utils, AutoModelForCausalLM, AutoModelForImageTextToText
import json
from copy import deepcopy

from constants import ALL_LANGUAGES

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


@torch.no_grad()
def broadcast_cache(cache: cache_utils.DynamicCache, batch_size: int) -> tuple[cache_utils.DynamicCache, torch.Tensor]:
    local_cache = deepcopy(cache)
    for layer in local_cache.layers:
        layer.keys = layer.keys[0:1].expand((batch_size, -1, -1, -1)).clone()
        layer.values = layer.values[0:1].expand((batch_size, -1, -1, -1)).clone()
    return local_cache, torch.ones((batch_size, layer.values.size(2)), device=layer.values.device)

@torch.no_grad()
def cache_prefix(model, tokenizer, lang, glossary, batch_size):
    prompt = "\n\n".join([f"{gloss['description']}\n{gloss['label']}" for gloss in glossary[lang]]) + "\n\n"
    tokenized = tokenizer([prompt], return_tensors="pt")
    t_ids = tokenized.input_ids.to("cuda:0")
    at_mask = tokenized.attention_mask.to("cuda:0")
    outputs = model(t_ids, attention_mask=at_mask, use_cache=True)
    return broadcast_cache(outputs.past_key_values, batch_size)

@torch.no_grad()
def encode_batch(model, tokenizer, kv_cache, cache_mask, sentences):
    out_dict = {}
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    prompts = [f"{sent}\n" for sent in sentences]
    inputs = tokenizer(prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
    t_ids = inputs.input_ids.to(cache_mask.device)
    at_mask = inputs.attention_mask.to(cache_mask.device)
    
    outputs = model(t_ids,
                    attention_mask=torch.cat((cache_mask, at_mask), dim=1),
                    past_key_values = kv_cache,
                    output_hidden_states=True)
    
    last_indices = at_mask.sum(dim=1) - 1

    for layer in range(model.config.num_hidden_layers + 1):
        hidden = outputs.hidden_states[layer]
        rep = hidden[torch.arange(hidden.size(0)), last_indices, :]
        out_dict[f"fewshot_{layer}"] = rep.cpu()

    return out_dict

@torch.no_grad()
def get_fewshot_embeds(model, tokenizer, dataloader, lang, flores_code, sent_rep):
    assert sent_rep == "fewshot"

    with open("definitions.json", "r") as gloss_file:
        glossary = json.load(gloss_file)

    results = {f"{sent_rep}_{layer}": [] for layer in range(model.config.num_hidden_layers + 1)}

    cached_kv_1, _ = cache_prefix(model, tokenizer, lang, glossary, 1)

    for batch in tqdm(dataloader, desc=f"Processing {lang} / {flores_code}"):
        out_of_mem = False
        try:
            batch_results = encode_batch(model, tokenizer, *broadcast_cache(cached_kv_1, len(batch["sentence_"+flores_code])), sentences=batch["sentence_"+flores_code])
        except torch.OutOfMemoryError:
            print("Ran out of memory, feeding the batch one sample at a time.")
            out_of_mem = True
        
        if out_of_mem:
            split_batch = [encode_batch(model, 
                                        tokenizer, 
                                        *broadcast_cache(cached_kv_1, 1), 
                                        sentences=[batch["sentence_"+flores_code][i]]) for i in range(len(batch["sentence_"+flores_code]))]
            batch_results = dict()
            for key in split_batch[0]:
                batch_results[key] = [split_batch[i][key][0] for i in range(len(batch["sentence_"+flores_code]))]

        for key, value in batch_results.items():
            results[key].append(value)
    
    return results

@torch.no_grad()
def main(dataset_name, model_name, langs, batch_size, sent_rep, overwrite=False):
    if dataset_name != "flores":  # happy path or w/e. bit more effort if I do implement other datasets
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                     attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto",
                                                            attn_implementation="flash_attention_2",
                                                            dtype=torch.bfloat16)
    model = torch.compile(model)
    model.eval()

    if "num_hidden_layers" not in model.config:
        model.config.num_hidden_layers = model.config.text_config.num_hidden_layers

    model_short_name = model_name.split("/")[-1]
    out_path = f"embeds/{dataset_name}"
    os.makedirs(out_path, exist_ok=True)

    # We may have saved a previous version where some languages were already processed.
    out_file = f"{out_path}/{model_short_name}-{sent_rep}.pickle"
    if os.path.exists(out_file) and not overwrite:
        print(f"Loading existing pickle file {out_file}")
        with open(out_file, "rb") as fin:
            out_dict = pickle.load(fin)
    else:
        out_dict = {}

    print("Loading the entire dataset")
    dataset = load_dataset("facebook/flores", data_dir="all", revision="refs/convert/parquet")["validation"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for lang in langs:
        if lang in out_dict and sent_rep+"_0" in out_dict[lang]:
            print(f"Already processed {lang}. Skipping.")
            continue

        with open("definitions.json", "r") as gloss_file:
            glossary = json.load(gloss_file)

        flores_code = get_flores_code(lang)
        print(f"Processing {lang}. Trying to use flores code {flores_code}")
        
        results = {f"{sent_rep}_{layer}": [] for layer in range(model.config.num_hidden_layers + 1)}

        cached_kv_1, _ = cache_prefix(model, tokenizer, lang, glossary, 1)

        for batch in tqdm(dataloader, desc=f"Processing {lang} / {flores_code}"):
            out_of_mem = False
            try:
                batch_results = encode_batch(model, tokenizer, *broadcast_cache(cached_kv_1, len(batch["sentence_"+flores_code])), sentences=batch["sentence_"+flores_code])
            except torch.OutOfMemoryError:
                print("Ran out of memory, feeding the batch one sample at a time.")
                out_of_mem = True
            
            if out_of_mem:
                split_batch = [encode_batch(model, tokenizer, *broadcast_cache(cached_kv_1, 1), sentences=[batch["sentence_"+flores_code][i]]) for i in range(batch_size)]
                batch_results = dict()
                for key in split_batch[0]:
                    batch_results[key] = [split_batch[i][key][0] for i in batch_size]

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
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B", help="Model name. Assuming an HF decoder")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for embedding calculation")
    # might do model short names again
    parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                        default=ALL_LANGUAGES)
    parser.add_argument("--sent-rep", type=str, default="fewshot", help="How to sentence rep",
                        choices=["fewshot"])

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    # layers? probably just do all of them?

    args = parser.parse_args()
    main(args.dataset, args.model, args.langs, args.batch_size, args.sent_rep, args.overwrite)
