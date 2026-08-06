import argparse
import os
import pickle

import torch
from datasets import load_dataset, Dataset
from langcodes import Language
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

from save_embeds_fewshot import get_fewshot_embeds

from attn_utils import get_attn_implementation
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


def masked_mean(data, mask, weighted = False):
    if not weighted:
        return (data * mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True)
    
    weights = torch.arange(1, mask.size(1) + 1, device=data.device).unsqueeze(0)
    weights = weights * mask
    weights = weights / weights.sum(1, keepdims=True)
    weighted_emb = (data * weights.unsqueeze(2)).sum(1)
    return weighted_emb

def extract_last(data, mask):
    last_indices = mask.sum(dim=1) - 1

    return data[torch.arange(data.size(0)), last_indices, :]

def get_prompt_conversation(text: str, is_thinking: bool):
    return [
        {"role": "user", "content": f"Summarize the following sentence in one word: {text}"},
        {"role": "assistant", "content": ("<think>\n\n</think>\n\n" if is_thinking else "") + "The sentence can be summarized in one word as: `"},
    ]

def encode_batch(batch, model, tokenizer, sent_rep, is_base: bool, is_thinking: bool, key="sentence"):
    out_dict = {}
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if sent_rep == "prompt":
        tokenizer.padding_side = "left"
        if is_base:
            batch[key] = [SENT_SUMM_TEMPLATE.format(sent=sent) for sent in batch[key]]
            inputs = tokenizer(batch[key], padding=True, truncation=True, return_tensors="pt")
        else:
            prompts = [get_prompt_conversation(sent, is_thinking) for sent in batch[key]]
            inputs = tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=False, continue_final_message=True, return_tensors="pt", padding="longest")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            for layer in range(len(outputs.hidden_states)):
                rep = outputs.hidden_states[layer][:, -1, :].cpu()
                out_dict[f"prompt_{layer}"] = rep
    elif sent_rep in ("mean", "weighted-mean", "last-token"):
        inputs = tokenizer(batch[key], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            mask = inputs["attention_mask"]
            for layer in range(len(outputs.hidden_states)):
                hidden_states = outputs.hidden_states[layer]
                if sent_rep == "last-token":
                    rep = extract_last(hidden_states, mask).cpu()
                else:
                    rep = masked_mean(hidden_states, mask, weighted=(sent_rep == "weighted-mean")).cpu()
                out_dict[f"{sent_rep}_{layer}"] = rep

    return out_dict


def main(dataset_name, model_name, langs, sent_rep, batch_size, overwrite=False):
    is_base = "base" in model_name.lower() or "pt" in model_name.lower()
    is_thinking = model_name in ["Qwen/Qwen3-14B"]
    print("Available CUDA devices:", torch.cuda.device_count())

    if dataset_name != "flores":  # happy path or w/e. bit more effort if I do implement other datasets
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    attn_implementation = get_attn_implementation()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", attn_implementation=attn_implementation, dtype=torch.bfloat16)
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", attn_implementation=attn_implementation, dtype=torch.bfloat16)
    model.eval()
    if "num_hidden_layers" not in model.config:
        model.config.num_hidden_layers = model.config.text_config.num_hidden_layers

    model_short_name = model_name.split("/")[-1]
    out_path = f"embeds/{dataset_name}"
    os.makedirs(out_path, exist_ok=True)

    print("Loading the entire dataset")

    dataset = load_dataset("facebook/flores", "all")["dev"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    


    for lang in langs:
        flores_code = get_flores_code(lang)
        print(f"Processing {lang}. Trying to use flores code {flores_code}")

        out_file = f"{out_path}/{model_short_name}-{sent_rep}-{lang}.pickle"
        if os.path.exists(out_file) and not overwrite:
            print(f"Embeddings seem to be already saved, testing validity...")
            try:
                with open(out_file, "rb") as fin:
                    pickle.load(fin)
            except:
                print("File invalid!")
            else:
                print("Skipping...")
                continue

        out_dict = {}

        if lang in out_dict and sent_rep+"_0" in out_dict[lang]:
            print(f"Already processed {lang}. Skipping.")
            continue
        

        if sent_rep == "fewshot":
            results = get_fewshot_embeds(model, tokenizer, dataloader, lang, flores_code, sent_rep, is_base, is_thinking)
        else:
            results = {f"{sent_rep}_{layer}": [] for layer in range(model.config.num_hidden_layers + 1)}
            for batch in tqdm(dataloader, desc=f"Processing {lang} / {flores_code}"):
                batch_results = (encode_batch(batch, model, tokenizer, sent_rep, is_base, is_thinking, key="sentence_"+flores_code))
                for key, value in batch_results.items():
                    results[key].append(value)
        out_dict[lang] = {k: torch.cat(v, dim=0) for k, v in results.items()}

        with open(out_file, "wb") as fout:
            pickle.dump(out_dict, fout)

    missing_langs = set(langs) - set(out_dict.keys())
    print(f"Still missing: {missing_langs}.")


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.dataset, 
             snakemake.params.model, 
             snakemake.params.langs, 
             snakemake.params.sent_rep, 
             snakemake.params.batch_size, 
             snakemake.params.overwrite)
    
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--dataset", type=str, default="flores", help="Dataset name. Just FLORES atm")
        parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", help="Model name. Assuming an HF decoder")
        # might do model short names again
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                            default=ALL_LANGUAGES)

        parser.add_argument("--sent-rep", type=str, default="prompt", help="How to sentence rep",
                            choices=["mean", "weighted-mean", "prompt", "last-token", "fewshot"])
        parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
        parser.add_argument("--batch-size", type=int, default=10, help="Batch size for embedding calculation")

        args = parser.parse_args()
        main(args.dataset, args.model, args.langs, args.sent_rep, args.batch_size, args.overwrite)
