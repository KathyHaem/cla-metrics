from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import cache_utils
from copy import deepcopy
from scipy.stats import pearsonr

from constants_translate import ALL_LANGUAGES, PROMPT
from langcodes import Language
from datasets import load_dataset
import argparse
import json
from tqdm import tqdm
import os

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


def make_few_shot_prefix(reference_src, reference_tgt):
    return "\n\n".join([f"{src}\n{tgt}" for src, tgt in zip(reference_src, reference_tgt)]) + "\n\n"

def make_prior_few_shot_prefix(reference_tgt):
    return "\n".join([f"{tgt}" for tgt in reference_tgt]) + "\n"

@torch.no_grad()
def broadcast_cache(cache: cache_utils.DynamicCache, batch_size: int, device) -> tuple[cache_utils.DynamicCache, torch.Tensor]:
    local_cache = deepcopy(cache)
    for layer in local_cache.layers:
        layer.keys = layer.keys[0:1].expand((batch_size, -1, -1, -1)).clone()
        layer.values = layer.values[0:1].expand((batch_size, -1, -1, -1)).clone()
    return local_cache, torch.ones((batch_size, layer.values.size(2)), device=device)

@torch.no_grad()
def cache_prefix(model, tokenizer, prompt):
    tokenized = tokenizer([prompt], return_tensors="pt")
    t_ids = tokenized.input_ids.to(model.device)
    at_mask = tokenized.attention_mask.to(model.device)
    outputs = model(t_ids, at_mask, use_cache=True)
    return outputs.past_key_values, at_mask

@torch.no_grad()
def loglik_from_logits(logits, mask, target_ids, normalize=True, debug=False):
    log_probs = logits.log_softmax(dim=-1)
    shift_log_probs = log_probs[:, :-1, :]

    token_loglik = shift_log_probs.gather(dim=-1, index=target_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    token_loglik[~(mask[:, 1:].bool())] = 0
    loglik = token_loglik.sum(dim=1)

    if not normalize:
        return loglik
    return loglik / mask[:, 1:].sum(dim=1)

@torch.no_grad()
def get_batch_mutinf(sources, targets, tokenizer, model, kv_cache_prior, cache_mask_prior, kv_cache_posterior, cache_mask_posterior, debug=False):
    # compute priors
    tokenized_ans = tokenizer([t for t in targets], add_special_tokens=False, return_tensors="pt", padding="longest")
    tokens_ans = tokenized_ans.input_ids.to(model.device)
    mask_ans = tokenized_ans.attention_mask.to(model.device)
    logits_ans = model(tokens_ans, torch.cat((cache_mask_prior, mask_ans), dim=1), past_key_values = kv_cache_prior).logits
    ans_prior_loglik = loglik_from_logits(logits_ans, mask_ans, tokens_ans, debug=debug)

    prompt_suffix = [f"{src}\n{tgt}" for src, tgt in zip(sources, targets)]
    tokens_suffix = tokenizer(prompt_suffix, add_special_tokens=False, return_tensors="pt", padding="longest")
    tok_ids = tokens_suffix.input_ids.to(model.device)
    att_mask = tokens_suffix.attention_mask.to(model.device)

    logits = model(tok_ids, torch.cat((cache_mask_posterior, att_mask), dim=1), past_key_values = kv_cache_posterior).logits

    source_lengths = [len(f"{src}\n") for src in sources]
    answer_start_token = torch.tensor([tokens_suffix.char_to_token(i, ref_len) for i, ref_len in enumerate(source_lengths)])
    arange = torch.arange(tok_ids.size(1), dtype=int)
    source_mask = (arange.unsqueeze(0) < answer_start_token.unsqueeze(1)).to(att_mask.device)
    answers_mask = att_mask * (~source_mask)

    ans_loglik = loglik_from_logits(logits, answers_mask, tok_ids, debug=debug)
    mutinf = ans_loglik - ans_prior_loglik

    return mutinf, answers_mask.sum(-1)

@torch.no_grad()
def main(model_id: str, src_lang: str, target_langs: list[str], batch_size: int, debug=False):
    out_path = f"translation/translation_mutinf/{model_id.split('/')[1]}-{src_lang}.json"

    if not debug:
        try:
            with open(out_path) as f:
                json.load(f)
            print("Target already exists, skipping...")
            return
        except:
            pass

    dataset = load_dataset("facebook/flores", data_dir="all", data_files="flores-devtest.parquet", revision="refs/convert/parquet")["train"]
    torch.set_float32_matmul_precision('high')
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    model = torch.compile(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    result_dict = dict()
    src_flores = get_flores_code(src_lang)
    sentences = dataset[f"sentence_{src_flores}"]
    for tgt_lang in target_langs:
        if tgt_lang == src_lang:
            continue
        tgt_flores = get_flores_code(tgt_lang)

        assert all(["\n" not in sentence for sentence in sentences])
        targets = dataset[f"sentence_{tgt_flores}"]

        prompt_prefix_prior = make_prior_few_shot_prefix(targets[:3])
        prompt_prefix = make_few_shot_prefix(sentences[:3], targets[:3])
        tokenizer.padding_side = 'left'
        prefix_cache_prior, _ = cache_prefix(model, tokenizer, prompt_prefix_prior)
        prefix_cache, _ = cache_prefix(model, tokenizer, prompt_prefix)
        tokenizer.padding_side = 'right'

        mutinfs = []
        seq_lens = []
        for batch_start in tqdm(range(3, len(sentences), batch_size)):
            batch_end = min(batch_start + batch_size, len(sentences))
            
            oom = False
            try:
                batch_mutinf, batch_seq_len = get_batch_mutinf(sentences[batch_start:batch_end], 
                                                targets[batch_start:batch_end],
                                                tokenizer,
                                                model,
                                                *broadcast_cache(prefix_cache_prior, batch_end - batch_start, device=model.device),
                                                *broadcast_cache(prefix_cache, batch_end - batch_start, device=model.device), 
                                                debug=debug)
                mutinfs += batch_mutinf.cpu().tolist()
                seq_lens += batch_seq_len.cpu().tolist()
            except torch.OutOfMemoryError:
                oom = True
            
            if oom:
                print("Ran out of memory, feeding one sample at a time")
                for i in range(batch_start, batch_end):
                    single_mutinf, single_seq_len = get_batch_mutinf(sentences[i:i+1], 
                                                targets[i:i+1],
                                                tokenizer,
                                                model,
                                                *broadcast_cache(prefix_cache_prior, 1, device=model.device),
                                                *broadcast_cache(prefix_cache, 1, device=model.device))
                    mutinfs.append(single_mutinf.cpu().item())
                    seq_lens.append(single_seq_len.cpu().item())
        
        if debug:
            print("Mean MI:", torch.tensor(mutinfs).mean().item())
            print("Mean seq len:", torch.tensor(seq_lens, dtype=float).mean().item())
            print("Correlation:", pearsonr(mutinfs, seq_lens))
            
        result_dict[tgt_lang] = {
            "mean": sum(mutinfs)/len(mutinfs),
            "sentence-level": mutinfs
            }

    os.makedirs("translation/translation_mutinf", exist_ok=True)
    if not debug:
        with open(out_path, "w") as f:
            json.dump(result_dict, f, indent=True, ensure_ascii=False)
                
            


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model, 
             snakemake.params.src_lang, 
             snakemake.params.langs,
             snakemake.params.batch_size)
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--src-lang", type=str, help="Language to translate from", default="en")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to translate into",
                            default=ALL_LANGUAGES)
        parser.add_argument("--batch-size", type=int, help="Batch size for inference.", default=10)
        parser.add_argument("--debug", action="store_true", help="Turn on debug mode: do not save anything")
        args = parser.parse_args()

        main(args.model, args.src_lang, args.langs, args.batch_size, args.debug)