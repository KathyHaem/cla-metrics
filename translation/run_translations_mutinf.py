from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
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

torch._dynamo.config.capture_scalar_outputs = True

def get_flores_code(short_code: str, dataset: str):
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

    if dataset == "bouquet":
        custom_codes["ar"] = "arz_Arab"
        custom_codes["ko"] = "kor_Kore"
        custom_codes["pt"] = "por_Latn_braz1246"
        custom_codes["zh"] = "cmn_Hant"

    if short_code in custom_codes:
        return custom_codes[short_code]

    lang = Language.get(short_code)
    script = lang.script or lang.assume_script().script
    return f"{lang.to_alpha3()}_{script}"

def get_sentences(dataset: str, src_lang: str):
    match dataset:
        case "flores":
            dataset = load_dataset("facebook/flores", get_flores_code(src_lang, dataset="flores"), split="devtest").shuffle(seed=42)
            return dataset["sentence"]
        case "bouquet":
            dataset = load_dataset("facebook/bouquet", get_flores_code(src_lang, dataset="bouquet"), split="dev").shuffle(seed=42)
            return dataset["src_text"]
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")

def make_few_shot_prefix(reference_src, reference_tgt, tokenizer, is_base, is_thinking):
    if is_base:
        return "\n\n".join([f"{src}\n{tgt}" for src, tgt in zip(reference_src, reference_tgt)]) + "\n\n", "\n"
    else:
        conversation = [{"role": "system", "content": ""}]
        for src, tgt in zip(reference_src, reference_tgt):
            conversation.append({
                "role": "user",
                "content":  src
            })
            conversation.append({
                "role": "assistant",
                "content": ("<think>\n\n</think>\n\n" if is_thinking else "") + tgt,
            })
        conversation.append({
            "role": "user",
            "content": "<SPLIT>"
            })
        if is_thinking:
            conversation.append({
                "role": "assistant",
                "content": "<think>\n\n</think>\n\n"
                })
        
        chat_applied: str = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=not is_thinking, continue_final_message=is_thinking)
        prefix, suffix = chat_applied.split("<SPLIT>")[:2]
        return prefix, suffix

def make_prior_few_shot_prefix(reference_tgt, tokenizer, is_base):
    newlined = "\n".join([f"{tgt}" for tgt in reference_tgt]) + "\n"
    if is_base:
        return newlined
    else:
        conversation = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": newlined,
            }]
        chat_applied = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, continue_final_message=True)
        return chat_applied
    
@torch.no_grad()
def broadcast_cache(cache: cache_utils.DynamicCache, batch_size: int, device) -> tuple[cache_utils.DynamicCache, torch.Tensor]:
    local_cache = deepcopy(cache)
    for layer in local_cache.layers:
        layer.keys = layer.keys[0:1].expand((batch_size, -1, -1, -1)).clone()
        layer.values = layer.values[0:1].expand((batch_size, -1, -1, -1)).clone()
    return local_cache, torch.ones((batch_size, layer.values.size(2)), device=device)

@torch.no_grad()
def cache_prefix(model, tokenizer, prompt):
    tokenized = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
    t_ids = tokenized.input_ids.to(model.device)
    at_mask = tokenized.attention_mask.to(model.device)
    outputs = model(t_ids, attention_mask=at_mask, use_cache=True)
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
def get_batch_mutinf(sources, 
                     targets, 
                     tokenizer, 
                     prompt_middlepart, 
                     model, kv_cache_prior, 
                     cache_mask_prior, 
                     kv_cache_posterior, 
                     cache_mask_posterior, 
                     debug=False):
    # compute priors
    tokenized_ans = tokenizer([t for t in targets], add_special_tokens=False, return_tensors="pt", padding="longest")
    tokens_ans = tokenized_ans.input_ids.to(model.device)
    mask_ans = tokenized_ans.attention_mask.to(model.device)
    logits_ans = model(tokens_ans, attention_mask=torch.cat((cache_mask_prior, mask_ans), dim=1), past_key_values = kv_cache_prior).logits
    ans_prior_loglik = loglik_from_logits(logits_ans, mask_ans, tokens_ans, debug=debug)

    prompt_suffix = [f"{src}{prompt_middlepart}{tgt}" for src, tgt in zip(sources, targets)]
    tokens_suffix = tokenizer(prompt_suffix, add_special_tokens=False, return_tensors="pt", padding="longest")
       
    tok_ids = tokens_suffix.input_ids.to(model.device)
    att_mask = tokens_suffix.attention_mask.to(model.device)

    logits = model(tok_ids, attention_mask=torch.cat((cache_mask_posterior, att_mask), dim=1), past_key_values = kv_cache_posterior).logits

    source_lengths = [len(f"{src}{prompt_middlepart}") for src in sources]
    answer_start_token = torch.tensor([tokens_suffix.char_to_token(i, ref_len) for i, ref_len in enumerate(source_lengths)])
    arange = torch.arange(tok_ids.size(1), dtype=int)
    source_mask = (arange.unsqueeze(0) < answer_start_token.unsqueeze(1)).to(att_mask.device)

    answers_mask = att_mask * (~source_mask)

    ans_loglik = loglik_from_logits(logits, answers_mask, tok_ids, debug=debug)
    mutinf = ans_loglik - ans_prior_loglik



    return mutinf, answers_mask.sum(-1)

@torch.no_grad()
def main(model_id: str, src_lang: str, target_langs: list[str], dataset: str, batch_size: int, n_shots: int, debug=False):
    is_base = "base" in model_id.lower() or "pt" in model_id.lower()
    is_thinking = model_id in ["Qwen/Qwen3-14B"]
    out_path = f"translation/translation_{dataset}_mutinf/{model_id.split('/')[1]}-{src_lang}.json"

    if not debug:
        try:
            with open(out_path) as f:
                json.load(f)
            print("Target already exists, skipping...")
            return
        except:
            pass


    torch.set_float32_matmul_precision('high')
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    result_dict = dict()
    sentences = get_sentences(dataset, src_lang)
    for tgt_lang in target_langs:
        if tgt_lang == src_lang:
            continue
        assert all(["\n" not in sentence for sentence in sentences])
        targets = get_sentences(dataset, tgt_lang)

        prompt_prefix_prior = make_prior_few_shot_prefix(targets[:n_shots], tokenizer, is_base)
        prompt_prefix, prompt_suffix = make_few_shot_prefix(sentences[:n_shots], targets[:n_shots], tokenizer, is_base, is_thinking)
        tokenizer.padding_side = 'left'
        prefix_cache_prior, _ = cache_prefix(model, tokenizer, prompt_prefix_prior)
        prefix_cache, _ = cache_prefix(model, tokenizer, prompt_prefix)
        tokenizer.padding_side = 'right'

        mutinfs = []
        seq_lens = []
        for batch_start in tqdm(range(n_shots, len(sentences), batch_size)):
            batch_end = min(batch_start + batch_size, len(sentences))
            
            oom = False
            try:
                batch_mutinf, batch_seq_len = get_batch_mutinf(sentences[batch_start:batch_end], 
                                                targets[batch_start:batch_end],
                                                tokenizer,
                                                prompt_suffix,
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
                                                prompt_suffix[i:i+1],
                                                model,
                                                *broadcast_cache(prefix_cache_prior, 1, device=model.device),
                                                *broadcast_cache(prefix_cache, 1, device=model.device),
                                                debug=debug)
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

    os.makedirs(f"translation/translation_{dataset}_mutinf", exist_ok=True)
    if not debug:
        with open(out_path, "w") as f:
            json.dump(result_dict, f, indent=True, ensure_ascii=False)
                
            


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model, 
             snakemake.params.src_lang, 
             snakemake.params.langs,
             snakemake.params.dataset,
             snakemake.params.batch_size,
             snakemake.params.n_shots)
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--src-lang", type=str, help="Language to translate from", default="en")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to translate into",
                            default=ALL_LANGUAGES)
        parser.add_argument("--dataset", type=str, help="Dataset to use for translation", choices=["flores", "bouquet"], default="flores")
        parser.add_argument("--batch-size", type=int, help="Batch size for inference.", default=10)
        parser.add_argument("--n-shots", type=int, help="Number of shots for few-shot prompting.", default=3)
        parser.add_argument("--debug", action="store_true", help="Turn on debug mode: do not save anything")
        args = parser.parse_args()

        main(args.model, args.src_lang, args.langs, args.dataset, args.batch_size, args.n_shots, args.debug)