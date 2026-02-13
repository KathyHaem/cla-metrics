from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, LlamaForCausalLM
from datasets import load_dataset, Value
from tqdm import tqdm
import torch
import json
import copy
import gc
from langcodes import Language
from constants import ALL_LANGUAGES
import os
import argparse

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

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

def get_ans_dict(log_probs, targets, max_samples):
    return {
            "accuracy": ((log_probs.argmax(dim=1) == targets[:max_samples]).sum() / max_samples).item(),
            "logliks": log_probs.tolist(),
            "chosen_answer": log_probs.argmax(dim=1).tolist(),
            "target": targets.tolist(),
        }

@torch.no_grad()
def compute_answers_logliks(
    model: LlamaForCausalLM, 
    tokenizer: PreTrainedTokenizerFast, 
    contexts: list[str], 
    questions: list[str], 
    answers: list[list[str]]):
    """Computes the per-token mean log-likelihood of each answers
    """
    
    def loglik_from_logprobs(log_probs, mask, target_ids, mean=True):
        shift_log_probs = log_probs[:, :-1, :]
        token_loglik = shift_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        token_loglik[~(mask.bool())] = 0
        loglik = token_loglik.sum(dim=1)
        if not mean:
            return loglik
        return loglik / mask.sum(dim=1)

    context_questions = [c + "\n" + q + "\n" for c, q in zip(contexts, questions)]

    answers_loglik = torch.zeros((len(questions)), len(answers[0]), device=model.device)
    answers_mutinf = torch.zeros((len(questions)), len(answers[0]), device=model.device)

    # change padding side to left so that there are no "holes" between the context and the answer
    tokenizer.padding_side = 'left'

    cq_tokens = tokenizer(context_questions, padding="longest", return_tensors="pt")
    cq_tok_ids = cq_tokens.input_ids.to(model.device)
    cq_mask = cq_tokens.attention_mask.to(model.device)

    cq_forward = model(cq_tok_ids, cq_mask, use_cache=True)
    cached_cq = cq_forward.past_key_values
    # NOTE: the predictions for the first answer tokens are stored in the cq logits
    first_ans_token_logits = cq_forward.logits[:,-1,:].unsqueeze(1)

    tokenizer.padding_side = "right"
    for ans in range(len(answers[0])):
        ans_tokens = tokenizer([anslist[ans] for anslist in answers], padding="longest", return_tensors="pt", add_special_tokens=True)
        ans_ids = ans_tokens.input_ids.to(model.device)
        ans_mask = ans_tokens.attention_mask.to(model.device)

        # compute priors
        logits = model(ans_ids, ans_mask).logits
        log_probs = logits.log_softmax(dim=-1)
        prior_loglik = loglik_from_logprobs(log_probs, ans_mask[:, 1:], ans_ids[:, 1:], mean=False)

        # drop the spcial BOS token
        ans_ids = ans_ids[:, 1:]
        ans_mask = ans_mask[:, 1:]

        cloned_cache = copy.deepcopy(cached_cq)
        logits = model(ans_ids, 
                    torch.cat((cq_mask, ans_mask), dim=1), # IMPORTANT: we also need the attention mask of the context
                    past_key_values = cloned_cache # IMPORTANT: we need to copy the original object to prevent mutation
                    ).logits 
        log_probs = torch.cat((first_ans_token_logits, logits), dim=1).log_softmax(dim=-1)

        cond_loglik = loglik_from_logprobs(log_probs, ans_mask, ans_ids)
        answers_loglik[:, ans] = cond_loglik
        answers_mutinf[:, ans] = loglik_from_logprobs(log_probs, ans_mask, ans_ids, mean=False) - prior_loglik

    return answers_loglik, answers_mutinf

def main(model_id, langs, batch_size, max_samples = None):
    torch.set_float32_matmul_precision('medium')
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    model = torch.compile(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    results_loglik = dict()
    results_mutinf = dict()
    for lang in tqdm(langs):
        try:
            belebele = load_dataset("facebook/belebele", get_flores_code(lang), split="test")
        except ValueError:
            print(f"Language {lang}/{get_flores_code(lang)} not found! Skipping.")
            continue

        contexts = belebele["flores_passage"]
        questions = belebele["question"]
        options = list(zip(*[belebele[f"mc_answer{i+1}"] for i in range(4)]))

        belebele = belebele.cast_column("correct_answer_num", Value(dtype="int32"))
        targets = torch.tensor(belebele["correct_answer_num"]) - 1

        if max_samples is None:
            max_samples = len(contexts)
        answers_logliks = torch.empty((max_samples, 4))
        answers_mutinf = torch.empty((max_samples, 4))
        for batch_start in tqdm(range(0, max_samples, batch_size)):
            batch_end = min(batch_start + batch_size, max_samples)
            batch_contexts = contexts[batch_start:batch_end]
            batch_questions = questions[batch_start:batch_end]
            batch_options = options[batch_start:batch_end]

            try:
                oom = False
                batch_answers_logliks, batch_answers_mutinf = compute_answers_logliks(model, tokenizer, batch_contexts, batch_questions, batch_options)
                answers_logliks[batch_start:batch_end] = batch_answers_logliks.cpu()
                answers_mutinf[batch_start:batch_end] = batch_answers_mutinf.cpu()
            except torch.OutOfMemoryError as e:
                oom = True
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
            
            # for some reason I had to put this outside the except block because the memory from try does not get cleared until the end of except
            if oom:
                print(f"Ran out of memory for lang {lang}, feeding the batch one sample at a time!")
                for i in range(batch_start, batch_end):
                    single_loglik, single_mutinf = compute_answers_logliks(model, tokenizer, [contexts[i]], [questions[i]], [options[i]])
                    answers_logliks[i] = single_loglik.cpu()
                    answers_mutinf[i] = single_mutinf.cpu()


        results_loglik[lang] = get_ans_dict(answers_logliks, targets, max_samples)
        results_mutinf[lang] = get_ans_dict(answers_mutinf, targets, max_samples)

    os.makedirs("scores/belebele/", exist_ok=True)

    with open(f"scores/belebele/{model_id.split("/")[1]}_loglik.json", "w") as f:
        json.dump(results_loglik, f, indent=True)
    
    with open(f"cores/belebele/{model_id.split("/")[1]}_mutinf.json", "w") as f:
        json.dump(results_mutinf, f, indent=True)


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model,
             snakemake.params.langs,
             snakemake.params.batch_size)
    else:
        parser = argparse.ArgumentParser(description="Calculate the belebele accuracy.")
        parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", help="Model name. Assuming an HF decoder")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                            default=ALL_LANGUAGES)
        parser.add_argument("--batch-size", type=int, default=10, help="Batch size for embedding calculation")
        args = parser.parse_args()

        main(args.model, args.langs, args.batch_size)