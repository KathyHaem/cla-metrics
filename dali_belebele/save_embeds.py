from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, LlamaForCausalLM, AutoModelForImageTextToText
from datasets import load_dataset
from tqdm import tqdm
import torch
import argparse
import copy
from langcodes import Language
import os

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

@torch.no_grad()
def compute_representations(
    model: LlamaForCausalLM, 
    tokenizer: PreTrainedTokenizerFast, 
    contexts: list[str], 
    questions: list[str], 
    answers: list[list[str]]):
    """Computes the last token embeddings of each premise-option pair, shape (num_questions, num_options, num_hidden_layers, hidden_size)
    """

    context_questions = [c + "\n" + q + "\n" for c, q in zip(contexts, questions)]

    options_emb = torch.empty((len(questions), len(answers[0]), model.config.num_hidden_layers + 1, model.config.hidden_size), device=model.device)

    # change padding side to left so that there are no "holes" between the context and the answer
    tokenizer.padding_side = 'left'

    cq_tokens = tokenizer(context_questions, padding="longest", return_tensors="pt")
    cq_tok_ids = cq_tokens.input_ids.to(model.device)
    cq_mask = cq_tokens.attention_mask.to(model.device)

    cq_forward = model(cq_tok_ids, attention_mask=cq_mask, use_cache=True)
    cached_cq = cq_forward.past_key_values

    tokenizer.padding_side = "right"
    for ans in range(len(answers[0])):
        ans_tokens = tokenizer([anslist[ans] for anslist in answers], padding="longest", return_tensors="pt", add_special_tokens=False)
        ans_ids = ans_tokens.input_ids.to(model.device)
        ans_mask = ans_tokens.attention_mask.to(model.device)

        cloned_cache = copy.deepcopy(cached_cq)
        hidden = torch.stack(model.forward(ans_ids, 
                    attention_mask=torch.cat((cq_mask, ans_mask), dim=1), # IMPORTANT: we also need the attention mask of the context
                    past_key_values = cloned_cache, # IMPORTANT: we need to copy the original object to prevent mutation
                    output_hidden_states = True
                    ).hidden_states)
        last_indices = ans_mask.sum(dim=1) - 1

        last_token_embs = hidden[:, torch.arange(len(questions)), last_indices] # shape (n_layers, n_questions, hidden_size)
        options_emb[:, ans, :, :] = torch.permute(last_token_embs, (1, 0, 2))

    return options_emb

def main(model_id, lang, batch_size=10):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                     attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto",
                                                            attn_implementation="flash_attention_2",
                                                            dtype=torch.bfloat16)
    # model = torch.compile(model)
    model.eval()
    if "num_hidden_layers" not in model.config:
        model.config.num_hidden_layers = model.config.text_config.num_hidden_layers
    if "hidden_size" not in model.config:
        model.config.hidden_size = model.config.text_config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs("embeds/dali-belebele/", exist_ok=True)
    if os.path.isfile(f"embeds/dali-belebele/{model_id.split('/')[1]}-{lang}.pt"):
        print(f"Lang {lang} already computed. skipping")
        return
    try:
        belebele = load_dataset("facebook/belebele", get_flores_code(lang), split="test").sort(["link", "question_number"])
    except ValueError:
        print(f"Language {lang}/{get_flores_code(lang)} not found! Skipping.")
        return

    contexts = belebele["flores_passage"]
    questions = belebele["question"]
    options = list(zip(*[belebele[f"mc_answer{i+1}"] for i in range(4)]))

    max_samples = len(contexts)
    answers_embs = torch.empty((len(questions), 4, model.config.num_hidden_layers + 1, model.config.hidden_size))
    
    for batch_start in tqdm(range(0, max_samples, batch_size)):
        batch_end = min(batch_start + batch_size, max_samples)
        batch_contexts = contexts[batch_start:batch_end]
        batch_questions = questions[batch_start:batch_end]
        batch_options = options[batch_start:batch_end]

        oom = False
        try:
            batch_answers_embs = compute_representations(model, tokenizer, batch_contexts, batch_questions, batch_options)
            answers_embs[batch_start:batch_end] = batch_answers_embs.cpu()
        except torch.OutOfMemoryError as e:
            oom = True
        
        # for some reason I had to put this outside the except block because the memory from try does not get cleared until the end of except
        if oom:
            print(f"Ran out of memory for lang {lang}, feeding the batch one sample at a time!")
            for i in range(batch_start, batch_end):
                single_emb = compute_representations(model, tokenizer, [contexts[i]], [questions[i]], [options[i]])
                answers_embs[i] = single_emb.cpu()

    torch.save(answers_embs, f"embeds/dali-belebele/{model_id.split('/')[1]}-{lang}.pt")

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model, 
             snakemake.params.lang,
             snakemake.params.batch_size)
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--lang", type=str, help="Language to embed", default="en")
        parser.add_argument("--batch-size", type=int, help="Batch size for inference.", default=10)
        args = parser.parse_args()

        main(args.model, args.lang, args.batch_size)