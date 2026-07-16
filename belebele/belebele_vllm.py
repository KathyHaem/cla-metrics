from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from datasets import load_dataset, Value
from tqdm import tqdm
import torch
import json
from langcodes import Language
from constants import ALL_LANGUAGES
import os
import argparse
import tempfile
import shutil

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


def process_one_sample(context, question, options, target=None, is_base=False, is_thinking=False):
    if is_base:
        return f"{context}\n\n{question}\n\n" + "\n".join(options) + "\n\n" + (options[target] if target is not None else "")
    conversation = [
        {"role": "user", "content": f"{context}\n\n{question}\n\n" + "\n".join(options)},
    ]
    if is_thinking:
        conversation.append({"role": "assistant", "content": "<think>\n\n</think>\n\n" + (options[target] if target is not None else "")})
    elif target is not None:
        conversation.append({"role": "assistant", "content": options[target]})
    return conversation

def make_n_shot_prompt(contexts: list[str], questions: list[str], options: list[tuple[str, ...]], targets: list[int], n_shots, is_base, is_thinking) -> str:
    if is_base:
        return "\n\n\n".join([
            process_one_sample(c, q, o, t, is_base=is_base, is_thinking=is_thinking) for 
            c, q, o, t in zip(contexts[:n_shots], 
                            questions[:n_shots], 
                            options[:n_shots], 
                            targets[:n_shots])]) + "\n\n\n"
    conversation = [{"role": "system", "content": ""}]
    for c, q, o, t in zip(contexts[:n_shots], 
                            questions[:n_shots], 
                            options[:n_shots], 
                            targets[:n_shots]):
        conversation += process_one_sample(c, q, o, t, is_base=is_base, is_thinking=is_thinking)
    return conversation

def get_min_tokens_to_generate(answers: tuple[str, ...], tokenizer: AutoTokenizer):
    tokenized_answers: torch.Tensor = tokenizer(answers, add_special_tokens=False, return_tensors="pt", padding="longest").input_ids
    # get the smallest amount of tokens that distinguishes the answers
    n, k = tokenized_answers.shape
    # find the smallest prefix length that makes all rows unique
    min_tokens = k
    for kp in range(1, k + 1):
        if tokenized_answers[:, :kp].unique(dim=0).shape[0] == n:
            min_tokens = kp + 1
            break
    if min_tokens == 0:
        raise ValueError(f"Could not find a prefix length that makes the answers unique: {answers}")
    return min_tokens

def main(model_id, langs, n_shots = 3):
    is_base = "base" in model_id.lower() or "pt" in model_id.lower()
    is_thinking = model_id in ["Qwen/Qwen3-14B"]
    cache_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cache_dir
    model = LLM(model_id, max_model_len=15000, dtype="bfloat16")
    if is_base:
        tokenizer = model.get_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    acc_dict = {}

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

        few_shot_prefix = make_n_shot_prompt(contexts, questions, options, targets, n_shots, is_base, is_thinking)

        prompts = [few_shot_prefix + process_one_sample(c, q, o, is_base=is_base, is_thinking=is_thinking) for c, q, o in zip(contexts[n_shots:], questions[n_shots:], options[n_shots:])]

        if is_base:
            sampling_params = [SamplingParams(temperature=0.0, max_tokens=get_min_tokens_to_generate(options_set, tokenizer), stop_token_ids=[tokenizer.eos_token_id], structured_outputs=StructuredOutputsParams(choice=options_set)) for options_set in options[n_shots:]]
        else:
            sampling_params = [SamplingParams(temperature=0.0, max_tokens=get_min_tokens_to_generate(options_set, tokenizer), stop_token_ids=[tokenizer.eos_token_id], structured_outputs=StructuredOutputsParams(choice=options_set)) for options_set in options[n_shots:]]
        if not is_base:
            if is_thinking:
                prompts = tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=False, continue_final_message=True).input_ids
            else:
                prompts = tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True).input_ids
                # decoded = tokenizer.batch_decode(prompts) # for debugging
                # print(decoded)
                # print([len(p) for p in prompts])
        
        generated = model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        correct = 0
        for answer, options, target in zip(generated, options[n_shots:], targets[n_shots:]):
            if (answer.outputs[0].finish_reason == "length" and options[target].startswith(answer.outputs[0].text.strip())) or answer.outputs[0].text.strip() == options[target]:
                correct += 1
        
        acc_dict[lang] = correct / len(targets[n_shots:])

    os.makedirs("scores/belebele/", exist_ok=True)

    with open(f"scores/belebele/{model_id.split('/')[1]}_acc.json", "w") as f:
        json.dump(acc_dict, f, indent=True)
    
    shutil.rmtree(cache_dir)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model,
             snakemake.params.langs)
    else:
        parser = argparse.ArgumentParser(description="Calculate the belebele accuracy.")
        parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", help="Model name. Assuming an HF decoder")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                            default=ALL_LANGUAGES)
        args = parser.parse_args()

        main(args.model, args.langs)