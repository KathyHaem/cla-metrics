from numpy import dtype
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
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


def process_one_sample(context, question, options, target=None):
        return f"{context}\n\n{question}\n\n" + "\n".join(options) + "\n\n" + (options[target] if target is not None else "")

def make_n_shot_prompt(contexts: list[str], questions: list[str], options: list[tuple[str, ...]], targets: list[int], n_shots) -> str:
    return "\n\n\n".join([
        process_one_sample(c, q, o, t) for 
        c, q, o, t in zip(contexts[:n_shots], 
                          questions[:n_shots], 
                          options[:n_shots], 
                          targets[:n_shots])]) + "\n\n\n"

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
    return min_tokens

def main(model_id, langs, n_shots = 2):
    cahce_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cahce_dir
    model = LLM(model_id, max_model_len=10000, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

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

        few_shot_prefix = make_n_shot_prompt(contexts, questions, options, targets, n_shots)

        prompts = [few_shot_prefix + process_one_sample(c, q, o) for c, q, o in zip(contexts[n_shots:], questions[n_shots:], options[n_shots:])]

        sampling_params = [SamplingParams(temperature=0.0, max_tokens=get_min_tokens_to_generate(options_set, tokenizer)) for options_set in options[n_shots:]]

        generated = model.generate(prompts, sampling_params=sampling_params)

        correct = 0
        for answer, options, target in zip(generated, options[n_shots:], targets[n_shots:]):
            if answer.outputs[0].finish_reason == "length" and options[target].startswith(answer.outputs[0].text.strip()):
                correct += 1
        
        acc_dict[lang] = correct / len(targets[n_shots:])

    os.makedirs("scores/belebele/", exist_ok=True)

    with open(f"scores/belebele/{model_id.split('/')[1]}_acc.json", "w") as f:
        json.dump(acc_dict, f, indent=True)
    
    shutil.rmtree(cahce_dir)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model,
             snakemake.params.langs)
    else:
        parser = argparse.ArgumentParser(description="Calculate the belebele accuracy.")
        parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", help="Model name. Assuming an HF decoder")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                            default=ALL_LANGUAGES)
        args = parser.parse_args()

        main(args.model, args.langs)