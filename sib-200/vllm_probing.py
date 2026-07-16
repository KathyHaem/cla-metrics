import re
import os
import json
import argparse
import tempfile
import shutil
import torch

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
from sklearn.metrics import f1_score

from langcodes import Language
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from constants import ALL_LANGUAGES

TOPICS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
PROMPT = """Classify the following text into one of these topics: "science/technology", "travel", "politics", "sports", "health", "entertainment", "geography". Provide only the topic in English as your response.

text: `{}`"""
PROMPT_BASE_SUFFIX = "\ntopic: `"

def prepare_prompt(text: str, is_base: bool, is_thinking: bool):
    if is_base:
        return PROMPT.format(text) + PROMPT_BASE_SUFFIX
    return [
    {"role": "system", "content": ""},
    {
        "role": "user",
        "content": PROMPT.format(text),
    },
    {
        "role": "assistant",
        "content": ("<think>\n\n</think>\n\n" if is_thinking else "") + "The topic is: `",
    }]

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

def process_answer(ans: str):
    ans = ans.strip()
    if ans == "":
        return ""
    if ans[-1] == "`":
        ans = ans[:-1]
    re_match = re.match(r"(?:[0-9\.]+ )?(.+)", ans) # NOTE: for some reasons the model often writes "1. [topic]"
    if re_match:
        return re_match.group(1)
    return ""


def main(model_name, langs):
    is_base = "base" in model_name.lower() or "pt" in model_name.lower()
    is_thinking = model_name in ["Qwen/Qwen3-14B"]

    os.makedirs(os.path.join("scores", "sib-200"), exist_ok=True)
    cahce_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cahce_dir

    llm = LLM(
        model_name, 
        max_model_len=2048, 
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.8,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=16, stop=["\n", "`"], stop_token_ids=[tokenizer.eos_token_id])

    f1s = dict()
    for i, lang in tqdm(enumerate(langs), total=len(langs)):
        print(f"Running on languege {i}/{len(langs)}: {lang}")
        full_code = get_flores_code(lang)
        dataset = load_dataset("Davlan/sib200", full_code, split="test")
        prompts = list(map(lambda x: prepare_prompt(x, is_base=is_base, is_thinking=is_thinking), dataset["text"]))
        if not is_base:
            prompts = tokenizer.apply_chat_template(prompts, tokenize=True, continue_final_message=True).input_ids
        targets = dataset["category"]

        outputs = llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=False
            )


        answers = [process_answer(output.outputs[0].text) for output in outputs]
        macro_f1 = f1_score(targets, answers, labels=TOPICS, average='macro', zero_division=0)
        f1s[lang] = macro_f1

    with open(os.path.join("scores", "sib-200", f"{model_name.split("/")[1]}.json"), "w") as f:
        json.dump(f1s, f, indent=4)

    shutil.rmtree(cahce_dir)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model,
             snakemake.params.langs)
    else:
        parser = argparse.ArgumentParser(description="Calculate the SIB-200 scores.")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                            default=ALL_LANGUAGES)

        args = parser.parse_args()
        main(args.model, args.langs)