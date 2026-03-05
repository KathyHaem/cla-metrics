import re
import os
import json
import argparse
import tempfile
import shutil

from numpy import dtype
from sklearn.metrics import f1_score

from langcodes import Language
from datasets import load_dataset
from vllm import LLM, SamplingParams

from constants import ALL_LANGUAGES

TOPICS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
PROMPT = """Classify the following text into one of these topics: "science/technology", "travel", "politics", "sports", "health", "entertainment", "geography". Provide only the topic in English as your response.

text: `{}`
topic: `"""

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
    os.makedirs(os.path.join("scores", "sib-200"), exist_ok=True)
    cahce_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cahce_dir

    llm = LLM(model_name, max_model_len=10000, dtype="bfloat16")
    basic_sampling = SamplingParams(temperature=0, max_tokens=16, stop=["\n", "`"])

    f1s = dict()
    for i, lang in enumerate(langs):
        print(f"Running on languege {i}/{len(langs)}: {lang}")
        full_code = get_flores_code(lang)
        dataset = load_dataset("Davlan/sib200", full_code, split="test")
        prompts = list(map(PROMPT.format, dataset["text"]))
        targets = dataset["category"]

        outputs = llm.generate(
            prompts=prompts,
            sampling_params=basic_sampling#sampling_params,
        )


        answers = [process_answer(output.outputs[0].text) for output in outputs]
        # correct = sum(1 for a, t in zip(answers, targets) if a == t)
        # accuracy = correct / len(targets)
        macro_f1 = f1_score(targets, answers, labels=TOPICS, average='macro', zero_division=0)
        f1s[lang] = macro_f1

    with open(os.path.join("scores", "sib-200", f"{model_name.split("/")[1]}.json"), "w") as f:
        json.dump(f1s, f, indent=4)

    shutil.rmtree(cahce_dir)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model,
             snakemake.params.langs)
    else:
        parser = argparse.ArgumentParser(description="Calculate the SIB-200 scores.")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                            default=ALL_LANGUAGES)

        args = parser.parse_args()
        main(args.model, args.langs)