from vllm import LLM, SamplingParams
from constants_translate import ALL_LANGUAGES, PROMPT
from langcodes import Language
from datasets import load_dataset
import argparse
import json
from tqdm import tqdm
import os
import tempfile
import shutil

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

def process_answer(ans: str):
    ans = ans.strip()
    if ans[-1] == "`":
        ans = ans[:-1]
    return ans

def make_few_shot_prefix(reference_src, reference_tgt):
    return "\n\n".join([f"> {src}\n> {tgt}" for src, tgt in zip(reference_src, reference_tgt)]) + "\n\n> "

def main(model: str, src_lang: str, langs: list[str], few_shot = True):
    out_path = f"translation/translations/{model.split("/")[1]}_{src_lang}.json"

    try:
        with open(out_path) as f:
            json.load(f)
        print("File already exists, skipping.")
        return
    except:
        pass

    cahce_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cahce_dir

    dataset = load_dataset("facebook/flores", data_dir="all", data_files="flores-devtest.parquet", revision="refs/convert/parquet")["train"]
    llm = LLM(model)
    tokenizer = llm.get_tokenizer()

    result_dict = dict()
    src_flores = get_flores_code(src_lang)
    sentences = dataset[f"sentence_{src_flores}"]
    for tgt_lang in tqdm(langs):
        if src_lang == tgt_lang:
            continue
        tgt_flores = get_flores_code(tgt_lang)

        if not few_shot:
            raise AssertionError("Prompt translation is deprecated")
            prompts = [PROMPT.substitute(
                source=Language.get(src_flores[:3]).language_name(), 
                target=Language.get(tgt_flores[:3]).language_name(),
                text=sentence) for sentence in sentences]

            targets = dataset[f"sentence_{tgt_flores}"]
            sampling_params_batch = [SamplingParams(temperature=0, max_tokens=int(1.5*len(tokenizer.tokenize(t))), stop=["\n", "`"]) for t in targets]
        
        else:
            assert all(["\n" not in sentence for sentence in sentences])
            targets = dataset[f"sentence_{tgt_flores}"]
            prompt_prefix = make_few_shot_prefix(sentences[:3], targets[:3])
            
            targets = targets[3:]

            prompts = [prompt_prefix + sentence + "\n>" for sentence in sentences[3:]]
            sampling_params_batch = [SamplingParams(temperature=0, max_tokens=int(1.5*len(tokenizer.tokenize(t))), stop=["\n"]) for t in targets]

        
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params_batch,
            use_tqdm=False
        )

        answers = [process_answer(output.outputs[0].text) for output in outputs]

        result_dict[tgt_lang] = [{"output": o, "target": t} for o, t in zip(answers, targets)]

    os.makedirs("translation/translations", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=True, ensure_ascii=False)

    shutil.rmtree(cahce_dir)
            


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model, 
             snakemake.params.src_lang, 
             snakemake.params.langs)
    
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--src-lang", type=str, help="Language to translate from", default="en")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to translate into",
                            default=ALL_LANGUAGES)

        args = parser.parse_args()
        main(args.model, args.src_lang, args.langs)