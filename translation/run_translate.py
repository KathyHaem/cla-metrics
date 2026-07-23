from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
from transformers import AutoTokenizer
from constants_translate import ALL_LANGUAGES, PROMPT
from langcodes import Language
from datasets import load_dataset
import argparse
import json
from tqdm import tqdm
import os
import tempfile
import shutil

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

def process_answer(ans: str):
    ans = ans.strip()
    ans = ans.replace("*", "") # strip markdown bold
    if "</think>" in ans:
        ans = ans.split("</think>")[-1]
        ans = ans.strip()
    if ans and ans[-1] == "`":
        ans = ans[:-1]
    return ans

def make_few_shot_prefix(reference_src, reference_tgt, is_base=False, is_thinking=False):
    if is_base:
        return "\n\n".join([f"{src}\n{tgt}" for src, tgt in zip(reference_src, reference_tgt)]) + "\n\n"
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
        return conversation


def main(model: str, src_lang: str, langs: list[str], dataset: str = "flores", n_shots: int = 3):
    is_base = "base" in model.lower() or "pt" in model.lower()
    if is_base:
        print("Using base model")
    is_thinking = model in ["Qwen/Qwen3-14B"]
    out_path = f"translation/translations_{dataset}/{model.split('/')[1]}_{src_lang}.json"

    try:
        with open(out_path) as f:
            json.load(f)
        print("File already exists, skipping.")
        return
    except:
        pass

    cache_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cache_dir

    llm = LLM(model, 
        max_model_len=15000, 
        dtype="bfloat16",
        # tensor_parallel_size=torch.cuda.device_count(),
        # gpu_memory_utilization=0.8,
        reasoning_config=None if not is_thinking else ReasoningConfig(
            reasoning_start_str="<think>",
            reasoning_end_str="</think>"),
        limit_mm_per_prompt={"image": 0}
        )
    if is_base:
        tokenizer = llm.get_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)


    result_dict = dict()
    sentences = get_sentences(dataset, src_lang)
    for tgt_lang in tqdm(langs):
        if src_lang == tgt_lang:
            continue
        
        assert all(["\n" not in sentence for sentence in sentences])
        targets = get_sentences(dataset, tgt_lang)
        prompt_prefix = make_few_shot_prefix(sentences[:n_shots], targets[:n_shots], is_base=is_base, is_thinking=is_thinking)
        
        targets = targets[n_shots:]

        if is_base:
            prompts = [prompt_prefix + sentence + "\n" for sentence in sentences[n_shots:]]
        else:
            prompts = []
            for sentence in sentences[n_shots:]:
                if is_thinking:
                    # enforce empty reasoning
                    conversation = prompt_prefix + [
                        {"role": "user", "content": sentence},
                        {"role": "assistant", "content": "<think>\n\n</think>\n\n"}
                    ]
                    chat_applied = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=False, continue_final_message=True)
                else:
                    conversation = prompt_prefix + [
                        {"role": "user", "content": sentence},
                    ]
                    chat_applied = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
                prompts.append(chat_applied.input_ids)

        sampling_params_batch = [SamplingParams(temperature=0, max_tokens=int(1.5*len(tokenizer(t).input_ids)), stop=["\n"], stop_token_ids=[tokenizer.eos_token_id]) for t in targets]
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params_batch,
            use_tqdm=False
        )

        answers = [process_answer(output.outputs[0].text) for output in outputs]

        result_dict[tgt_lang] = [{"output": o, "target": t} for o, t in zip(answers, targets)]

    os.makedirs(f"translation/translations_{dataset}", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=True, ensure_ascii=False)

    shutil.rmtree(cache_dir)
            


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model, 
             snakemake.params.src_lang, 
             snakemake.params.langs,
             snakemake.params.dataset,
             snakemake.params.n_shots)
    
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="google/gemma-3-12b-pt")
        parser.add_argument("--src-lang", type=str, help="Language to translate from", default="en")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to translate into",
                            default=ALL_LANGUAGES)
        parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["flores", "bouquet"], default="flores")
        parser.add_argument("--n-shots", type=int, help="Number of few-shot examples to use", default=5)

        args = parser.parse_args()
        main(args.model, args.src_lang, args.langs, args.dataset, args.n_shots)