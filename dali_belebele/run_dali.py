from datasets import load_dataset, Value
from tqdm import tqdm
import torch
import torch.nn.functional as F
import json
import argparse
from langcodes import Language
from constants import ALL_LANGUAGES
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
def get_dali_score(tesnor1: torch.Tensor, tensor2: torch.Tensor, correct_idx: torch.Tensor, strong = False):
    if not strong:
        correct_source = tesnor1[torch.arange(tesnor1.size(0)), correct_idx].unsqueeze(1)
        sim_with_correct = F.cosine_similarity(correct_source, tensor2, dim = -1)
        closest_option = sim_with_correct.argmax(1)
        success = closest_option == correct_idx.unsqueeze(1)
        success_rate = success.sum(0) / success.size(0)
        return success_rate
    else:
        raise NotImplementedError


def main(model_id): 
    ds = load_dataset("facebook/belebele", get_flores_code("en"), split="test").sort(["link", "question_number"]).cast_column("correct_answer_num", Value(dtype="int32"))
    correct_indices = torch.tensor(ds["correct_answer_num"], device="cuda") - 1
    
    res_dict = dict()
    for i, src_lang in enumerate(ALL_LANGUAGES):
        print(f"Running on {src_lang} ({i+1}/{len(ALL_LANGUAGES)})")
        try:
            src_emb = torch.load(f"embeds/dali-belebele/{model_id.split('/')[1]}-{src_lang}.pt", map_location="cuda")
        except FileNotFoundError:
            continue
        for tgt_lang in tqdm(ALL_LANGUAGES):
            if tgt_lang == src_lang:
                continue
            try:
                tgt_emb = torch.load(f"embeds/dali-belebele/{model_id.split('/')[1]}-{tgt_lang}.pt", map_location="cuda")
            except FileNotFoundError:
                continue
            res_dict[f"{src_lang}-{tgt_lang}"] = get_dali_score(src_emb, tgt_emb, correct_indices).cpu().tolist()

    os.makedirs("scores/dali_belebele/", exist_ok=True)

    with open(f"scores/dali_belebele/{model_id.split('/')[1]}.json", "w") as f:
        json.dump(res_dict, f, indent=True)
    
if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model)
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        args = parser.parse_args()

        main(args.model)