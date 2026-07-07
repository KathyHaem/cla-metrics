# modified from https://github.com/KathyHaem/token-alignability/blob/main/prep_tok_files.py

import argparse
import os

from datasets import Dataset
from langcodes import Language
from transformers import AutoTokenizer



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

def make_tok_file_flores(model_id, src_short, tgt_short):
    """
    :param model_id: Full model name (e.g., "bert-base-uncased")
    :param src_short: Source language name, two-letter code (or three-letter code if needed)
    :param tgt_short: Target language name, same format
    """
    src_tag = get_flores_code(src_short)
    tgt_tag = get_flores_code(tgt_short)

    dataset = Dataset.from_parquet("eflomal/computations/dataset/flores/all/flores-dev.parquet")

    return get_tokens(dataset, model_id, src_short, tgt_short, src_key=f"sentence_{src_tag}",
                      tgt_key=f"sentence_{tgt_tag}")


def get_tokens(dataset, model_id, src_lang, tgt_lang, src_key=None, tgt_key=None):
    if src_key is None:
        src_key = src_lang
    if tgt_key is None:
        tgt_key = tgt_lang

    tokenizer = AutoTokenizer.from_pretrained(f"./eflomal/computations/{model_id.split('/')[1]}/tokenizer", fix_mistral_regex=True)
    src_tokens = dataset.map(lambda x: {
        'tok': tokenizer.convert_ids_to_tokens(tokenizer(x[src_key])['input_ids'], skip_special_tokens=True)})
    tgt_tokens = dataset.map(lambda x: {
        'tok': tokenizer.convert_ids_to_tokens(tokenizer(x[tgt_key])['input_ids'], skip_special_tokens=True)})
    
    return src_tokens, tgt_tokens



def write_tok_file(model_id: str, src_lang: str, src_tokens: Dataset,
                   tgt_lang: str,
                   tgt_tokens: Dataset):
    zip_tokens = zip(list(src_tokens), list(tgt_tokens))
    file_name = f'eflomal/computations/{model_id.split("/")[1]}/{src_lang}-{tgt_lang}.tok.fast_align'
    with open(file_name, "w+") as f:
        for src, tgt in zip_tokens:
            src = src["tok"]
            tgt = tgt["tok"]
            f.write(" ".join(src) + " ||| " + " ".join(tgt) + "\n")


def tok_file_exists(model_id, src_lang, tgt_lang):
    file_name = f'eflomal/computations/{model_id.split("/")[1]}/{src_lang}-{tgt_lang}.tok.fast_align'
    return os.path.isfile(file_name)


def main(model_id: str, src_lang: str, tgt_lang: str, overwrite = False):
    # add other datasets as needed
    assert (src_lang != tgt_lang)


    out_dir = f"eflomal/computations/{model_id.split('/')[1]}"
    os.makedirs(out_dir, exist_ok=True)

    if tok_file_exists(model_id, src_lang, tgt_lang) and not overwrite:
        return
    
    src_tokens, tgt_tokens = make_tok_file_flores(model_id, src_lang, tgt_lang)

    write_tok_file(model_id, src_lang, src_tokens, tgt_lang, tgt_tokens)


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.model, snakemake.params.src, snakemake.params.tgt)
    else:
        parser = argparse.ArgumentParser(description="Prepare tokenized files for fast_align.")
        parser.add_argument("--model_id", type=str, help="Model ID, as in HuggingFace (e.g., bert-base-uncased).")
        parser.add_argument("--src_lang", type=str, help="Source language name, two-letter code (or three-letter code if needed)")
        parser.add_argument("--tgt_lang", type=str, help="Target language name, same format")
        parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing files.")
        args = parser.parse_args()
        main(args.model_id, args.src_lang, args.tgt_lang, args.overwrite)
