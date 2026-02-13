import argparse
import os
import json

from datasets import load_dataset
from langcodes import Language
from transformers import AutoTokenizer

from constants import ALL_LANGUAGES


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


def main(model_name, langs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_short_name = model_name.split("/")[-1]
    out_path = "scores/num_tokens_flores/"
    os.makedirs(out_path, exist_ok=True)

    dataset = load_dataset("facebook/flores", data_dir="all", revision="refs/convert/parquet")["validation"]
    avg_lengths = dict()

    for lang in langs:
        flores_code = get_flores_code(lang)
        print(f"Processing {lang}. Trying to use flores code {flores_code}")

        tokens = tokenizer(list(dataset["sentence_"+flores_code])).input_ids
        lengths = map(len, tokens)
        avg_len = sum(lengths) / len(tokens)
        avg_lengths[lang] = avg_len

    with open(os.path.join(out_path, f"{model_short_name}.json"), "w") as out_file:
        json.dump(avg_lengths, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
    parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                        default=ALL_LANGUAGES)

    args = parser.parse_args()
    main(args.model, args.langs)
