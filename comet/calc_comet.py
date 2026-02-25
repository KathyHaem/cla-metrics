from comet import download_model, load_from_checkpoint
import json
import os
import torch
import argparse

torch.set_float32_matmul_precision('high')

xlm_langs = ['af', 'sq', 'am', 'ar', 'hy', 'as', 'az', 'eu', 'be', 'bn', 'bn', 'bs', 'br', 'bg', 'my', 'my', 'ca', 'zh', 'zh-Hant', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et', 'fil', 'fi', 'fr', 'gl', 'ka', 'de', 'el', 'gu', 'ha', 'he', 'hi', 'hi', 'hu', 'is', 'id', 'ga', 'it', 'ja', 'jv', 'kn', 'kk', 'km', 'ko', 'ku', 'ky', 'lo', 'la', 'lv', 'lt', 'mk', 'mg', 'ms', 'ml', 'mr', 'mn', 'ne', 'no', 'or', 'om', 'ps', 'fa', 'pl', 'pt', 'pa', 'ro', 'ru', 'gd', 'gd', 'sr', 'sd', 'si', 'sk', 'sl', 'so', 'es', 'su', 'sw', 'sv', 'ta', 'ta', 'te', 'te', 'th', 'tr', 'uk', 'ur', 'ur', 'ug', 'uz', 'vi', 'cy', 'fy', 'xh', 'yi']

our_langs = ['af', 'am', 'ar', 'az', 'bg', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'he',
                 'hi', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'km', 'kn', 'ko', 'lt', 'ml', 'mr', 'nl',
                 'pl', 'pt', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'zh']

leftover_lang = "tl"

def main(model_id, src_lang, tgt_langs):
    os.makedirs("comet/scores", exist_ok=True)
    _, model_short = model_id.split("/")

    available_langs = set(tgt_langs).intersection(xlm_langs)

    if src_lang not in xlm_langs:
        print("Source language not supported")
        with open(f"comet/scores/{model_short}_{src_lang}.json", "w"):
            pass
        return

    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)


    with open(f"translation/translations/{model_short}_{leftover_lang}.json") as f:
        original_dataset = json.load(f)

    result = {}
    with open(f"translation/translations/{model_short}_{src_lang}.json") as f:
        src_dataset = json.load(f)
    
    for tgt_lang in available_langs:
        if src_lang == tgt_lang:
            continue
        records = src_dataset[tgt_lang]
        data = []
        for i in range(len(records)):
            data.append({
                "src": original_dataset[src_lang][i]["target"],
                "ref": records[i]["target"],
                "mt": records[i]["output"]
            })

        model_output = model.predict(data, batch_size=32, gpus=1)
        result[f"{src_lang}-{tgt_lang}"] = {"mean": model_output.system_score, "sentence-level": model_output.scores}

    with open(f"comet/scores/{model_short}_{src_lang}.json", "w") as f:
        json.dump(result, f, indent=True)


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model,
             snakemake.params.src_lang,
             snakemake.params.langs)
    else:
        parser = argparse.ArgumentParser(description="Calculate comet scores from the translations.")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--src-lang", type=str, help="Source language", default="en")
        parser.add_argument("--langs", type=str, nargs="+", help="Target languages.",
                            default=our_langs)
        args = parser.parse_args()

        main(args.model, args.src_lang, args.langs)