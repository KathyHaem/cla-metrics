import json
import argparse
import os
from constants import ALL_LANGUAGES

def main(model: str, langs: list[str]):
    result_dict = dict()

    for src_lang in langs:
        with open(f"translation/translation_mutinf/{model.split('/')[1]}-{src_lang}.json") as f:
            scores = json.load(f)
        
        for tgt_lang in langs:
            if tgt_lang != src_lang:
                result_dict[f"{src_lang}-{tgt_lang}"] = scores[tgt_lang]

    os.makedirs("scores/translation_mutinf", exist_ok=True)
    with open(f"scores/translation_mutinf/{model.split('/')[1]}.json", "w") as f:
        json.dump(result_dict, f, indent=True)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model, 
             snakemake.params.langs)
    
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to translate from/into",
                            default=ALL_LANGUAGES)

        args = parser.parse_args()
        main(args.model, args.langs)