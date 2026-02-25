import json
import argparse
import os

def main(model_id, langs):
    _, model_short = model_id.split("/")

    all_scores = dict()
    for src_lang in langs:
        try:
            with open(f"comet/scores/{model_short}_{src_lang}.json") as f:
                all_scores.update(json.load(f))
        except:
            continue
    
    os.makedirs("scores/comet/", exist_ok=True)
    with open(f"scores/comet/{model_short}.json", "w") as f:
        json.dump(all_scores, f, indent=True)
            

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model,
             snakemake.params.langs)
    else:
        parser = argparse.ArgumentParser(description="Calculate comet scores from the translations.")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--langs", type=str, help="Languages to aggregate")
        args = parser.parse_args()

        main(args.model, args.src_lang, args.langs)