import json
from chrFpp import computeChrF
from tqdm import tqdm
from constants import ALL_LANGUAGES
import argparse
import os
from multiprocessing import Pool, cpu_count

def sentence_level_chrf(refs, outs):
    return [computeChrF([r], [o], nworder=2, ncorder=6, beta=2)[1] for r, o in zip(refs, outs)]

def score_one_src(args):
    model_short, src_lang = args

    with open(f"translation/translations/{model_short}_{src_lang}.json") as f:
        translations = json.load(f)

    local = {}
    for tgt_lang, results in translations.items():
        outs = [res["output"] for res in results]
        refs = [res["target"] for res in results]
        sentence_chrfs = sentence_level_chrf(refs, outs)
        mean_chrf = sum(sentence_chrfs) / len(sentence_chrfs)

        local[f"{src_lang}-{tgt_lang}"] = {"mean": mean_chrf, "sentence-level": sentence_chrfs}

    return local


def main(model: str, langs: list[str]):
    _, model_short = model.split("/")

    tasks = [(model_short, src_lang) for src_lang in langs]

    mean_chrf = {}

    with Pool(processes=cpu_count()) as pool:
        # tqdm over completion of src_lang jobs
        for local in tqdm(pool.imap_unordered(score_one_src, tasks), total=len(tasks)):
            mean_chrf.update(local)

    os.makedirs("scores/translation_chrf", exist_ok=True)
    with open(f"scores/translation_chrf/{model_short}.json", "w") as f:
        json.dump(mean_chrf, f, indent=True)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model,  
             snakemake.params.langs)
    
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--model", type=str, help="Model name. Assuming an HF decoder", default="meta-llama/Llama-3.2-3B")
        parser.add_argument("--src-lang", type=str, help="Language to translate from", default="en")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to translate into",
                            default=ALL_LANGUAGES)

        args = parser.parse_args()
        main(args.model, args.src_lang, args.langs)