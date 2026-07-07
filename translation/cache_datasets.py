from langcodes import Language
from datasets import load_dataset
import argparse
from constants_translate import ALL_LANGUAGES

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


def main(langs: list[str], dataset: str = "flores"):
    for lang in langs:
        load_dataset(f"facebook/{dataset}", get_flores_code(lang, dataset=dataset), split="devtest" if dataset == "flores" else "dev")
            
    with open(f"translation/dataset_{dataset}_cached.ok", "w") as f:
        f.write("done")

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(snakemake.params.langs, snakemake.params.dataset)
    else:
        parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
        parser.add_argument("--langs", type=str, nargs="+", help="Languages to translate into",
                            default=ALL_LANGUAGES)
        parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["flores", "bouquet"], default="flores")

        args = parser.parse_args()
        main(args.langs, args.dataset)