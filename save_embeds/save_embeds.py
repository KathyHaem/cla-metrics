import argparse

from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoTokenizer
from langcodes import Language

ALL_LANGUAGES = "ar tr zh el es en sw hi mr ur ta te th ru bg he ka vi fr de".split()  # TODO add others


def get_flores_code(short_code: str):
    lang = Language.get(short_code)
    script = lang.script or lang.assume_script().script
    return "arb_Arab" if short_code == "ar" else "zho_Hant" if short_code == "zh" else "swh_Latn" if short_code == "sw" else f"{lang.to_alpha3()}_{script}"


def encode_batch(batch, model, tokenizer, sent_rep):
    pass


def main(dataset_name, model_name, langs, sent_rep):
    if dataset_name != "flores":  # happy path or w/e. bit more effort if I do implement other datasets
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    out_dict = {}
    for lang in langs:
        flores_code = get_flores_code(lang)
        dataset = load_dataset("facebook/flores", flores_code, trust_remote_code=True)["devtest"]  # todo use dev?

        # todo check that shit works
        dataset.map(encode_batch, batched=True,
                    fn_kwargs={"model": model, "tokenizer": tokenizer, "sent_rep": sent_rep},
                    remove_columns=dataset.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save model embeddings for parallel data")
    parser.add_argument("--dataset", type=str, default="flores", help="Dataset name. Just FLORES atm")
    parser.add_argument("--model", type=str, required=True, help="Model name. Assuming an HF decoder")
    # might do model short names again
    parser.add_argument("--langs", type=str, nargs="+", help="Languages to embed",
                        default=ALL_LANGUAGES)

    parser.add_argument("--sent-rep", type=str, default="mean", help="How to sentence rep",
                        choices=["mean", "prompt"])
    # layers? probably just do all of them?

    args = parser.parse_args()
    main(args.dataset, args.model, args.langs, args.sent_rep)
