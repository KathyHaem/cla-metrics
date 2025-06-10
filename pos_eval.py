import argparse
import json
import os
from typing import List, Tuple, Dict, Any

from datasets import load_dataset
from seqeval.metrics import f1_score
from vllm import LLM, SamplingParams

from constants import ALL_LANGUAGES

UDPOS_TAGS_TO_IDS = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "AUX": 3,
    "CCONJ": 4,
    "DET": 5,
    "INTJ": 6,
    "NOUN": 7,
    "NUM": 8,
    "PART": 9,
    "PRON": 10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM": 14,
    "VERB": 15,
    "X": 16
}

IDS_TO_UDPOS_TAGS = {v: k for k, v in UDPOS_TAGS_TO_IDS.items()}


def parse_labels_from_string(response: str) -> List[Tuple[str, str]]:
    """
    Parse the model response to extract the POS tags.
    The expected format is a list of tuples, where each tuple contains a word and its corresponding POS tag.
    """
    try:
        # Assuming the response is a string representation of a list of tuples
        # I suspect this will raise a lot of errors due to chatty models. Let's see.
        pos_tuples = eval(response)
        if isinstance(pos_tuples, list) and all(isinstance(item, tuple) and len(item) == 2 for item in pos_tuples):
            pos_tags = [y[1] for y in pos_tuples]
            return pos_tags
        else:
            raise ValueError("Invalid format for POS tags.")
    except ValueError as e:
        print(f"Error parsing model response: {response}")
        return []


def load_pos_data(langs):
    datasets = {}
    for lang in langs:
        try:
            lang_dataset = load_dataset("wietsedv/udpos28", lang, trust_remote_code=True)
        except KeyError as e:
            print(f"No UDPOS data for {lang} in the dataset. Skipping.")
            continue
        datasets[lang] = lang_dataset["test"]
    return datasets


def build_model_inputs(batch: List[Dict[str, Any]], prompt: str):
    # TODO actually: include the chat template where applicable!
    inputs = [prompt.format(input=item["tokens"]) for item in batch]
    return inputs


def get_sampling_params():
    return SamplingParams(
        use_beam_search=False,
        best_of=1,
        # Since we are doing sentence level we can stop at newlines. (?)
        stop=["<|im_end|>", "\n", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        min_p=0.02,
        temperature=1,
        max_tokens=4096,
    )


def eval_and_save(data, generations, lang, model_short_name, output_file):
    outputs_to_save = [{} for _ in data]
    for idx, (item, generation) in enumerate(zip(data, generations)):
        outputs_to_save[idx]["tokens"] = item["tokens"]
        outputs_to_save[idx]["label_ids"] = item["labels"]
        outputs_to_save[idx]["label_tags"] = [IDS_TO_UDPOS_TAGS[label_id] for label_id in item["labels"]]
        outputs_to_save[idx]["model_output"] = generation
        outputs_to_save[idx]["prediction"] = parse_labels_from_string(generation)

    y_true = [item["label_tags"] for item in outputs_to_save]
    y_pred = [item["prediction"] for item in outputs_to_save]
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"F1 score for {lang}: Micro: {micro_f1}, Macro: {macro_f1}")
    out_dict = {
        "model": model_short_name,
        "lang": lang,
        "f1_macro": macro_f1,
        "f1_micro": micro_f1,
        "model_outputs": outputs_to_save,
    }
    print(f"Saving results to {output_file}")
    with open(output_file, 'w') as file:
        json.dump(out_dict, file, indent=4)


def run_pos_eval(model_name, langs, prompt, gpus, overwrite=False):
    model_short_name = model_name.split("/")[-1]
    output_directory = os.path.join("outputs", "pos", model_short_name)
    os.makedirs(output_directory, exist_ok=True)

    llm = LLM(model=model_name, tensor_parallel_size=gpus)
    sampling_params = get_sampling_params()

    pos_data = load_pos_data(langs)
    for lang, data in pos_data.items():
        output_file = os.path.join(output_directory, f"{lang}_pos_outputs.json")
        if os.path.exists(output_file) and not overwrite:
            print(f"Output file {output_file} already exists. Skipping {lang}.")
            continue

        model_inputs = build_model_inputs(data, prompt)
        outputs = llm.generate(model_inputs, sampling_params)
        generations = [o.outputs[0].text for o in outputs]

        eval_and_save(data, generations, lang, model_short_name, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance on POS tagging")
    parser.add_argument("--model", type=str, required=True, help="The model to evaluate.")
    parser.add_argument("--langs", type=str, nargs="+", default=ALL_LANGUAGES,
                        help="The languages to evaluate the model on.")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please provide the POS tags for each word in the input sentence. The input will be a list of words in \
        the sentence. The output format should be a list of tuples, where each tuple consists of a word from the input \
        text and its corresponding POS tag label from the tag label set: ['ADJ', 'ADP', 'ADV', 'AUX','CCONJ', 'DET', \
        'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']. \n \
        Your response should include only a list of tuples, in the order that the words appear in the input \
        sentence, with each tuple containing the corresponding POS tag label for a word.\n {input}",
        help="The prompt to use for evaluation."
    )
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()

    run_pos_eval(args.model, args.langs, args.prompt, args.gpus, args.overwrite)
