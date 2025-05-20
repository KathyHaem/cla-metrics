import argparse
import json
import os
from collections import defaultdict
from typing import Set


def compute_overlap_coefficient(set_a: Set[int], set_b: Set[int]) -> float:
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def collect_probe_dims_overlaps(results_dir, output_dir, model, experiment_name, langs_file, attributes):
    raw_results = {x: {} for x in args.attributes}  # will i need this? who knows!
    probe_dims = {x: defaultdict(list) for x in args.attributes}
    # Read the languages from the file
    with open(langs_file, "r") as f:
        langs = [line.strip() for line in f.readlines()]
    # Iterate over the languages and attributes
    for lang in langs:
        for attribute in attributes:
            # Construct the file path
            file_path = os.path.join(
                results_dir,
                f"{model}/{experiment_name}/{lang}/{attribute}/loginfo.json"
            )

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            with open(file_path, "r") as f:
                data = json.load(f)
                raw_results[attribute][lang] = data
                probe_dims[attribute][lang] = [d["iteration_dimension"] for d in data if "iteration_dimension" in d]
    # Extract dimension overlaps
    dim_overlaps = {x: defaultdict(dict) for x in attributes}
    for attribute in attributes:
        for lang, dimensions in probe_dims[attribute].items():
            for other_lang, other_dimensions in probe_dims[attribute].items():
                if lang == other_lang:
                    continue
                if (lang, other_lang) in dim_overlaps[attribute]:
                    continue
                overlap_coefficient = compute_overlap_coefficient(set(dimensions), set(other_dimensions))
                overlap_set = sorted(list(set(dimensions & other_dimensions)))
                overlap_num = len(overlap_set)
                results_dict = {
                    "overlap_coefficient": overlap_coefficient,
                    "overlap_set": overlap_set,
                    "overlap_num": overlap_num
                }
                dim_overlaps[attribute][lang][other_lang] = results_dict
                dim_overlaps[attribute][other_lang][lang] = results_dict

    output_dir = os.path.join(output_dir, model, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "results.json")
    with open(out_file, "w") as f:
        # TODO maybe save raw results too?
        json.dump(dim_overlaps, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate morph probing scores.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../probing-multilingual-dynamics/multilingual-typology-probing/results/",
        help="Directory containing probing results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scores/morph_probe",
        help="Directory to save aggregated results.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name as in the results file path.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name.")
    parser.add_argument(
        "--langs",
        type=str,
        help="Languages list. Default is relative path to morph probing directory, coming from base dir of this repo.",
        default="../probing-multilingual-dynamics/multilingual-typology-probing/scripts/languages_experiment.lst"
    )
    parser.add_argument(
        "--attributes",
        type=str,
        nargs="+",
        default=["POS"],
        help="Attributes to aggregate scores for."
    )
    args = parser.parse_args()

    collect_probe_dims_overlaps(args.results_dir, args.output_dir, args.model, args.experiment_name, args.langs,
                                args.attributes)
