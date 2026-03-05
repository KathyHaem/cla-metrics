from huggingface_hub import hf_hub_download, snapshot_download
import os
import sys
import argparse

def main(model_id: str, resource: str):
    match resource:
        case "tokenizer":
            dir_name = f"eflomal/computations/{model_id.split('/')[-1]}/tokenizer"
            os.makedirs(dir_name, exist_ok=True)
            hf_hub_download(repo_id=model_id, filename="tokenizer.json",
                                             local_dir=dir_name)
            hf_hub_download(repo_id=model_id, filename="tokenizer_config.json",
                                             local_dir=dir_name)
            hf_hub_download(repo_id=model_id, filename="config.json",
                            local_dir=dir_name)
        case "dataset":
            dir_name_flores = f"eflomal/computations/dataset/flores"
            os.makedirs(dir_name_flores, exist_ok=True)
            print(hf_hub_download("facebook/flores", subfolder="all",
                            filename="flores-dev.parquet", repo_type="dataset",
                            revision="refs/convert/parquet", local_dir=dir_name_flores), file=sys.stderr)
if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        main(snakemake.params.model, snakemake.params.resource)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
        parser.add_argument("--resource", type=str, required=True, choices=["tokenizer", "dataset"], help="Resource to download")
        args = parser.parse_args()
        main(args.model, args.resource)