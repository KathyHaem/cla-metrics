# Snakefile

# Run with snakemake==9.3.0 snakemake-executor-plugin-slurm==0.14.2

from constants import ALL_LANGUAGES, MUTUALLY_INTELLIGIBLE
import os
import time
import random
import hashlib

CPU_PARTITION="cpu-troja"
GPU_PARTITION="gpu-troja,gpu-ms"
GPU_CONSTRAINT="gpuram40G"#|gpuram48G"
GPU_CONSTRAINT_A100="gpuram40G"

def gres_gpu(num):
    return f"--gres 'gpu:{num}'"

def get_free_gpus(node):
    allocated = os.popen(fr'scontrol show node {node} | grep -Po "AllocTRES[^ ]*(?<=gpu=)\K[0-9]+"').read().strip()
    if allocated == "":
        allocated = 0
    else:
        allocated = int(allocated)
    total = int(os.popen(fr'scontrol show node {node} | grep -Po "CfgTRES[^ ]*(?<=gpu=)\K[0-9]+"').read().strip())
    return total - allocated

assigned_resources = dict()
def select_24_or_40G(wildcards, base_num_gpus=2):
    key = "_".join(str(v) for v in wildcards.__dict__.values() if isinstance(v, str) or isinstance(v, int))
    hash_key = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    
    use_24G = hash_key % 2 == 0
    if use_24G:
        res = {
            "constraint": "gpuram24G",
            "num_gpus": base_num_gpus*2
        }
    else:
        res = {
            "constraint": "gpuram40G",
            "num_gpus": base_num_gpus
        }

    return res 

    # nodes = ["tdll-8gpu1", "tdll-8gpu2", "dll-8gpu1", "dll-8gpu2"]
    # weights = [3, 4, 2, 2]

    # for node in nodes:
    #     free_gpus = get_free_gpus(node)
    #     required_gpus = base_num_gpus*(1 if node.startswith("tdll") else 2)
    #     if free_gpus >= required_gpus:
    #         res = {
    #             "constraint": "gpuram40G" if node.startswith("tdll") else "gpuram24G",
    #             "num_gpus": base_num_gpus*(1 if node.startswith("tdll") else 2)
    #         }
    #         break
    # else:
    #     res = {
    #         "constraint": "gpuram40G",
    #         "num_gpus": base_num_gpus
    #     }

    # assigned_resources[key] = res
    # return res
        

FULL_MODELS = [
    # "Qwen/Qwen3-14B-Base",
    # "Qwen/Qwen3-14B",
    # "mistralai/Ministral-3-14B-Base-2512",
    # "mistralai/Ministral-3-14B-Instruct-2512",
    "google/gemma-3-12b-pt",
    # "google/gemma-3-12b-it"
]

def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]

MODELS = { short_name(m): m for m in FULL_MODELS }

SCORES = ["cosine", "anc", "dist", "ratio", "nn-abs"]

SENT_REPS = ["mean", "weighted-mean", "prompt", "last-token", "fewshot"]

TRANSLATION_DATASETS = ["flores", "bouquet"]

rule all:
    input:
        # expand("scores/flores/{short}_{sent_rep}_{score}.json",
        #        short=MODELS.keys(),
        #        sent_rep=SENT_REPS,
        #        score=SCORES),
        # expand("scores/translation_{dataset}_chrf/{short}.json",
        #        short=MODELS.keys(),
        #        dataset=TRANSLATION_DATASETS),
        expand("scores/translation_{dataset}_mutinf/{short}.json",
               short=MODELS.keys(),
               dataset=TRANSLATION_DATASETS),
        # expand("scores/dali_belebele/{short}.json",
        #        short=MODELS.keys()),
        # expand("scores/belebele/{short}_acc.json",
        #        short=MODELS.keys()),
        # expand("scores/sib-200/{short}.json",
        #        short=MODELS.keys()),
        # expand("scores/eflomal/{model}.json",
        #        model=MODELS.keys())

rule belebele:
    output:
        "scores/belebele/{short}_acc.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        constraint=GPU_CONSTRAINT,
        gpu=2,
        gpu_use=2
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES
    priority:
        -1
    conda:
        "envs/vllm.yaml"
    script:
        "belebele/belebele_vllm.py"

rule sib_200:
    output:
        "scores/sib-200/{short}.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        constraint=GPU_CONSTRAINT,
        gpu=1,
        gpu_use=1
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES
    conda:
        "envs/vllm.yaml"
    script:
        "sib-200/vllm_probing.py"

rule aggregate_comet:
    input:
        expand("comet/scores/{{short}}_{lang}.json", lang=ALL_LANGUAGES)
    output:
        "scores/comet/{short}.json"
    resources:
        slurm_partition=CPU_PARTITION,
        mem_mb=4000
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES
    script:
        "comet/comet_aggregate.py"

rule translations_comet:
    input:
        "translation/translations/{short}_{lang}.json",
        "translation/translations/{short}_tl.json" # this will be used to extract the untranslated sentences for all the supported languages
    output:
        "comet/scores/{short}_{lang}.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=40000,
        constraint=GPU_CONSTRAINT,
        gpu=1,
        gpu_use=1
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        src_lang=lambda wildcards: wildcards.lang,
        langs=ALL_LANGUAGES
    conda:
        "envs/comet.yaml"
    script:
        "comet/calc_comet.py"

rule cache_translation_datasets:
    output:
        "translation/dataset_{dataset}_cached.ok"
    resources:
        slurm_partition=CPU_PARTITION,
        mem_mb=1000
    params:
        langs=ALL_LANGUAGES,
        dataset=lambda wildcards: wildcards.dataset
    conda:
        "envs/transformers.yaml"
    script:
        "translation/cache_datasets.py"

rule translations_chrf:
    input:
        expand("translation/translations_{{dataset}}/{{short}}_{lang}.json", lang=ALL_LANGUAGES)
    resources:
        slurm_partition=CPU_PARTITION,
        mem_mb=8000,
        cpus_per_task=45,
        tasks=1
    output:
        "scores/translation_{dataset}_chrf/{short}.json"
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES,
        dataset=lambda wildcards: wildcards.dataset
    priority:
        10
    conda:
        "envs/transformers.yaml"
    script:
        "translation/evaluate.py"

rule translations_vllm:
    input:
        "translation/dataset_{dataset}_cached.ok"
    output:
        "translation/translations_{dataset}/{short}_{lang}.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=16000,
        constraint=GPU_CONSTRAINT,
        gpu=2,
        gpu_use=2
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        src_lang=lambda wildcards: wildcards.lang,
        langs=ALL_LANGUAGES,
        dataset=lambda wildcards: wildcards.dataset,
        n_shots=5
    conda:
        "envs/vllm.yaml"
    script:
        "translation/run_translate.py"

rule translations_mutinf_agregate:
    input:
        expand("translation/translation_{{dataset}}_mutinf/{{short}}-{lang}.json", lang = ALL_LANGUAGES)
    output:
        "scores/translation_{dataset}_mutinf/{short}.json"
    resources:
        slurm_partition=CPU_PARTITION,
        mem_mb=2000
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES,
        dataset=lambda wildcards: wildcards.dataset
    priority:
        10
    script:
        "translation/aggregate_mutinf.py"

rule translations_mutinf:
    input:
        "translation/dataset_{dataset}_cached.ok"
    output:
        "translation/translation_{dataset}_mutinf/{short}-{lang}.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=16000,
        constraint=GPU_CONSTRAINT_A100,
        gpu=2,
        gpu_use=2
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        src_lang=lambda wildcards: wildcards.lang,
        langs=ALL_LANGUAGES,
        batch_size=10,
        dataset=lambda wildcards: wildcards.dataset,
        n_shots=5
    priority:
        -1
    conda:
        "envs/transformers.yaml"
    script:
        "translation/run_translations_mutinf.py"

rule calc_scores_dali:
    input:
        expand("embeds/dali-belebele/{{short}}-{lang}.pt", lang = ALL_LANGUAGES)
    output:
        "scores/dali_belebele/{short}.json"
    params:
        model=lambda wildcards: MODELS[wildcards.short]
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=16000,
        gpu=1,
        gpu_use=1,
        constraint=GPU_CONSTRAINT
    conda:
        "envs/transformers.yaml"
    script:
        "dali_belebele/run_dali.py"

rule save_embeds_dali:
    output:
        "embeds/dali-belebele/{short}-{lang}.pt"
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        batch_size=10,
        lang=lambda wildcards: wildcards.lang
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        gpu=2,
        gpu_use=2,
        constraint=GPU_CONSTRAINT
    conda:
        "envs/transformers.yaml"
    script:
        "dali_belebele/save_embeds.py"

rule calc_scores:
    input:
        expand("embeds/flores/{{short}}-{{sent_rep}}-{lang}.pickle", lang=ALL_LANGUAGES)
    output:
        "scores/flores/{short}_{sent_rep}_{score}.json"
    resources:
        slurm_partition=lambda wildcards: CPU_PARTITION if wildcards.score in ["dist", "ratio", "nn-abs"] else GPU_PARTITION,
        mem_mb=16000,
        constraint=lambda wildcards: "" if wildcards.score in ["dist", "ratio", "nn-abs"] else GPU_CONSTRAINT,
        gpu=lambda wildcards: 0 if wildcards.score in ["dist", "ratio", "nn-abs"] else 1,
        gpu_use=lambda wildcards: 0 if wildcards.score in ["dist", "ratio", "nn-abs"] else 1,
        tasks=1,
        cpus_per_task=lambda wildcards: 10 if wildcards.score in ["dist", "ratio", "nn-abs"] else 2,
        mutually_intelligible=False
    priority:
        1
    threads:
        10
    params:
        dataset="flores",
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES,
        sent_rep=lambda wildcards: wildcards.sent_rep,
        overwrite=False,
        score=lambda wildcards: [wildcards.score]
    conda:
        "envs/transformers.yaml"
    script:
        "calc_scores.py"

rule save_embeds:
    input:
        "definitions.json"
    output:
        expand("embeds/flores/{{short}}-{{sent_rep}}-{lang}.pickle", lang=ALL_LANGUAGES),
    params:
        dataset="flores",
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES,
        sent_rep=lambda wildcards: wildcards.sent_rep,
        batch_size=10,
        overwrite=False
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        gpu=2,
        gpu_use=2,
        constraint=GPU_CONSTRAINT
    conda:
        "envs/transformers.yaml"
    script:
        "save_embeds.py"

rule save_definitions:
    output:
        "definitions.json"
    resources:
        mem_mb=2000,
        slurm_partition=CPU_PARTITION
    conda:
        "envs/transformers.yaml"
    script:
        "get_word_definitions.py"

# Eflomal

rule score_eflomal:
    resources:
        mem_mb=8000,
        slurm_partition=CPU_PARTITION
    input:
        ["eflomal/computations/{{model}}/{src}-{tgt}.eflomal.sym".format(src=s, tgt=t) for s in ALL_LANGUAGES for t in ALL_LANGUAGES if s != t],
        ["eflomal/computations/{{model}}/{src}-{tgt}.eflomal.scores.fwd".format(src=s, tgt=t) for s in ALL_LANGUAGES for t in ALL_LANGUAGES if s != t],
        ["eflomal/computations/{{model}}/{src}-{tgt}.eflomal.scores.rev".format(src=s, tgt=t) for s in ALL_LANGUAGES for t in ALL_LANGUAGES if s != t]
    output:
        "scores/eflomal/{model}.json"
    params:
        model=lambda wildcards: MODELS[wildcards.model],
        langs=ALL_LANGUAGES
    conda:
        "envs/eflomal.yaml"
    script:
        "eflomal/alignability.py"

rule symmetrize:
    resources:
        mem_mb=2000,
        slurm_partition=CPU_PARTITION
    input:
        "eflomal/computations/fast_align/build/atools",
        "eflomal/computations/{model}/{src}-{tgt}.eflomal.fwd",
        "eflomal/computations/{model}/{src}-{tgt}.eflomal.rev"
    output:
        "eflomal/computations/{model}/{src}-{tgt}.eflomal.sym"
    shell:
        """eflomal/computations/fast_align/build/atools -i "eflomal/computations/{wildcards.model}/{wildcards.src}-{wildcards.tgt}.eflomal.fwd" \
        -j "eflomal/computations/{wildcards.model}/{wildcards.src}-{wildcards.tgt}.eflomal.rev" \
        -c grow-diag-final-and > "eflomal/computations/{wildcards.model}/{wildcards.src}-{wildcards.tgt}.eflomal.sym" """

rule run_eflomal:
    resources:
        mem_mb=2000,
        slurm_partition=CPU_PARTITION
    input:
        "eflomal/computations/{model}/{src}-{tgt}.tok.fast_align"
    output:
        "eflomal/computations/{model}/{src}-{tgt}.eflomal.fwd",
        "eflomal/computations/{model}/{src}-{tgt}.eflomal.rev",
        "eflomal/computations/{model}/{src}-{tgt}.eflomal.scores.fwd",
        "eflomal/computations/{model}/{src}-{tgt}.eflomal.scores.rev"
    conda:
        "envs/eflomal.yaml"
    shell:
        """
        eflomal-align \
        -i "eflomal/computations/{wildcards.model}/{wildcards.src}-{wildcards.tgt}.tok.fast_align" \
        -f="eflomal/computations/{wildcards.model}/{wildcards.src}-{wildcards.tgt}.eflomal.fwd" \
        -r="eflomal/computations/{wildcards.model}/{wildcards.src}-{wildcards.tgt}.eflomal.rev" \
        --forward-scores "eflomal/computations/{wildcards.model}/{wildcards.src}-{wildcards.tgt}.eflomal.scores.fwd" \
        --reverse-scores "eflomal/computations/{wildcards.model}/{wildcards.src}-{wildcards.tgt}.eflomal.scores.rev" --overwrite
        """

rule prep_corpus:
    resources:
        mem_mb=2000,
        slurm_partition=CPU_PARTITION
    input:
        "eflomal/computations/{model}/tokenizer",
        "eflomal/computations/dataset/flores/all/flores-dev.parquet"
    output:
        "eflomal/computations/{model}/{src}-{tgt}.tok.fast_align"
    params:
        model=lambda wildcards: MODELS[wildcards.model],
        src=lambda wildcards: wildcards.src,
        tgt=lambda wildcards: wildcards.tgt
    conda:
        "envs/eflomal.yaml"
    script:
        "eflomal/prep_tok_files.py"

rule download_dataset:
    resources:
        mem_mb=2000,
        slurm_partition=CPU_PARTITION
    output:
        "eflomal/computations/dataset/flores/all/flores-dev.parquet"
    params:
        model=None,
        resource="dataset"
    conda:
        "envs/transformers.yaml"
    script:
        "eflomal/download_resources.py"

rule download_tokenizer:
    resources:
        mem_mb=2000,
        slurm_partition=CPU_PARTITION
    output:
        directory("eflomal/computations/{model}/tokenizer"),
    params:
        model = lambda wildcards: MODELS[wildcards.model],
        resource="tokenizer"
    conda:
        "envs/transformers.yaml"
    script:
        "eflomal/download_resources.py"
        
rule build_fast_align:
    resources:
        mem_mb=2000,
        slurm_partition=CPU_PARTITION
    output:
        "eflomal/computations/fast_align/build/atools"
    shell:
        """
        cd eflomal/computations
        rm -rf fast_align
        git clone https://github.com/clab/fast_align.git
        mkdir fast_align/build
        cd fast_align/build
        cmake .. && make"""