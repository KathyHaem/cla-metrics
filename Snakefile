# Snakefile

# Run with snakemake==9.3.0 snakemake-executor-plugin-slurm==0.14.2

from constants import ALL_LANGUAGES

CPU_PARTITION="cpu-troja"
GPU_PARTITION="gpu-troja,gpu-ms"
GPU_CONSTRAINT="gpuram40G|gpuram48G"

def gres_gpu(num):
    return f"--gres 'gpu:{num}'"

FULL_MODELS = [
    "Qwen/Qwen3-14B",
    "mistralai/Ministral-3-14B-Base-2512",
    "google/gemma-3-12b-pt"
]

def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]

MODELS = { short_name(m): m for m in FULL_MODELS }

SCORES = ["cosine", "anc", "dist", "ratio", "nn-abs"]

SENT_REPS = ["mean", "weighted-mean", "prompt", "last-token", "fewshot"]

rule all:
    input:
        expand("scores/flores/{short}_{sent_rep}_{score}.json",
               short=MODELS.keys(),
               sent_rep=SENT_REPS,
               score=SCORES),
        expand("scores/translation_chrf/{short}.json",
               short=MODELS.keys()),
        expand("scores/translation_mutinf/{short}.json",
               short=MODELS.keys()),
        expand("scores/dali_belebele/{short}.json",
               short=MODELS.keys()),
        # expand("scores/comet/{short}.json",
        #        short=MODELS.keys()),
        expand("scores/belebele/{short}_acc.json",
               short=MODELS.keys()),
        expand("scores/sib-200/{short}.json",
               short=MODELS.keys()),
        # expand("scores/eflomal/{model}.json",
        #        model=MODELS.keys())

rule belebele:
    output:
        "scores/belebele/{short}_acc.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        constraint=GPU_CONSTRAINT,
        slurm_extra=gres_gpu(2)
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES
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
        slurm_extra=gres_gpu(1)
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
        slurm_extra=gres_gpu(1)
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        src_lang=lambda wildcards: wildcards.lang,
        langs=ALL_LANGUAGES
    conda:
        "envs/comet.yaml"
    script:
        "comet/calc_comet.py"

rule translations_chrf:
    input:
        expand("translation/translations/{{short}}_{lang}.json", lang=ALL_LANGUAGES)
    resources:
        slurm_partition=CPU_PARTITION,
        mem_mb=8000,
        cpus_per_task=45,
        tasks=1
    output:
        "scores/translation_chrf/{short}.json"
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES
    priority:
        10
    conda:
        "envs/transformers.yaml"
    script:
        "translation/evaluate.py"

rule translations_vllm:
    output:
        "translation/translations/{short}_{lang}.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=16000,
        constraint=GPU_CONSTRAINT,
        slurm_extra=gres_gpu(2)
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        src_lang=lambda wildcards: wildcards.lang,
        langs=ALL_LANGUAGES
    conda:
        "envs/vllm.yaml"
    script:
        "translation/run_translate.py"

rule translations_mutinf_agregate:
    input:
        expand("translation/translation_mutinf/{{short}}-{lang}.json", lang = ALL_LANGUAGES)
    output:
        "scores/translation_mutinf/{short}.json"
    resources:
        slurm_partition=CPU_PARTITION,
        mem_mb=2000
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES
    priority:
        10
    script:
        "translation/aggregate_mutinf.py"

rule translations_mutinf:
    output:
        "translation/translation_mutinf/{short}-{lang}.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=16000,
        constraint=GPU_CONSTRAINT,
        slurm_extra=gres_gpu(2)
    params:
        model=lambda wildcards: MODELS[wildcards.short],
        src_lang=lambda wildcards: wildcards.lang,
        langs=ALL_LANGUAGES,
        batch_size=10
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
        slurm_extra=gres_gpu(1),
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
        slurm_extra=gres_gpu(2),
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
        slurm_extra=lambda wildcards: "" if wildcards.score in ["dist", "ratio", "nn-abs"] else gres_gpu(1),
        tasks=1,
        cpus_per_task=lambda wildcards: 10 if wildcards.score in ["dist", "ratio", "nn-abs"] else 2
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
    output:
        expand("embeds/flores/{{short}}-{{sent_rep}}-{lang}.pickle", lang=ALL_LANGUAGES)
    # noinspection PyUnresolvedReferences
    params:
        dataset="flores",
        model=lambda wildcards: MODELS[wildcards.short],
        langs=ALL_LANGUAGES,
        sent_rep=lambda wildcards: wildcards.sent_rep,
        batch_size=10,
        overwrite=False
    # noinspection PyUnresolvedReferences
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        slurm_extra=lambda wildcards: gres_gpu(2) if wildcards.sent_rep=="fewshot" else gres_gpu(2),
        constraint=GPU_CONSTRAINT
    conda:
        "envs/transformers.yaml"
    script:
        "save_embeds.py"

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