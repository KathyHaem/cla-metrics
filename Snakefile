# Snakefile

# Run with snakemake==9.3.0 snakemake-executor-plugin-slurm==0.14.2

from constants import ALL_LANGUAGES

CPU_PARTITION="cpu-troja"
GPU_PARTITION="gpu-troja,gpu-ms"
GPU_CONSTRAINT="gpuram40G|gpuram48G"

def gres_gpu(num):
    return f"--gres 'gpu:{num}'"

FULL_MODELS = [
    # "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen3-14B"
    # "CohereLabs/aya-expanse-8b",
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
        expand("scores/comet/{short}.json",
               short=MODELS.keys())

rule belebele:
    output:
        "scores/sib-200/{short}_loglik.json",
        "scores/sib-200/{short}_mutinf.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        constraint=GPU_CONSTRAINT,
        slurm_extra=gres_gpu(2)
    params:
        model=lambda wc: MODELS[wc.short],
        langs=ALL_LANGUAGES,
        batch_size=10
    conda:
        "envs/transformers.yaml"
    script:
        "belebele/run.py"

rule sib_200:
    output:
        "scores/sib-200/{short}.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        constraint=GPU_CONSTRAINT,
        slurm_extra=gres_gpu(1)
    params:
        model=lambda wc: MODELS[wc.short],
        langs=ALL_LANGUAGES
    conda:
        "envs/vllm.yaml"
    script:
        "sib-200/vllm_probing.py"


rule translations_comet:
    input:
        expand("translation/translations/{{short}}_{lang}.json", lang=ALL_LANGUAGES)
    output:
        "scores/comet/{short}.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=40000,
        constraint=GPU_CONSTRAINT,
        slurm_extra=gres_gpu(1)
    params:
        model=lambda wc: MODELS[wc.short],
        langs=ALL_LANGUAGES,
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
        model=lambda wc: MODELS[wc.short],
        langs=ALL_LANGUAGES,
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
        slurm_extra="--gres 'gpu:2'"
    params:
        model=lambda wc: MODELS[wc.short],
        src_lang=lambda wc: wc.lang,
        langs=ALL_LANGUAGES,
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
        model=lambda wc: MODELS[wc.short],
        langs=ALL_LANGUAGES
    script:
        "translation/aggregate_mutinf.py"

rule translations_mutinf:
    output:
        "translation/translation_mutinf/{short}-{lang}.json"
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=16000,
        constraint=GPU_CONSTRAINT,
        slurm_extra="--gres 'gpu:2'"
    params:
        model=lambda wc: MODELS[wc.short],
        src_lang=lambda wc: wc.lang,
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
        model=lambda wc: MODELS[wc.short],
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=16000,
        slurm_extra="--gres 'gpu:1'",
        constraint=GPU_CONSTRAINT,
    conda:
        "envs/transformers.yaml"
    script:
        "dali_belebele/run_dali.py"

rule save_embeds_dali:
    output:
        "embeds/dali-belebele/{short}-{lang}.pt"
    params:
        model=lambda wc: MODELS[wc.short],
        batch_size=10,
        lang=lambda wc: wc.lang
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        slurm_extra="--gres 'gpu:2'",
        constraint=GPU_CONSTRAINT,
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
        slurm_partition=lambda wc: CPU_PARTITION if wc.score in ["dist", "ratio", "nn-abs"] else GPU_PARTITION,
        mem_mb=16000,
        constraint=lambda wc: "" if wc.score in ["dist", "ratio", "nn-abs"] else GPU_CONSTRAINT,
        slurm_extra=lambda wc: "" if wc.score in ["dist", "ratio", "nn-abs"] else "--gres 'gpu:1'",
        tasks=1,
        cpus_per_task=lambda wc: 10 if wc.score in ["dist", "ratio", "nn-abs"] else 2
    threads:
        10
    params:
        dataset="flores",
        model=lambda wc: MODELS[wc.short],
        langs=ALL_LANGUAGES,
        sent_rep=lambda wc: wc.sent_rep,
        overwrite=False,
        score=lambda wc: [wc.score]
    conda:
        "envs/transformers.yaml"
    script:
        "calc_scores.py"

rule save_embeds:
    output:
        expand("embeds/flores/{{short}}-{{sent_rep}}-{lang}.pickle", lang=ALL_LANGUAGES)
    params:
        dataset="flores",
        model=lambda wc: MODELS[wc.short],
        langs=ALL_LANGUAGES,
        sent_rep=lambda wc: wc.sent_rep,
        batch_size=10,
        overwrite=False
    resources:
        slurm_partition=GPU_PARTITION,
        mem_mb=20000,
        slurm_extra=lambda wc: "--gres 'gpu:3'" if wc.sent_rep=="fewshot" else "--gres 'gpu:2'",
        constraint=GPU_CONSTRAINT,
    conda:
        "envs/transformers.yaml"
    script:
        "save_embeds.py"