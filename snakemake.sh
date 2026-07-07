#!/bin/bash
#SBATCH --mem=4G

set -e

python3 -m snakemake --executor slurm \
    --jobs 40 \
    --latency-wait 1 \
    --keep-incomplete \
    --use-conda \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --scheduler ilp --scheduler-ilp-solver PULP_CBC_CMD \
    --resources gpu_use=24 \
    --default-resources tasks=0 slurm_extra='-q low' nodes=1 runtime=1080  \
    --slurm-jobname-prefix "cla-metrics" \
    --keep-going \
    --retries 2 \
    --slurm-requeue \
    --slurm-no-account \
    --slurm-exclude-failed-nodes tdll-3gpu2,tdll-8gpu2 \
    # --touch \
    # --verbose \
    # --default-resources tasks=0 slurm_extra='-q low --exclude tdll-3gpu3' max_nodes=1 \
