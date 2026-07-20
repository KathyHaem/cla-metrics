#!/bin/bash
#SBATCH --mem=4G

set -e

python3 -m snakemake --executor slurm \
    --jobs 400 \
    --latency-wait 10 \
    --keep-incomplete \
    --use-conda \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --scheduler ilp --scheduler-ilp-solver PULP_CBC_CMD \
    --resources gpu_use=24 \
    --default-resources tasks=0 slurm_extra='-q low' nodes=1 runtime=1080  \
    --keep-going \
    --retries 2 \
    --slurm-requeue \
    --slurm-jobname-prefix "cla-metrics" \
    --slurm-no-account \
    --slurm-exclude-failed-nodes tdll-3gpu2 \
    # --touch \
    # --verbose \
    # --default-resources tasks=0 slurm_extra='-q low --exclude tdll-3gpu3' max_nodes=1 \
