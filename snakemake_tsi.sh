#!/bin/bash

set -e

python3 -m snakemake --executor slurm \
    --jobs 10 \
    --latency-wait 10 \
    --keep-incomplete \
    --use-conda \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --scheduler ilp --scheduler-ilp-solver PULP_CBC_CMD \
    --resources gpu_use=4 \
    --default-resources tasks=0 slurm_extra='-q low' nodes=1 runtime=1440  \
    --keep-going \
    --retries 2 \
    --slurm-requeue \
    --slurm-jobname-prefix "cla-metrics-tsi" \
    --slurm-no-account \
    --slurm-exclude-failed-nodes tdll-3gpu2 \
    tsi \
    --dry-run \
    --touch \
    --verbose \
