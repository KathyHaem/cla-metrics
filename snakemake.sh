#!/bin/bash
#SBATCH --mem=4G

set -e

snakemake --executor slurm \
    --jobs 6 \
    --latency-wait 1 \
    --keep-incomplete \
    --use-conda \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --scheduler ilp --scheduler-ilp-solver PULP_CBC_CMD \
    --keep-going
    # --retries 2 \