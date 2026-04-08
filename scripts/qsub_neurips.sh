#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# NeurIPS 2026 — Grid Engine submission script
# ═══════════════════════════════════════════════════════════════════
#
# Submit with:  qsub scripts/qsub_neurips.sh
#
# This runs the orchestrator on the compute node.
# Edit STEP below to run specific steps.
#
#$ -N neurips_eval
#$ -q mld.q@gridnode016
#$ -l gpu=1
#$ -pe smp 4
#$ -o /home/ferpaa/SemTalk/outputs/qsub_neurips.log
#$ -j y
#$ -cwd

set -euo pipefail

echo "=== Job started: $(date) ==="
echo "=== Host: $(hostname) ==="

cd /home/ferpaa/SemTalk

# Activate environment
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif command -v conda &>/dev/null; then
    conda activate semtalk
fi

echo "=== Python: $(which python) ==="
echo "=== GPU status ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null || true

# ── Choose what to run ──
# Options: gen_allspk_cache | eval_best | train_3seed | train_3seed_parallel
#          train_svib_only | ablation_eval | collect_3seed | all
STEP="${1:-all}"
GPU="${2:-0}"

./scripts/neurips_orchestrator.sh --step "$STEP" --gpu "$GPU"

echo "=== Job finished: $(date) ==="
