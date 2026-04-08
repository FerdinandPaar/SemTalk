#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# NeurIPS — Train S-VIB only (no physics) for ablation condition (c)
# Submit with: qsub scripts/qsub_train_svib_only.sh
# ═══════════════════════════════════════════════════════════════════
#$ -S /bin/bash
#$ -N svib_only
#$ -q mld.q@gridnode016
#$ -pe smp 4
#$ -cwd
#$ -o /home/ferpaa/SemTalk/outputs/qsub_train_svib_only.log
#$ -e /home/ferpaa/SemTalk/outputs/qsub_train_svib_only.err
#$ -j y

set -euo pipefail
echo "=== Job started: $(date) ==="
echo "=== Host: $(hostname) ==="

cd /home/ferpaa/SemTalk
source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1
export NCCL_IB_DISABLE=1

echo "=== Training S-VIB only (phys_enabled=false) ==="

python train.py \
    --config configs/semtalk_moclip_svib_only.yaml \
    --ddp false \
    --gpus 0 \
    --project "neurips_ablation" \
    --stat ts

echo "=== Job finished: $(date) ==="
