#!/bin/bash
#$ -S /bin/bash
#$ -N gate_abl_C
#$ -q mld.q@gridnode016
#$ -pe smp 4
#$ -cwd
#$ -o outputs/gate_abl_C.log
#$ -e outputs/gate_abl_C.err
#$ -j y

# ═══════════════════════════════════════════════════════════════════════
# Experiment C: Gate-complementary physics
#
# Identical to gate_abl_A EXCEPT:
#   phys_enabled: true
#   phys_gate_grad: true   ← (1-ψ) per-frame weight, gradient through ψ
#   phys_upper_only: true  ← upper body only (best from prior ablation)
#
# HYPOTHESIS: weight = (1-ψ) → jerk penalty strong on beat frames, weak on
# semantic frames → gate learns to lower ψ where output would be jerky.
# Direct comparison: gate_abl_A achieves FGD 0.4056; C may improve further
# AND produce a more differentiated ψ distribution (not all 0.99).
# ═══════════════════════════════════════════════════════════════════════

set -u

source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/ferpaa/SemTalk
mkdir -p outputs/rank_logs

CKPT="outputs/custom/0311_102115_s01_base_r05_r05base_s01_base_r05_b0.010_fb0.30_pl0.08_tb0.50_tf0.10/best_184.bin"
PYTHON=/home/ferpaa/miniconda3/envs/semtalk/bin/python

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Experiment C (gate-complementary physics) started: $(date)"
echo "  Host: $(hostname)"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
echo "═══════════════════════════════════════════════════════════════"

MASTER_PORT_C=$((29600 + (JOB_ID % 500)))
$PYTHON -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT_C \
    train.py \
    --config configs/gate_abl_C_gate_phys.yaml \
    --ddp True \
    --load_ckpt "$CKPT" \
    --start_epoch 184 \
    2>&1 | tee outputs/gate_abl_C_run.log
STATUS_C=${PIPESTATUS[0]}

echo "═══════════════════════════════════════════════════════════════"
echo "  Experiment C finished: $(date)"
echo "  Exit code: C=$STATUS_C"
echo "═══════════════════════════════════════════════════════════════"
