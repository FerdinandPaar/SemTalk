#!/bin/bash
#$ -S /bin/bash
#$ -N gate_ablation_AB
#$ -q mld.q@gridnode016
#$ -pe smp 4
#$ -cwd
#$ -o outputs/gate_ablation_AB.log
#$ -e outputs/gate_ablation_AB.err
#$ -j y

# ═══════════════════════════════════════════════════════════════════════
# Clean gate ablation: A (no VIB) vs B (S-VIB) — 4 GPU DDP, sequential.
# The ONLY difference between configs is vib_enabled.
# Both start from s01_base_r05 best_184.bin, 80 epochs, same trainer.
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
echo "  Gate ablation started: $(date)"
echo "  Host: $(hostname)"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
echo "═══════════════════════════════════════════════════════════════"

# ── Experiment A: NO VIB (linear gate only) — all 4 GPUs ──
echo "--- Starting A (no VIB) on 4 GPUs ---"
MASTER_PORT_A=$((28600 + (JOB_ID % 500)))
$PYTHON -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT_A \
    train.py \
    --config configs/gate_abl_A_no_vib.yaml \
    --ddp True \
    --load_ckpt "$CKPT" \
    --start_epoch 184 \
    2>&1 | tee outputs/gate_abl_A.log
STATUS_A=${PIPESTATUS[0]}
echo "--- Experiment A finished: status=$STATUS_A at $(date) ---"

# ── Experiment B: WITH VIB (S-VIB gate) — all 4 GPUs ──
echo "--- Starting B (S-VIB) on 4 GPUs ---"
MASTER_PORT_B=$((29100 + (JOB_ID % 500)))
$PYTHON -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT_B \
    train.py \
    --config configs/gate_abl_B_with_vib.yaml \
    --ddp True \
    --load_ckpt "$CKPT" \
    --start_epoch 184 \
    2>&1 | tee outputs/gate_abl_B.log
STATUS_B=${PIPESTATUS[0]}
echo "--- Experiment B finished: status=$STATUS_B at $(date) ---"

echo "═══════════════════════════════════════════════════════════════"
echo "  Gate ablation finished: $(date)"
echo "  Exit codes: A=$STATUS_A  B=$STATUS_B"
echo "═══════════════════════════════════════════════════════════════"
