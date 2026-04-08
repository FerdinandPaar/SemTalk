#!/bin/bash
#$ -S /bin/bash
#$ -N fair_eval_AB
#$ -q mld.q@gridnode016
#$ -pe smp 2
#$ -cwd
#$ -o outputs/fair_eval_AB.log
#$ -e outputs/fair_eval_AB.err
#$ -j y

# ── Fair comparison: Experiment A (baseline-continued) vs B (contrast-margin) ──
# Both start from best_190.bin, train 40 epochs, 1 GPU each, same node.

source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd /home/ferpaa/SemTalk
mkdir -p outputs/rank_logs

CKPT=outputs/custom/0223_224354_semtalk_moclip_sparse_1gpu/best_190.bin

echo "=== Fair eval started: $(date) ==="
echo "=== Host: $(hostname) ==="
echo "=== GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',') ==="

# ── Experiment A: baseline continued (GPU 0) ──
echo "--- Starting Experiment A (baseline continued) on GPU 0 ---"
CUDA_VISIBLE_DEVICES=0 /home/ferpaa/miniconda3/envs/semtalk/bin/python train.py \
    --config configs/semtalk_moclip_baseline_continued.yaml \
    --ddp False \
    --load_ckpt "$CKPT" \
    --start_epoch 190 \
    2>&1 | tee outputs/fair_eval_A.log &
PID_A=$!

# ── Experiment B: contrastive margin (GPU 1) ──
echo "--- Starting Experiment B (contrastive margin) on GPU 1 ---"
CUDA_VISIBLE_DEVICES=1 /home/ferpaa/miniconda3/envs/semtalk/bin/python train.py \
    --config configs/semtalk_moclip_contrast_margin.yaml \
    --ddp False \
    --load_ckpt "$CKPT" \
    --start_epoch 190 \
    2>&1 | tee outputs/fair_eval_B.log &
PID_B=$!

echo "PID_A=$PID_A  PID_B=$PID_B"

# Wait for both
wait $PID_A
STATUS_A=$?
echo "--- Experiment A finished with status $STATUS_A ---"

wait $PID_B
STATUS_B=$?
echo "--- Experiment B finished with status $STATUS_B ---"

echo "=== Fair eval finished: $(date) ==="
echo "=== Exit codes: A=$STATUS_A  B=$STATUS_B ==="
