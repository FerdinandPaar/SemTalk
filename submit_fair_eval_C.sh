#!/bin/bash
#$ -S /bin/bash
#$ -N fair_eval_C
#$ -q mld.q@gridnode016
#$ -pe smp 1
#$ -cwd
#$ -o outputs/fair_eval_C.log
#$ -e outputs/fair_eval_C.err
#$ -j y

# ── Experiment C: S-VIB + Physics with tuned hyperparameters ──
# beta_target: 0.001→0.5 (KL now 8% of loss), lambda: 0.01→1.0 (jerk 6% of loss)
# gate_class_weight: 9.0 (fixes 10:1 semantic imbalance)
# Same starting point as A and B: best_190.bin, 40 epochs

source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1
export NCCL_IB_DISABLE=1

cd /home/ferpaa/SemTalk
mkdir -p outputs/rank_logs

echo "=== Experiment C started: $(date) ==="
echo "=== Host: $(hostname) ==="

CUDA_VISIBLE_DEVICES=2 /home/ferpaa/miniconda3/envs/semtalk/bin/python train.py \
    --config configs/semtalk_moclip_svib_phys_tuned.yaml \
    --ddp False \
    --load_ckpt outputs/custom/0223_224354_semtalk_moclip_sparse_1gpu/best_190.bin \
    --start_epoch 190

echo "=== Experiment C finished: $(date) ==="
