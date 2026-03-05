#!/bin/bash
#$ -S /bin/bash
#$ -N semtalk_svib_phys
#$ -q mld.q@gridnode016
#$ -pe smp 4
#$ -cwd
#$ -o outputs/qsub_train.log
#$ -e outputs/qsub_train.err
#$ -j y

# ── environment ───────────────────────────────────────────────────────
source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1          # block ~/.local from polluting sys.path
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd /home/ferpaa/SemTalk

mkdir -p outputs/rank_logs

echo "=== Job started: $(date) ==="
echo "=== Host: $(hostname) ==="
echo "=== GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',') ==="

/home/ferpaa/miniconda3/envs/semtalk/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29502 \
    train.py \
    --config configs/semtalk_moclip_sparse.yaml \
    --ddp True \
    --load_ckpt outputs/custom/0304_230546_semtalk_moclip_sparse_4gpu/best_0.bin \
    --start_epoch 1

echo "=== Job finished: $(date) ==="
