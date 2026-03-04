#!/bin/bash
# ============================================================
# MoCLIP SemTalk training launcher — run ON gridnode016
#
#   ssh gridnode016
#   bash run_train_moclip_4gpu.sh
#
# Automatically re-launches itself inside a tmux session so
# training survives SSH disconnects.  Re-attach any time with:
#   tmux attach -t semtalk_train
# ============================================================

set -e
cd /home/ferpaa/SemTalk

# ── tmux guard: if not already inside tmux, re-launch there ──────────
SESSION="semtalk_train"
if [ -z "$TMUX" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session '$SESSION' already exists."
        echo "Re-attaching — use Ctrl-B D to detach without killing it."
        tmux attach -t "$SESSION"
    else
        echo "Launching inside tmux session '$SESSION'..."
        echo "Re-attach any time with:  tmux attach -t $SESSION"
        tmux new-session -s "$SESSION" "bash $(realpath "$0"); exec bash"
    fi
    exit 0
fi
# ── Everything below runs inside tmux ────────────────────────────────

source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

# --------------- GPU inventory ----------------------------------------
echo ""
echo "=== GPUs available ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo ""

# --------------- Mode selection ---------------------------------------
echo "How do you want to train?"
echo "  1) 1 GPU  — full WandB loss logging  (slower, proven stable)"
echo "  2) 4 GPUs — DDP, per-rank log files  (faster, logs in rank_*.log)"
echo ""
read -rp "Enter 1 or 2 [default 1]: " MODE_CHOICE
MODE_CHOICE=${MODE_CHOICE:-1}

if [[ "$MODE_CHOICE" != "1" && "$MODE_CHOICE" != "2" ]]; then
    echo "Invalid choice. Defaulting to 1 GPU."
    MODE_CHOICE=1
fi

# --------------- Checkpoint resume -----------------------------------
# Set RESUME_CKPT=path/to/ckpt.bin and START_EPOCH=N to resume from a checkpoint.
# Leave unset (or set to "none") to start a fresh run.
RESUME_CKPT=${RESUME_CKPT:-none}
START_EPOCH=${START_EPOCH:-0}

LOAD_CKPT_ARG=""
START_EPOCH_ARG=""
EFFECTIVE_START_EPOCH=0
if [ -n "$RESUME_CKPT" ] && [ "$RESUME_CKPT" != "none" ] && [ -f "$RESUME_CKPT" ]; then
    echo "Resuming from : $RESUME_CKPT  (start_epoch=$START_EPOCH)"
    LOAD_CKPT_ARG="--load_ckpt $RESUME_CKPT"
    START_EPOCH_ARG="--start_epoch $START_EPOCH"
    EFFECTIVE_START_EPOCH=$START_EPOCH
else
    echo "Starting fresh (checkpoint '$RESUME_CKPT' not found or not set)"
fi

# --------------- NCCL (single-node only) ------------------------------
export MASTER_PORT=${MASTER_PORT:-29503}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
# Disable InfiniBand so NCCL uses NVLink/SHM only
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# Propagate NCCL errors as Python exceptions (instead of SIGABRT)
# so non-rank-0 tracebacks appear in their per-rank log files.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# Fail fast: 2-minute NCCL timeout instead of default 10 minutes
export NCCL_TIMEOUT=600   # test() takes ~150s; 600s gives 10x headroom
# Note: CUDA_LAUNCH_BLOCKING=1 is set per-mode below (only 4-GPU needs it)

# =====================================================================
if [ "$MODE_CHOICE" = "1" ]; then
# ----------------------- 1-GPU mode (with WandB logs) ----------------
    echo ""
    echo "=== 1-GPU mode (GPU 0) — WandB logs enabled ==="
    echo "  batch_size   : 64"
    echo "  resume from  : ${RESUME_CKPT:-none}"
    echo ""

    CUDA_LAUNCH_BLOCKING=0 \
    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
        --config configs/semtalk_moclip_sparse.yaml \
        --ddp False \
        --gpus 0 \
        --notes "_1gpu" \
        $LOAD_CKPT_ARG \
        $START_EPOCH_ARG

else
# ----------------------- 4-GPU DDP mode ------------------------------
    GPU_IDS=$(seq 0 $(( NUM_GPUS - 1 )) | tr '\n' ' ')
    CUDA_MAP=$(seq 0 $(( NUM_GPUS - 1 )) | tr '\n' ',' | sed 's/,$//')

    echo ""
    echo "=== 4-GPU torchrun mode === per-rank logs → outputs/rank_logs/rank_*.log ==="
    echo "  GPUs             : $CUDA_MAP"
    echo "  master_addr:port : $MASTER_ADDR:$MASTER_PORT"
    echo "  per-GPU batch    : 64  (effective total: $(( 64 * NUM_GPUS )))"
    echo "  resume from      : ${RESUME_CKPT:-none}"
  echo "  start_epoch      : ${EFFECTIVE_START_EPOCH}"
    echo ""
    echo "  Tail logs with:  tail -f outputs/rank_logs/rank_1.log"
    echo ""

    export RANK_LOG_DIR="$(pwd)/outputs/rank_logs"
    mkdir -p "$RANK_LOG_DIR"

    # torchrun (python -m torch.distributed.run) creates one subprocess per rank.
    # Non-rank-0 Python stdout+stderr is redirected to rank_N.log by train.py itself.
    CUDA_LAUNCH_BLOCKING=1 \
    CUDA_VISIBLE_DEVICES=$CUDA_MAP \
    /home/ferpaa/miniconda3/envs/semtalk/bin/python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py \
        --config configs/semtalk_moclip_sparse.yaml \
        --ddp True \
        --gpus $GPU_IDS \
        --notes "_4gpu" \
        $LOAD_CKPT_ARG \
        $START_EPOCH_ARG
fi

echo ""
echo "=== Training complete ==="
