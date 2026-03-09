#!/bin/bash
# ============================================================
# SemTalk P2+P3 Fine-Tune Launcher — run ON gridnode016
#
# Fine-tunes best_116.bin (S-VIB + physics) with three fixes:
#   P1: sqrt τ formula + τ_floor (already applied at inference, now active in training)
#   P2: pose-level physics jerk loss (decoded rot6d) — phys_lambda=0.08
#   P3: stronger S-VIB β=0.010, free_bits=0.20 — harder gate commitment
#
# Config: configs/semtalk_moclip_sparse_ft.yaml
#   load_ckpt:   ./outputs/custom/0305_073649_semtalk_moclip_sparse/best_116.bin
#   start_epoch: 116
#   epochs:      220  (→ 104 fine-tune epochs)
#   lr_base:     5e-5 (half of original 1e-4)
#
# Usage (on gridnode016):
#   bash run_finetune_moclip_4gpu.sh
#
# Re-attach:
#   screen -r semtalk_ft
# ============================================================

set -e
cd /home/ferpaa/SemTalk

SESSION="semtalk_ft"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# ── persistence via screen (prefer system screen; avoid non-exec conda binaries)
SCREEN_BIN=""
if [ -x "/usr/bin/screen" ]; then
    SCREEN_BIN="/usr/bin/screen"
elif command -v screen &>/dev/null; then
    # ensure the discovered binary is executable
    CANDIDATE=$(command -v screen)
    if [ -x "$CANDIDATE" ]; then
        SCREEN_BIN="$CANDIDATE"
    else
        echo "Note: discovered screen at $CANDIDATE but it's not executable — skipping screen usage."
    fi
fi

if [ -n "$SCREEN_BIN" ]; then
    if [ -z "$STY" ]; then
        if $SCREEN_BIN -ls "$SESSION" 2>/dev/null | grep -q "$SESSION"; then
            echo "screen session '$SESSION' already exists — re-attaching."
            $SCREEN_BIN -r "$SESSION"
        else
            echo "Launching fine-tune inside screen session '$SESSION'..."
            echo "Re-attach any time with:  $SCREEN_BIN -r $SESSION"
            $SCREEN_BIN -S "$SESSION" bash "$SCRIPT_PATH"
        fi
        exit 0
    fi
    echo "Running inside screen session: $STY"
else
    echo "WARNING: screen not available or not executable. Training will run with nohup and may die on SSH disconnect."
fi

source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

# --------------- GPU inventory ----------------------------------------
echo ""
echo "=== GPUs available ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo ""

# --------------- Mode selection ---------------------------------------
echo "Fine-tune mode:"
echo "  1) 1 GPU  — full WandB logging  (slower)"
echo "  2) 4 GPUs — DDP, per-rank logs  (faster, recommended)"
echo ""
read -rp "Enter 1 or 2 [default 2]: " MODE_CHOICE
MODE_CHOICE=${MODE_CHOICE:-2}

# --------------- NCCL -------------------------------------------------
export MASTER_PORT=${MASTER_PORT:-29505}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=600

# =====================================================================
if [ "$MODE_CHOICE" = "1" ]; then
# ----------------------- 1-GPU ---------------------------------------
    echo ""
    echo "=== 1-GPU fine-tune (GPU 0) ==="

    CUDA_LAUNCH_BLOCKING=0 \
    CUDA_VISIBLE_DEVICES=0 \
    nohup /home/ferpaa/miniconda3/envs/semtalk/bin/python train.py \
        --config configs/semtalk_moclip_sparse_ft.yaml \
        --ddp False \
        --gpus 0 \
        --notes "_ft_1gpu" \
        > outputs/finetune_nohup.log 2>&1 &
    echo "Fine-tune PID: $! — tail -f outputs/finetune_nohup.log"

else
# ----------------------- 4-GPU DDP -----------------------------------
    CUDA_MAP=$(seq 0 $(( NUM_GPUS - 1 )) | tr '\n' ',' | sed 's/,$//')
    GPU_IDS=$(seq 0 $(( NUM_GPUS - 1 )) | tr '\n' ' ')

    echo ""
    echo "=== 4-GPU torchrun fine-tune === logs → outputs/rank_logs/rank_*.log ==="
    echo "  GPUs             : $CUDA_MAP"
    echo "  master_addr:port : $MASTER_ADDR:$MASTER_PORT"
    echo "  config           : configs/semtalk_moclip_sparse_ft.yaml"
    echo "  checkpoint       : outputs/custom/0305_073649_semtalk_moclip_sparse/best_116.bin"
    echo "  start_epoch      : 116   epochs → 220  (104 fine-tune epochs)"
    echo ""

    export RANK_LOG_DIR="$(pwd)/outputs/rank_logs"
    mkdir -p "$RANK_LOG_DIR"

    nohup \
    env CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$CUDA_MAP \
    /home/ferpaa/miniconda3/envs/semtalk/bin/python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py \
        --config configs/semtalk_moclip_sparse_ft.yaml \
        --ddp True \
        --gpus $GPU_IDS \
        --notes "_ft_4gpu" \
        > outputs/finetune_nohup.log 2>&1 &
    echo "Fine-tune PID: $! — tail -f outputs/finetune_nohup.log"
fi

echo ""
echo "=== Fine-tune launched in background (nohup) ==="
echo "Monitor:  tail -f outputs/finetune_nohup.log"
echo "Kill all: pkill -f 'train.py.*sparse_ft'"
