#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# NeurIPS — 3-seed S-VIB+physics training (parallel on 3 GPUs)
# Submit with: qsub scripts/qsub_3seed_training.sh
# ═══════════════════════════════════════════════════════════════════
#$ -S /bin/bash
#$ -N neurips_3seed
#$ -q mld.q@gridnode016
#$ -pe smp 3
#$ -cwd
#$ -o /home/ferpaa/SemTalk/outputs/qsub_3seed_training.log
#$ -e /home/ferpaa/SemTalk/outputs/qsub_3seed_training.err
#$ -j y

set -euo pipefail
echo "=== Job started: $(date) ==="
echo "=== Host: $(hostname) ==="

cd /home/ferpaa/SemTalk
source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1
export NCCL_IB_DISABLE=1

CONFIG="configs/semtalk_moclip_sparse.yaml"
BASE_CKPT="outputs/sweeps/20260310_180145_svib_phys_10runs_r05base_v1/safe_models/s01_base_r05/best_184.bin"
START_EPOCH=184
END_EPOCH=220
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/neurips_3seed_${TS}"
mkdir -p "$LOG_DIR"

SEEDS=(42 123 7)
PIDS=()
BASE_PORT=8682

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU_ID=$((i))
    PORT=$((BASE_PORT + i))
    SEED_LOG="${LOG_DIR}/seed_${SEED}_gpu${GPU_ID}.log"

    echo "[LAUNCH] seed=$SEED gpu=$GPU_ID port=$PORT log=$SEED_LOG"

    CUDA_VISIBLE_DEVICES=$GPU_ID MASTER_PORT=$PORT python train.py \
        --config "$CONFIG" \
        --load_ckpt "$BASE_CKPT" \
        --start_epoch "$START_EPOCH" \
        --epochs "$END_EPOCH" \
        --random_seed "$SEED" \
        --ddp false \
        --gpus 0 \
        --project "neurips_3seed" \
        > "$SEED_LOG" 2>&1 &
    PIDS+=($!)
done

echo "[INFO] Waiting for all seeds to finish..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "[DONE] PID=$pid exit=$?"
done

# Collect results
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  3-Seed Training Results"
echo "═══════════════════════════════════════════════════════════"
for seed_log in "$LOG_DIR"/seed_*.log; do
    SEED=$(basename "$seed_log" | grep -oP 'seed_\K\d+')
    BEST_FGD=$(grep "fid score:" "$seed_log" | awk '{print $NF}' | sort -n | head -1)
    BEST_BC=$(grep "align score:" "$seed_log" | tail -1 | awk '{print $NF}')
    echo "  seed=$SEED: best FGD=$BEST_FGD  BC=$BEST_BC"
done

# Compute mean ± std
python3 -c "
import sys, re
fgds, bcs = [], []
for f in sorted([f for f in sys.argv[1:] if f.endswith('.log')]):
    with open(f) as fh:
        txt = fh.read()
    fid_vals = [float(m) for m in re.findall(r'fid score: ([\d.]+)', txt)]
    bc_vals = [float(m) for m in re.findall(r'align score: ([\d.]+)', txt)]
    if fid_vals:
        fgds.append(min(fid_vals))
    if bc_vals:
        bcs.append(bc_vals[-1])
import numpy as np
if fgds:
    print(f'FGD: {np.mean(fgds):.4f} ± {np.std(fgds):.4f}  (n={len(fgds)})')
if bcs:
    print(f'BC:  {np.mean(bcs):.4f} ± {np.std(bcs):.4f}  (n={len(bcs)})')
" "$LOG_DIR"/seed_*.log

echo "=== Job finished: $(date) ==="
