#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# NeurIPS — Ablation table: matched evaluation on full 265-seq test set
#
# Conditions:
#   (a) Vanilla SemTalk — no MoCLIP, no S-VIB, no physics
#   (b) +MoCLIP        — TMR semantic encoder added
#   (c) +S-VIB         — S-VIB gate, no physics smoothing  [NEEDS TRAINED CKPT]
#   (d) +S-VIB+Physics — full proposed system
#
# Submit with: qsub scripts/qsub_ablation_eval.sh
# ═══════════════════════════════════════════════════════════════════
#$ -S /bin/bash
#$ -N neurips_abl
#$ -q mld.q@gridnode016
#$ -pe smp 4
#$ -cwd
#$ -o /home/ferpaa/SemTalk/outputs/qsub_ablation_eval.log
#$ -e /home/ferpaa/SemTalk/outputs/qsub_ablation_eval.err
#$ -j y

set -euo pipefail
echo "=== Job started: $(date) ==="
echo "=== Host: $(hostname) ==="

cd /home/ferpaa/SemTalk
source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1
export NCCL_IB_DISABLE=1

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/neurips_ablation_${TS}"
mkdir -p "$LOG_DIR"

GPU=0

# ── Checkpoints ──
CKPT_A="weights/best_semtalk_sparse.bin"
CKPT_B="outputs/custom/0223_224354_semtalk_moclip_sparse_1gpu/best_190.bin"
# Condition (c): find latest S-VIB only checkpoint if available
CKPT_C=$(find outputs -path "*svib_only*" -name "best_*.bin" 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
CKPT_D="outputs/sweeps/20260310_180145_svib_phys_10runs_r05base_v1/safe_models/s01_base_r05/best_184.bin"

ALLSPK_MOCLIP_CONFIG="configs/semtalk_moclip_sparse_allspk_eval.yaml"

eval_condition() {
    local COND="$1"
    local CONFIG="$2"
    local CKPT="$3"
    local EXTRA_ARGS="${4:-}"
    local COND_LOG="${LOG_DIR}/ablation_${COND}.log"

    echo ""
    echo "═══ Condition ${COND} ═══"
    echo "  Config: $CONFIG"
    echo "  Checkpoint: $CKPT"

    if [[ ! -f "$CKPT" ]]; then
        echo "  WARNING: Checkpoint not found, skipping"
        return 1
    fi

    python train.py \
        --test_state \
        --config "$CONFIG" \
        --load_ckpt "$CKPT" \
        --ddp false \
        --gpus "$GPU" \
        $EXTRA_ARGS \
        > "$COND_LOG" 2>&1

    # Extract metrics
    local FGD=$(grep "fid score:" "$COND_LOG" | tail -1 | awk '{print $NF}')
    local BC=$(grep "align score:" "$COND_LOG" | tail -1 | awk '{print $NF}')
    local L1DIV=$(grep "l1div score:" "$COND_LOG" | tail -1 | awk '{print $NF}')
    echo "  FGD=${FGD:-N/A}  BC=${BC:-N/A}  L1div=${L1DIV:-N/A}"
}

echo "═══════════════════════════════════════════════════════════"
echo "  Ablation Table — Full 265-seq Test Set"
echo "═══════════════════════════════════════════════════════════"

# (a) Vanilla SemTalk
eval_condition "a_vanilla" \
    "configs/semtalk_sparse.yaml" \
    "$CKPT_A" \
    "--training_speakers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 --test_path datasets/beat2_semtalk_test_allspk.pkl"

# (b) +MoCLIP
eval_condition "b_moclip" \
    "$ALLSPK_MOCLIP_CONFIG" \
    "$CKPT_B"

# (c) +S-VIB (no physics)
if [[ -n "${CKPT_C:-}" ]]; then
    eval_condition "c_svib" \
        "$ALLSPK_MOCLIP_CONFIG" \
        "$CKPT_C" \
        "--phys_enabled false"
else
    echo ""
    echo "═══ Condition c_svib ═══"
    echo "  SKIPPED: No S-VIB only checkpoint found. Train with:"
    echo "  python train.py --config configs/semtalk_moclip_svib_only.yaml --ddp false --gpus 0"
fi

# (d) +S-VIB+Physics (full system)
eval_condition "d_svib_phys" \
    "$ALLSPK_MOCLIP_CONFIG" \
    "$CKPT_D"

# Summary table
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ABLATION RESULTS SUMMARY"
echo "═══════════════════════════════════════════════════════════"
printf "%-20s %10s %10s %10s\n" "Condition" "FGD↓" "BC↑" "L1div"
printf "%-20s %10s %10s %10s\n" "---" "---" "---" "---"
for cond in a_vanilla b_moclip c_svib d_svib_phys; do
    LOG_FILE="$LOG_DIR/ablation_${cond}.log"
    if [[ -f "$LOG_FILE" ]]; then
        FGD=$(grep "fid score:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        BC=$(grep "align score:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        L1DIV=$(grep "l1div score:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        printf "%-20s %10s %10s %10s\n" "$cond" "${FGD:-N/A}" "${BC:-N/A}" "${L1DIV:-N/A}"
    else
        printf "%-20s %10s %10s %10s\n" "$cond" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "=== Job finished: $(date) ==="
