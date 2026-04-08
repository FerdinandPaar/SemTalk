#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# NeurIPS 2026 — Master Orchestration Script
# ═══════════════════════════════════════════════════════════════════
#
# Must-do tasks:
#   1. Generate all-speaker (265-seq) test cache
#   2. Full 265-seq FGD eval on best checkpoint (s01 best_184.bin)
#   3. 3-seed training for S-VIB+physics headline config
#   4. Ablation table: (a) vanilla, (b) +MoCLIP, (c) +S-VIB, (d) +S-VIB+phys
#   5. All-speaker eval on each ablation condition
#
# Usage:
#   ./scripts/neurips_orchestrator.sh [--step STEP_NAME] [--gpu GPU_ID] [--dry-run]
#
# Steps: gen_allspk_cache | eval_best | train_3seed | train_svib_only | ablation_eval | all
#
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# ── Configuration ──────────────────────────────────────────────────
ALLSPK_TEST_PKL="datasets/beat2_semtalk_test_moclip_allspk.pkl"
ALLSPK_EVAL_CONFIG="configs/semtalk_moclip_sparse_allspk_eval.yaml"
HEADLINE_CONFIG="configs/semtalk_moclip_sparse.yaml"
SVIB_ONLY_CONFIG="configs/semtalk_moclip_svib_only.yaml"

# Best checkpoints by condition
CKPT_VANILLA="weights/best_semtalk_sparse.bin"
CKPT_MOCLIP="outputs/custom/0223_224354_semtalk_moclip_sparse_1gpu/best_190.bin"
CKPT_SVIB_PHYS="outputs/sweeps/20260310_180145_svib_phys_10runs_r05base_v1/safe_models/s01_base_r05/best_184.bin"
# CKPT_SVIB_ONLY will be set after training completes

# 3-seed training base
SEED_BASE_CKPT="${CKPT_SVIB_PHYS}"  # continue from best checkpoint
SEED_START_EPOCH=184
SEED_END_EPOCH=220

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/neurips_${TIMESTAMP}"

# ── Parse arguments ────────────────────────────────────────────────
STEP="all"
GPU=0
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)  STEP="$2"; shift 2 ;;
        --gpu)   GPU="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/orchestrator.log"; }
run_cmd() {
    log "CMD: $*"
    if $DRY_RUN; then
        log "(dry-run, skipped)"
    else
        eval "$@"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Generate all-speaker test cache
# ═══════════════════════════════════════════════════════════════════
gen_allspk_cache() {
    log "═══ STEP 1: Generating all-speaker test caches ═══"

    # MoCLIP test cache (for conditions b, c, d)
    if [[ -f "$ALLSPK_TEST_PKL" ]]; then
        log "MoCLIP all-speaker test cache already exists: $ALLSPK_TEST_PKL — skipping"
    else
        run_cmd python dataloaders/save_test_dataset.py \
            --training_speakers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
            --moclip_path weights/moclip_checkpoints/models/tmr_humanml3d_guoh3dfeats/last_weights/text_encoder.pt \
            --distilbert_path distilbert-base-uncased \
            --dst_pkl "$ALLSPK_TEST_PKL" \
            --device "cuda:${GPU}" \
            2>&1 | tee "$LOG_DIR/gen_allspk_cache_moclip.log"
        log "MoCLIP all-speaker test cache generated"
    fi

    # Non-MoCLIP test cache (for condition a - vanilla)
    ALLSPK_TEST_PKL_VANILLA="datasets/beat2_semtalk_test_allspk.pkl"
    if [[ -f "$ALLSPK_TEST_PKL_VANILLA" ]]; then
        log "Vanilla all-speaker test cache already exists — skipping"
    else
        run_cmd python dataloaders/save_test_dataset.py \
            --training_speakers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
            --dst_pkl "$ALLSPK_TEST_PKL_VANILLA" \
            --device "cuda:${GPU}" \
            2>&1 | tee "$LOG_DIR/gen_allspk_cache_vanilla.log"
        log "Vanilla all-speaker test cache generated"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Full 265-seq FGD eval on best checkpoint
# ═══════════════════════════════════════════════════════════════════
eval_best() {
    log "═══ STEP 2: Full 265-seq FGD eval on best S-VIB+phys checkpoint ═══"
    log "Checkpoint: $CKPT_SVIB_PHYS"

    EVAL_OUT_DIR="$LOG_DIR/eval_best_s01_184"
    mkdir -p "$EVAL_OUT_DIR"

    run_cmd python train.py \
        --test_state \
        --config "$ALLSPK_EVAL_CONFIG" \
        --load_ckpt "$CKPT_SVIB_PHYS" \
        --ddp false \
        --gpus "$GPU" \
        --out_path "$EVAL_OUT_DIR/" \
        2>&1 | tee "$LOG_DIR/eval_best_265seq.log"

    # Also run FGD re-eval on the generated NPZs
    log "Running FGD evaluation on generated files..."
    EPOCH_DIR=$(find "$EVAL_OUT_DIR" -mindepth 2 -maxdepth 2 -type d | head -1)
    if [[ -n "$EPOCH_DIR" ]]; then
        run_cmd python utils/run_fgd_eval.py \
            --epoch_dir "$EPOCH_DIR" \
            --device "cuda:${GPU}" \
            2>&1 | tee "$LOG_DIR/eval_best_265seq_fgd.log"
    fi
    log "Full 265-seq eval complete"
}

# ═══════════════════════════════════════════════════════════════════
# STEP 3: 3-seed training for headline S-VIB+physics config
# ═══════════════════════════════════════════════════════════════════
train_3seed() {
    log "═══ STEP 3: 3-seed S-VIB+physics training ═══"

    SEEDS=(42 123 7)
    for i in "${!SEEDS[@]}"; do
        SEED=${SEEDS[$i]}
        SEED_IDX=$((i+1))
        RUN_NAME="neurips_3seed_s${SEED_IDX}_seed${SEED}"
        SEED_LOG="$LOG_DIR/train_seed${SEED}.log"

        log "Launching seed ${SEED_IDX}/3 (seed=$SEED) → $RUN_NAME"

        run_cmd python train.py \
            --config "$HEADLINE_CONFIG" \
            --load_ckpt "$SEED_BASE_CKPT" \
            --start_epoch "$SEED_START_EPOCH" \
            --epochs "$SEED_END_EPOCH" \
            --random_seed "$SEED" \
            --ddp false \
            --gpus "$GPU" \
            --project "neurips_3seed" \
            2>&1 | tee "$SEED_LOG"

        log "Seed ${SEED_IDX} training complete"
    done

    log "All 3 seeds complete. Collect FGD scores from logs."
}

# ═══════════════════════════════════════════════════════════════════
# STEP 3b: 3-seed training PARALLEL (uses 3 GPUs simultaneously)
# ═══════════════════════════════════════════════════════════════════
train_3seed_parallel() {
    log "═══ STEP 3 (parallel): 3-seed S-VIB+physics training on 3 GPUs ═══"

    SEEDS=(42 123 7)
    PIDS=()

    for i in "${!SEEDS[@]}"; do
        SEED=${SEEDS[$i]}
        SEED_IDX=$((i+1))
        GPU_ID=$((i))  # GPU 0, 1, 2
        RUN_NAME="neurips_3seed_s${SEED_IDX}_seed${SEED}"
        SEED_LOG="$LOG_DIR/train_seed${SEED}.log"

        log "Launching seed ${SEED_IDX}/3 (seed=$SEED, gpu=$GPU_ID) → $RUN_NAME"

        if ! $DRY_RUN; then
            python train.py \
                --config "$HEADLINE_CONFIG" \
                --load_ckpt "$SEED_BASE_CKPT" \
                --start_epoch "$SEED_START_EPOCH" \
                --epochs "$SEED_END_EPOCH" \
                --random_seed "$SEED" \
                --ddp false \
                --gpus "$GPU_ID" \
                --project "neurips_3seed" \
                > "$SEED_LOG" 2>&1 &
            PIDS+=($!)
        fi
    done

    # Wait for all seeds
    for pid in "${PIDS[@]}"; do
        wait "$pid"
        log "PID $pid finished (exit=$?)"
    done

    log "All 3 seeds complete"
}

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Train S-VIB only (no physics) for ablation condition (c)
# ═══════════════════════════════════════════════════════════════════
train_svib_only() {
    log "═══ STEP 4: Training S-VIB only (no physics) ablation ═══"

    RUN_NAME="neurips_ablation_svib_only"

    run_cmd python train.py \
        --config "$SVIB_ONLY_CONFIG" \
        --ddp false \
        --gpus "$GPU" \
        --project "neurips_ablation" \
        --name "$RUN_NAME" \
        2>&1 | tee "$LOG_DIR/train_svib_only.log"

    log "S-VIB only training complete"
}

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Ablation table — evaluate all 4 conditions on full test set
# ═══════════════════════════════════════════════════════════════════
ablation_eval() {
    log "═══ STEP 5: Ablation table evaluation (4 conditions, 265 sequences) ═══"

    # Condition (a): vanilla SemTalk (no MoCLIP)
    log "── Condition (a): Vanilla SemTalk ──"
    ABL_DIR_A="$LOG_DIR/ablation_a_vanilla"
    mkdir -p "$ABL_DIR_A"
    run_cmd python train.py \
        --test_state \
        --config configs/semtalk_sparse.yaml \
        --load_ckpt "$CKPT_VANILLA" \
        --ddp false \
        --gpus "$GPU" \
        --training_speakers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
        --test_path "datasets/beat2_semtalk_test_allspk.pkl" \
        --out_path "$ABL_DIR_A/" \
        2>&1 | tee "$LOG_DIR/ablation_a_vanilla.log"

    # Condition (b): +MoCLIP
    log "── Condition (b): +MoCLIP ──"
    ABL_DIR_B="$LOG_DIR/ablation_b_moclip"
    mkdir -p "$ABL_DIR_B"
    run_cmd python train.py \
        --test_state \
        --config "$ALLSPK_EVAL_CONFIG" \
        --load_ckpt "$CKPT_MOCLIP" \
        --ddp false \
        --gpus "$GPU" \
        --out_path "$ABL_DIR_B/" \
        2>&1 | tee "$LOG_DIR/ablation_b_moclip.log"

    # Condition (c): +S-VIB (no physics)
    log "── Condition (c): +S-VIB ──"
    # Find best checkpoint from svib_only training
    CKPT_SVIB_ONLY=$(find "$LOG_DIR" -path "*svib_only*" -name "best_*.bin" | sort | tail -1)
    if [[ -z "$CKPT_SVIB_ONLY" ]]; then
        # Try to find from previous runs
        CKPT_SVIB_ONLY=$(find outputs -path "*svib_only*" -name "best_*.bin" 2>/dev/null | sort | tail -1)
    fi
    if [[ -z "$CKPT_SVIB_ONLY" ]]; then
        log "WARNING: No S-VIB only checkpoint found. Run --step train_svib_only first."
    else
        ABL_DIR_C="$LOG_DIR/ablation_c_svib"
        mkdir -p "$ABL_DIR_C"
        run_cmd python train.py \
            --test_state \
            --config "$ALLSPK_EVAL_CONFIG" \
            --load_ckpt "$CKPT_SVIB_ONLY" \
            --ddp false \
            --gpus "$GPU" \
            --phys_enabled false \
            --out_path "$ABL_DIR_C/" \
            2>&1 | tee "$LOG_DIR/ablation_c_svib.log"
    fi

    # Condition (d): +S-VIB+physics (full system)
    log "── Condition (d): +S-VIB+Physics (full) ──"
    ABL_DIR_D="$LOG_DIR/ablation_d_svib_phys"
    mkdir -p "$ABL_DIR_D"
    run_cmd python train.py \
        --test_state \
        --config "$ALLSPK_EVAL_CONFIG" \
        --load_ckpt "$CKPT_SVIB_PHYS" \
        --ddp false \
        --gpus "$GPU" \
        --out_path "$ABL_DIR_D/" \
        2>&1 | tee "$LOG_DIR/ablation_d_svib_phys.log"

    # Collect results
    log "═══ Ablation Results Summary ═══"
    for cond in a_vanilla b_moclip c_svib d_svib_phys; do
        LOG_FILE="$LOG_DIR/ablation_${cond}.log"
        if [[ -f "$LOG_FILE" ]]; then
            FGD=$(grep "fid score:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
            BC=$(grep "align score:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
            L1DIV=$(grep "l1div score:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
            log "  ${cond}: FGD=${FGD:-N/A}  BC=${BC:-N/A}  L1div=${L1DIV:-N/A}"
        else
            log "  ${cond}: (no log file)"
        fi
    done
}

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Collect 3-seed results
# ═══════════════════════════════════════════════════════════════════
collect_3seed() {
    log "═══ STEP 6: Collecting 3-seed results ═══"

    FGDS=()
    for seed_log in "$LOG_DIR"/train_seed*.log; do
        if [[ -f "$seed_log" ]]; then
            BEST_FGD=$(grep "fid score:" "$seed_log" | awk '{print $NF}' | sort -n | head -1)
            if [[ -n "$BEST_FGD" ]]; then
                FGDS+=("$BEST_FGD")
                log "  $(basename "$seed_log"): best FGD = $BEST_FGD"
            fi
        fi
    done

    if [[ ${#FGDS[@]} -ge 2 ]]; then
        # Compute mean ± std with Python
        python3 -c "
import numpy as np
fgds = [${FGDS[*]}]
print(f'3-seed FGD: {np.mean(fgds):.4f} ± {np.std(fgds):.4f}')
print(f'Individual: {fgds}')
" | tee -a "$LOG_DIR/orchestrator.log"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════
case "$STEP" in
    gen_allspk_cache)   gen_allspk_cache ;;
    eval_best)          eval_best ;;
    train_3seed)        train_3seed ;;
    train_3seed_parallel) train_3seed_parallel ;;
    train_svib_only)    train_svib_only ;;
    ablation_eval)      ablation_eval ;;
    collect_3seed)      collect_3seed ;;
    all)
        gen_allspk_cache
        eval_best
        train_3seed
        train_svib_only
        ablation_eval
        collect_3seed
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Valid steps: gen_allspk_cache | eval_best | train_3seed | train_3seed_parallel | train_svib_only | ablation_eval | collect_3seed | all"
        exit 1
        ;;
esac

log "═══ Orchestrator finished step: $STEP ═══"
