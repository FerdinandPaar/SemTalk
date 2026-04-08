#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# NeurIPS — Step 1: Generate all-speaker test caches
# Submit with: qsub scripts/qsub_gen_allspk_cache.sh
# ═══════════════════════════════════════════════════════════════════
#$ -S /bin/bash
#$ -N gen_allspk
#$ -q mld.q@gridnode016
#$ -pe smp 2
#$ -cwd
#$ -o /home/ferpaa/SemTalk/outputs/qsub_gen_allspk_cache.log
#$ -e /home/ferpaa/SemTalk/outputs/qsub_gen_allspk_cache.err
#$ -j y

set -euo pipefail
echo "=== Job started: $(date) ==="
echo "=== Host: $(hostname) ==="

cd /home/ferpaa/SemTalk
source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1
export NCCL_IB_DISABLE=1
export MASTER_PORT=8681

echo "=== Python: $(which python) ==="

# ── 1. MoCLIP all-speaker test cache ──
if [[ -f datasets/beat2_semtalk_test_moclip_allspk.pkl ]]; then
    echo "[Skip] MoCLIP allspk test cache exists"
else
    echo "[Gen] MoCLIP allspk test cache..."
    python dataloaders/save_test_dataset.py \
        --training_speakers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
        --moclip_path weights/moclip_checkpoints/models/tmr_humanml3d_guoh3dfeats/last_weights/text_encoder.pt \
        --distilbert_path distilbert-base-uncased \
        --dst_pkl datasets/beat2_semtalk_test_moclip_allspk.pkl \
        --device cuda:0
    echo "[Done] MoCLIP allspk test cache"
fi

# ── 2. Vanilla (no MoCLIP) all-speaker test cache ──
if [[ -f datasets/beat2_semtalk_test_allspk.pkl ]]; then
    echo "[Skip] Vanilla allspk test cache exists"
else
    echo "[Gen] Vanilla allspk test cache..."
    python dataloaders/save_test_dataset.py \
        --training_speakers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
        --dst_pkl datasets/beat2_semtalk_test_allspk.pkl \
        --device cuda:0
    echo "[Done] Vanilla allspk test cache"
fi

# ── 3. Full 265-seq FGD eval on best S-VIB+physics checkpoint ──
BEST_CKPT="outputs/sweeps/20260310_180145_svib_phys_10runs_r05base_v1/safe_models/s01_base_r05/best_184.bin"
EVAL_CONFIG="configs/semtalk_moclip_sparse_allspk_eval.yaml"

echo "=== Running full 265-seq FGD eval ==="
echo "=== Checkpoint: $BEST_CKPT ==="

python train.py \
    --test_state \
    --config "$EVAL_CONFIG" \
    --load_ckpt "$BEST_CKPT" \
    --ddp false \
    --gpus 0

echo "=== Full 265-seq FGD eval complete ==="
echo "=== Job finished: $(date) ==="
