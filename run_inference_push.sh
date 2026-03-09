#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Inference + GitHub push for the S-VIB + physics fine-tuned checkpoint.
#
# Generates:  demo/2_scott_0_1_1_test_moclip_svib_phy_<YYYYMMDD_HHMMSS>.npz
# Commits it to branch feat/svib-bottleneck and pushes to origin.
#
# Usage (run on gridnode016):
#   bash run_inference_push.sh
#   bash run_inference_push.sh best_124    # pick a different checkpoint
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /home/ferpaa/SemTalk

# ── conda env ─────────────────────────────────────────────────────────────────
source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

PYTHON=/home/ferpaa/miniconda3/envs/semtalk/bin/python
# torchrun is not executable on gridnode016 (permission issue); use python -m instead
GIT=/usr/bin/git

# ── checkpoint selection ───────────────────────────────────────────────────────
CKPT_DIR="outputs/custom/0308_220319_semtalk_moclip_sparse_ft_ft_4gpu"
# Default: best_126.bin (highest-FGD checkpoint of the fine-tune run).
# Override by passing first arg, e.g.:  bash run_inference_push.sh best_124
CKPT_STEM="${1:-best_126}"
CKPT="${CKPT_DIR}/${CKPT_STEM}.bin"

if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    echo "        Available: $(ls ${CKPT_DIR}/best_*.bin)"
    exit 1
fi

# ── paths ─────────────────────────────────────────────────────────────────────
CONFIG="configs/semtalk_moclip_sparse_ft.yaml"
AUDIO="demo/2_scott_0_1_1_test.wav"
OUT_STEM="demo/2_scott_0_1_1_test_moclip_svib_phy"   # timestamp appended by trainer

echo "================================================================"
echo "  Checkpoint : $CKPT"
echo "  Audio      : $AUDIO"
echo "  Output stem: ${OUT_STEM}_<timestamp>.npz"
echo "================================================================"

# ── run inference ─────────────────────────────────────────────────────────────
# Use python -m torch.distributed.run (torchrun is not executable on this node).
# Master port 29600 avoids collisions with any residual training processes.
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29600 \
$PYTHON -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=29600 \
    train.py \
        --config "$CONFIG" \
        --inference \
        --load_ckpt "$CKPT" \
        --audio_infer_path "$AUDIO" \
        --out_name "$OUT_STEM"

# ── find the generated file ───────────────────────────────────────────────────
# The trainer saves to demo/<OUT_STEM>_<timestamp>.npz — pick the newest npz.
GENERATED=$(ls -t demo/2_scott_0_1_1_test_moclip_svib_phy_*.npz 2>/dev/null | head -1)

if [[ -z "$GENERATED" ]]; then
    echo "[ERROR] No output NPZ found in demo/. Inference may have failed."
    exit 1
fi

echo ""
echo "Generated: $GENERATED  ($(du -sh "$GENERATED" | cut -f1))"

# ── git commit + push ─────────────────────────────────────────────────────────
BRANCH="feature/physics-smoother-svib"

$GIT -C /home/ferpaa/SemTalk checkout "$BRANCH"
# demo/ and *.npz are in .gitignore — force-add the specific demo output file
$GIT -C /home/ferpaa/SemTalk add -f "$GENERATED"

COMMIT_MSG="demo: add $(basename $GENERATED)

Checkpoint : $CKPT_STEM (S-VIB + physics fine-tune from best_116)
Audio      : $(basename $AUDIO)
P1  tau_floor=0.10 sqrt-compressed mass formula (physics_smoother.py)
P2  pose-level rot6d jerk loss via VQ decoders (phys_lambda=0.08)
P3  vib_beta_target=0.010 free_bits=0.20 (stronger VIB bottleneck)
FGD after fine-tune: see outputs/finetune_nohup.log epoch 128/134 ~0.428"

$GIT -C /home/ferpaa/SemTalk commit -m "$COMMIT_MSG"
$GIT -C /home/ferpaa/SemTalk push origin "$BRANCH"

echo ""
echo "================================================================"
echo "  Pushed: $GENERATED → origin/$BRANCH"
echo "================================================================"
