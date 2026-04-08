#!/usr/bin/env bash
# Submit weight-embedding ablation as an SGE job (disconnect-safe).

set -euo pipefail

CONFIG="configs/semtalk_moclip_sparse.yaml"
BEAT2_DIR="BEAT2/beat_english_v2.0.0"
SPLIT="test"
EPOCHS=5
MASS_SCALE=0.30
MIN_WINDOW_FRAMES=15
BOOTSTRAP=300
MODE_LIST="arm_combined_core,split_arm_core,all_joints_core"
EXTRA_ARGS=""
PYTHON_BIN="/home/ferpaa/miniconda3/envs/semtalk/bin/python"
FORCE_RETRAIN=false
TRAIN_BASE_JOINTLY=false
COUPLE_BASE_MASS_MODE=false
BASE_MASS_SCALE=""

NAME=""
QUEUE="mld.q@gridnode016"
SLOTS="1"
GPUS="0"
CONDA_ENV="semtalk"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --beat2_dir)
            BEAT2_DIR="$2"; shift 2 ;;
        --split)
            SPLIT="$2"; shift 2 ;;
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --mass_scale)
            MASS_SCALE="$2"; shift 2 ;;
        --min_window_frames)
            MIN_WINDOW_FRAMES="$2"; shift 2 ;;
        --bootstrap)
            BOOTSTRAP="$2"; shift 2 ;;
        --modes)
            MODE_LIST="$2"; shift 2 ;;
        --extra)
            EXTRA_ARGS="$2"; shift 2 ;;
        --python)
            PYTHON_BIN="$2"; shift 2 ;;
        --force_retrain)
            FORCE_RETRAIN=true; shift ;;
        --train_base_jointly)
            TRAIN_BASE_JOINTLY=true; shift ;;
        --couple_base_mass_mode)
            COUPLE_BASE_MASS_MODE=true; shift ;;
        --base_mass_scale)
            BASE_MASS_SCALE="$2"; shift 2 ;;
        --name)
            NAME="$2"; shift 2 ;;
        --queue)
            QUEUE="$2"; shift 2 ;;
        --slots)
            SLOTS="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --conda_env)
            CONDA_ENV="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub not found on this host. Submit from gridmaster, e.g.:"
    echo "  ssh gridmaster 'cd ~/SemTalk && bash scripts/submit_weight_embedding_ablation.sh --epochs ${EPOCHS} --modes ${MODE_LIST}'"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

CONFIG_STEM="$(basename "$CONFIG" .yaml)"
if [[ -z "$NAME" ]]; then
    TS="$(date +%m%d_%H%M%S)"
    NAME="${TS}_${CONFIG_STEM}_ablation_e${EPOCHS}"
fi

OUTPUT_DIR="${ROOT_DIR}/outputs/experiments/${NAME}"
mkdir -p "$OUTPUT_DIR"
cp "$CONFIG" "${OUTPUT_DIR}/config.yaml"

RUN_CMD=(
    bash scripts/run_weight_embedding_ablation.sh
    --foreground
    --config "$CONFIG"
    --beat2_dir "$BEAT2_DIR"
    --split "$SPLIT"
    --epochs "$EPOCHS"
    --mass_scale "$MASS_SCALE"
    --min_window_frames "$MIN_WINDOW_FRAMES"
    --bootstrap "$BOOTSTRAP"
    --modes "$MODE_LIST"
    --python "$PYTHON_BIN"
)
if [[ -n "$EXTRA_ARGS" ]]; then
    RUN_CMD+=(--extra "$EXTRA_ARGS")
fi
if [[ "$FORCE_RETRAIN" == "true" ]]; then
    RUN_CMD+=(--force_retrain)
fi
if [[ "$TRAIN_BASE_JOINTLY" == "true" ]]; then
    RUN_CMD+=(--train_base_jointly)
fi
if [[ "$COUPLE_BASE_MASS_MODE" == "true" ]]; then
    RUN_CMD+=(--couple_base_mass_mode)
fi
if [[ -n "$BASE_MASS_SCALE" ]]; then
    RUN_CMD+=(--base_mass_scale "$BASE_MASS_SCALE")
fi
RUN_CMD_STR="$(printf '%q ' "${RUN_CMD[@]}")"

JOB_SCRIPT="${OUTPUT_DIR}/qsub_ablation_job.sh"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -N ${NAME}
#$ -q ${QUEUE}
#$ -pe smp ${SLOTS}
#$ -cwd
#$ -o ${OUTPUT_DIR}/qsub_ablation.log
#$ -e ${OUTPUT_DIR}/qsub_ablation.err
#$ -j y

set -euo pipefail

source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

cd ${ROOT_DIR}

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${GPUS}
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1
# train.py initializes torch.distributed even for single-rank runs;
# set a unique rendezvous port per scheduler job to avoid collisions.
if [[ -n "\${JOB_ID:-}" ]]; then
    export MASTER_PORT=\$((15000 + (JOB_ID % 20000)))
else
    export MASTER_PORT=29500
fi

echo "=== Ablation job started: \$(date) ==="
echo "=== Host: \$(hostname) ==="
echo "=== GPUs: ${GPUS} ==="
echo "=== MASTER_ADDR: \${MASTER_ADDR}  MASTER_PORT: \${MASTER_PORT} ==="

${RUN_CMD_STR}

echo "=== Ablation job finished: \$(date) ==="
EOF
chmod +x "$JOB_SCRIPT"

echo "========================================="
echo "Submitting weight-embedding ablation"
echo "Name: $NAME"
echo "Config: $CONFIG"
echo "Modes: $MODE_LIST"
echo "Train base jointly: $TRAIN_BASE_JOINTLY"
echo "Couple base mass mode: $COUPLE_BASE_MASS_MODE"
echo "Base mass scale: ${BASE_MASS_SCALE:-auto(${MASS_SCALE})}"
echo "Queue: $QUEUE"
echo "Slots: $SLOTS"
echo "GPUs: $GPUS"
echo "Output: $OUTPUT_DIR"
echo "Job script: $JOB_SCRIPT"
echo "========================================="

qsub "$JOB_SCRIPT" | tee "${OUTPUT_DIR}/qsub_submit.txt"

echo "Submitted. Track with: qstat -u \$USER"
echo "Logs: ${OUTPUT_DIR}/qsub_ablation.log"
