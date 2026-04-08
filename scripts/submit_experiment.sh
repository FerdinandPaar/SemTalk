#!/bin/bash
# Submit a persistent SemTalk training job through SGE/qsub.
# This is disconnect-safe: the job is managed by the scheduler.

set -e

CONFIG=""
NAME=""
QUEUE="mld.q@gridnode016"
SLOTS="4"
GPUS="0,1,2,3"
WANDB_MODE="online"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --slots)
            SLOTS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --offline)
            WANDB_MODE="offline"
            shift
            ;;
        --test)
            EXTRA_ARGS="$EXTRA_ARGS --test_state"
            shift
            ;;
        --resume)
            EXTRA_ARGS="$EXTRA_ARGS --resume"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config is required"
    echo "Usage: ./scripts/submit_experiment.sh --config configs/your_config.yaml --name your_run"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

if [ -z "$NAME" ]; then
    TS=$(date +%m%d_%H%M%S)
    CONFIG_NAME=$(basename "$CONFIG" .yaml)
    NAME="${TS}_${CONFIG_NAME}_qsub"
fi

ROOT_DIR="$(pwd)"
OUTPUT_DIR="outputs/experiments/${NAME}"
mkdir -p "$OUTPUT_DIR"
cp "$CONFIG" "$OUTPUT_DIR/config.yaml"

JOB_SCRIPT="$OUTPUT_DIR/qsub_job.sh"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#$ -S /bin/bash
#$ -N ${NAME}
#$ -q ${QUEUE}
#$ -pe smp ${SLOTS}
#$ -cwd
#$ -o ${OUTPUT_DIR}/qsub_train.log
#$ -e ${OUTPUT_DIR}/qsub_train.err
#$ -j y

set -e

source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

cd ${ROOT_DIR}

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${GPUS}
export WANDB_MODE=${WANDB_MODE}
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Job started: \\$(date) ==="
echo "=== Host: \\$(hostname) ==="
echo "=== CWD: \\$(pwd) ==="
echo "=== GPUs: ${GPUS} ==="

python train.py --config ${CONFIG} --notes _${NAME} ${EXTRA_ARGS}

echo "=== Job finished: \\$(date) ==="
EOF
chmod +x "$JOB_SCRIPT"

echo "========================================="
echo "Submitting qsub experiment"
echo "Name: $NAME"
echo "Config: $CONFIG"
echo "Queue: $QUEUE"
echo "Slots: $SLOTS"
echo "GPUs: $GPUS"
echo "Output: $OUTPUT_DIR"
echo "Job script: $JOB_SCRIPT"
echo "========================================="

qsub "$JOB_SCRIPT" | tee "$OUTPUT_DIR/qsub_submit.txt"

echo "Submitted. Track with: qstat -u \$USER"
echo "Logs: $OUTPUT_DIR/qsub_train.log"
