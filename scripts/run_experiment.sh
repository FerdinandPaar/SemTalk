#!/bin/bash
# Standardized experiment runner for SemTalk
# Usage: ./scripts/run_experiment.sh --config configs/your_config.yaml --name "experiment_name"

set -e

# ========================================
# Parse Arguments
# ========================================
CONFIG=""
NAME=""
GPUS="0"
WANDB_MODE="online"
EXTRA_ARGS=""
DETACH=true

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
        --foreground)
            DETACH=false
            shift
            ;;
        --detach)
            DETACH=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ========================================
# Validation
# ========================================
if [ -z "$CONFIG" ]; then
    echo "❌ Error: --config is required"
    echo "Usage: ./scripts/run_experiment.sh --config configs/your_config.yaml --name 'experiment_name'"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

# Generate name from config if not provided
if [ -z "$NAME" ]; then
    TIMESTAMP=$(date +%m%d_%H%M%S)
    CONFIG_NAME=$(basename "$CONFIG" .yaml)
    NAME="${TIMESTAMP}_${CONFIG_NAME}"
fi

# ========================================
# Environment Setup
# ========================================
export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE="$WANDB_MODE"

# DDP stability settings
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

is_python_38_plus() {
    local py="$1"
    "$py" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (sys.version_info.major > 3 or (sys.version_info.major == 3 and sys.version_info.minor >= 8)) else 1)
PY
}

USER_PYTHON_BIN="${PYTHON_BIN:-}"
PYTHON_BIN=""
for CAND in "$USER_PYTHON_BIN" python python3 /usr/bin/python3; do
    if [ -z "$CAND" ]; then
        continue
    fi
    if command -v "$CAND" >/dev/null 2>&1; then
        CAND_PATH="$(command -v "$CAND")"
        if is_python_38_plus "$CAND_PATH"; then
            PYTHON_BIN="$CAND_PATH"
            break
        fi
    elif [ -x "$CAND" ]; then
        if is_python_38_plus "$CAND"; then
            PYTHON_BIN="$CAND"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Error: Python >=3.8 not found. Activate the SemTalk environment first."
    echo "   Example: source scripts/setup_env.sh"
    exit 1
fi

# ========================================
# Create Output Directory
# ========================================
OUTPUT_DIR="outputs/experiments/${NAME}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"
PID_FILE="$OUTPUT_DIR/train.pid"
LAUNCH_LOG="$OUTPUT_DIR/launcher.log"
RUN_SCRIPT="$OUTPUT_DIR/run_training.sh"

echo "========================================="
echo "🚀 SemTalk Experiment Runner"
echo "========================================="
echo "Config: $CONFIG"
echo "Name: $NAME"
echo "GPUs: $GPUS"
echo "Output: $OUTPUT_DIR"
echo "WandB: $WANDB_MODE"
echo "Python: $PYTHON_BIN"
echo "Detach mode: $DETACH"
echo "Extra args: $EXTRA_ARGS"
echo "========================================="

# Copy config for reproducibility
cp "$CONFIG" "$OUTPUT_DIR/config.yaml"
echo "📋 Config saved to $OUTPUT_DIR/config.yaml"

# ========================================
# Build Run Script
# ========================================
cat > "$RUN_SCRIPT" <<EOF
#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE="$WANDB_MODE"
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "$(pwd)"

"$PYTHON_BIN" train.py \
    --config "$CONFIG" \
    --notes "_$NAME" \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_FILE"
EOF
chmod +x "$RUN_SCRIPT"

# ========================================
# Run Training
# ========================================
echo ""
echo "🏃 Starting training..."
echo "   Log: $LOG_FILE"
echo ""

if [ "$DETACH" = true ]; then
    nohup bash "$RUN_SCRIPT" > "$LAUNCH_LOG" 2>&1 < /dev/null &
    TRAIN_PID=$!
    echo "$TRAIN_PID" > "$PID_FILE"
    echo "✅ Training launched in background"
    echo "   PID: $TRAIN_PID"
    echo "   PID file: $PID_FILE"
    echo "   Launcher log: $LAUNCH_LOG"
    echo ""
    echo "To monitor:"
    echo "  ./scripts/monitor_training.sh --output_dir $OUTPUT_DIR"
    exit 0
fi

bash "$RUN_SCRIPT"
EXIT_CODE=$?

# ========================================
# Post-Training Summary
# ========================================
echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
fi
echo "   Output directory: $OUTPUT_DIR"
echo "   Log file: $LOG_FILE"
echo "========================================="

exit $EXIT_CODE
