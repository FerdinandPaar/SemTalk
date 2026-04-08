#!/bin/bash
# Training monitor with auto-stop capability
# Monitors training progress and can auto-stop if metrics degrade
#
# Usage: 
#   ./scripts/monitor_training.sh --output_dir outputs/experiments/my_run/
#   ./scripts/monitor_training.sh --output_dir outputs/experiments/my_run/ --auto_stop --patience 10

set -e

# ========================================
# Parse Arguments
# ========================================
OUTPUT_DIR=""
AUTO_STOP=false
PATIENCE=10
METRIC="fgd"  # fgd, loss, or val_loss
CHECK_INTERVAL=60  # seconds

while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --auto_stop)
            AUTO_STOP=true
            shift
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    echo "❌ Error: --output_dir is required"
    echo "Usage: ./scripts/monitor_training.sh --output_dir outputs/experiments/my_run/"
    exit 1
fi

# ========================================
# Helper Functions
# ========================================

extract_fgd() {
    # Extract SemTalk FGD/FID values from log file.
    grep -oPi '(?:fid score|fgd)[:\s]+\K[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?' "$1" 2>/dev/null | tail -20
}

extract_loss() {
    # Extract loss values from log file
    grep -oP 'loss[:\s]+\K[0-9.]+' "$1" 2>/dev/null | tail -50
}

find_training_pid() {
    # Prefer PID file from run_experiment.sh detached mode.
    PID_FILE="$OUTPUT_DIR/train.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "$PID"
            return
        fi
    fi

    # Fallback: scan process list.
    pgrep -f "python.*train.py" 2>/dev/null | head -1
}

# ========================================
# Main Monitor Loop
# ========================================

LOG_FILE="$OUTPUT_DIR/train.log"
BEST_METRIC=999999
NO_IMPROVE_COUNT=0

echo "========================================="
echo "📊 SemTalk Training Monitor"
echo "========================================="
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Auto-stop: $AUTO_STOP"
echo "Patience: $PATIENCE epochs"
echo "Metric: $METRIC"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "========================================="
echo ""

while true; do
    # Check if log file exists
    if [ ! -f "$LOG_FILE" ]; then
        echo "⏳ Waiting for log file..."
        sleep "$CHECK_INTERVAL"
        continue
    fi

    # Get latest metrics
    case $METRIC in
        fgd)
            LATEST=$(extract_fgd "$LOG_FILE" | tail -1)
            ;;
        loss)
            LATEST=$(extract_loss "$LOG_FILE" | tail -1)
            ;;
        *)
            LATEST=$(grep -oP "${METRIC}[:\s]+\K[0-9.]+" "$LOG_FILE" 2>/dev/null | tail -1)
            ;;
    esac

    # Get current epoch
    CURRENT_EPOCH=$(grep -oP 'Epoch[:\s]+\K[0-9]+' "$LOG_FILE" 2>/dev/null | tail -1 || echo "?")
    
    # Check GPU status
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")

    # Display status
    TIMESTAMP=$(date '+%H:%M:%S')
    echo "[$TIMESTAMP] Epoch: $CURRENT_EPOCH | $METRIC: ${LATEST:-N/A} | Best: $BEST_METRIC | GPU: $GPU_INFO"

    # Check for improvement
    if [ -n "$LATEST" ]; then
        # Compare (lower is better for FGD/loss)
        IS_BETTER=$(echo "$LATEST < $BEST_METRIC" | bc -l 2>/dev/null || echo "0")
        
        if [ "$IS_BETTER" = "1" ]; then
            BEST_METRIC="$LATEST"
            NO_IMPROVE_COUNT=0
            echo "    ✨ New best $METRIC!"
        else
            NO_IMPROVE_COUNT=$((NO_IMPROVE_COUNT + 1))
            
            if [ "$AUTO_STOP" = true ] && [ "$NO_IMPROVE_COUNT" -ge "$PATIENCE" ]; then
                echo ""
                echo "========================================="
                echo "⚠️  No improvement for $PATIENCE checks"
                echo "    Triggering auto-stop..."
                echo "========================================="
                
                PID=$(find_training_pid)
                if [ -n "$PID" ]; then
                    echo "    Sending SIGTERM to PID $PID"
                    kill -TERM "$PID" 2>/dev/null || true
                    sleep 5
                    # Force kill if still running
                    if kill -0 "$PID" 2>/dev/null; then
                        echo "    Process still running, sending SIGKILL"
                        kill -KILL "$PID" 2>/dev/null || true
                    fi
                    echo "    Training stopped."
                else
                    echo "    Could not find training process"
                fi
                
                exit 0
            fi
        fi
    fi

    # Check if training is still running
    PID=$(find_training_pid)
    if [ -z "$PID" ]; then
        # Check if process recently finished
        if grep -q "Training completed\|finished\|done" "$LOG_FILE" 2>/dev/null; then
            echo ""
            echo "✅ Training completed!"
            echo "   Best $METRIC: $BEST_METRIC"
            exit 0
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
