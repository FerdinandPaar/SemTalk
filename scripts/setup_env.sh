#!/bin/bash
# Agent environment setup script for SemTalk
# Run this after SSH'ing into the compute node
# Usage: source scripts/setup_env.sh

# ========================================
# Configuration
# ========================================
CONDA_ENV="semtalk"
PROJECT_DIR="$HOME/semtalk"

# ========================================
# Activate Conda Environment
# ========================================
echo "🔧 Activating conda environment: $CONDA_ENV"

# Try common conda initialization paths
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "⚠️  Could not find conda.sh, trying 'conda activate' directly"
fi

conda activate "$CONDA_ENV" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment '$CONDA_ENV'"
    echo "   Available environments:"
    conda env list
    return 1
fi

echo "✅ Conda environment '$CONDA_ENV' activated"
echo "   Python: $(which python)"
echo "   Version: $(python --version)"

# ========================================
# Navigate to Project Directory
# ========================================
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "📁 Changed to project directory: $PROJECT_DIR"
else
    echo "⚠️  Project directory not found: $PROJECT_DIR"
    echo "   Current directory: $(pwd)"
fi

# ========================================
# Display Status
# ========================================
echo ""
echo "========================================="
echo "🚀 SemTalk Environment Ready"
echo "========================================="
echo "Working directory: $(pwd)"
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""
