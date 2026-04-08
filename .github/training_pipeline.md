# Training Pipeline for Agents

This document describes the standardized pipeline for agents to implement ideas, run experiments, and iterate based on results.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agent Experiment Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. IMPLEMENT ──► 2. VALIDATE ──► 3. RUN ──► 4. MONITOR ──► 5. DECIDE  │
│       │               │              │            │              │      │
│   Modify code    Quick test     Launch       Watch metrics    Improve   │
│   or config      (1 batch)     training     (auto-stop)      or stop   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Step 0: Activate Environment

Before launching any run, activate a Python 3.8+ environment:

```bash
source scripts/setup_env.sh
# or: conda activate semtalk
```

## Step 1: Implement Changes

### Option A: Config-only changes (preferred)
```bash
# Copy existing config as template
cp configs/semtalk_moclip_sparse.yaml configs/my_experiment.yaml

# Edit the config (change hyperparameters, enable/disable features)
# Example changes:
# - use_flow_matching: true
# - semantic_weight: 2.0
# - batch_size: 64
```

### Option B: Code changes
```bash
# 1. Identify the right file to modify
# 2. Make minimal, surgical changes
# 3. Add config flags for toggleability
# 4. Test immediately (Step 2)
```

## Step 2: Validate Changes

Before running a full experiment, validate that the code runs without errors:

```bash
# Quick validation run (1-2 batches)
python train.py \
    --config configs/my_experiment.yaml \
    --test_mode  # If supported, runs minimal batches

# Or manually limit epochs in config:
# epochs: 1
# Then run and check for errors
```

**Check for:**
- Import errors
- Shape mismatches
- CUDA OOM (reduce batch_size if needed)
- Config parsing errors

## Step 3: Run Experiment

### Basic run:
```bash
./scripts/run_experiment.sh \
    --config configs/my_experiment.yaml \
    --name "descriptive_experiment_name"
```

This now launches in detached mode by default, so training continues if the VS Code window or SSH session closes.
The script writes:
- outputs/experiments/<name>/train.log
- outputs/experiments/<name>/launcher.log
- outputs/experiments/<name>/train.pid

### Recommended for cluster persistence (qsub)

For guaranteed continuation independent of terminal/VS Code lifecycle, submit via scheduler:

```bash
# Run on gridmaster (or through an SSH session to gridmaster)
./scripts/submit_experiment.sh \
    --config configs/my_experiment.yaml \
    --name "my_experiment_qsub"
```

This creates a scheduler-managed job, so closing VS Code does not stop training.

### With specific GPU:
```bash
./scripts/run_experiment.sh \
    --config configs/my_experiment.yaml \
    --name "experiment_gpu2" \
    --gpus 2
```

### Foreground mode (optional)
```bash
./scripts/run_experiment.sh \
    --config configs/my_experiment.yaml \
    --name "debug_foreground" \
    --foreground
```

Use this only when you want live stdout in the current terminal and do not need disconnect safety.

### Resuming interrupted training:
```bash
./scripts/run_experiment.sh \
    --config configs/my_experiment.yaml \
    --name "experiment_resumed" \
    --resume
```

### Offline mode (no WandB):
```bash
./scripts/run_experiment.sh \
    --config configs/my_experiment.yaml \
    --name "experiment_offline" \
    --offline
```

## Step 4: Monitor Training

### Basic monitoring:
```bash
./scripts/monitor_training.sh \
    --output_dir outputs/experiments/my_experiment/
```

In detached mode, monitor reads outputs/experiments/<name>/train.pid to track the exact process.

### With auto-stop on degradation:
```bash
./scripts/monitor_training.sh \
    --output_dir outputs/experiments/my_experiment/ \
    --auto_stop \
    --patience 15 \
    --metric fgd
```

### Monitor multiple experiments:
```bash
# Terminal 1
./scripts/monitor_training.sh --output_dir outputs/experiments/exp1/

# Terminal 2
./scripts/monitor_training.sh --output_dir outputs/experiments/exp2/
```

## Step 5: Decide Next Steps

### If training is going well (metrics improving):
- Let it continue
- Consider extending epochs if still improving at limit
- Take note of successful hyperparameters

### If training degrades:
```bash
# Auto-stop is triggered, or manually stop:
# 1. Note the best checkpoint before degradation
# 2. Analyze what went wrong (check logs)
# 3. Modify approach and retry from Step 1
```

### If training is stuck (no improvement for many epochs):
- Check learning rate (might be too low or too high)
- Check batch size (might need adjustment)
- Check if loss is NaN (numerical instability)
- Consider different initialization

## Experiment Tracking

### Log format:
All experiments are saved in:
```
outputs/experiments/{experiment_name}/
├── config.yaml          # Copy of config used
├── train.log            # Full training log
├── checkpoints/         # Model checkpoints
│   ├── best_{epoch}.bin
│   └── latest.bin
└── tensorboard/         # If using TensorBoard
```

### Naming convention:
```
{MMDD}_{HHMMSS}_{config_name}_{variant_description}

Examples:
- 0326_1430_moclip_sparse_lr5e5
- 0326_1545_fm_sparse_batch64
- 0327_0900_ablation_no_semantic
```

## Common Experiment Types

### 1. Hyperparameter Tuning
```yaml
# Experiment: Test different learning rates
# Create configs with variations:
# - configs/hp_lr_1e4.yaml (lr_base: 1e-4)
# - configs/hp_lr_5e5.yaml (lr_base: 5e-5)
# - configs/hp_lr_1e5.yaml (lr_base: 1e-5)

# Run each and compare final FGD
```

### 2. Ablation Study
```yaml
# Experiment: Measure impact of MoCLIP conditioning
# Config A: use_moclip: true (baseline)
# Config B: use_moclip: false (ablation)

# Compare FGD difference to measure MoCLIP contribution
```

### 3. Architecture Comparison
```yaml
# Experiment: Compare base motion generators
# Config A: use_flow_matching: false (original)
# Config B: use_flow_matching: true (GestureLSM-style)

# Run both, compare FGD and motion quality
```

## Interpreting Results

### Key Metrics:

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| FGD | < 0.42 | Lower = more realistic motion |
| Loss | Decreasing | Should steadily decrease |
| Val Loss | Close to Train | Large gap = overfitting |

### Warning Signs:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss = NaN | Numerical instability | Lower LR, add gradient clipping |
| Loss not decreasing | LR too low | Increase LR |
| Loss oscillating | LR too high | Decrease LR |
| Val loss increasing | Overfitting | Stop early, add regularization |
| GPU OOM | Batch too large | Reduce batch_size |

## Best Practices

1. **Always start from a working config** - Don't change too many things at once
2. **Document your hypothesis** - Note what you're testing and why
3. **Save successful configs** - Copy working configs to a `configs/successful/` folder
4. **Track experiment results** - Keep a log of experiment → result mappings
5. **Clean up failed runs** - Remove outputs from experiments that crashed early

## Quick Reference

```bash
# Full workflow example:
# 1. Create config variation
cp configs/semtalk_moclip_sparse.yaml configs/test_higher_lr.yaml
# Edit: lr_base: 1e-4 (was 5e-5)

# 2. Validate
python train.py --config configs/test_higher_lr.yaml --epochs 1

# 3. Run
./scripts/run_experiment.sh --config configs/test_higher_lr.yaml --name "higher_lr_test"

# 4. Monitor (in another terminal)
./scripts/monitor_training.sh --output_dir outputs/experiments/higher_lr_test/ --auto_stop

# 5. Evaluate best checkpoint
python train.py --test_state --config configs/test_higher_lr.yaml \
    --load_ckpt outputs/experiments/higher_lr_test/checkpoints/best_*.bin
```
