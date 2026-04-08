# Common Tasks for Agents

Quick reference for frequent development tasks in SemTalk.

## Checking Training Status

```bash
# View recent log output
tail -100 outputs/experiments/*/train.log

# Check GPU usage
nvidia-smi

# Check running experiments
ps aux | grep train.py
```

## Finding Best Checkpoints

```bash
# List all checkpoints sorted by FGD (if in filename)
ls -la outputs/*/checkpoints/best_*.bin

# Or search logs for best FGD values
grep -r "FGD" outputs/*/train.log | grep -oP "FGD[:\s]+[0-9.]+" | sort -t: -k2 -n | head
```

## Modifying Hyperparameters

### Learning Rate
```yaml
# In config YAML:
lr_base: 5e-5        # For base model parameters
lr_sparse: 1e-4      # For sparse model parameters

# Or via command line:
python train.py --config configs/X.yaml --lr_base 1e-4
```

### Batch Size
```yaml
batch_size: 128      # Default
# Reduce if OOM:
batch_size: 64
```

### Epochs
```yaml
epochs: 200          # Total training epochs
save_period: 10      # Checkpoint save frequency
```

## Evaluating a Checkpoint

```bash
# Run test evaluation
python train.py --test_state \
    --config configs/semtalk_moclip_sparse.yaml \
    --load_ckpt weights/best_190.bin

# Run FGD evaluation specifically
python utils/run_fgd_eval.py \
    --checkpoint weights/best_190.bin \
    --config configs/semtalk_moclip_sparse.yaml
```

## Generating Motion from Audio

```bash
# Inference on a single audio file
python train.py --inference \
    --config configs/semtalk_moclip_sparse.yaml \
    --audio_infer_path ./demo/my_audio.wav

# Output will be in outputs/inference/
```

## Comparing Two Models

```bash
# Run test on model A
python train.py --test_state --config configs/model_a.yaml --load_ckpt weights/model_a.bin > /tmp/eval_a.txt 2>&1

# Run test on model B  
python train.py --test_state --config configs/model_b.yaml --load_ckpt weights/model_b.bin > /tmp/eval_b.txt 2>&1

# Compare FGD
echo "Model A:" && grep FGD /tmp/eval_a.txt
echo "Model B:" && grep FGD /tmp/eval_b.txt
```

## Debugging Training Issues

### Loss is NaN
```bash
# 1. Check for numerical issues
# Add to train.py or trainer temporarily:
torch.autograd.set_detect_anomaly(True)

# 2. Try lower learning rate
# 3. Check data preprocessing (NaN in inputs?)
python -c "import torch; d = torch.load('datasets/beat2_semtalk_train/data.pt'); print('Has NaN:', torch.isnan(d).any())"
```

### CUDA OOM
```bash
# 1. Reduce batch size
# 2. Reduce pose_length (try 32 instead of 64)
# 3. Use gradient checkpointing (if implemented)
# 4. Use smaller model variant
```

### DDP Hangs
```bash
# 1. Set environment variables
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 2. Check rank logs
ls outputs/rank_logs/
cat outputs/rank_logs/rank_1.log  # Non-rank-0 errors

# 3. Try single GPU first to isolate issue
CUDA_VISIBLE_DEVICES=0 python train.py --config ...
```

## Adding a New Feature

### 1. Config-driven feature (recommended)
```yaml
# configs/my_config.yaml
my_new_feature: true
my_feature_weight: 0.5
```

```python
# In trainer or model:
if self.args.my_new_feature:
    # Apply feature
    loss += self.args.my_feature_weight * new_loss
```

### 2. New loss function
```python
# optimizers/loss_factory.py
def my_custom_loss(pred, target):
    return F.mse_loss(pred, target)

# Register in get_loss_func() if needed
```

### 3. New model component
```python
# models/my_module.py
class MyModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        # ...
    
    def forward(self, x):
        # ...
```

## Quick Config Reference

### Most Important Fields
```yaml
# Model architecture
model: semtalk_sparse
g_name: SemTalkSparse
trainer: semtalk_sparse

# Training
batch_size: 128
epochs: 200
lr_base: 5e-5

# Data
pose_length: 64
dataset: semtalk_dataloader

# Checkpoints
base_ckpt: ./weights/best_semtalk_base.bin
load_ckpt: null  # Set to resume training

# Logging
stat: wandb  # or 'ts' for tensorboard
project: SemTalk
```

## File Locations

| What | Where |
|------|-------|
| Configs | `configs/*.yaml` |
| Model checkpoints | `weights/` or `outputs/*/checkpoints/` |
| Training logs | `outputs/*/train.log` |
| Generated motions | `outputs/inference/` |
| Test results | `outputs/test_*/` |
| Development notes | `models/DEVELOPMENT_LOG.md` |
