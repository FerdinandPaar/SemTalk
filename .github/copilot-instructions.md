# GitHub Copilot Instructions for SemTalk

## Project Summary

SemTalk is a **co-speech gesture motion generation** research project for **ICCV 2025**. Built with **Python, PyTorch, and PyTorch DDP**, it generates holistic full-body motion (face, upper body, hands, lower body) from speech audio using a two-stage hierarchical approach:
1. **Base Motion**: Rhythmic audio-driven motion generation
2. **Sparse Semantic**: Frame-level semantic emphasis using MoCLIP embeddings

The project uses BEAT2 dataset and trains on GPU cluster (gridmaster → qrsh → gridnode016).

## Quick Reference

### Remote Execution (GPU Cluster)
```bash
# Run any command on the GPU cluster:
./scripts/agent_connect.sh "your_command_here"

# Examples:
./scripts/agent_connect.sh "nvidia-smi"
./scripts/agent_connect.sh "python train.py --config configs/semtalk_moclip_sparse.yaml"

# Or set up global command (run once):
bash scripts/setup_global_command.sh && source ~/.bashrc
# Then from anywhere:
semtalk "your_command"
```
See `.github/GLOBAL_COMMAND_SETUP.md` for details.

### Environment Setup
```bash
# After SSH to compute node (qrsh16 via gridmaster):
source scripts/setup_env.sh

# Or use the agent connection script:
./scripts/agent_connect.sh "your_command_here"
```

### Key Commands
| Task | Command |
|------|---------|
| Train base motion | `python train.py --config configs/semtalk_base.yaml` |
| Train sparse (Stage 2) | `python train.py --config configs/semtalk_sparse.yaml` |
| Train with MoCLIP | `python train.py --config configs/semtalk_moclip_sparse.yaml` |
| Test a model | `python train.py --test_state --config configs/semtalk_sparse.yaml` |
| Inference | `python train.py --inference --config configs/semtalk_sparse.yaml --audio_infer_path ./demo/audio.wav` |
| Multi-GPU training | `./run_train_moclip_4gpu.sh` |

### Key Configuration Files
- `configs/semtalk_base.yaml` - Base motion generation config
- `configs/semtalk_sparse.yaml` - Sparse semantic emphasis config  
- `configs/semtalk_moclip_sparse.yaml` - MoCLIP-conditioned training
- `configs/semtalk_fm_sparse.yaml` - Flow-matching base + sparse

## Build & Test

```bash
# Run quick validation (1 epoch to check for errors)
python train.py --config configs/semtalk_moclip_sparse.yaml --epochs 1

# Full training
python train.py --config configs/semtalk_moclip_sparse.yaml

# Test existing checkpoint
python train.py --test_state --config configs/semtalk_moclip_sparse.yaml

# Evaluate FGD metric
python utils/run_fgd_eval.py --checkpoint weights/best_190.bin --config configs/semtalk_moclip_sparse.yaml

# Multi-GPU training (4 GPUs)
./run_train_moclip_4gpu.sh
```

Always validate your changes with a quick 1-epoch run before launching full experiments.

## Commit Conventions

- Use **Conventional Commits**: `type(scope): short description`
  - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `train`
  - Scope: component name — e.g. `base_model`, `sparse_trainer`, `dataloader`, `flow_matching`, `config`
  - Examples:
    - `feat(sparse_trainer): add MoCLIP semantic conditioning`
    - `fix(flow_matching): correct velocity field computation`
    - `train: best FGD 0.4189 at epoch 190 with MoCLIP`
    - `docs(development_log): add flow matching integration notes`
- **One logical change per commit.** Don't batch unrelated changes.
- Keep subject line ≤ 72 characters.
- Never commit secrets, generated outputs, or large checkpoint files (use `weights/` with `.gitignore`).

## Code Style & Python Rules

- **Python 3.8+** — Use type hints for function signatures.
- **PyTorch conventions** — Models inherit from `nn.Module`, trainers from `BaseTrainer`.
- Every new model must be registered in its config YAML (`model:`, `g_name:` fields).
- Config-driven features preferred — add YAML flags before hardcoding behavior.
- **No hardcoded paths** — Use `args.data_path`, `args.out_path` from config.
- **Device agnostic** — Use `self.device` not `.cuda()` hardcoded.
- Checkpoints use `.bin` extension and include full state dict + optimizer state.

## File Layout

```
SemTalk/
├── train.py                    # @main entry, dynamic trainer import
├── train_torchrun.py           # DDP launcher (torchrun --nproc_per_node=N)
│
├── Trainers (one file per training mode)
├── semtalk_base_trainer.py     # Stage 1: base motion
├── semtalk_sparse_trainer.py   # Stage 2: semantic sparse
├── flowmatch_base_trainer.py   # Alternative: flow matching base
├── ae_trainer.py               # RVQ-VAE training (face/body parts)
│
├── models/
│   ├── semtalk_base.py         # Base motion generator
│   ├── semtalk_sparse.py       # Sparse semantic overlay
│   ├── flow_matching_base.py   # GestureLSM-style OT flow matching
│   ├── quantizer.py            # RVQ-VAE codebook
│   └── DEVELOPMENT_LOG.md      # Research notes & experiment history
│
├── configs/                    # YAML configs (model, training, data)
├── dataloaders/               # BEAT2 data pipeline
├── utils/                     # Metrics (FGD, beat align, L1 vel)
├── optimizers/                # Loss functions, schedulers
├── scripts/                   # Agent helper scripts
│   ├── agent_connect.sh       # SSH → qrsh → activate env
│   ├── run_experiment.sh      # Standardized training runner
│   └── monitor_training.sh    # Monitor with auto-stop
│
├── weights/                   # Checkpoints (.gitignore'd, not committed)
├── BEAT2/                     # Dataset (symlink or download)
└── outputs/                   # Training logs, generated motions
```

## Secrets & Sensitive Files

- **WandB API key** — Set as environment variable `WANDB_API_KEY`, never commit.
- **Checkpoint files** (`*.bin`, `*.pth`) — Do not commit to git (already in `.gitignore`).
- **SSH passwords** — NEVER hardcode. Use SSH keys (see `.github/SSH_SETUP.md`).

## Workflow for Agents

### 1. Implementing New Ideas
When asked to implement a new feature or experiment:

1. **Read existing code first** - Check `models/DEVELOPMENT_LOG.md` for context
2. **Find similar implementations** - Look in existing trainers/models for patterns
3. **Make minimal changes** - Prefer modifying configs over code when possible
4. **Add config flags** - Use YAML config to toggle features (e.g., `use_flow_matching: true`)

### 2. Running Training Experiments

```bash
# Standard training workflow:
./scripts/run_experiment.sh --config configs/your_config.yaml --name "experiment_name"

# The script will:
# - Create timestamped output directory
# - Launch training with WandB logging
# - Save checkpoints and logs
```

### 3. Monitoring Training

```bash
# Monitor running experiment:
./scripts/monitor_training.sh --output_dir outputs/your_experiment/

# Auto-stop if FGD/loss degrades for N epochs:
./scripts/monitor_training.sh --output_dir outputs/your_experiment/ --auto_stop --patience 10
```

### 4. Evaluating Results

```bash
# Run FGD evaluation on a checkpoint:
python utils/run_fgd_eval.py --checkpoint weights/best_model.bin --config configs/semtalk_sparse.yaml
```

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **FGD** (Fréchet Gesture Distance) | Lower is better | < 0.42 |
| **L1 Velocity** | Motion smoothness | Lower is better |
| **Beat Alignment** | Audio-motion sync | Higher is better |

Current best FGD: **0.4189** (epoch 190, MoCLIP sparse)

## Model Architecture

### Two-Stage Pipeline
```
Audio + Text → [Stage 1: Base Motion] → q_b (rhythmic latents)
                        ↓
           [Stage 2: Sparse Semantic] → q_s (semantic latents)
                        ↓
              q_m = MLP(ψ·q_s + (1−ψ)·q_b)  # Adaptive fusion
                        ↓
                  RVQ Decode → Full body motion
```

### Input/Output Shapes
- **Input audio**: `[B, T, 3]` (onset + amplitude beats)
- **HuBERT features**: `[B, T, 1024]`
- **Motion**: `[B, T, 337]` (55 joints × 6 rot + 3 trans + 4 contact)
- **Output latents**: `[B, T', 256]` where T'=T//4

## Common Issues & Solutions

### DDP Training Hangs
- Set `NCCL_IB_DISABLE=1` in environment
- Add `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
- Check `outputs/rank_logs/` for per-rank errors

### CUDA OOM
- Reduce `batch_size` in config
- Reduce `pose_length` (default 64)
- Use gradient checkpointing

### WandB Issues
- Set `stat: ts` in config to use TensorBoard instead
- Or set `WANDB_MODE=offline`

## When Making Changes

1. **Read `models/DEVELOPMENT_LOG.md` first** — Understand previous experiments and current state
2. **Test with 1 epoch** — Run `--epochs 1` to catch errors before full training
3. **Config-first approach** — Add YAML flags for toggleable features
4. **Backward compatibility** — Existing configs must still work
5. **Update DEVELOPMENT_LOG.md** — Document experiments, FGD scores, findings
6. **One change per commit** — Don't batch unrelated modifications

## Experiment / Branch Etiquette

- **Branch naming**: `feat/<short-slug>`, `fix/<short-slug>`, `train/<experiment-name>`
  - Examples: `feat/moclip-gating`, `fix/ddp-hang`, `train/fm-base-ablation`
- **Before committing**: Run validation to ensure code runs without errors
- **Track experiments**: Note FGD scores, hyperparameters, and findings in DEVELOPMENT_LOG.md

## Files to NEVER Modify Without Discussion
- `train.py` (main entry point - very stable, used by many configs)
- `dataloaders/*.py` (data pipeline - subtle bugs hard to catch, affects all experiments)
- Anything in `weights/` (pretrained models, read-only)
- `BEAT2/` dataset files (canonical data, never modify)

## Contact & Resources
- Paper: [SemTalk: Holistic Co-speech Motion Generation](https://arxiv.org/abs/2412.16563)
- Project: https://xiangyue-zhang.github.io/SemTalk/
- Dataset: [BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2)
