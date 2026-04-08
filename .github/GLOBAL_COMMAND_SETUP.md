# Global SemTalk Command Setup

## One-Time Setup (Run Once)

```bash
# From anywhere, run:
bash /home/ferpaa/SemTalk/scripts/setup_global_command.sh

# Then activate:
source ~/.bashrc   # or source ~/.zshrc if you use zsh
```

This creates a global `semtalk` command you can run from **anywhere**.

## Usage

### From your local machine (`ferpaa@ssh`):
```bash
# Interactive shell on gridnode016 with environment ready
semtalk

# Run a command
semtalk "nvidia-smi"

# Run training
semtalk "python train.py --config configs/semtalk_moclip_sparse.yaml"
```

### For agents in their terminal:
Same commands work:
```bash
semtalk "your_command_here"
```

## What it does

The `semtalk` command:
1. SSH to `gridmaster`
2. Request `qrsh -q mld.q@gridnode016`
3. Activate venv (`.venv`) or conda env (`semtalk`)
4. `cd ~/SemTalk`
5. Run your command (or start interactive shell)

## Manual command (without alias)

If you don't want to set up the alias:
```bash
/home/ferpaa/SemTalk/scripts/agent_connect.sh "your_command"
```

## Alternative: Direct script usage

```bash
# Full control with options
/home/ferpaa/SemTalk/scripts/connect_gridnode.sh -- "python train.py ..."

# Different node
/home/ferpaa/SemTalk/scripts/connect_gridnode.sh --node gridnode015 -- "nvidia-smi"
```
