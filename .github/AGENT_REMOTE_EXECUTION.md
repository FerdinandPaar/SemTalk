# Agent Remote Execution Guide

## Quick Start for Agents

Use the canonical grid script:

```bash
./scripts/connect_gridnode.sh -- "your_command_here"
```

Or open an interactive shell directly on the compute node:

```bash
./scripts/connect_gridnode.sh
```

`scripts/agent_connect.sh` remains available as a backward-compatible wrapper.

### Examples

```bash
# Check GPU status
./scripts/connect_gridnode.sh -- "nvidia-smi"

# Run training
./scripts/connect_gridnode.sh -- "python train.py --config configs/semtalk_moclip_sparse.yaml --ddp false --gpus 0"

# Run tests
./scripts/connect_gridnode.sh -- "python train.py --test_state --config configs/semtalk_moclip_sparse.yaml --ddp false --gpus 0"

# Interactive shell (for debugging)
./scripts/connect_gridnode.sh
```

## What the Script Does

The script automatically:
1. SSH to `gridmaster`
2. Request compute node: `qrsh -q mld.q@gridnode016`
3. Change directory: `cd ~/SemTalk`
4. Activate environment on compute node:
	- tries `.venv` (`/home/ferpaa/SemTalk/.venv/bin/activate`)
	- falls back to `conda activate semtalk`
5. Run your command

The setup finishes on the compute node (for example `gridnode016.mpi.nl`), not on `gridmaster`.

## Connection Details

| Step | Command |
|------|---------|
| Login server | `ssh gridmaster` |
| Compute node | `qrsh -q mld.q@gridnode016` |
| Project folder | `cd ~/SemTalk` |
| Environment | `source /home/ferpaa/SemTalk/.venv/bin/activate` (preferred) or `conda activate semtalk` |

## Useful Options

```bash
# Different node
./scripts/connect_gridnode.sh --node gridnode015

# Different queue
./scripts/connect_gridnode.sh --queue mld.q --node gridnode016

# Non-interactive command
./scripts/connect_gridnode.sh -- "hostname && pwd"
```

## Manual Connection (if script fails)

```bash
ssh ferpaa@gridmaster
qrsh -q mld.q@gridnode016
cd ~/SemTalk
source /home/ferpaa/SemTalk/.venv/bin/activate
# or: conda activate semtalk
# Now run your commands
```

## Troubleshooting

### Password prompts
Set up SSH keys once: `ssh-copy-id ferpaa@gridmaster`

### "No matching job" from qrsh
The node may be busy. Try without specific node: `qrsh -q mld.q`

### Environment not found
Check available envs: `conda env list`
