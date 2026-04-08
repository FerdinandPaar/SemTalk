# SSH Setup for Passwordless Authentication

⚠️ **SECURITY**: Never store passwords in scripts, config files, or code.

## One-Time Setup: SSH Key Authentication

### Step 1: Generate SSH Key (if you don't have one)
```bash
# On your local machine (the one you're running copilot from)
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter for default location, optionally set passphrase
```

### Step 2: Copy Key to GridMaster
```bash
# This will prompt for your password ONE TIME
ssh-copy-id ferpaa@gridmaster
```

### Step 3: Test Passwordless Login
```bash
ssh gridmaster  # Should log in without password prompt
```

### Step 4: Setup SSH Config for Easy Access
Add to `~/.ssh/config`:
```
Host gridmaster
    HostName gridmaster
    User ferpaa
    IdentityFile ~/.ssh/id_ed25519
```

## Quick Connect Scripts

After SSH keys are set up:

```bash
# Connect to gridmaster with env activated
./scripts/agent_connect.sh

# Connect to compute node via qrsh
./scripts/agent_connect.sh --qrsh

# Run a command on gridmaster
./scripts/agent_connect.sh "nvidia-smi"

# Run a command on compute node
./scripts/agent_connect.sh --qrsh "python train.py --config configs/semtalk_moclip_sparse.yaml"
```

## Manual Workflow (Without Scripts)

If you prefer manual steps:
```bash
# 1. SSH to gridmaster
ssh gridmaster

# 2. Request compute node (if needed for GPU)
qrsh

# 3. Activate environment
conda activate semtalk

# 4. Go to project
cd ~/semtalk

# 5. Run your command
python train.py --config configs/semtalk_moclip_sparse.yaml
```

## Troubleshooting

### "Name or service not known"
The compute node name (like qrsh16) is dynamically assigned. Use `qrsh` to get a node instead of SSH'ing directly.

### Permission denied (publickey)
Your SSH key isn't set up. Run `ssh-copy-id ferpaa@gridmaster`.

### Connection timed out
Check VPN connection or network access to the cluster.
