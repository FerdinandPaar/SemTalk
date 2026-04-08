#!/usr/bin/env bash
set -euo pipefail

# One-time SSH key setup helper for cluster access.
# This script intentionally avoids storing passwords anywhere.

HOST="gridmaster"
USER_NAME="${USER:-}"
IDENTITY_FILE="$HOME/.ssh/id_ed25519"
ADD_CONFIG=true
OVERWRITE_CONFIG=false

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup_ssh_keys.sh [options]

Options:
  --host <host>                 SSH host alias or hostname (default: gridmaster)
  --user <user>                 SSH username (default: current local user)
  --identity-file <path>        Private key path (default: ~/.ssh/id_ed25519)
  --no-config                   Do not add/update ~/.ssh/config entry
  --overwrite-config            Replace an existing matching Host block
  -h, --help                    Show this help

What it does:
  1) Ensures an SSH key exists (ed25519)
  2) Copies the public key to remote host using ssh-copy-id
  3) Optionally adds/updates ~/.ssh/config host entry
  4) Tests passwordless SSH login

Notes:
  - You may be prompted for your remote password exactly once by ssh-copy-id.
  - No password is stored in any file or script.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --user)
      USER_NAME="$2"
      shift 2
      ;;
    --identity-file)
      IDENTITY_FILE="$2"
      shift 2
      ;;
    --no-config)
      ADD_CONFIG=false
      shift
      ;;
    --overwrite-config)
      OVERWRITE_CONFIG=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$USER_NAME" ]]; then
  echo "Error: username is empty. Pass --user <user>." >&2
  exit 1
fi

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ ! -f "$IDENTITY_FILE" ]]; then
  echo "[1/4] Generating SSH key: $IDENTITY_FILE"
  ssh-keygen -t ed25519 -f "$IDENTITY_FILE" -C "$USER_NAME@$HOST" -N ""
else
  echo "[1/4] SSH key already exists: $IDENTITY_FILE"
fi

PUB_KEY="${IDENTITY_FILE}.pub"
if [[ ! -f "$PUB_KEY" ]]; then
  echo "Error: public key not found at $PUB_KEY" >&2
  exit 1
fi

echo "[2/4] Copying key to ${USER_NAME}@${HOST} (may prompt for password once)"
ssh-copy-id -i "$PUB_KEY" "${USER_NAME}@${HOST}"

if [[ "$ADD_CONFIG" == true ]]; then
  CONFIG_FILE="$HOME/.ssh/config"
  touch "$CONFIG_FILE"
  chmod 600 "$CONFIG_FILE"

  HOST_BLOCK=$(cat <<EOF
Host $HOST
    HostName $HOST
    User $USER_NAME
    IdentityFile $IDENTITY_FILE
    IdentitiesOnly yes
EOF
)

  if grep -Eq "^Host[[:space:]]+$HOST$" "$CONFIG_FILE"; then
    if [[ "$OVERWRITE_CONFIG" == true ]]; then
      echo "[3/4] Replacing existing Host $HOST block in $CONFIG_FILE"
      awk -v host="$HOST" '
        BEGIN {skip=0}
        $1=="Host" && $2==host {skip=1; next}
        $1=="Host" && skip==1 {skip=0}
        skip==0 {print}
      ' "$CONFIG_FILE" > "$CONFIG_FILE.tmp"
      mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
      printf "\n%s\n" "$HOST_BLOCK" >> "$CONFIG_FILE"
    else
      echo "[3/4] Host $HOST already exists in $CONFIG_FILE (keeping existing block)"
      echo "      Use --overwrite-config to replace it."
    fi
  else
    echo "[3/4] Adding Host $HOST block to $CONFIG_FILE"
    printf "\n%s\n" "$HOST_BLOCK" >> "$CONFIG_FILE"
  fi
else
  echo "[3/4] Skipping ~/.ssh/config update (--no-config)"
fi

echo "[4/4] Testing passwordless SSH login"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST" "echo 'SSH setup OK on $(hostname)'" || {
  echo "SSH key copied, but passwordless login test failed." >&2
  echo "Try: ssh ${USER_NAME}@${HOST}" >&2
  exit 1
}

echo "Done. You can now use: ./scripts/agent_connect.sh \"nvidia-smi\""
