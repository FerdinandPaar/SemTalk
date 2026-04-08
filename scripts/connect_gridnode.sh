#!/usr/bin/env bash
# Connect to gridmaster, allocate a compute node with qrsh,
# then enter SemTalk and activate environment on the compute node.

set -euo pipefail

GRIDMASTER="${GRIDMASTER:-gridmaster}"
NODE="${NODE:-gridnode016}"
QUEUE="${QUEUE:-mld.q}"
CONDA_ENV="${CONDA_ENV:-semtalk}"
PROJECT_DIR="${PROJECT_DIR:-SemTalk}"
VENV_PATH="${VENV_PATH:-/home/ferpaa/SemTalk/.venv/bin/activate}"
PRINT_HOST=true

print_help() {
    cat <<'EOF'
Usage:
  scripts/connect_gridnode.sh [options] [-- command...]

Options:
  --gridmaster <host>   Login host (default: gridmaster)
  --queue <queue>       SGE queue (default: mld.q)
  --node <node>         Target node (default: gridnode016)
  --project <dir>       Project dir under home (default: SemTalk)
  --conda-env <name>    Conda env fallback (default: semtalk)
  --venv <path>         Venv activate path (default: /home/ferpaa/SemTalk/.venv/bin/activate)
  --no-host-print       Do not print hostname after connect
  -h, --help            Show help

Examples:
  scripts/connect_gridnode.sh
  scripts/connect_gridnode.sh -- node -v
  scripts/connect_gridnode.sh -- "python train.py --config configs/semtalk_moclip_sparse.yaml --ddp false --gpus 0"
EOF
}

REMOTE_CMD=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gridmaster)
            GRIDMASTER="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --node)
            NODE="$2"
            shift 2
            ;;
        --project)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --no-host-print)
            PRINT_HOST=false
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        --)
            shift
            REMOTE_CMD="$*"
            break
            ;;
        *)
            REMOTE_CMD="$*"
            break
            ;;
    esac
done

SETUP_CMD="cd ~/${PROJECT_DIR}"
if [[ "$PRINT_HOST" == "true" ]]; then
    SETUP_CMD+=" && echo Connected host: \$(hostname)"
fi
SETUP_CMD+=" && if [ -f \"${VENV_PATH}\" ]; then source \"${VENV_PATH}\"; fi"
SETUP_CMD+=" && source ~/.bashrc 2>/dev/null || true"
SETUP_CMD+=" && source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true"
SETUP_CMD+=" && if command -v conda >/dev/null 2>&1; then conda activate ${CONDA_ENV} 2>/dev/null || true; fi"
SETUP_CMD+=" && cd ~/${PROJECT_DIR}"

echo "Connecting: ${GRIDMASTER} -> ${QUEUE}@${NODE} -> ~/${PROJECT_DIR}"

if [[ -n "$REMOTE_CMD" ]]; then
    REMOTE_PAYLOAD="${SETUP_CMD} && ${REMOTE_CMD}"
else
    REMOTE_PAYLOAD="${SETUP_CMD} && exec bash -l"
fi

REMOTE_QUOTED="$(printf '%q' "$REMOTE_PAYLOAD")"
ssh -t "${GRIDMASTER}" "qrsh -q ${QUEUE}@${NODE} bash -l -c ${REMOTE_QUOTED}"
