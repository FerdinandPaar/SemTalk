#!/bin/bash
set -euo pipefail

# ============================================================================
# SemTalk 10-run S-VIB + Physics fine-tune sweep — Round 2, r05-base
#
# Base: r05_fb030 best_146.bin  (vib_beta=0.010, vib_free=0.30, phys_lambda=0.08,
#                                 tau_base=0.50, tau_floor=0.10)
#
# New 10-run sensitivity matrix around r05 winner:
#   s01  base: exact r05 params                 → confirms round-2 baseline
#   s02  vib_free  0.25  (between 0.20 and 0.30)→ does tighter free-bits still help?
#   s03  vib_free  0.40  (looser than r05)       → explore upper end
#   s04  vib_beta  0.005 (×0.5 of r05 beta)
#   s05  vib_beta  0.020 (×2 of r05 beta)
#   s06  phys_lambda 0.05 (lighter physics)
#   s07  phys_lambda 0.12 (heavier physics)
#   s08  tau_base  0.35  (light EMA — most diverse gestures in round 1)
#   s09  tau_base  0.65  (heavy EMA)
#   s10  tau_floor 0.05  (tighter floor)
#
# Usage:
#   bash run_sweep_svib_phys_10_r05base.sh
#
# Optional env overrides (same as run_sweep_svib_phys_10.sh):
#   SWEEP_NAME, RESUME_SWEEP_ROOT, BASE_CKPT, START_EPOCH, END_EPOCH,
#   START_RUN_IDX, END_RUN_IDX, TRAIN_MODE, DDP_NPROC, VALIDATE_DEVICE,
#   BRANCH, PUSH_EVERY, PUSH_CHECKPOINTS, RUN_STAT, WANDB_API_KEY,
#   AUTO_DETACH, EMAIL_TO, EMAIL_FROM, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
# ============================================================================

ROOT_DIR="/home/ferpaa/SemTalk"
cd "$ROOT_DIR"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

PYTHON="/home/ferpaa/miniconda3/envs/semtalk/bin/python"

SWEEP_NAME="${SWEEP_NAME:-svib_phys_10runs_r05base_v1}"
RESUME_SWEEP_ROOT="${RESUME_SWEEP_ROOT:-}"
BASE_CONFIG="${BASE_CONFIG:-configs/semtalk_moclip_sparse_ft.yaml}"
BASE_CKPT="${BASE_CKPT:-outputs/custom/0310_073539_r05_fb030_sw10_r05_fb030_b0.010_fb0.30_pl0.08_tb0.50_tf0.10/best_146.bin}"
START_EPOCH="${START_EPOCH:-146}"
END_EPOCH="${END_EPOCH:-186}"
START_RUN_IDX="${START_RUN_IDX:-1}"
END_RUN_IDX="${END_RUN_IDX:-10}"
TRAIN_MODE="${TRAIN_MODE:-single}"
SINGLE_GPU="${SINGLE_GPU:-0}"
DDP_NPROC="${DDP_NPROC:-4}"
VALIDATE_DEVICE="${VALIDATE_DEVICE:-cpu}"
BRANCH="${BRANCH:-feature/physics-smoother-svib}"
PUSH_EVERY="${PUSH_EVERY:-1}"
PUSH_CHECKPOINTS="${PUSH_CHECKPOINTS:-0}"
RUN_STAT="${RUN_STAT:-wandb}"
AUTO_DETACH="${AUTO_DETACH:-1}"
EMAIL_TO="${EMAIL_TO:-}"
EMAIL_FROM="${EMAIL_FROM:-}"
SMTP_HOST="${SMTP_HOST:-}"
SMTP_PORT="${SMTP_PORT:-587}"
SMTP_USER="${SMTP_USER:-}"
SMTP_PASS="${SMTP_PASS:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$RESUME_SWEEP_ROOT" ]]; then
  SWEEP_ROOT="$RESUME_SWEEP_ROOT"
else
  SWEEP_ROOT="outputs/sweeps/${STAMP}_${SWEEP_NAME}"
fi
CFG_DIR="${SWEEP_ROOT}/configs"
SAFE_DIR="${SWEEP_ROOT}/safe_models"
LOG_DIR="${SWEEP_ROOT}/logs"
VAL_DIR="${SWEEP_ROOT}/validation"
SUMMARY_CSV="${SWEEP_ROOT}/summary.csv"
SUMMARY_MD="${SWEEP_ROOT}/summary.md"
DEVLOG="models/DEVELOPMENT_LOG.md"

mkdir -p "$CFG_DIR" "$SAFE_DIR" "$LOG_DIR" "$VAL_DIR"

MASTER_LOG="${LOG_DIR}/orchestrator.log"
PID_FILE="${SWEEP_ROOT}/orchestrator.pid"

# Auto-detach so the sweep survives terminal closure.
if [[ "${AUTO_DETACH}" == "1" ]] && [[ -z "${SWEEP_LAUNCHED:-}" ]] && [[ -z "${STY:-}" ]] && [[ -z "${TMUX:-}" ]]; then
  export SWEEP_LAUNCHED=1
  { echo ""; echo "=== RESTART $(date '+%Y-%m-%d %H:%M:%S') ==="; } >> "$MASTER_LOG" 2>/dev/null || true
  nohup bash "$SCRIPT_PATH" "$@" >> "$MASTER_LOG" 2>&1 &
  bg_pid=$!
  echo "$bg_pid" > "$PID_FILE"
  echo "[LAUNCHED] Detached sweep started: pid=${bg_pid}"
  echo "[LAUNCHED] Master log: ${MASTER_LOG}"
  echo "[LAUNCHED] PID file  : ${PID_FILE}"
  exit 0
fi

if [[ "${SWEEP_LAUNCHED:-0}" == "1" ]]; then
  echo $$ > "$PID_FILE"
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[ERROR] BASE_CONFIG not found: $BASE_CONFIG"
  exit 1
fi
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[ERROR] BASE_CKPT not found: $BASE_CKPT"
  exit 1
fi
if (( START_RUN_IDX < 1 || START_RUN_IDX > 10 || END_RUN_IDX < 1 || END_RUN_IDX > 10 || START_RUN_IDX > END_RUN_IDX )); then
  echo "[ERROR] Invalid run window: START_RUN_IDX=${START_RUN_IDX}, END_RUN_IDX=${END_RUN_IDX} (expected 1..10 and start<=end)"
  exit 1
fi

if [[ "$RUN_STAT" == "wandb" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
  fi

  if ! "$PYTHON" - <<'PY'
import sys
try:
    import wandb
    key = getattr(wandb.api, 'api_key', None)
    sys.exit(0 if key else 1)
except Exception:
    sys.exit(1)
PY
  then
    echo "[ERROR] RUN_STAT=wandb but no non-interactive W&B login is available."
    echo "        Set WANDB_API_KEY before launching, e.g.:"
    echo "        WANDB_API_KEY=... RUN_STAT=wandb bash run_sweep_svib_phys_10_r05base.sh"
    exit 1
  fi
fi

GIT_BIN=""
for candidate in /usr/bin/git /bin/git "$(command -v git 2>/dev/null || true)"; do
  [[ -z "$candidate" ]] && continue
  [[ ! -x "$candidate" ]] && continue
  if "$candidate" --version >/dev/null 2>&1; then
    GIT_BIN="$candidate"
    break
  fi
done

append_devlog() {
  local msg="$1"
  {
    echo ""
    echo "$msg"
  } >> "$DEVLOG"
}

send_update_mail() {
  local subject="$1"
  local body="$2"

  if [[ -z "$EMAIL_TO" ]]; then
    return 0
  fi

  if command -v mail >/dev/null 2>&1; then
    printf "%s\n" "$body" | mail -s "$subject" "$EMAIL_TO" || true
    return 0
  fi

  if [[ -n "$SMTP_HOST" ]]; then
    "$PYTHON" - <<PY || true
import smtplib
from email.message import EmailMessage

subject = """$subject"""
body = """$body"""
to_addr = """$EMAIL_TO"""
from_addr = """${EMAIL_FROM:-$EMAIL_TO}"""
host = """$SMTP_HOST"""
port = int("$SMTP_PORT")
user = """$SMTP_USER"""
password = """$SMTP_PASS"""

msg = EmailMessage()
msg["Subject"] = subject
msg["From"] = from_addr
msg["To"] = to_addr
msg.set_content(body)

with smtplib.SMTP(host, port, timeout=20) as s:
    s.ehlo()
    try:
        s.starttls()
        s.ehlo()
    except Exception:
        pass
    if user:
        s.login(user, password)
    s.send_message(msg)
PY
    return 0
  fi

  echo "[WARN] EMAIL_TO is set (${EMAIL_TO}) but no mail transport is configured." >&2
  echo "[WARN] Install local 'mail' command or set SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS." >&2
}

git_push_metadata() {
  local label="$1"
  if [[ -z "$GIT_BIN" ]]; then
    echo "[WARN] git not found; skipping push for $label"
    return 0
  fi

  set +e
  "$GIT_BIN" -C "$ROOT_DIR" checkout "$BRANCH" >/dev/null 2>&1
  "$GIT_BIN" -C "$ROOT_DIR" add "$CFG_DIR" "$LOG_DIR" "$VAL_DIR" "$SUMMARY_CSV" "$SUMMARY_MD" "$DEVLOG" >/dev/null 2>&1
  if [[ "$PUSH_CHECKPOINTS" == "1" ]]; then
    "$GIT_BIN" -C "$ROOT_DIR" add -f "$SAFE_DIR"/*.bin >/dev/null 2>&1
  fi
  if ! "$GIT_BIN" -C "$ROOT_DIR" diff --cached --quiet; then
    "$GIT_BIN" -C "$ROOT_DIR" commit -m "sweep(${SWEEP_NAME}): ${label}" >/dev/null 2>&1
    "$GIT_BIN" -C "$ROOT_DIR" push origin "$BRANCH" >/dev/null 2>&1
    echo "[GIT] pushed metadata: $label"
  else
    echo "[GIT] no staged metadata changes for: $label"
  fi
  set -e
}

extract_last_metric() {
  local pattern="$1"
  local file="$2"
  grep "$pattern" "$file" | tail -1 | awk -F': ' '{print $2}' | tr -d ' ' \
    || true
}

best_ckpt_from_run_dir() {
  local run_dir="$1"
  ls "$run_dir"/best_*.bin 2>/dev/null | sort -V | tail -1
}

find_run_dir_for_run() {
  local run_id="$1"
  local notes="$2"
  local run_log="$3"
  local run_dir=""

  run_dir=$(ls -td outputs/custom/*"${notes}"* 2>/dev/null | head -1 || true)
  if [[ -n "$run_dir" ]]; then echo "$run_dir"; return 0; fi

  run_dir=$(ls -td outputs/custom/*"_${run_id}_"*"${notes}"* 2>/dev/null | head -1 || true)
  if [[ -n "$run_dir" ]]; then echo "$run_dir"; return 0; fi

  run_dir=$(ls -td outputs/custom/*"_${run_id}_"* 2>/dev/null | head -1 || true)
  if [[ -n "$run_dir" ]]; then echo "$run_dir"; return 0; fi

  run_dir=$(grep -Eo 'outputs/custom/[^ ]+' "$run_log" 2>/dev/null | tail -1 || true)
  if [[ -n "$run_dir" ]] && [[ -d "$run_dir" ]]; then echo "$run_dir"; return 0; fi

  echo ""
  return 0
}

# ---------------------------------------------------------------------------
# Round-2 sensitivity matrix — centred on r05 winner
# (vib_beta=0.010, vib_free=0.30, phys_lambda=0.08, tau_base=0.50, tau_floor=0.10)
#
# Columns: id  vib_beta  vib_free  phys_lambda  phys_tau_base  phys_tau_floor
# ---------------------------------------------------------------------------
MATRIX=$(cat <<'EOF'
s01_base_r05        0.010 0.30 0.08 0.50 0.10
s02_fb025           0.010 0.25 0.08 0.50 0.10
s03_fb040           0.010 0.40 0.08 0.50 0.10
s04_b005            0.005 0.30 0.08 0.50 0.10
s05_b020            0.020 0.30 0.08 0.50 0.10
s06_pl005           0.010 0.30 0.05 0.50 0.10
s07_pl012           0.010 0.30 0.12 0.50 0.10
s08_tb035           0.010 0.30 0.08 0.35 0.10
s09_tb065           0.010 0.30 0.08 0.65 0.10
s10_tf005           0.010 0.30 0.08 0.50 0.05
EOF
)

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "run_id,run_dir,best_ckpt,best_epoch,fgd_subset,fid,bc,l1div,status" > "$SUMMARY_CSV"
fi

append_devlog "---

## [$(date +%Y-%m-%d)] Auto Sweep Started — ${SWEEP_NAME} (Round 2 / r05-base)

- Start time   : $(date '+%Y-%m-%d %H:%M:%S')
- Base checkpoint: ${BASE_CKPT}  (r05 best_146, vib_free=0.30 winner)
- Epoch schedule : start_epoch=${START_EPOCH}, end_epoch=${END_EPOCH}  (40 epochs)
- Train mode   : ${TRAIN_MODE}
- Detached mode: ${AUTO_DETACH} (master log: ${MASTER_LOG})
- Email updates: ${EMAIL_TO:-disabled}"

send_update_mail \
  "[SemTalk Sweep R2] Started: ${SWEEP_NAME}" \
  "Round-2 sweep started on $(hostname)\n\nName: ${SWEEP_NAME}\nBase ckpt: ${BASE_CKPT}\nEpochs: ${START_EPOCH} -> ${END_EPOCH}\nMode: ${TRAIN_MODE}\nMaster log: ${MASTER_LOG}\n"

matrix_idx=0
run_count=0
while read -r run_id vib_beta vib_free phys_lambda tau_base tau_floor; do
  [[ -z "$run_id" ]] && continue
  matrix_idx=$((matrix_idx+1))

  if (( matrix_idx < START_RUN_IDX || matrix_idx > END_RUN_IDX )); then
    continue
  fi

  partial_resume_ckpt=""
  partial_resume_start_epoch=""
  if grep -q "^${run_id}," "$SUMMARY_CSV"; then
    prev_status=$(grep "^${run_id}," "$SUMMARY_CSV" | tail -1 | awk -F, '{print $9}')
    if [[ "$prev_status" == "TRAINED" || "$prev_status" == "VALIDATED" ]]; then
      echo "[SKIP] ${run_id} already recorded as ${prev_status} in summary"
      continue
    elif [[ "$prev_status" == "TRAINED_PARTIAL" ]]; then
      partial_resume_ckpt=$(grep "^${run_id}," "$SUMMARY_CSV" | tail -1 | awk -F, '{print $3}')
      partial_epoch_num=$(grep "^${run_id}," "$SUMMARY_CSV" | tail -1 | awk -F, '{print $4}')
      if [[ -f "$partial_resume_ckpt" && -n "$partial_epoch_num" ]]; then
        partial_resume_start_epoch=$((partial_epoch_num + 1))
        echo "[RESUME] ${run_id}: TRAINED_PARTIAL at epoch ${partial_epoch_num}, resuming from epoch ${partial_resume_start_epoch}"
      else
        echo "[WARN] ${run_id}: TRAINED_PARTIAL but checkpoint not found (${partial_resume_ckpt}), restarting from base"
        partial_resume_ckpt=""
        partial_resume_start_epoch=""
      fi
    fi
    awk -F, -v OFS=, -v rid="$run_id" 'NR==1 || $1!=rid {print}' "$SUMMARY_CSV" > "${SUMMARY_CSV}.tmp"
    mv "${SUMMARY_CSV}.tmp" "$SUMMARY_CSV"
  fi

  run_count=$((run_count+1))

  notes="_r05base_${run_id}"
  notes="${notes}_b${vib_beta}_fb${vib_free}_pl${phys_lambda}_tb${tau_base}_tf${tau_floor}"
  cfg_path="${CFG_DIR}/${run_id}.yaml"
  run_log="${LOG_DIR}/${run_id}_train.log"

  echo "================================================================"
  echo "[TRAIN ${run_count}/$((END_RUN_IDX-START_RUN_IDX+1))] ${run_id} (matrix_idx=${matrix_idx})"
  echo "================================================================"

  cp -f "$BASE_CONFIG" "$cfg_path"
  cat >> "$cfg_path" <<EOF

# --- sweep overrides (${SWEEP_NAME}/${run_id}) ---
load_ckpt: ${partial_resume_ckpt:-${BASE_CKPT}}
start_epoch: ${partial_resume_start_epoch:-${START_EPOCH}}
epochs: ${END_EPOCH}

vib_enabled: true
phys_enabled: true

vib_beta_target: ${vib_beta}
vib_free_bits: ${vib_free}
phys_lambda: ${phys_lambda}
phys_tau_base: ${tau_base}
phys_tau_floor: ${tau_floor}
EOF
  echo "[OK] wrote ${cfg_path}"

  # Run training — capture exit code so set -e does NOT kill the orchestrator.
  train_exit=0
  if [[ "$TRAIN_MODE" == "ddp" ]]; then
    GPU_IDS=$(seq 0 $((DDP_NPROC-1)) | tr '\n' ' ')
    CUDA_MAP=$(seq 0 $((DDP_NPROC-1)) | tr '\n' ',' | sed 's/,$//')
    read -r -a gpu_arr <<< "$GPU_IDS"
    cmd=(
      "$PYTHON" -m torch.distributed.run
      --nproc_per_node="$DDP_NPROC"
      --master_addr=127.0.0.1
      --master_port=$((29710 + run_count))
      train.py
      --config "$cfg_path"
      --ddp True
      --gpus "${gpu_arr[@]}"
      --project "semtalk_svib_phys_sweep_r2"
      --stat "$RUN_STAT"
      --wandb_name "${SWEEP_NAME}_${run_id}"
      --notes "$notes"
    )
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29710 + run_count)) CUDA_VISIBLE_DEVICES="$CUDA_MAP" "${cmd[@]}" > "$run_log" 2>&1 || train_exit=$?
  else
    cmd=(
      "$PYTHON" train.py
      --config "$cfg_path"
      --ddp False
      --gpus 0
      --project "semtalk_svib_phys_sweep_r2"
      --stat "$RUN_STAT"
      --wandb_name "${SWEEP_NAME}_${run_id}"
      --notes "$notes"
    )
    CUDA_VISIBLE_DEVICES="$SINGLE_GPU" "${cmd[@]}" > "$run_log" 2>&1 || train_exit=$?
  fi
  if [[ $train_exit -ne 0 ]]; then
    echo "[WARN] ${run_id}: training process exited with code ${train_exit} (SIGKILL=137, SIGTERM=143, crash=1) — scanning for checkpoints"
  fi

  # Locate run directory.
  run_dir=$(find_run_dir_for_run "$run_id" "$notes" "$run_log")
  if [[ -z "$run_dir" ]]; then
    echo "[ERROR] could not find run_dir for ${run_id}" | tee -a "$run_log"
    echo "${run_id},,,-1,,,,,,FAILED_NO_RUNDIR" >> "$SUMMARY_CSV"
    append_devlog "- ${run_id}: FAILED (run_dir not found)"
    (( run_count % PUSH_EVERY == 0 )) && git_push_metadata "run ${run_id} failed (no run_dir)"
    continue
  fi

  # Safety archive.
  run_safe_dir="${SAFE_DIR}/${run_id}"
  mkdir -p "$run_safe_dir"
  cp -f "$cfg_path" "$run_safe_dir/config.yaml"
  cp -f "$run_log"  "$run_safe_dir/train.log"
  cp -f "$run_dir"/*.txt  "$run_safe_dir/" 2>/dev/null || true
  cp -f "$run_dir"/*.yaml "$run_safe_dir/" 2>/dev/null || true
  cp -f "$run_dir"/best_*.bin "$run_safe_dir/" 2>/dev/null || true
  cp -f "$run_dir"/last_*.bin "$run_safe_dir/" 2>/dev/null || true

  best_ckpt="$(best_ckpt_from_run_dir "$run_dir")"
  if [[ -z "$best_ckpt" ]]; then
    best_ckpt=$(ls "$run_dir"/last_*.bin 2>/dev/null | sort -V | tail -1 || true)
  fi

  best_epoch=""
  if [[ -n "$best_ckpt" ]]; then
    best_epoch="$(basename "$best_ckpt" | sed -E 's/[^0-9]*([0-9]+)\.bin/\1/')"
  fi

  # Determine status: TRAINED | TRAINED_PARTIAL | FAILED_KILLED
  run_status="TRAINED"
  if [[ $train_exit -ne 0 ]]; then
    wandb_synced=$(grep -c "wandb: Synced" "$run_log" 2>/dev/null || echo 0)
    if [[ -f "${run_dir}/last_${END_EPOCH}.bin" || "${wandb_synced}" -gt 0 ]]; then
      run_status="TRAINED"
      echo "[INFO] ${run_id}: exit ${train_exit} but epoch ${END_EPOCH} checkpoint found — TRAINED"
    elif [[ -n "$best_ckpt" ]]; then
      last_ckpt_path=$(ls "$run_dir"/last_*.bin 2>/dev/null | sort -V | tail -1 || true)
      if [[ -n "$last_ckpt_path" ]]; then
        last_ckpt_epoch="$(basename "$last_ckpt_path" | sed -E 's/[^0-9]*([0-9]+)\.bin/\1/')"
        best_ckpt="$last_ckpt_path"
        best_epoch="$last_ckpt_epoch"
      fi
      run_status="TRAINED_PARTIAL"
      echo "[WARN] ${run_id}: interrupted (exit ${train_exit}), last epoch=${best_epoch} — TRAINED_PARTIAL (will auto-resume)"
    else
      run_status="FAILED_KILLED"
      echo "[ERROR] ${run_id}: killed (exit ${train_exit}), no checkpoints — FAILED_KILLED"
    fi
  fi

  echo "${run_id},${run_dir},${best_ckpt},${best_epoch},,,,,$run_status" >> "$SUMMARY_CSV"
  append_devlog "- ${run_id}: ${run_status}. run_dir=${run_dir}, best_ckpt=${best_ckpt}"
  send_update_mail \
    "[SemTalk Sweep R2] TRAIN done: ${run_id} (${run_status})" \
    "Run ${run_id} finished with status ${run_status}.\nRun dir: ${run_dir}\nBest ckpt: ${best_ckpt}\n\nSweep: ${SWEEP_NAME}\n"

  (( run_count % PUSH_EVERY == 0 )) && git_push_metadata "run ${run_id} training completed"

done <<< "$MATRIX"

# ---------------------------------------------------------------------------
# Post-training validation phase
# ---------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "[VALIDATION] All training runs finished. Starting inference + FGD..."
echo "================================================================"

while IFS=, read -r run_id run_dir best_ckpt best_epoch fgd_old fid_old bc_old l1div_old status; do
  [[ "$run_id" == "run_id" ]] && continue
  [[ -z "$run_id" ]] && continue

  if [[ "$status" == "TRAINED_PARTIAL" || "$status" == "FAILED_KILLED" || "$status" == "FAILED_NO_RUNDIR" ]]; then
    echo "[SKIP-VAL] ${run_id}: status=${status} — not fully trained, skipping validation"
    continue
  fi

  if [[ ! -f "$best_ckpt" ]]; then
    echo "[WARN] ${run_id}: best_ckpt missing, skipping validation"
    continue
  fi

  cfg_path="${CFG_DIR}/${run_id}.yaml"
  infer_stem="demo/2_scott_0_1_1_${SWEEP_NAME}_${run_id}"
  val_log="${VAL_DIR}/${run_id}_validation.log"

  echo "[VAL] ${run_id}  ckpt=$(basename "$best_ckpt")"

  MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29810 + ${best_epoch:-0} % 100)) \
  "$PYTHON" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=$((29810 + ${best_epoch:-0} % 100)) \
    train.py \
      --config "$cfg_path" \
      --inference \
      --load_ckpt "$best_ckpt" \
      --audio_infer_path demo/2_scott_0_1_1_test.wav \
      --out_name "$infer_stem" \
      > "$val_log" 2>&1

  fgd_line=$(
    "$PYTHON" utils/run_fgd_eval.py --epoch_dir "${run_dir}/${best_epoch}" --device "$VALIDATE_DEVICE" 2>&1 | \
    tee -a "$val_log" | grep "FGD (SemTalk FIDCalculator)" | tail -1 || true
  )
  fgd_val=$(echo "$fgd_line" | awk -F':' '{print $2}' | tr -d ' ')

  run_txt=$(ls "$run_dir"/*.txt 2>/dev/null | head -1 || true)
  if [[ -n "$run_txt" ]]; then
    fid_val=$(extract_last_metric "fid score"   "$run_txt")
    bc_val=$(extract_last_metric  "align score" "$run_txt")
    l1div_val=$(extract_last_metric "l1div score" "$run_txt")
  else
    fid_val=""; bc_val=""; l1div_val=""
  fi

  awk -F, -v OFS=, -v rid="$run_id" -v fgd="$fgd_val" -v fid="$fid_val" -v bc="$bc_val" -v l1="$l1div_val" '
    NR==1 {print; next}
    $1==rid {$5=fgd; $6=fid; $7=bc; $8=l1; $9="VALIDATED"}
    {print}
  ' "$SUMMARY_CSV" > "${SUMMARY_CSV}.tmp"
  mv "${SUMMARY_CSV}.tmp" "$SUMMARY_CSV"

  append_devlog "- ${run_id}: VALIDATED. best_epoch=${best_epoch}, fgd=${fgd_val}, fid=${fid_val}, bc=${bc_val}, l1div=${l1div_val}"
  send_update_mail \
    "[SemTalk Sweep R2] VALIDATED: ${run_id} | FGD ${fgd_val}" \
    "Run ${run_id} validated.\nBest epoch: ${best_epoch}\nFGD: ${fgd_val}\nFID: ${fid_val}\nBC: ${bc_val}\nL1DIV: ${l1div_val}\n\nSweep: ${SWEEP_NAME}\n"

  git_push_metadata "run ${run_id} validated"
done < "$SUMMARY_CSV"

# Build markdown summary sorted by FGD
{
  echo "# ${SWEEP_NAME} — 10-run summary (Round 2 / r05-base)"
  echo ""
  echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
  echo "| run_id | best_epoch | FGD | fid | bc | l1div | status |"
  echo "|---|---:|---:|---:|---:|---:|---|"
  tail -n +2 "$SUMMARY_CSV" | sort -t, -k5,5g | while IFS=, read -r run_id run_dir best_ckpt best_epoch fgd fid bc l1div status; do
    echo "| ${run_id} | ${best_epoch} | ${fgd} | ${fid} | ${bc} | ${l1div} | ${status} |"
  done
} > "$SUMMARY_MD"

append_devlog "
### Sweep Final Summary (${SWEEP_NAME})
- Summary CSV: ${SUMMARY_CSV}
- Summary MD : ${SUMMARY_MD}
- End time   : $(date '+%Y-%m-%d %H:%M:%S')
"

best_line=$(tail -n +2 "$SUMMARY_CSV" | sort -t, -k5,5g | head -1)
best_run=$(echo "$best_line" | awk -F, '{print $1}')
best_fgd=$(echo "$best_line" | awk -F, '{print $5}')

send_update_mail \
  "[SemTalk Sweep R2] COMPLETE: ${SWEEP_NAME}" \
  "Sweep completed.\nName: ${SWEEP_NAME}\nBest run: ${best_run}\nBest FGD: ${best_fgd}\n\nSummary CSV: ${SUMMARY_CSV}\nSummary MD: ${SUMMARY_MD}\n"

git_push_metadata "final sweep summary ${SWEEP_NAME}"

echo ""
echo "================================================================"
echo "Sweep completed: ${SWEEP_NAME}"
echo "Summary: ${SUMMARY_CSV}"
echo "Markdown: ${SUMMARY_MD}"
echo "Safe models: ${SAFE_DIR}"
echo "================================================================"
