#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/semtalk_moclip_sparse.yaml"
BEAT2_DIR="BEAT2/beat_english_v2.0.0"
SPLIT="test"
EPOCHS=1
MASS_SCALE=0.30
MIN_WINDOW_FRAMES=15
BOOTSTRAP=300
MODE_LIST="arm_combined_core,split_arm_core,all_joints_core"
EXTRA_ARGS=""
PYTHON_BIN="${PYTHON_BIN:-}"
FORCE_RETRAIN=false
DETACH=true
LAUNCHER_LOG=""
TRAIN_BASE_JOINTLY=false
COUPLE_BASE_MASS_MODE=false
BASE_MASS_SCALE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --beat2_dir)
            BEAT2_DIR="$2"; shift 2 ;;
        --split)
            SPLIT="$2"; shift 2 ;;
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --mass_scale)
            MASS_SCALE="$2"; shift 2 ;;
        --min_window_frames)
            MIN_WINDOW_FRAMES="$2"; shift 2 ;;
        --bootstrap)
            BOOTSTRAP="$2"; shift 2 ;;
        --modes)
            MODE_LIST="$2"; shift 2 ;;
        --extra)
            EXTRA_ARGS="$2"; shift 2 ;;
        --python)
            PYTHON_BIN="$2"; shift 2 ;;
        --force_retrain)
            FORCE_RETRAIN=true; shift ;;
        --foreground)
            DETACH=false; shift ;;
        --detach)
            DETACH=true; shift ;;
        --launcher_log)
            LAUNCHER_LOG="$2"; shift 2 ;;
        --train_base_jointly)
            TRAIN_BASE_JOINTLY=true; shift ;;
        --couple_base_mass_mode)
            COUPLE_BASE_MASS_MODE=true; shift ;;
        --base_mass_scale)
            BASE_MASS_SCALE="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

CONFIG_STEM="$(basename "$CONFIG" .yaml)"

if [[ "$DETACH" == "true" ]] && [[ "${ABLATION_CHILD:-0}" != "1" ]]; then
    TS="$(date +%m%d_%H%M%S)"
    if [[ -z "$LAUNCHER_LOG" ]]; then
        LAUNCHER_LOG="outputs/ablations/${CONFIG_STEM}_weight_embedding_ablation_e${EPOCHS}_${TS}.log"
    fi
    mkdir -p "$(dirname "$LAUNCHER_LOG")"

    RELAUNCH=(
        bash "$0"
        --foreground
        --config "$CONFIG"
        --beat2_dir "$BEAT2_DIR"
        --split "$SPLIT"
        --epochs "$EPOCHS"
        --mass_scale "$MASS_SCALE"
        --min_window_frames "$MIN_WINDOW_FRAMES"
        --bootstrap "$BOOTSTRAP"
        --modes "$MODE_LIST"
    )
    if [[ -n "$PYTHON_BIN" ]]; then
        RELAUNCH+=(--python "$PYTHON_BIN")
    fi
    if [[ -n "$EXTRA_ARGS" ]]; then
        RELAUNCH+=(--extra "$EXTRA_ARGS")
    fi
    if [[ "$FORCE_RETRAIN" == "true" ]]; then
        RELAUNCH+=(--force_retrain)
    fi
    if [[ "$TRAIN_BASE_JOINTLY" == "true" ]]; then
        RELAUNCH+=(--train_base_jointly)
    fi
    if [[ "$COUPLE_BASE_MASS_MODE" == "true" ]]; then
        RELAUNCH+=(--couple_base_mass_mode)
    fi
    if [[ -n "$BASE_MASS_SCALE" ]]; then
        RELAUNCH+=(--base_mass_scale "$BASE_MASS_SCALE")
    fi

    nohup env ABLATION_CHILD=1 "${RELAUNCH[@]}" > "$LAUNCHER_LOG" 2>&1 < /dev/null &
    PID=$!
    PID_FILE="outputs/ablations/${CONFIG_STEM}_weight_embedding_ablation_e${EPOCHS}.pid"
    echo "$PID" > "$PID_FILE"

    echo "Ablation launched in background"
    echo "pid=$PID"
    echo "pid_file=$PID_FILE"
    echo "launcher_log=$LAUNCHER_LOG"
    exit 0
fi

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "/home/ferpaa/miniconda3/envs/semtalk/bin/python" ]]; then
        PYTHON_BIN="/home/ferpaa/miniconda3/envs/semtalk/bin/python"
    elif [[ -x "/home/ferpaa/anaconda3/envs/semtalk/bin/python" ]]; then
        PYTHON_BIN="/home/ferpaa/anaconda3/envs/semtalk/bin/python"
    else
        PYTHON_BIN="/home/ferpaa/SemTalk/.venv/bin/python"
    fi
fi

if ! "$PYTHON_BIN" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 8) else 1)" >/dev/null 2>&1; then
    echo "Python >=3.8 interpreter check failed: $PYTHON_BIN"
    exit 1
fi

if [[ -z "$BASE_MASS_SCALE" ]]; then
    BASE_MASS_SCALE="$MASS_SCALE"
fi

# Prevent ~/.local site-packages from shadowing the target environment.
export PYTHONNOUSERSITE=1

normalize_mode() {
    local mode="$1"
    mode="${mode//[[:space:]]/}"
    echo "$mode"
}

find_latest_epoch_with_generated() {
    local run_dir="$1"
    local max_epoch=0
    local d
    local ep
    while IFS= read -r d; do
        ep="$(basename "$d")"
        if [[ "$ep" =~ ^[0-9]+$ ]] && compgen -G "$d/res_*.npz" > /dev/null; then
            if (( ep > max_epoch )); then
                max_epoch=$ep
            fi
        fi
    done < <(find "$run_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
    echo "$max_epoch"
}

IFS=',' read -r -a MODES_RAW <<< "$MODE_LIST"
MODES=()
for raw_mode in "${MODES_RAW[@]}"; do
    mode="$(normalize_mode "$raw_mode")"
    if [[ -z "$mode" ]]; then
        continue
    fi
    case "$mode" in
        arm_combined_core|split_arm_core|all_joints_core|none)
            MODES+=("$mode") ;;
        *)
            echo "Invalid mode '$mode'. Allowed: arm_combined_core, split_arm_core, all_joints_core, none"
            exit 1 ;;
    esac
done

if [[ ${#MODES[@]} -eq 0 ]]; then
    echo "No valid modes requested. Use --modes with at least one valid entry."
    exit 1
fi

REPORTS=()

echo "Running weight-embedding ablation"
echo "config=$CONFIG split=$SPLIT epochs=$EPOCHS mass_scale=$MASS_SCALE"
echo "python=$PYTHON_BIN"
echo "detach=$DETACH"
echo "modes=${MODES[*]}"
echo "train_base_jointly=$TRAIN_BASE_JOINTLY"
echo "couple_base_mass_mode=$COUPLE_BASE_MASS_MODE base_mass_scale=$BASE_MASS_SCALE"

for MODE in "${MODES[@]}"; do
    NOTES="_abl_${MODE}_e${EPOCHS}"
    REPORT_PATH="outputs/ablations/${CONFIG_STEM}_${MODE}_e${EPOCHS}_generated_semantic_eval.json"

    echo ""
    echo "=== Mode: $MODE ==="

    RUN_DIR="$(ls -dt outputs/custom/*_${CONFIG_STEM}${NOTES} 2>/dev/null | head -n 1 || true)"
    GEN_GLOB=""
    if [[ -n "$RUN_DIR" ]]; then
        GEN_GLOB="${RUN_DIR}/${EPOCHS}/res_*.npz"
    fi

    START_EPOCH=1
    RESUME_CKPT=""
    if [[ "$FORCE_RETRAIN" != "true" ]] && [[ -n "$RUN_DIR" ]]; then
        LATEST_EPOCH="$(find_latest_epoch_with_generated "$RUN_DIR")"
        if [[ "$LATEST_EPOCH" =~ ^[0-9]+$ ]] && (( LATEST_EPOCH > 0 )) && (( LATEST_EPOCH < EPOCHS )); then
            CANDIDATE_CKPT="${RUN_DIR}/last_${LATEST_EPOCH}.bin"
            if [[ -f "$CANDIDATE_CKPT" ]]; then
                START_EPOCH=$((LATEST_EPOCH + 1))
                RESUME_CKPT="$CANDIDATE_CKPT"
                echo "Found partial run: epoch=${LATEST_EPOCH}; resuming from ${RESUME_CKPT}"
            fi
        fi
    fi

    if [[ "$FORCE_RETRAIN" == "true" ]] || [[ -z "$RUN_DIR" ]] || ! compgen -G "$GEN_GLOB" > /dev/null; then
        TRAIN_CMD=(
            "$PYTHON_BIN" train.py
            --config "$CONFIG" \
            --ddp false \
            --gpus 0 \
            --start_epoch "$START_EPOCH" \
            --epochs "$EPOCHS" \
            --test_period 1 \
            --mass_cond_mode "$MODE" \
            --mass_cond_scale "$MASS_SCALE" \
            --notes "$NOTES"
        )

        if [[ -n "$RESUME_CKPT" ]]; then
            TRAIN_CMD+=(--load_ckpt "$RESUME_CKPT")
        fi
        if [[ "$TRAIN_BASE_JOINTLY" == "true" ]]; then
            TRAIN_CMD+=(--joint_train_base_in_sparse true)
        fi
        if [[ "$COUPLE_BASE_MASS_MODE" == "true" ]]; then
            TRAIN_CMD+=(--base_mass_cond_mode "$MODE" --base_mass_cond_scale "$BASE_MASS_SCALE")
        fi
        if [[ -n "$EXTRA_ARGS" ]]; then
            # shellcheck disable=SC2206
            EXTRA_ARR=($EXTRA_ARGS)
            TRAIN_CMD+=("${EXTRA_ARR[@]}")
        fi

        "${TRAIN_CMD[@]}"
    else
        echo "Existing generated outputs found; skipping retrain for mode=$MODE"
    fi

    RUN_DIR="$(ls -dt outputs/custom/*_${CONFIG_STEM}${NOTES} 2>/dev/null | head -n 1 || true)"
    if [[ -z "$RUN_DIR" ]]; then
        echo "Could not locate run directory for mode=$MODE notes=$NOTES"
        exit 1
    fi

    GEN_GLOB="${RUN_DIR}/${EPOCHS}/res_*.npz"
    if ! compgen -G "$GEN_GLOB" > /dev/null; then
        echo "Generated files still missing for mode=$MODE epoch=$EPOCHS"
        exit 1
    fi

    echo "run_dir=$RUN_DIR"
    echo "generated_glob=$GEN_GLOB"

    "$PYTHON_BIN" utils/run_generated_semantic_beat_eval.py \
        --generated_glob "$GEN_GLOB" \
        --beat2_dir "$BEAT2_DIR" \
        --split "$SPLIT" \
        --min_window_frames "$MIN_WINDOW_FRAMES" \
        --bootstrap "$BOOTSTRAP" \
        --output "$REPORT_PATH"

    REPORTS+=("$REPORT_PATH")
done

SUMMARY_JSON="outputs/ablations/${CONFIG_STEM}_weight_embedding_ablation_e${EPOCHS}_summary.json"

"$PYTHON_BIN" - <<'PY' "${SUMMARY_JSON}" "${REPORTS[@]}"
import json
import os
import sys

summary_path = sys.argv[1]
reports = sys.argv[2:]
rows = []
for rp in reports:
    if not os.path.exists(rp):
        continue
    with open(rp, "r") as f:
        d = json.load(f)
    base = os.path.basename(rp).replace("_generated_semantic_eval.json", "")
    mode = base
    for candidate in ("arm_combined_core", "split_arm_core", "all_joints_core", "none"):
        if candidate in base:
            mode = candidate
            break

    beat = d["category_reports"]["beat"]["metrics"]
    sem = d["category_reports"]["semantic"]["metrics"]

    row = {
        "mode": mode,
        "beat_peak_hz_dw": beat["shoulder_peak_hz"]["duration_weighted"]["mean"],
        "sem_peak_hz_dw": sem["shoulder_peak_hz"]["duration_weighted"]["mean"],
        "beat_plv_dw": beat["plv"]["duration_weighted"]["mean"],
        "sem_plv_dw": sem["plv"]["duration_weighted"]["mean"],
        "beat_delta_arm_combined_dw": beat["delta_r2_arm_combined_vs_shoulder"]["duration_weighted"]["mean"],
        "sem_delta_arm_combined_dw": sem["delta_r2_arm_combined_vs_shoulder"]["duration_weighted"]["mean"],
        "combined_recommendation": d.get("feature_group_recommendation", {}).get("combined", {}).get("recommended_feature_group"),
        "report_path": rp,
    }
    rows.append(row)

out = {
    "num_reports": len(rows),
    "rows": rows,
}

os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w") as f:
    json.dump(out, f, indent=2)

print("Saved ablation summary to {}".format(summary_path))
for r in rows:
    print(r)
PY

echo ""
echo "Ablation complete"
echo "summary=$SUMMARY_JSON"
