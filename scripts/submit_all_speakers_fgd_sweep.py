#!/usr/bin/env python3
"""
Submit a multi-model SemTalk base sweep that trains/tests on all speakers and
optionally re-runs FGD after training for each model.

Key features:
- Auto-detect target speakers from train_test_split.csv (default: all TEST speakers).
- Launch one model per GPU in a single qsub job (default: 4 models on 4 GPUs).
- Estimate runtime per model by scaling from a reference log.
- Optional post-run FGD re-eval for each model's best epoch.

Example (default 4-model hypothesis, submit):
  python scripts/submit_all_speakers_fgd_sweep.py \
      --config configs/semtalk_base.yaml \
      --seed-ckpt outputs/custom/0404_123353_semtalk_base_0404_123320_masshyp_on_l003_s75_semtalk_base/best_10.bin

Example (custom variants, dry run only):
  python scripts/submit_all_speakers_fgd_sweep.py \
      --dry-run \
      --variant "off_s70::--base_mass_cond_scale 0.70 --base_mass_cond_gate_init -1.2 --base_phys_enabled false" \
      --variant "on_l003_s75::--base_mass_cond_scale 0.75 --base_mass_cond_gate_init -1.0 --base_phys_enabled true --base_phys_lambda 0.003"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_CSV = ROOT / "BEAT2" / "beat_english_v2.0.0" / "train_test_split.csv"
DEFAULT_CONFIG = ROOT / "configs" / "semtalk_base.yaml"
DEFAULT_CONDA_ENV = "semtalk"
DEFAULT_QUEUE = "mld.q@gridnode016"
DEFAULT_GPUS = "0,1,2,3"
DEFAULT_SLOTS = 4
DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_PORT_BASE = 9100


@dataclass
class Variant:
    name: str
    extra_args: List[str]


def default_variants() -> List[Variant]:
    return [
        Variant(
            "allspk_off_s70_gi12",
            [
                "--base_mass_cond_scale", "0.70",
                "--base_mass_cond_gate_init", "-1.2",
                "--base_phys_enabled", "false",
            ],
        ),
        Variant(
            "allspk_off_s75_gi10",
            [
                "--base_mass_cond_scale", "0.75",
                "--base_mass_cond_gate_init", "-1.0",
                "--base_phys_enabled", "false",
            ],
        ),
        Variant(
            "allspk_off_s85_gi08",
            [
                "--base_mass_cond_scale", "0.85",
                "--base_mass_cond_gate_init", "-0.8",
                "--base_phys_enabled", "false",
            ],
        ),
        Variant(
            "allspk_on_l003_s70",
            [
                "--base_mass_cond_scale", "0.70",
                "--base_mass_cond_gate_init", "-1.2",
                "--base_phys_enabled", "true",
                "--base_phys_lambda", "0.003",
                "--base_phys_tau_base", "0.50",
                "--base_phys_tau_floor", "0.10",
                "--base_phys_alpha", "1.00",
                "--base_phys_warmup_start", "1",
                "--base_phys_warmup_end", "30",
            ],
        ),
    ]


def parse_variant_specs(raw_specs: Sequence[str]) -> List[Variant]:
    if not raw_specs:
        return default_variants()

    variants: List[Variant] = []
    for spec in raw_specs:
        if "::" not in spec:
            raise ValueError(
                f"Invalid --variant '{spec}'. Use format: name::--arg1 v1 --arg2 v2"
            )
        name, extra = spec.split("::", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --variant '{spec}': empty name")
        extra_args = shlex.split(extra.strip()) if extra.strip() else []
        variants.append(Variant(name=name, extra_args=extra_args))
    return variants


def read_split_rows(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_speaker_id(sample_id: str) -> int:
    return int(sample_id.split("_", 1)[0])


def select_target_speakers(rows: Sequence[dict], source: str) -> List[int]:
    speakers = set()
    for row in rows:
        if source == "test" and row.get("type") != "test":
            continue
        sid = parse_speaker_id(row["id"])
        speakers.add(sid)
    return sorted(speakers)


def count_rows(rows: Sequence[dict], row_type: str, speakers: Sequence[int]) -> int:
    sset = set(speakers)
    c = 0
    for row in rows:
        if row.get("type") != row_type:
            continue
        sid = parse_speaker_id(row["id"])
        if sid in sset:
            c += 1
    return c


def parse_last_sane_time_info_minutes(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None

    # Example:
    # Time info >>>>  elapsed: 134.50 mins  remain: 441.93 mins
    pattern = re.compile(r"elapsed:\s*([0-9.]+)\s*mins.*remain:\s*([0-9.]+)\s*mins")
    best: Optional[float] = None
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            elapsed = float(m.group(1))
            remain = float(m.group(2))
            # Ignore early pathological estimates that show absurd remaining time.
            if remain > 10000.0:
                continue
            best = elapsed + remain
    return best


def pick_default_reference_log() -> Optional[Path]:
    candidates = [
        ROOT / "outputs" / "0404_123320_masshyp_off_s70_gi12_semtalk_base.log",
        ROOT / "outputs" / "0404_002632_massphys_s75_gi10_l005_semtalk_base.log",
        ROOT / "outputs" / "0403_135254_physon_hp_l005_semtalk_base.log",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def shell_quote_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def build_job_script(
    job_name: str,
    queue: str,
    slots: int,
    root_dir: Path,
    output_dir: Path,
    conda_env: str,
    config_path: Path,
    seed_ckpt: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    speaker_ids: Sequence[int],
    variants: Sequence[Variant],
    gpu_ids: Sequence[str],
    port_base: int,
    run_fgd_after: bool,
) -> str:
    lines: List[str] = []
    lines.extend(
        [
            "#!/bin/bash",
            "#$ -S /bin/bash",
            f"#$ -N {job_name}",
            f"#$ -q {queue}",
            f"#$ -pe smp {slots}",
            "#$ -cwd",
            f"#$ -o {output_dir}/qsub_allspeaker.log",
            f"#$ -e {output_dir}/qsub_allspeaker.err",
            "#$ -j y",
            "",
            "set -euo pipefail",
            "",
            "source /home/ferpaa/miniconda3/etc/profile.d/conda.sh",
            f"conda activate {conda_env}",
            "",
            f"cd {root_dir}",
            "",
            "export PYTHONNOUSERSITE=1",
            "export WANDB_MODE=online",
            "export NCCL_IB_DISABLE=1",
            "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
            "",
            f"BASE_CKPT={shlex.quote(str(seed_ckpt))}",
            "TS=$(date +%m%d_%H%M%S)",
            "TRAIN_SPEAKERS=(" + " ".join(str(s) for s in speaker_ids) + ")",
            "",
            "COMMON_ARGS=(",
            f"  --config {shlex.quote(str(config_path))}",
            "  --trainer semtalk_base",
            "  --ddp false",
            "  --gpus 0",
            f"  --batch_size {batch_size}",
            f"  --num_workers {num_workers}",
            f"  --epochs {epochs}",
            "  --load_ckpt \"$BASE_CKPT\"",
            "  --speaker_id onehot",
            "  --training_speakers \"${TRAIN_SPEAKERS[@]}\"",
            "  --base_mass_cond_mode arm_combined_core",
            "  --mass_cond_mode none",
            "  --stat wandb",
            ")",
            "",
            "launch_run() {",
            "  local gpu=\"$1\"",
            "  local port=\"$2\"",
            "  local tag=\"$3\"",
            "  shift 3",
            "  local run_name=\"${TS}_${tag}_semtalk_base\"",
            "  local log_file=\"/home/ferpaa/SemTalk/outputs/${run_name}.log\"",
            "",
            "  echo \"[LAUNCH] gpu=${gpu} port=${port} tag=${tag} name=${run_name} log=${log_file}\"",
            "  MASTER_ADDR=127.0.0.1 MASTER_PORT=\"${port}\" CUDA_VISIBLE_DEVICES=\"${gpu}\" \\",
            "  python train.py \"${COMMON_ARGS[@]}\" \\",
            "    --wandb_name \"${run_name}\" \\",
            "    --notes \"_${run_name}\" \\",
            "    \"$@\" > \"${log_file}\" 2>&1 &",
            "",
            "  PIDS+=(\"$!\")",
            "  TAGS+=(\"${tag}\")",
            "  RUN_NAMES+=(\"${run_name}\")",
            "}",
            "",
            "declare -a PIDS=()",
            "declare -a TAGS=()",
            "declare -a RUN_NAMES=()",
            "",
            "echo \"=== Job started: $(date) ===\"",
            "echo \"=== Host: $(hostname) ===\"",
            "echo \"=== Base checkpoint: ${BASE_CKPT} ===\"",
            "echo \"=== Target speakers: ${TRAIN_SPEAKERS[*]} ===\"",
            "",
        ]
    )

    if len(variants) > len(gpu_ids):
        raise ValueError(
            f"Number of variants ({len(variants)}) exceeds GPU count ({len(gpu_ids)}). "
            "Submit in batches or pass fewer variants."
        )

    for i, v in enumerate(variants):
        gpu = gpu_ids[i]
        port = port_base + i
        extra = shell_quote_join(v.extra_args)
        lines.append(f"launch_run {shlex.quote(gpu)} {port} {shlex.quote(v.name)} {extra}".rstrip())

    lines.extend(
        [
            "",
            "FAIL=0",
            "for i in \"${!PIDS[@]}\"; do",
            "  pid=\"${PIDS[$i]}\"",
            "  tag=\"${TAGS[$i]}\"",
            "  if wait \"${pid}\"; then",
            "    echo \"[DONE] ${tag} (pid=${pid})\"",
            "  else",
            "    echo \"[FAIL] ${tag} (pid=${pid})\"",
            "    FAIL=1",
            "  fi",
            "done",
            "",
        ]
    )

    if run_fgd_after:
        lines.extend(
            [
                "echo \"=== Post-run FGD re-eval (all selected speakers) ===\"",
                "for rn in \"${RUN_NAMES[@]}\"; do",
                "  run_dir=$(ls -dt outputs/custom/*_${rn} 2>/dev/null | head -n 1 || true)",
                "  if [[ -z \"${run_dir}\" ]]; then",
                "    echo \"[FGD_SKIP] run dir not found for ${rn}\"",
                "    continue",
                "  fi",
                "  best_epoch=$(ls \"${run_dir}\"/best_*.bin 2>/dev/null | sed -E 's/.*best_([0-9]+)\\.bin/\\1/' | sort -n | tail -n 1)",
                "  if [[ -z \"${best_epoch}\" ]]; then",
                "    echo \"[FGD_SKIP] no best checkpoint for ${rn}\"",
                "    continue",
                "  fi",
                "  eval_out=\"outputs/${rn}_fgd_eval.txt\"",
                "  echo \"[FGD] ${rn} epoch=${best_epoch} run_dir=${run_dir}\"",
                "  python utils/run_fgd_eval.py --epoch_dir \"${run_dir}/${best_epoch}\" --device cpu > \"${eval_out}\" 2>&1 || true",
                "done",
                "",
            ]
        )

    lines.extend(
        [
            "echo \"=== Job finished: $(date) ===\"",
            "exit ${FAIL}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit all-speaker multi-model FGD sweep")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training config path")
    parser.add_argument("--seed-ckpt", required=True, help="Checkpoint used to initialize all variants")
    parser.add_argument("--split-csv", default=str(DEFAULT_SPLIT_CSV), help="train_test_split.csv path")
    parser.add_argument(
        "--speaker-source",
        choices=["test", "all"],
        default="test",
        help="Which speakers to include for --training_speakers",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant spec: name::--arg1 v1 --arg2 v2 (repeatable)",
    )
    parser.add_argument(
        "--reference-log",
        default=None,
        help="Reference log for runtime estimation (default: best available known log)",
    )
    parser.add_argument(
        "--reference-speakers",
        default="2",
        help="Comma-separated reference speaker IDs used in reference-log run (for scaling)",
    )
    parser.add_argument("--name", default=None, help="Sweep name (default auto-generated)")
    parser.add_argument("--submit-host", default="gridmaster", help="Host used if qsub is unavailable locally")
    parser.add_argument("--skip-fgd-rerun", action="store_true", help="Do not run post-training FGD re-eval")
    parser.add_argument("--dry-run", action="store_true", help="Create scripts and estimates, do not submit")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    seed_ckpt = Path(args.seed_ckpt).resolve()
    split_csv = Path(args.split_csv).resolve()

    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    if not seed_ckpt.exists():
        raise SystemExit(f"Seed checkpoint not found: {seed_ckpt}")
    if not split_csv.exists():
        raise SystemExit(f"Split CSV not found: {split_csv}")

    variants = parse_variant_specs(args.variant)
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise SystemExit("No GPU IDs provided.")
    if len(variants) > len(gpu_ids):
        raise SystemExit(
            f"Variants ({len(variants)}) > GPUs ({len(gpu_ids)}). "
            "Pass fewer --variant values or a larger --gpus list."
        )

    rows = read_split_rows(split_csv)
    target_speakers = select_target_speakers(rows, args.speaker_source)
    if not target_speakers:
        raise SystemExit("No target speakers selected from split CSV.")

    ref_speakers = [int(x.strip()) for x in args.reference_speakers.split(",") if x.strip()]
    target_train_count = count_rows(rows, "train", target_speakers)
    ref_train_count = count_rows(rows, "train", ref_speakers)

    ref_log = Path(args.reference_log).resolve() if args.reference_log else pick_default_reference_log()
    ref_total_minutes = parse_last_sane_time_info_minutes(ref_log) if ref_log else None

    scale = None
    est_per_model = None
    est_wall = None
    if ref_total_minutes is not None and ref_train_count > 0:
        scale = target_train_count / float(ref_train_count)
        est_per_model = ref_total_minutes * scale
        waves = math.ceil(len(variants) / float(len(gpu_ids)))
        est_wall = est_per_model * waves

    ts = dt.datetime.now().strftime("%m%d_%H%M%S")
    name = args.name or f"{ts}_allspeaker_base_sweep"
    output_dir = ROOT / "outputs" / "experiments" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not sanitized or not sanitized[0].isalpha():
        sanitized = f"aspk_{sanitized}"
    job_name = sanitized[:12]
    job_script_path = output_dir / "qsub_allspeaker_job.sh"

    job_text = build_job_script(
        job_name=job_name,
        queue=args.queue,
        slots=args.slots,
        root_dir=ROOT,
        output_dir=output_dir,
        conda_env=args.conda_env,
        config_path=config_path,
        seed_ckpt=seed_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        speaker_ids=target_speakers,
        variants=variants,
        gpu_ids=gpu_ids,
        port_base=args.port_base,
        run_fgd_after=not args.skip_fgd_rerun,
    )
    job_script_path.write_text(job_text)
    job_script_path.chmod(job_script_path.stat().st_mode | stat.S_IXUSR)

    estimate_path = output_dir / "estimate.txt"
    with estimate_path.open("w") as f:
        f.write(f"name: {name}\n")
        f.write(f"config: {config_path}\n")
        f.write(f"seed_ckpt: {seed_ckpt}\n")
        f.write(f"target_speakers ({len(target_speakers)}): {target_speakers}\n")
        f.write(f"target_train_count: {target_train_count}\n")
        f.write(f"reference_speakers: {ref_speakers}\n")
        f.write(f"reference_train_count: {ref_train_count}\n")
        f.write(f"reference_log: {ref_log}\n")
        f.write(f"reference_total_minutes: {ref_total_minutes}\n")
        f.write(f"scale_factor: {scale}\n")
        f.write(f"estimated_minutes_per_model: {est_per_model}\n")
        f.write(f"estimated_wall_minutes_for_batch: {est_wall}\n")
        f.write("variants:\n")
        for v in variants:
            f.write(f"  - {v.name}: {' '.join(v.extra_args)}\n")

    print("=========================================")
    print("All-speaker FGD sweep setup")
    print("=========================================")
    print(f"Name: {name}")
    print(f"Output dir: {output_dir}")
    print(f"Job script: {job_script_path}")
    print(f"Estimate file: {estimate_path}")
    print(f"Target speakers ({len(target_speakers)}): {target_speakers}")
    print(f"Train clips (target): {target_train_count}")
    print(f"Variants ({len(variants)}): {[v.name for v in variants]}")

    if est_per_model is not None and est_wall is not None:
        print(f"Estimated time/model: {est_per_model:.1f} min ({est_per_model/60.0:.2f} h)")
        print(f"Estimated wall time:  {est_wall:.1f} min ({est_wall/60.0:.2f} h)")
    else:
        print("Estimated time/model: N/A (missing/invalid reference log or speaker counts)")

    if args.dry_run:
        print("Dry run enabled: not submitting.")
        return 0

    if shutil.which("qsub"):
        submit_cmd = ["qsub", str(job_script_path)]
    else:
        remote_cmd = f"cd ~/SemTalk && qsub {shlex.quote(str(job_script_path))}"
        submit_cmd = ["ssh", args.submit_host, remote_cmd]

    print("Submitting:", " ".join(shlex.quote(x) for x in submit_cmd))
    result = subprocess.run(submit_cmd, check=False, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    print("Submitted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
