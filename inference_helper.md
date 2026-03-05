# SemTalk Inference Helper

> Model: **SemTalk-sparse + S-VIB + Physics Smoother** (`semtalk_svib_phys`)  
> Trained run: `0305_073649_semtalk_moclip_sparse` | Best epoch: **116**

---

## 1. Quick-Start Command (Single GPU)

```bash
cd /home/ferpaa/SemTalk

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
  /home/ferpaa/miniconda3/envs/semtalk/bin/python \
  -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29601 \
  train.py \
  --config configs/semtalk_moclip_sparse.yaml \
  --load_ckpt outputs/custom/0305_073649_semtalk_moclip_sparse/best_116.bin \
  --inference \
  --audio_infer_path demo/2_scott_0_1_1_test.wav
```

After the run, rename the output:

```bash
mv demo/2_scott_0_1_1_test.npz demo/2_scott_0_1_1_test_moclip_svib_phy_v1.npz
```

---

## 2. Cluster (qsub) Inference Script

The grid-cluster has no GPUs on the login node. Use `qsub`:  
Save as `submit_inference.sh`:

```bash
#!/bin/bash
#$ -N semtalk_infer
#$ -q mld.q@gridnode016
#$ -pe smp 1
#$ -l gpu=1
#$ -o /home/ferpaa/outputs/qsub_infer.log
#$ -e /home/ferpaa/outputs/qsub_infer.log
#$ -j y
#$ -cwd

echo "=== Inference started: $(date) ==="
source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

cd /home/ferpaa/SemTalk

/home/ferpaa/miniconda3/envs/semtalk/bin/python \
  -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29601 \
  train.py \
  --config configs/semtalk_moclip_sparse.yaml \
  --load_ckpt outputs/custom/0305_073649_semtalk_moclip_sparse/best_116.bin \
  --inference \
  --audio_infer_path demo/2_scott_0_1_1_test.wav

mv demo/2_scott_0_1_1_test.npz demo/2_scott_0_1_1_test_moclip_svib_phy_v1.npz
echo "=== Inference finished: $(date) ==="
echo "Output: demo/2_scott_0_1_1_test_moclip_svib_phy_v1.npz"
```

Submit with:

```bash
ssh gridmaster 'cd /home/ferpaa/SemTalk && qsub submit_inference.sh'
```

Monitor:

```bash
ssh gridmaster 'qstat -u ferpaa'
tail -f /home/ferpaa/outputs/qsub_infer.log
```

---

## 3. CLI Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `--config` | — | Path to YAML config (`configs/semtalk_moclip_sparse.yaml`) |
| `--load_ckpt` | — | Path to `.bin` checkpoint to load |
| `--inference` | `False` | Flag: run `trainer.inference()` then exit (no training) |
| `--audio_infer_path` | `demo/2_scott_0_1_1.wav` | Path to audio file (WAV, 16 kHz) |
| `--test_state` | `False` | Flag: run `trainer.test()` for FGD evaluation then exit |

---

## 4. Checkpoint Registry

| NPZ output file | Checkpoint | Config | Epoch | Notes |
|---|---|---|---|---|
| `2_scott_0_1_1_test_backup.npz` | `best_semtalk_sparse.bin` (legacy) | `semtalk_sparse.yaml` | — | Original sparse baseline |
| `2_scott_0_1_1_test_moclip.npz` | `best_semtalk_sparse.bin` | `semtalk_moclip_sparse.yaml` | — | MoCLIP TMR encoder, no VIB/phys |
| `2_scott_0_1_1_test_moclip_svib_v1.npz` | (prior run) | `semtalk_moclip_sparse.yaml` | — | S-VIB gate, no physics smoother |
| **`2_scott_0_1_1_test_moclip_svib_phy_v1.npz`** | `best_116.bin` (`0305_073649_...`) | `semtalk_moclip_sparse.yaml` | **116** | **S-VIB + Physics Smoother** ← current |

---

## 5. Inference Pipeline (What Happens Inside)

```
audio WAV
    │
    ├─ WhisperX → word-aligned text + sentence tokens   (Systran/faster-whisper-large-v3)
    ├─ HuBERT   → frame-level acoustic features          (facebook/hubert-large-ls960-ft)
    ├─ beat     → rhythmic beat signal                   (from audio)
    └─ emotion  → NRC VAD emotion estimate per window    (from audio energy/text)

For each temporal chunk:
    ├─ TMR text encoder (moclip_ckpt)  → semantic CLIP embedding
    ├─ semtalk_base  → rhythmic token indices (upper/lower/hands/face)
    ├─ semtalk_svib_phys (main model) → semantic token indices + gate logits
    │       ├─ SemanticVIB: μ/σ → z ~ q(z|ctx) → reparameterize
    │       ├─ gate classifier: p(semantic | z) → ψ ∈ [0,1]
    │       └─ gate: merge semtalk_base & semantic indices
    └─ VQ decoders → pose 6D rotation sequence

Post-loop:
    ├─ GlobalMotion decoder → translation trajectory
    └─ PhysicsSmootherSMPLX.forward_inference():
            τ_j(ψ) = τ_base · exp(-α·ψ)   (semantic joints: small τ → low smoothing)
            EMA per joint over time
    
Output: NPZ with poses [N×165], trans [N×3], expressions [N×100], betas, gender, fps=30
```

---

## 6. Output Format

The saved `.npz` contains:

| Key | Shape | Description |
|---|---|---|
| `poses` | `(N, 165)` | SMPL-X body pose, axis-angle, 55 joints × 3 |
| `trans` | `(N, 3)` | Global translation (x, y, z) in metres |
| `expressions` | `(N, 100)` | FLAME expression coefficients |
| `betas` | `(1, 300)` | SMPL-X shape coefficients (copied from GT) |
| `gender` | `"neutral"` | Body model gender |
| `model` | `"smplx2020"` | Body model identifier |
| `mocap_frame_rate` | `30` | Frames per second |

Visualise with `demo/smplx_viewer.ipynb`.

---

## 7. Model Architecture Summary

```
semtalk_svib_phys  =  semtalk_sparse  (alias in models/semtalk.py)
├── semtalk_base              (rhythmic context; frozen from base_ckpt)
├── SemanticVIB               (VIB bottleneck gate)
│   ├── timing_proj: 256 → 64
│   ├── fc_mu / fc_logvar: 320 → 16
│   └── classifier: 16 → 32 → 2  (binary gate)
└── PhysicsSmootherSMPLX      (per-joint EMA at inference)
    ├── τ_j = τ_base · exp(−α · ψ_j)
    └── De Leva mass fractions for all 55 SMPL-X joints
```

### Key Hyperparameters (semtalk_moclip_sparse.yaml)

| Component | Param | Value |
|---|---|---|
| VIB | `vib_z_dim` | 16 |
| VIB | `vib_beta_target` | 0.001 |
| VIB | warmup | epochs 20 → 100 |
| Physics | `phys_tau_base` | 0.15 |
| Physics | `phys_alpha` | 1.0 |
| Physics | `phys_lambda` (jerk) | 0.01 |
| Physics | warmup | epochs 30 → 80 |

### Final Training Metrics (epoch 156, best at 116)

| Metric | Value |
|---|---|
| `vib_beta` | 0.00095 (target 0.001) |
| `phys_beta` | 0.00953 (≈ λ=0.01, fully ramped) |
| `phys_jerk` | 0.82255 |
| `sem` gate activity | 0.04064 |

---

## 8. Troubleshooting

**`ImportError: typing_extensions.ParamSpec`**  
→ Do NOT use `~/.local/bin/torchrun`. Use `python -m torch.distributed.run` with the full conda python path.

**`FileNotFoundError: Checkpoint does not exist`**  
→ Pass `--load_ckpt` with the full absolute or relative-from-CWD path. Run from `/home/ferpaa/SemTalk/`.

**`AttributeError: module 'models.semtalk' has no attribute 'semtalk_svib_phys'`**  
→ Ensure `models/semtalk.py` has the alias `semtalk_svib_phys = semtalk_sparse` at the bottom.

**OOM on single GPU**  
→ Inference uses batch_size=1 and should fit in ~8 GB. If OOM, reduce `vae_test_len` in config.

**Output NPZ has wrong shape**  
→ Check `pos_len % 8 != 0` trimming; the function pads/trims to multiples of 8 automatically.
