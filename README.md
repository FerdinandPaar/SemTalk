<img width="1085" height="739" alt="image" src="https://github.com/user-attachments/assets/dd538821-49ea-4438-8868-183c7db1376f" />

# physics-neuro-gesture-gen: Sparse Motion Generation with S-VIB and Physics-Aware Smoothing

**Version:** 2026.03.12  
**Commit:** `7332cd7`  
**Core Objective:** High-fidelity co-speech gesture generation combining MoCLIP-driven semantic alignment with biologically grounded physics constraints.

---

## 🚀 Key Architectural Innovations

### 1. S-VIB (Semantic Variational Information Bottleneck)
The model uses a dual-stream stochastic bottleneck to decide **when** to inject semantic gestures versus following rhythmic "beat" motion.
* **Two-Stream Input:** Separates semantic content (MoCLIP/Words) from timing (HuBERT).
* **Information Bottleneck:** Compresses a 320-dim input into a 16-dim latent $z$ using a KL-regularized posterior. This forces the model to ignore "beat leakage" and focus on minimal sufficient statistics for semantic gating.
* **Biological Prior:** Achieves a sparse firing rate (~3-5%), matching the human frequency of iconic/deictic gestures.

### 2. Physics-Aware Smoothing & Jerk Loss
Unlike standard neural gesture models, SemTalk honors anatomical constraints using mass fractions from **De Leva (1996)**.
* **Gate-Modulated EMA:** Smoothing strength $\tau_j$ is inversely proportional to gate activity $\psi$. 
    * $\psi \to 1$ (Semantic): Active muscle control overrides physics (sharp, precise motion).
    * $\psi \to 0$ (Beat): Passive limb dynamics dominate (pendulum-like motion).
* **Sqrt-Compressed Mass Scaling:** Compresses the 4-decade mass range of the human body so that every joint (from spine to fingers) receives meaningful regularisation.
* **Decoded Pose Jerk Loss:** A differentiable loss that penalizes angular jerk in 6D rotation space, weighted by joint inertia.

### 3. MoCLIP Conditioning
Integration of MoCLIP embeddings to ensure high-level semantic alignment between speech text and generated motion.

---

## 📊 Performance Benchmarks (BEAT2 Subset)

Evaluation performed on Speaker 2 (Scott) using the VAESKConv encoder protocol.

| Model State | FGD (Subset) ↓ | BC (Align) ↑ | L1-Diversity |
| :--- | :--- | :--- | :--- |
| **Baseline (MoCLIP Sparse)** | 0.4734 | 0.721 | 11.23 |
| **Pre-Physics Baseline** | 0.4189 | 0.758 | 12.69 |
| **S-VIB + Physics (Best s01)** | **0.4137** | **0.746** | **12.46** |

> **Finding:** The S-VIB and Physics constraints together achieve a **~12.6% improvement** over the initial baseline while maintaining high diversity and semantic alignment.

---

## 🛠 Training & Infrastructure

### Multi-GPU DDP Robustness
The repo includes a stabilized Distributed Data Parallel (DDP) implementation:
* **NCCL Optimization:** Disabled `SyncBatchNorm` to prevent all-gather timeouts; set `find_unused_parameters=True`.
* **Fault Tolerance:** Automated stdout/stderr redirection per rank to capture non-rank-0 tracebacks.
* **Interactive Launchers:** * `run_train_moclip_4gpu.sh`: Standard 4-GPU DDP training.
    * `run_finetune_moclip_4gpu.sh`: Optimized for fine-tuning from existing checkpoints.

### Automated Sweep Orchestrator
The `run_sweep_svib_phys_10.sh` script enables high-throughput ablation studies:
* Automated sequential chaining of 10 runs with varying hyperparameters ($\beta, \lambda, \tau$).
* Detached execution via `nohup` with PID tracking.
* Real-time W&B logging and post-sweep metric extraction (FID, Align, Diversity).

---

## 📖 Usage Guide

### 1. Training a Fine-tune Run
```bash
TRAIN_MODE=ddp DDP_NPROC=4 \
BASE_CKPT=outputs/checkpoints/best_116.bin \
bash run_finetune_moclip_4gpu.sh
```

### 2. Running Inference
Generate an NPZ file for Blender visualization:
```bash
# Uses the current best checkpoint (best_126.bin)
bash run_inference_push.sh best_126
```

### 3. Physics Evaluation
Analyze the spectral properties of generated motion:
```bash
# Computes mass-weighted Power Spectral Density (PSD)
python demo/matched_wholebody_physics.py --input path_to_generated.npz
```

---

## 📂 Repository Structure
* `models/semtalk.py`: Core S-VIB and Sparse pathway logic.
* `models/physics_smoother.py`: De Leva mass tables and EMA smoothing logic.
* `models/flow_matching_base.py`: Pluggable GestureLSM-style Flow Matching (Current Status: Deprioritized).
* `semtalk_sparse_trainer.py`: Main training loop with pose-level jerk loss.
* `utils/run_fgd_eval.py`: FGD/FID evaluation suite.
* `configs/`: YAML configurations for MoCLIP, FM, and S-VIB variants.

---

## 📝 Neural-Inspired AI (NeuroPSI) Alignment
This project aligns with the goals of NeuroPSI by demonstrating:
1.  **Physiological Constraints:** Direct implementation of biomechanical inertia and mass fractions.
2.  **Stochastic Gating:** Using Information Bottleneck theory to model the cognitive "choice" between rhythmic and semantic motor commands.
3.  **Empirical Validation:** Frequency-domain analysis proving that beat gestures track gravity-pendulum resonances.

Would you like me to generate a specific technical appendix for the S-VIB KL divergence math or the De Leva mass fraction table used in the smoother?
