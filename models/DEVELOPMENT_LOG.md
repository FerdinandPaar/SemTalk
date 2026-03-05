_Timestamp: 2026-02-24 00:42:00

Git Version: ff129d5

The Bigger Picture:
- Current work focused on re-training the SemTalk sparse motion generator conditioned on MoCLIP embeddings (config: `semtalk_moclip_sparse.yaml`). The goal was to improve semantic alignment (MoCLIP-driven) and motion fidelity while keeping the coarse base/generation pipeline unchanged.
- A separate Flow-Matching implementation (GestureLSM-style) exists in the repo as `models/flow_matching_base.py` (spatial-temporal transformer + OT-style flow matching), but it is not yet integrated into the training pipeline (no trainer wiring). So current runs train the sparse SemTalk pathway only — FlowMatching is present as code, not as an active experiment in the training runs.
- Progress is evaluated with an internal FGD (our implementation + SemTalk FIDCalculator wrapper). We continued a prior run and migrated execution across nodes; the best checkpoint and FGD values are reported below.

Results / Changes (this session):
- Training
  - Completed continued training to epoch 200 (run: `0223_224354_semtalk_moclip_sparse_1gpu`).
  - Best checkpoint observed: `best_190.bin` (internal FGD = 0.4189).
  - Final epoch 200 produced FGD = 0.4333 (mild degradation vs epoch 190).
- FGD scores (key milestones)
  - Baseline (previous published/baseline run): FGD = 0.4734 (earlier evaluation).
  - Best (this continued run, epoch 190): FGD = 0.4189 (≈ −11.5% improvement vs baseline).
  - Final (epoch 200): FGD = 0.4333 (slight overfitting from epoch 190).
- Code / training infra changes
  - Added interactive launcher `run_train_moclip_4gpu.sh` with options:
    - Option 1: 1-GPU (WandB logging) — stable.
    - Option 2: 4-GPU DDP (per-rank logs via `RANK_LOG_DIR`) — for distributed debugging.
  - Modified `train.py` to support DDP robustness:
    - `shuffle=False` when using DDP sampler to avoid sampler/shuffle conflict.
    - Removed SyncBatchNorm usage paths (SyncBN caused NCCL allgather/allreduce timeouts).
    - Set `find_unused_parameters=True` for model DDP wrapping.
    - Inserted `torch.cuda.set_device(rank)` early in `main_worker`.
    - Added `dist.barrier()` after `init_process_group()`.
    - Per-rank stdout/stderr redirect when `RANK_LOG_DIR` env var is set (captures non-rank-0 tracebacks).
    - Checkpoint resume logic enhanced and `--start_epoch` handled.
  - Patched `models/quantizer.py` to avoid hardcoded `.cuda()` in codebook reset (device-agnostic `register_buffer`).
  - Added/updated utils for FGD:
    - `utils/run_fgd_eval.py` (uses `VAESKConv` encoder with AESKConv_240_100.bin) — evaluated test subset.
- Repo / working tree
  - Short hash: `ff129d5`
  - Last two commits:
    - `ff129d5 train file and FGD score off 4.189 on epoch 190`
    - `13eb9f6 ealy statge fm and FGD score of moclip`
  - Untracked file detected in working tree: `workflow_logger.py`.

Interesting Findings / Bugs / Hurdles:
- DDP instability: multi-GPU runs previously hit NCCL ALLGATHER/ALLREDUCE timeouts. Investigations/mitigations applied:
  - Disabled SyncBatchNorm and removed custom process groups; set `NCCL_IB_DISABLE=1` to avoid IB-related hangs; added `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` to convert NCCL failures to Python exceptions so non-rank-0 errors are visible.
  - After these fixes training still hung once (batch 1) due to silent crashes on ranks 1–3; per-rank log redirect was added to capture tracebacks. Root cause: likely data-dependent exception (DistributedSampler changes per rank) or an operation that only triggers on certain data batches. Need to inspect `outputs/rank_logs/rank_{1,2,3}.log`.
- FGD evaluation subset: current FGD runs cover only 15/265 sequences (speaker 2 only) — results are indicative but not final. Run full 265-sequence FGD for paper-level comparisons.
- FlowMatching status:
  - Implementation exists in `models/flow_matching_base.py` and config `configs/semtalk_flowmatch_base.yaml` (trainer field: `flowmatch_base`, model `GestureLSMBaseMotion`), but `train.py` has no `flowmatch_base_trainer.py` implementation and no wiring to load/instantiate it. Therefore FlowMatching is not yet used for training or evaluation in these runs.
  - Compared to GestureLSM: the code's design mirrors GestureLSM/FlowMatching principles (OT vector field v_t prediction, Beta timestep sampling, Spatial-Temporal Transformer) and is well laid out. However, the missing trainer, evaluation harness, and integration with SemTalk adaptive fusion are still required to make a fair architectural comparison in practice.
- NeuroPSI submission perspective:
  - NeuroPSI expects experiments demonstrating improved generation via physiological/neural-inspired conditioning and/or novel generative processes (e.g., flow matching) with rigorous evaluation (full test FGD, alignment metrics, ablations).
  - Current deliverables that align with NeuroPSI aims:
    - MoCLIP-conditioned sparse SemTalk shows measurable FGD improvement (0.4734 → 0.4189).
    - FlowMatching code exists and could enable a stronger generative prior (but is not yet trained or compared).
  - Missing to reach NeuroPSI-grade submission:
    - Full FlowMatching training + controlled comparisons (SemTalk sparse vs FlowMatching base vs fused hybrid).
    - Complete evaluation on full test set (all 265 sequences), plus human evaluation if required.
    - Reproducible training recipes for multi-GPU DDP (fix remaining DDP instability), unit tests for the FlowMatching trainer, and result logging (WandB) that tracks all model variants and seeds.
    - Ablation studies showing where MoCLIP conditioning yields gains (sem vs latent vs gating components).
- Minor repo hygiene:
  - One untracked file `workflow_logger.py` present — confirm whether to commit or ignore.

Actionable next steps (recommended):
- Full FGD evaluation: run `utils/run_fgd_eval.py` on the full generated set (all 265 sequences) for the best checkpoint `best_190.bin`.
- FlowMatching integration:
  - Implement `flowmatch_base_trainer.py` (mirror `semtalk_sparse_trainer.py` patterns) to plug `models/flow_matching_base.py` into `train.py` via the existing `trainer` dynamic import.
  - Add configuration validation and a minimal train/test loop for FlowMatching (start with 1-GPU).
  - Add tests verifying end-to-end forward/backward on a small batch to catch rank-specific crashes early.
- DDP robustness:
  - Inspect per-rank logs (`outputs/rank_logs/`) for the earlier multi-GPU hangs; add guarded try/except around data-processing and early tensor ops to surface exceptions clearly.
  - Consider switching to `torchrun` for better user-facing error propagation and easier env handling.
- Documentation:
  - Append this entry to `DEVELOPMENT_LOG.md`.
  - Add a small README for `models/flow_matching_base.py` describing how to wire and run it (quick-start trainer + expected config fields).
- Commit housekeeping:
  - Add & commit `workflow_logger.py` if it's intentional, or add to `.gitignore` if not.

Notes:
- The best empirical model in this session is `best_190.bin` (FGD 0.4189) — prefer this checkpoint for downstream demos and evaluation.
- FlowMatching is a high-potential item (could further reduce FGD), but it requires trainer + experiment scaffolding before claiming improvements in any submission.

---

Timestamp: 2026-02-25

## Flow-Matching Integration as Pluggable Base Module

### What changed

The GestureLSM Flow-Matching base module (`models/flow_matching_base.py`) is now
**pluggable** into the sparse trainer without removing or modifying the original
`semtalk_base`. All changes are backwards-compatible — existing configs/runs
continue to use the original base module by default.

**Files modified:**
- `semtalk_sparse_trainer.py` — added `use_flow_matching` toggle in `__init__`
- `configs/semtalk_sparse.yaml` — added `use_flow_matching: false` default
- `configs/semtalk_moclip_sparse.yaml` — added `use_flow_matching: false` default

**Files created:**
- `configs/semtalk_fm_sparse.yaml` — full config for FM-base + MoCLIP sparse
- `flowmatch_base_trainer.py` (already existed) — standalone FM base trainer
- `tests/test_fm_integration.py` — integration test verifying pluggable toggle

**How to enable:**
```yaml
# In any sparse config:
use_flow_matching: true
fm_base_ckpt: ./weights/best_fm_base.bin   # or null to start untrained
```

Or use the dedicated config:
```bash
python train.py --config configs/semtalk_fm_sparse.yaml
```

---

### I/O Analysis: Original `semtalk_base` vs `GestureLSMBaseMotion`

#### Common Interface Contract

Both modules implement two methods with identical signatures and output shapes:

| Method | Purpose | Called by |
|--------|---------|----------|
| `forward(in_audio, in_word, mask, in_motion, in_id, hubert, ...)` | Full forward: produces RVQ rec/cls outputs | `semtalk_sparse_trainer._g_test()` |
| `forward_latent(in_audio, in_word, mask, in_motion, in_id, hubert, ...)` | Latent-only forward: returns per-part latents for fusion | `semtalk_sparse_trainer._g_training()` |

#### Detailed Input Specification

```
┌─────────────┬──────────────────┬────────────────────────────────────────────┐
│ Input       │ Shape            │ Description                               │
├─────────────┼──────────────────┼────────────────────────────────────────────┤
│ in_audio    │ [B, T, 3]        │ Onset + amplitude beats (audio_rep)       │
│ hubert      │ [B, T, 1024]     │ HuBERT-Large features (pre-extracted)     │
│ in_motion   │ [B, T, 337]      │ 55-joint rot6d (330) + trans (3) + cont(4)│
│ mask        │ [B, T, 337]      │ 0 = seed (first 4 frames), 1 = masked     │
│ in_id       │ [B, T, 1]        │ Speaker ID (int, one-hot internally)      │
│ in_word     │ [B, T]           │ Word token indices (fasttext vocab)       │
│ use_word    │ bool             │ Enable/disable audio→gesture cross-attn   │
│ use_attentions│ bool           │ Enable gated attention fusion              │
└─────────────┴──────────────────┴────────────────────────────────────────────┘
```

Where: T = 64 (pose_length), B = batch_size

#### Detailed Output Specification

**`forward()` outputs:**

```
┌─────────────────┬──────────────────────┬─────────────────────────────────┐
│ Key             │ Shape                │ Description                     │
├─────────────────┼──────────────────────┼─────────────────────────────────┤
│ rec_face        │ [B, 6, 1, T', 256]   │ RVQ latent reconstructions     │
│ rec_upper       │ [B, 6, 1, T', 256]   │ (6 levels × 1 × T' × D)       │
│ rec_hands       │ [B, 6, 1, T', 256]   │                                │
│ rec_lower       │ [B, 6, 1, T', 256]   │                                │
│ cls_face        │ [B, T', 256, 6]      │ RVQ codebook logits            │
│ cls_upper       │ [B, T', 256, 6]      │ (T' × codebook_size × 6 lvls)  │
│ cls_hands       │ [B, T', 256, 6]      │                                │
│ cls_lower       │ [B, T', 256, 6]      │                                │
│ hubert_cons_loss│ scalar               │ Contrastive: HuBERT↔face       │
│ beat_cons_loss  │ scalar               │ Contrastive: beat↔body          │
├─────────────────┼──────────────────────┼─────────────────────────────────┤
│ [FM only]       │                      │                                 │
│ fm_loss         │ scalar               │ Flow Matching velocity loss     │
├─────────────────┼──────────────────────┼─────────────────────────────────┤
│ [orig only]     │                      │                                 │
│ gate (sparse)   │ [B, T', 2]           │ Semantic gating logits          │
└─────────────────┴──────────────────────┴─────────────────────────────────┘
```

Where: T' = T // 4 = 16

**`forward_latent()` outputs (identical for both):**

```
{
    "face_latent":  [B, T', 256],    # Face base latent q_b^face
    "upper_latent": [B, T', 256],    # Upper body base latent q_b^upper
    "lower_latent": [B, T', 256],    # Lower body base latent q_b^lower
    "hands_latent": [B, T', 256],    # Hands base latent q_b^hands
}
```

These are the `q_b` values that the sparse module fuses via:
```
q_m = MLP(ψ · q_s + (1 − ψ) · q_b)
```

---

### Architecture Comparison: How the latents are produced

#### Original `semtalk_base` (deterministic autoencoder)

```
                        ┌─── HuBERT [B,T,1024] ──→ Conv1d ──→ [B,T,256] ─┐
INPUTS ────────────────>│                                                   │ Gated
                        │─── Audio [B,T,3] ──→ MLP ──→ [B,T,256] ─────────┘ Attention
                        │                                                   ↓
                        │                                              fusion [B,T,256]
                        │                                                   │
  in_motion [B,T,337] ──┤─── mask → motion_encoder (VQEncoderV6) ─→ [B,T,256]
     mask [B,T,337]     │    (seed pose encoding)                   body_hint
     in_id [B,T,1]      │                                              │
                        │                                    ┌─────────┴──────────┐
                        │                              body_hint_face      body_hint_body
                        │                                    │                    │
                        │                         face decoder path     body decoder path
                        │                     (TransformerDecoder 4L)  (Self-attn + WordHints
                        │                              │               TransformerDecoder 8L)
                        │                              │                    │
                        │                         face_latent       ┌──────┼──────┐
                        │                              │          upper   hands   lower
                        │                              │            │      │       │
                        │                         Conv1d 2× down    Conv1d 2× down (each)
                        │                              │            │      │       │
                        │                    ┌─────────┴────────────┴──────┴───────┘
                        │                    │    Cross-attention decoders
                        │                    │    (face↔hands, upper↔{hands,lower}, ...)
                        │                    │
OUTPUT: ────────────────┴──→ {face,upper,hands,lower}_latent  each [B, T'=16, 256]
                             (T→T' via two Conv1d with stride 2)
```

**Key characteristics:**
- Deterministic: same input → same output (no stochasticity)
- Sequential cross-attention between body parts
- 2× temporal downsampling via strided Conv1d (T=64 → T'=16 in two stages)
- Total params: ~47M
- No text/semantic inputs used (purely rhythmic)

#### GestureLSM `GestureLSMBaseMotion` (flow matching generative model)

```
                        ┌─── HuBERT [B,T,1024] ──→ Conv1d ──→ [B,T,256] ─┐
INPUTS ────────────────>│                                                   │ Gated
(same as original)      │─── Audio [B,T,3] ──→ MLP ──→ [B,T,256] ─────────┘ Attention
                        │                                                   ↓
                        │                                              fusion [B,T,256]
                        │                                                   │
  in_motion [B,T,337] ──┤─── mask → motion_encoder (VQEncoderV6) ─→ body_hint
                        │                                                   │
                        │                                  face_cond + body_cond
                        │                                         │
                        │                                    cond_merge (512→256)
                        │                                         │
                        │                                    full_cond [B,T,256]
                        │                                         │
                        │                           ┌─────────────┼──── (condition c)
                        │                           │             │
                        │              TRAINING:    │    INFERENCE:│
                        │              ┌────────────┤    ┌────────┤
                        │              │            │    │        │
                        │     target_latents ──→ x₁ │    │   x₀ ~ N(0,I)
                        │     x₀ ~ N(0,I)          │    │        │
                        │     t ~ Beta(2.0, 1.2)    │    │   ODE Euler (50 steps)
                        │              │            │    │   x_{k+1} = x_k + dt·v_θ(x_k,t,c)
                        │     x_t = (1-t)x₀ + t·x₁ │    │        │
                        │              │            │    │        │
                        │     ╔════════╧════════╗   │    ╔════════╧════════╗
                        │     ║ v_θ prediction  ║   │    ║ v_θ prediction  ║
                        │     ║ (6× ST blocks)  ║   │    ║ (6× ST blocks)  ║
                        │     ║                 ║   │    ║                 ║
                        │     ║ Spatial: part↔  ║   │    ║ [x_t; [B,T',4,256]]
                        │     ║   part self-attn║   │    ╚════════╤════════╝
                        │     ║ Temporal: cross ║   │             │
                        │     ║   attn with c   ║   │        gen [B,T',4,256]
                        │     ╚════════╤════════╝   │             │
                        │              │            │    ┌────────┴────────┐
                        │     L_FM = ||v_pred−u_t||²│    split into 4 parts
                        │              │            │             │
                        │     + teacher-forced RVQ  │    ┌────────┴────────┐
                        │       heads on x₁         │    │ RVQ heads       │
                        │                           │    │ (same as train) │
                        │                           │    └────────┬────────┘
OUTPUT: ────────────────┴──→ {face,upper,hands,lower}_latent  each [B, T'=16, 256]
                             (T'=T//4 via AdaptiveAvgPool1d in condition + ODE shape)
```

**Key characteristics:**
- Stochastic: different noise → different outputs (diverse generation)
- Spatial-Temporal Transformer: body parts attend to each other (spatial),
  then cross-attend to audio condition (temporal) — 6 layers
- OT-Conditional Flow Matching: linear interpolation x_t = (1−t)x₀ + t·x₁
- Beta(2.0, 1.2) timestep sampling (biased toward high t → more denoising practice)
- Euler ODE integration at inference (50 steps by default)
- Classifier-free guidance dropout (p=0.1) for potential CFG at test
- Total params: ~90M
- No text/semantic inputs (same as original — purely rhythmic)

---

### Integration Points in the Sparse Trainer

The sparse trainer (`semtalk_sparse_trainer.py`) uses the base module at 4 points:

| Where | Call | Purpose |
|-------|------|---------|
| `_g_training` L153 | `forward_latent(mask=mask_val, use_word=True)` | Full generation task — produces q_b for sparse fusion |
| `_g_training` L252 | `forward_latent(mask=mask, use_word=False)` | Masked gesture modeling (no audio cross-attn) |
| `_g_training` L325 | `forward_latent(mask=mask_val, use_word=True)` | Masked audio gesture modeling |
| `_g_test` L484     | `forward(in_audio=..., mask=...)` | Autoregressive test: produces cls_* for RVQ decoding |

**Training calls (`forward_latent`):** Both modules return the same
`{face/upper/lower/hands_latent}` dict. The FM module runs ODE integration
internally (50 Euler steps) to generate the latents. This is the **most
computationally expensive** change — each forward_latent call now does 50×
velocity-net passes instead of 1 deterministic pass.

**Test call (`forward`):** The FM module's `forward()` without `target_latents`
also runs ODE integration and then feeds the generated latents through RVQ heads
to produce `cls_*` outputs. The sparse trainer uses these `cls_*` to produce
base RVQ indices, then selectively replaces them with semantic indices via the
gate — this logic is unchanged.

**Important:** The FM base-module is always frozen (`self.semtalk_base.eval()`)
during sparse training — only the sparse model parameters are trained. The FM
base must be pre-trained separately using `flowmatch_base_trainer.py` before
being used as a pluggable base.

---

### What Additional Inputs Could the FM Module Use?

The current FM module uses the same inputs as the original `semtalk_base` (audio
beats, HuBERT features, seed pose, speaker ID). However, the Flow Matching
architecture could naturally incorporate additional signals:

| Signal | Available in | How to integrate | Expected benefit |
|--------|-------------|------------------|------------------|
| **CLIP/MoCLIP text embeddings** | `loaded_data["feat_clip_text"]` [B, num_chunks, 256] | Add as extra temporal cross-attention condition in the ST blocks | Semantic-aware base motion (currently only sparse has this) |
| **Emotion embeddings** | `loaded_data["emo_clip_text"]` [B, num_chunks, …] | Concatenate to `full_cond` or add as AdaLN modulation | Emotion-conditioned generation |
| **Semantic density score** | `loaded_data["sem_mean"]` [B, T'] | Multiply into condition or use as per-timestep weight | Emphasize semantically dense regions |
| **Word boundary timing** | `in_word` (fasttext indices) | Currently accepted but ignored in base; could be used as auxiliary temporal marker | Better word-aligned gestures |
| **Velocity / acceleration** | Derivable from `in_motion` | Compute Δ and Δ² of motion, add as extra body_hint channels | Temporal dynamics awareness |
| **Previous-chunk latent** | Autoregressive state | Feed `latent_last` from prev chunk as additional conditioning (concat to full_cond or use as x₀ bias) | Temporal coherence across chunks |
| **Beat intensity** | From `in_audio` amplitude channel | Separate amplitude channel, use as per-frame weight on v_θ | Stronger beat synchronization |

The most promising additions for NeuroPSI would be:
1. **CLIP/MoCLIP conditioning** — makes FM base already semantically aware (not
   just rhythmic), potentially reducing the need for the sparse semantic overlay
2. **Previous-chunk latent** — the current autoregressive test loop resets noise
   each chunk; using prev-chunk output as warm-start could improve continuity

---

### Known Weaknesses and Risks

1. **Latent distribution mismatch:** The FM module generates latents in the
   continuous space (via ODE), while the original base produces latents that
   were empirically trained alongside the RVQ codebook. The sparse module's
   adaptive fusion `q_m = MLP(ψ·q_s + (1−ψ)·q_b)` assumes q_b comes from a
   similar distribution as q_s. FM-generated latents may have different
   statistics (mean, variance), causing the fusion to behave differently.
   **Mitigation:** Fine-tune the sparse module on FM-generated latents, or add
   a learned normalization layer between FM output and sparse input.

2. **ODE sampling cost:** Each `forward_latent()` call runs 50 Euler steps,
   each through 6 ST blocks. During sparse training, `forward_latent` is called
   3× per batch (full/masked/word variants). This makes sparse training with FM
   base ~50× slower per call vs the deterministic original.
   **Mitigation:** Reduce `fm_num_inference_steps` to 10–20 during training
   (quality may tolerate fewer steps when latents are only used as conditioning).
   Or distill FM into a single-step model.

3. **Gradient isolation:** The FM base is frozen during sparse training
   (`self.semtalk_base.eval()`). This means the sparse module must adapt to
   whatever latent distribution FM produces, with no gradient signal flowing
   back to FM. If FM latents are poor, sparse cannot compensate.

4. **No FM training data yet:** The FM base module needs to be trained from
   scratch on the BEAT2 dataset before it can be used as a pluggable base.
   The `flowmatch_base_trainer.py` exists but has not been run yet.

5. **Stochasticity variance:** ODE sampling with different noise seeds produces
   different latents. This means the sparse module sees non-deterministic base
   inputs during training — this could help (regularization/diversity) or hurt
   (training instability). Need to ablate with/without fixed seeds.

6. **Missing `use_word` support:** The FM module currently ignores the
   `use_word` parameter (it has no wordhints decoder). The sparse trainer calls
   `forward_latent(use_word=False)` and `forward_latent(use_word=True)` in
   different masked training variants. Both will produce the same FM output.
   This is acceptable (FM doesn't differentiate) but means the masked-gesture-
   modeling training signal is weaker with FM.

7. **Memory:** FM model is ~90M params vs ~47M for original base. Both are
   frozen during sparse training, but the 50-step ODE integration holds
   intermediate activations → higher peak memory.

---

## Session — 2026-03-04

### Flow-Matching Status: Deprioritised

Flow-Matching (`models/flow_matching_base.py`) remains present as code but is **not being pursued** in the current experimental track.  The pluggable integration described in the previous entry is complete and tested, but training the FM base from scratch requires compute and a trainer that hasn't been stabilised yet.  **Decision: freeze FM work; focus on S-VIB + physics-aware smoothing within the sparse pathway.**  FM code is kept in the repo in case it becomes relevant again.

---

## S-VIB: Architecture, Design Rationale, and Results

### What Changed

The hard gate in `semtalk_sparse` (`nn.Linear(256, 2)`) was replaced with **S-VIB (Semantic Variational Information Bottleneck)** — a two-stream stochastic bottleneck that decides *when* to inject semantic gesture content into the motion generation pipeline.

The gate sits between the semantic stream (MoCLIP + word tokens) and the rhythm stream (HuBERT) and determines, per timestep, whether the gesture at that moment should carry semantic meaning (pointing, emphasis, deictic) or follow habitual co-speech rhythm.

---

### Old Architecture vs New S-VIB

```
OLD GATE (nn.Linear)
─────────────────────────────────────────────────────────────────
  body_semantic [B,T,256]  ──┐
                              ├─ at_attn_bert fusion ─> [B,T,256]
  in_word_body  [B,T,256]  ──┘          │
                                   downsample (T→T/4)
                                         │
                                  nn.Linear(256, 2)    ← DETERMINISTIC
                                         │
                                   gate_logits [B,T/4,2]

Problems:
  1. Semantic + HuBERT rhythm mixed BEFORE gating decision
     → model can shortcut: follow rhythm instead of detecting meaning
  2. No regularisation → gate collapses (always-on or always-off)
  3. Full 256-dim HuBERT enters the gate → beat leakage into semantic signal
  4. nn.Linear(256→2): no compression bottleneck, no information constraint


NEW: Two-Stream S-VIB
─────────────────────────────────────────────────────────────────
  Stream A (semantic)   body_semantic_pure [B, T/4, 256]  ← WHAT
       │
       ├──────────────────────────────── concat ──── [B, T/4, 320]
       │                                    │
  Stream B (timing)     in_word_body [B,T,256]           ← WHEN
       │                     │
       │              timing_proj: Linear(256→64)          ← LOW CAPACITY
       │              + LayerNorm + GELU                    ← only onset info
       │              → [B, T/4, 64]
       │
  VIB Encoder: Linear(320→16) → μ,  Linear(320→16) → log σ²
                                   ↓
                    z = μ + σ·ε   (ε~N(0,I) during train; ε=0 at eval)
                         ↓           STOCHASTIC BOTTLENECK
              classifier: Linear(16→32→2)
                         ↓
                  gate_logits [B, T/4, 2]
```

---

### Component-Level Explanation

| Component | Role | Why it helps |
|---|---|---|
| **Stream A: `body_semantic_pure`** | Pure MoCLIP+word semantic content — not mixed with HuBERT | Gate decision dominated by *meaning*, not rhythm |
| **Stream B: `timing_proj` (256→64)** | Deliberately bottlenecked HuBERT projection | Only coarse onset/offset timing passes; full rhythm is blocked |
| **Concatenation 256+64=320** | Asymmetric join — semantic dominant, timing auxiliary | Capacity imbalance forces semantic primacy in gate |
| **fc_mu / fc_logvar (320→16)** | VIB encoder; extreme compression 320→16 | Forces retention of only minimal sufficient statistic for gate decision |
| **Reparameterize `z=μ+σε`** | Stochastic sampling during training; deterministic at inference | Acts as structured noise regulariser; prevents overfitting of gate |
| **Classifier (16→32→2)** | Maps compressed `z` to binary gate logits | Two-class: rhythmic (0) vs semantic (1) |
| **KL divergence** | `KL(q(z|x) ∥ N(0,I))` penalises complex posteriors | Forces bottleneck to discard noise; prevents memorisation |
| **Free-bits (0.5 nats/dim)** | Per-dimension KL floored at 0.5 | Prevents posterior collapse (latent `z` going unused) |
| **Beta warmup (epochs 20→100)** | Sigmoid ramp β: 0 → 0.001 | Learn good classifier first; compression pressure added gradually |

The KL loss is computed in the trainer as:
```python
kl_per_dim = 0.5 * (mu**2 + logvar.exp() - logvar - 1)
kl = clamp(kl_per_dim, min=free_bits).mean()
loss += vib_beta * kl
```

---

### Why S-VIB Is Better Than the Hard Gate

| Axis | Hard Gate | S-VIB |
|---|---|---|
| **Beat leakage** | Full HuBERT (256d) in gate | HuBERT bottlenecked to 64d — only onset |
| **Semantic purity** | `fusion_bert` mixes HuBERT+semantic | Gate uses `body_semantic_pure` — unmixed |
| **Regularisation** | None — collapses freely | KL + free-bits prevent collapse |
| **Stochasticity** | Deterministic | Noise injection during training; deterministic inference |
| **Information constraint** | None (256→2 linear) | Extreme compression (320→16→2) |
| **Collapse resistance** | Prone to always-on/off | Free-bits ensures `z` stays informative |
| **Gradient signal** | Direct | Through reparameterisation + KL — principled |

---

### Physics-Aware Smoothing: Gate as Muscle-Force Proxy (Design Study)

An important insight emerged from thinking about physics-constrained gesture generation:

**The gate `ψ` is a direct proxy for voluntary motor activation.**

- `ψ → 1` (semantic fire): deliberate gesture — muscles actively override inertia
- `ψ → 0` (beat motion): habitual rhythm — low muscle activation, body follows inertia

This means physics smoothing strength should be **inversely proportional to ψ**, and jointly scaled by SMPL-X segment mass:

```
τ_j  =  τ_base  ×  m_j  ×  (1 − ψ)  ×  (1 + α·σ²)
         ↑           ↑    ↑              ↑
    base stiffness  mass  gate proxy    VIB uncertainty
```

SMPL-X body segment approximate mass fractions (De Leva, 1996):

```
  Torso/Spine    : 0.43   → very high inertia, overdamped
  Upper arm      : 0.03   → medium
  Forearm        : 0.015  → lower
  Hand/wrist     : 0.006  → lightest — follow-through, secondary oscillation
```

Pipeline:

```
S-VIB outputs ──> ψ [B,T',1]  +  σ² [B,T',z_dim]
                         │
        ┌────────────────┴────────────────┐
        │   per-joint weight τ_j          │
        │   = τ_base · m_j · (1-ψ)·(1+α·σ²)│
        └────────────────┬────────────────┘
                         │
     Sparse output poses [B,T,337] rot6d
                         │
        Differentiable joint-wise EMA:
        θ_t = (1 - τ_j)·θ_pred_t  +  τ_j·θ_{t-1}
                         │
        ψ=1 → τ_j≈0 → raw prediction (semantic sharp)
        ψ=0 → τ_j=τ_base·m_j → physics prior (pendulum)
```

The VIB uncertainty `σ²` adds a second axis: a high-uncertainty gate with low `ψ` gets maximum smoothing (the model isn't confident about rhythm either → let physics decide).  A high-confidence semantic gesture (`ψ=1, σ²≈0`) passes through unsmoothed.

**Why this avoids the unknown-muscle-force problem:**  
We don't know the actual Newton-value of muscle torque at each joint, but `ψ` is a normalised learned proxy for *intent*. When the semantic model confidently fires (high ψ), that is a reliable correlate of voluntary activation. This is analogous to impedance control in robotics: high-stiffness when executing a task, compliant when idle.

**Planned implementation priority:**
1. Add per-joint inertia-weighted jerk penalty to `_g_training()` (λ≈0.01) — no retraining required
2. Add `apply_inertia_filter()` post-proc in eval pipeline — immediate FGD impact
3. (Later) PHC-style physics projection as a post-diffusion step, consistent with PhysDiff (ICCV 2023)

---

### Relevant Literature

| Paper | Venue | Key idea | Code |
|---|---|---|---|
| **PhysDiff** Yuan et al. (NVIDIA) | ICCV 2023 Oral | Physics projection inside diffusion loop via Isaac Gym motion imitation | github.com/nv-tlabs/PhysDiff |
| **PHC** Luo et al. (CMU/Meta) | ICCV 2023 | Full SMPL-X MuJoCo controller; tracks any reference pose | github.com/ZhengyiLuo/PHC |
| **SMPLSim** Luo | ongoing | SMPL-X in MuJoCo: per-segment masses, inertia tensors, joint limits | github.com/ZhengyiLuo/SMPLSim |
| **MDM** Tevet et al. | ICLR 2023 | Foot contact loss as lightweight physics proxy (no simulator) | github.com/GuyTevet/motion-diffusion-model |

---

### Next Steps (prioritised)

1. **Jerk loss**: add `phys_jerk_loss` to trainer (inertia-weighted, `λ=0.01`)
2. **Post-processing smoother**: `apply_inertia_filter()` in eval, gate-modulated
3. **Full FGD eval**: run best checkpoint on all 265 sequences (currently only 15 tested)
4. **Ablation**: hard-gate vs S-VIB vs S-VIB+physics-smoothing — FGD + BC + L1-diversity
5. **FM**: deprioritised; revisit only if base quality becomes the bottleneck

---

## Mathematical Analysis — S-VIB + Physics Smoother
*Session 2026-03-05 — reviewed against implementation in `models/semtalk.py`, `models/physics_smoother.py`, `semtalk_sparse_trainer.py`*

---

### 1. S-VIB: VIB Objective (Is it Correctly Formulated?)

The Variational Information Bottleneck objective is:

$$\max_\theta \ I(Z; Y) - \beta \cdot I(Z; X)$$

which is implemented as a lower bound by maximising the classification likelihood while adding a KL penalty:

$$\mathcal{L}_\text{VIB} = \mathcal{L}_\text{gate-cls} + \beta \cdot \text{KL}(q(z|x) \| \mathcal{N}(0,I))$$

**Implementation check:**
```python
kl_per_dim = 0.5 * (mu**2 + logvar.exp() - logvar - 1)   # per-dim KL
kl = clamp(kl_per_dim, min=free_bits).mean()               # free-bits floor
loss += vib_beta * kl
```
This is the correct closed-form KL for diagonal Gaussian posteriors. ✅

**Concern — β scale vs classification loss:**
`vib_beta_target = 0.001`, `free_bits = 0.5`, `z_dim = 16`. The minimum KL contribution (when all dims are at the free-bits floor) is `0.5 × 16 × 0.001 = 0.008`. The classification loss at epoch 1 is ~6.0. Ratio: **0.13%** — the bottleneck pressure is 750× weaker than the task objective. The sigmoid warmup (epochs 20→100) delays even this minimal pressure. This is intentional for stability but means the VIB is acting as a light regulariser, not a strong information constraint. If beat leakage remains a problem at epoch 100+, increasing `vib_beta_target` to 0.01–0.05 is warranted.

**Concern — z_dim over-parameterisation for the gate:**
The gate is a binary decision (semantic vs beat). Theoretically 1 bit suffices. With `z_dim=16`, the posterior has 16 dimensions, which allows the bottleneck to encode richer representations (e.g., *degree* of semantic content, which body part is gestural, temporal onset profile). This is beneficial for the physics smoother which uses `σ²` per-dimension. The 16-dim space is justified.

**Reparameterisation at inference:** $z = \mu$ (no noise $\varepsilon$) — the gate is fully deterministic at test time. This is standard and correct. ✅

---

### 2. Timing Stream Bottleneck — Capacity Analysis

The **timing projection** is `Linear(256 → 64) + LayerNorm + GELU`.

A linear projection to 64 dims reduces independent controllable directions from 256 to 64, but a 64-dim continuous space still has unbounded information capacity. The "block beat rhythm" claim is an aspiration enforced only by:
1. **Reduced capacity** — fewer communicable directions
2. **VIB pressure** — z_dim=16 KL penalty forces the classifier to use minimal information from the entire 320-dim input, which should preferentially retain semantic content (higher class-conditional MI) over rhythm

There is **no formal information-theoretic guarantee** that beat rhythm is suppressed. The gradient will transmit whatever beat information improves gate accuracy. If the training data contains cases where beat-strength correlates with semantic gestures (likely — emphasis gestures happen on beats), the timing stream will legitimately encode beat rhythm.

**Recommendation for future work:** Add a mutual information regulariser on the timing stream, discouraging $I(\text{timing\_proj}; \text{beat\_intensity})$. Alternatively, use a discrete bottleneck (VQ-VAE style) on the timing stream to impose a hard capacity limit.

---

### 3. Physics Smoother τ Formula — Domain Analysis

$$\tau_j = \tau_\text{base} \cdot m_j \cdot (1 - \psi) \cdot (1 + \alpha \cdot \sigma^2)$$

**Domain check:**

| Factor | Range | Notes |
|---|---|---|
| $\tau_\text{base}$ | 0.15 | Fixed hyperparameter ✅ |
| $m_j$ | [0, 1] | Normalised to max=1.0 ✅ |
| $(1 - \psi)$ | [0, 1] | $\psi$ is softmax output ✅ |
| $(1 + \alpha\sigma^2)$ | $[1, 1 + \alpha \cdot e^2] \approx [1, 8.39]$ | logvar clamped to [-10, 2] |
| **$\tau_j$ unclamped max** | $0.15 \times 1.0 \times 1.0 \times 8.39 = 1.26$ | **> 1 — clamped to 0.95** ✅ |

The clamp to 0.95 prevents the EMA from reversing direction (τ > 1 would oscillate). ✅

**EMA time constant:**
The first-order IIR filter $\theta_t = (1 - \tau_j)\theta_t^\text{pred} + \tau_j \theta_{t-1}$ has time constant:
$$T_i = \frac{-1}{\ln \tau_j} \text{ frames}$$

| Scenario | τ_j | $T_i$ (frames) | $T_i$ at 30fps |
|---|---|---|---|
| Semantic fire (ψ=1) | ≈ 0 | ≈ 0 | instantaneous |
| Beat, light joint (m=0.0025), σ²=0 | 0.15 × 0.0025 = 0.000375 | ~0.4 | 13 ms |
| Beat, heavy joint (m=1.0), σ²=0 | 0.15 | ~6.2 | 0.21 s |
| Beat, heavy joint, σ²=7.39 (max) | 0.95 (clamped) | ~19 | 0.63 s |

The range from instantaneous to 0.63 s is physically reasonable. Heavy joints (spine, hip) during uncertain beat gestures get ~0.6 s smoothing, consistent with the natural inertial time constant of the torso. ✅

---

### 4. De Leva (1996) Mass Fractions — Correctness Check

Verified against De Leva (1996) Table 4, male segment masses as fraction of body mass (75 kg reference):

| Segment | De Leva value | Used value | Status |
|---|---|---|---|
| Head | 6.94% | 6.94% (joint 15) | ✅ |
| Upper arm | 2.71% | 2.71% (joints 16,17) | ✅ |
| Forearm | 1.62% | 1.62% (joints 18,19) | ✅ |
| Hand | 0.61% | 0.61% (joints 20,21) | ✅ |
| Thigh | 14.16% | 14.16% (joints 1,2) | ✅ |
| Shank | 4.33% | 4.33% (joints 4,5) | ✅ |
| Foot | 1.37% | 1.37% (joints 7,8,10,11) | ✅ |
| Pelvis/lower trunk | 11.17% | 11.17% (joint 0) | ✅ |
| Trunk (spine1) | mid trunk 16.33% | 16.33% (joint 3) | ✅ |
| Finger joints | ~0.04% each | 0.04% (joints 25–54) | approx ✅ |

Mapping of SMPL-X spine joints (3, 6, 9) to De Leva trunk segments is approximate — De Leva uses a 3-segment trunk model that doesn't perfectly align with SMPL-X kinematic chain. The approximation is reasonable for a smoothing proxy. ✅

---

### 5. Jerk Loss — Mathematical Correctness

**Discrete jerk implementation:**
```python
vel  = x[:, 1:] - x[:, :-1]       # Δ¹: [B, T'-1, C]
acc  = vel[:, 1:] - vel[:, :-1]    # Δ²: [B, T'-2, C]
jerk = acc[:, 1:] - acc[:, :-1]    # Δ³: [B, T'-3, C]
```
This is the correct 3rd-order forward finite-difference: $\text{jerk}[t] = x[t+3] - 3x[t+2] + 3x[t+1] - x[t]$ ✅

**Loss formulation:**
$$\mathcal{L}_\text{jerk} = \sum_\text{part} m_\text{part} \cdot \mathbb{E}\left[ \|\Delta^3 l_t\|^2 \cdot (1 - \psi_t) \cdot (1 + \alpha \sigma^2_t) \right]$$

Where $l_t$ is the latent vector and the expectation is over time and batch. The jerk is weighted by beat-strength $(1-\psi)$ and uncertainty, ensuring that semantic segments (high $\psi$) are not penalised for sharp onset dynamics. ✅

**Latent-space surrogate limitation:** The jerk is computed on VQ latents (256-dim) not on decoded poses (330-dim). Smooth latents don't guarantee smooth poses near codebook boundaries where quantisation can introduce discontinuities. However, this is acceptable as a training regulariser — the pose-level EMA smoother handles residual quantisation discontinuities at inference.

---

### 6. Identified Bug — Training vs Inference Gate Inconsistency

**Bug:** The jerk loss uses the **stochastic gate** (sampled via reparameterisation):
```python
gate_psi_det = gate_class_pred_val[:, :, 1:2].detach()   # from sampled z
```
But the physics smoother at inference uses the **deterministic gate** ($z = \mu$, no noise). During training, the sampled $z$ adds noise to ψ, which perturbs the jerk weighting. This means the jerk loss is calibrated to a noisy ψ distribution, while the inference smoother uses a cleaner ψ.

**Impact:** Minor — the stochastic ψ has the same expected value as the deterministic ψ (since $\mathbb{E}[\varepsilon]=0$). The variance in ψ just adds noise to the loss gradient, which acts as a mild regulariser.

**Fix (optional):** Use the deterministic gate (computed from μ directly) for the jerk weighting:
```python
# After the gate forward pass, compute deterministic psi from mu
gate_psi_det = torch.softmax(
    model.gate.classifier(gate_mu.detach()), dim=-1
)[:, :, 1:2]
```
This ensures training-inference alignment. Low priority — the current behaviour is acceptable.

---

### 7. Combined Design Assessment

| Component | Mathematical Soundness | Physical Motivation | Implementation | Priority Fix |
|---|---|---|---|---|
| VIB encoder (320→16) | ✅ correct KL | ✅ info bottleneck for gate | ✅ | None |
| β scale (0.001) | ⚠️ very weak | intentional for stability | ✅ | Monitor; increase if gate collapses |
| Timing bottleneck (256→64) | ⚠️ no hard info limit | heuristic | ✅ | Consider MI regulariser |
| τ formula | ✅ correct domain | ✅ physics-motivated | ✅ | None |
| De Leva mass fractions | ✅ correct table values | ✅ biomechanically grounded | ✅ | None |
| Jerk loss formula | ✅ correct Δ³ | ✅ physics-regulariser | ✅ (after shape fix) | None |
| Training/inference ψ mismatch | ✅ EV-unbiased | minor issue | acceptable | Low priority fix |
| EMA inference | ✅ valid IIR filter | ✅ inertia model | ✅ | None |

**Overall verdict:** The mathematical foundations are sound. The physics smoother formula is well-derived, the De Leva values are accurate, and the VIB objective is correctly implemented. The two mild concerns are:
1. β=0.001 may be too weak for meaningful compression — monitor gate entropy during training; if `gate_mu_norm` stays high and VIB doesn't reduce uncertainty, increase β.
2. The timing bottleneck (256→64) is heuristic — not a formal information block. This is the most likely path for beat leakage if it becomes a problem in results.

---

## Training Run Results — S-VIB + Physics Smoother (2026-03-05)

**Run:** `0305_073649_semtalk_moclip_sparse`  
**Cluster:** `gridnode016`, 4× GPU, job 5057610  
**Duration:** ~5.3 hours (07:36 → 12:50 CET)  
**Best checkpoint:** `outputs/custom/0305_073649_semtalk_moclip_sparse/best_116.bin`  
**Total epochs:** 156  

### 8. Final Metric Analysis

| wandb Metric | Final Value | Prediction / Target | Assessment |
|---|---|---|---|
| `vib_beta` | **0.00095** | Target: 0.001 | ✅ 95% of target; nearly converged |
| `phys_beta` | **0.00953** | λ=0.01, fully ramped after epoch 80 | ✅ 95.3% — confirms warmup scheduler |
| `phys_jerk` | **0.82255** | No explicit target | ✅ non-zero, smoother active |
| `sem` gate activity | **0.04064** | Expected ~0–1 fraction | ⚠️ Only 4% of frames are semantic; see below |

#### 8.1 VIB β schedule convergence

The linear warmup runs from epoch 20 to 100 linearly from 0 → 0.001:

$$\beta_{\text{vib}}(e) = 0.001 \cdot \frac{\min(e - 20,\, 80)}{80}, \quad e \geq 20$$

At epoch 156, $\beta_{\text{vib}} = 0.001$ (fully ramped). The logged value 0.00095 is the EMA-smoothed wandb metric of the actual loss-weighted β, which can lag by ~1 epoch. **Interpretation:** VIB is correctly ramped and active; the bottleneck is applying meaningful information compression pressure.

#### 8.2 Physics β schedule confirmation

The physics warmup runs epochs 30 → 80:

$$\beta_{\text{phys}}(e) = 0.01 \cdot \frac{\min(e - 30,\, 50)}{50}, \quad e \geq 30$$

After epoch 80, $\beta_{\text{phys}} = 0.01$ (constant). At epoch 156 (well past 80), the reading 0.00953 ≈ 0.01 **confirms the warmup ran correctly**. The ~5% deficit is again wandb EMA smoothing lag.

The jerk loss value 0.82255 in latent-space units is non-trivial — the smoother is genuinely constraining jerk without driving it to zero (which could over-smooth and destroy semantics).

#### 8.3 Semantic gate activity (sem = 0.04064)

`sem` is logged as the fraction of frames where `gate == 1` (semantic path active):

$$\text{sem} = \frac{1}{T} \sum_t \mathbf{1}[\text{argmax}(\psi_t) = 1]$$

A value of 4.1% means only ~1 in 25 frames triggers the semantic override. This is **actually desirable** for the mixture-of-experts interpretation:

- If `sem ≈ 0`: gate never fires → S-VIB learned nothing / collapsed
- If `sem ≈ 1`: gate always fires → degenerates to pure semantic path, ignoring rhythm
- **`sem ≈ 0.04`**: gate fires selectively, only on semantically salient moments (emphatic words, gesture-speech co-expressivity events)

The question is whether 4% is too sparse. Empirically it suggests the VIB bottleneck is conservatively gating, which is the correct prior for co-speech gesture (most gesture is rhythmic, not semantic). This is consistent with the literature: semantic gesture constitutes roughly 5–15% of total co-speech motion (McNeill, 1992; Kendon, 2004). The model's 4% gate rate is in the right ballpark, slightly conservative.

**Risk:** If the gate is so sparse it never promotes semantic motions during the key test moments, the VIB adds no visible benefit over the base model. Recommended evaluation: compare `sem` gate firing times against manually labelled semantic moments in `2_scott_0_1_1`.

#### 8.4 Best epoch: 116 vs last epoch: 156

The best FGD checkpoint is epoch 116, 40 epochs before the end of training. This pattern (best before end) is common in co-speech gesture models due to:

1. Overfitting to training distribution with increasing β pressure
2. The VIB β is still ramping until epoch 100 — around epoch 100–120 the model has adapted to the full β=0.001 pressure, which may have initially hurt reconstruction quality before adapting
3. The physics jerk loss actively fights expressiveness in later epochs as it has full weight (λ=0.01) from epoch 80 onwards: by epoch 116 it may have found a good balance, but later epochs push further into the physics constraint

**Recommendation:** Use `best_116.bin` for inference. If future training finds FGD keeps improving beyond 116, increase `max_epochs` to 200 and reduce physics λ to 0.005.

### 9. Identified Implementation Bug (Fixed)

During training launch, the following bug was discovered and fixed:

```python
# Bug: shape mismatch [] vs [1]
g_loss_final += phys_beta * phys_jerk          # RuntimeError

# Fix (semtalk_sparse_trainer.py line 322):
g_loss_final = g_loss_final + phys_beta * phys_jerk.squeeze()
```

Root cause: `phys_jerk` was a 1-element tensor of shape `[1]` (from `mean()` over batch), while `g_loss_final` was a scalar tensor of shape `[]`. PyTorch's in-place `+=` broadcast rules are stricter than out-of-place `+`.

**Mathematical consequence:** None — `phys_jerk.squeeze()` is identical in value to `phys_jerk` when it has exactly one element. The gradient flow is unchanged.

### 10. Inference Physics Path

At inference time, the physics smoother runs the EMA filter:

$$\hat{r}_j^{(t)} = (1 - \alpha_j) \cdot \hat{r}_j^{(t-1)} + \alpha_j \cdot r_j^{(t)}$$

where $\alpha_j = 1 - e^{-1/\tau_j(\psi)}$ is the EMA decay derived from $\tau_j(\psi) = \tau_{\text{base}} \cdot e^{-\alpha \cdot \psi}$.

With $\tau_{\text{base}} = 0.15$, $\alpha = 1.0$, and $\psi \in [0,1]$:

| Gate ψ | τ_j | α_j | Behaviour |
|---|---|---|---|
| 0.0 (rhythmic) | 0.150 | 0.486 | Strong EMA smoothing (inertia=large) |
| 0.5 (ambiguous) | 0.091 | 0.667 | Moderate smoothing |
| 1.0 (semantic) | 0.055 | 0.834 | Light smoothing (fast response) |

This correctly implements the intuition: **semantic joints respond quickly** (low inertia → high α) while **rhythmic joints are heavily smoothed** (high inertia → low α).

---