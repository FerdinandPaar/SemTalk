# SemTalk MoCLIP Sparse Integration

This repository contains the modified code for **SemTalk** to support **MoCLIP (TMR)** embeddings (256-dim) instead of the original CLIP embeddings (512-dim).

## Summary of Changes

We replaced the original CLIP-based text encoder with a **Text-Motion Retrieval (TMR)** model. This change aligns the semantic space with motion generation tasks better than standard CLIP.

### 1. Architecture Updates
- **Text Encoder**: Switched from `CLIP` (Frozen) to `TMR` (DistilBERT-based, Frozen).
- **Dimensionality**: Reduced embedding size from **512** (CLIP) to **256** (TMR).
- **Projection Layer**: Updated `SemTalk` model to accept 256-dim inputs. The semantic injection layer now projects `256 -> 256` (was `512 -> 256`).

### 2. Code Modifications
- **`configs/semtalk_moclip_sparse.yaml`**: 
  - Added `semantic_encoder: tmr`.
  - Changed `clip_dim` to `256`.
- **`src/utils/moclip_utils.py`**:
  - Implemented `TMRTextEncoder` class to wrap the TMR model.
  - Added logic to download/load TMR weights automatically.
- **`semtalk_sparse_trainer.py`**:
  - Updated initialization to load TMR encoder when configured.
  - Ensured inference logic correctly uses the new encoder.
- **`train.py`**:
  - Improved HuggingFace cache handling for cluster environments (checks `HF_HOME`, `~/.cache`, `/mnt/disk2T` etc.).

### 3. Training & Results
- **Training**: Successfully trained on `gridnode016` for 131 epochs.
- **Checkpoint**: `outputs/custom/0216_084918_semtalk_moclip_sparse/best_131.bin`.
- **Inference**: valid output generated at `demo/2_scott_0_1_1_test_moclip.npz`.
- **Verification**: Numerical comparison confirmed that the generated motion is distinct from the baseline CLIP model.

## Issues Resolved
- **Cluster Deployment**: Fixed `HF_HOME` cache paths to prevent permission errors on compute nodes.
- **Inference Crash (NCCL)**: Resolved by running inference on GPU nodes (`gridnode016`) instead of login nodes.
- **GLIBC Compatibility**: Addressed `git` version issues in the conda environment by downgrading/upgrading tools as needed.

## Setup & Usage

### Dependencies
Ensure you have the required packages:
```bash
pip install -r requirements_fixed.txt
# TMR dependencies
pip install transform-moclip-link-if-public-or-local
```

### Running Inference
To generate motion from text using the new model:
```bash
python semtalk_sparse_trainer.py --config configs/semtalk_moclip_sparse.yaml --gpu_id 0
```

The output will be saved to `demo/` as `.npz` files.
