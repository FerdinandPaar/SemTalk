"""
Full-test-set FGD evaluation using SemTalk's own evaluation protocol.

Correct evaluation pipeline (matches semtalk_sparse_trainer.py lines 693-794):
  1. Load axis-angle poses from npz (N_frames, 165) = 55 joints × 3
  2. Convert to rot6d: (N_frames, 330) = 55 joints × 6
  3. Chunk into NON-OVERLAPPING windows of vae_test_len=32 frames
     (discard the tail remainder — no padding)
  4. Encode via VAESKConv (AESKConv_240_100.bin) → latent (vae_length=240)
  5. Compute FGD on the concatenated latent vectors

⚠ Known limitations of this particular run:
  - Only 15 sequences available (speaker 2 / Scott only)
  - Full BEAT2 English test split has 265 sequences across 25 speakers
  - Scores cannot be directly compared to paper results (which use all 265)
  - To get paper-comparable scores, re-run inference on all test sequences
    and pass all gt_*.npz / res_*.npz files to this script

Usage:
    python utils/run_fgd_eval.py
    python utils/run_fgd_eval.py --epoch_dir outputs/custom/<run>/<epoch>
"""

import sys, os, argparse, glob, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
from utils.fgd import compute_statistics, calculate_fgd
from dataloaders.data_tools import FIDCalculator

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_EPOCH_DIR = "outputs/custom/0216_084918_semtalk_moclip_sparse/138"
EVAL_CKPT         = "BEAT2/beat_english_v2.0.0/weights/AESKConv_240_100.bin"
DATA_PATH_1       = "./weights/"         # for smplx_models inside VAESKConv
VAE_TEST_LEN      = 32                   # chunk size (matches semtalk_sparse.yaml)
VAE_LENGTH        = 240                  # latent dim  (matches AESKConv_240_100)
VAE_TEST_DIM      = 330                  # 55 joints × 6 rot6d
N_JOINTS          = 55


# ---------------------------------------------------------------------------
# Build a minimal args namespace for VAESKConv
# ---------------------------------------------------------------------------
def _make_eval_args(data_path_1: str = DATA_PATH_1) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        vae_test_dim     = VAE_TEST_DIM,
        vae_test_len     = VAE_TEST_LEN,
        vae_length       = VAE_LENGTH,
        vae_codebook_size= 256,
        vae_layer        = 4,
        vae_grow         = [1, 1, 2, 1],
        variational      = False,
        data_path_1      = data_path_1,
    )


# ---------------------------------------------------------------------------
# Load and freeze the evaluation encoder
# ---------------------------------------------------------------------------
def load_eval_encoder(ckpt_path: str, device: str = "cpu") -> nn.Module:
    from models.motion_representation import VAESKConv
    args  = _make_eval_args()
    model = VAESKConv(args).to(device).eval()

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Eval encoder checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # SemTalk saves: {"model_state": {...}}  or plain state dict
    sd = state.get("model_state", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    print(f"[INFO] Loaded eval encoder from: {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Preprocessing: axis-angle npz → rot6d chunks
# ---------------------------------------------------------------------------
def poses_to_rot6d(poses_aa: np.ndarray) -> torch.Tensor:
    """
    (N_frames, 165)  axis-angle  →  (N_frames, 330)  rot6d
    Matches the preprocessing in the SemTalk trainer test loop.
    """
    t = torch.from_numpy(poses_aa).float()   # (N, 165)
    N = t.shape[0]
    aa  = t.reshape(N, N_JOINTS, 3)
    mat = axis_angle_to_matrix(aa)            # (N, 55, 3, 3)
    r6d = matrix_to_rotation_6d(mat)          # (N, 55, 6)
    return r6d.reshape(N, N_JOINTS * 6)       # (N, 330)


def extract_latents(
    npz_paths: list,
    encoder: nn.Module,
    device: str = "cpu",
) -> np.ndarray:
    """
    Load all npz files, convert to rot6d, chunk into vae_test_len windows,
    encode, and return (M, vae_length) latent matrix — exactly as done in
    semtalk_sparse_trainer.py lines 688-694.
    """
    latents = []
    enc_device = next(encoder.parameters()).device

    for p in sorted(npz_paths):
        data    = np.load(p, allow_pickle=True)
        poses   = data["poses"]              # (N_frames, 165)
        r6d     = poses_to_rot6d(poses)      # (N_frames, 330)

        n       = r6d.shape[0]
        remain  = n % VAE_TEST_LEN
        n_trim  = n - remain                 # largest multiple of VAE_TEST_LEN

        if n_trim == 0:
            continue                         # sequence too short

        r6d_trim = r6d[:n_trim].unsqueeze(0)  # (1, n_trim, 330)
        r6d_trim = r6d_trim.to(enc_device)

        with torch.no_grad():
            z = encoder.map2latent(r6d_trim)  # (1, n_trim/?, vae_length) or similar
            z = z.reshape(-1, VAE_LENGTH)     # (M_seq, 240)

        latents.append(z.cpu().numpy())

    return np.concatenate(latents, axis=0)   # (M_total, 240)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FGD evaluation using SemTalk's own evaluation protocol."
    )
    parser.add_argument("--epoch_dir", default=DEFAULT_EPOCH_DIR,
                        help="Path to an epoch output folder containing gt_*.npz and res_*.npz")
    parser.add_argument("--eval_ckpt", default=EVAL_CKPT,
                        help="Path to AESKConv_240_100.bin evaluation encoder checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    epoch_dir = args.epoch_dir
    if not os.path.isabs(epoch_dir):
        epoch_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            epoch_dir,
        )

    eval_ckpt = args.eval_ckpt
    if not os.path.isabs(eval_ckpt):
        eval_ckpt = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            eval_ckpt,
        )

    gt_paths  = sorted(glob.glob(os.path.join(epoch_dir, "gt_*.npz")))
    res_paths = sorted(glob.glob(os.path.join(epoch_dir, "res_*.npz")))

    if not gt_paths:
        sys.exit(f"[ERROR] No gt_*.npz files found in {epoch_dir}")
    if len(gt_paths) != len(res_paths):
        sys.exit(f"[ERROR] Mismatched: {len(gt_paths)} GT vs {len(res_paths)} res")

    print(f"\n{'='*60}")
    print(f" FGD Evaluation — MoCLIP SemTalk")
    print(f"{'='*60}")
    print(f" Epoch dir   : {epoch_dir}")
    print(f" Sequences   : {len(gt_paths)}  (⚠ full test set = 265)")
    print(f" Encoder     : VAESKConv  ({os.path.basename(eval_ckpt)})")
    print(f" vae_test_len: {VAE_TEST_LEN}  vae_length: {VAE_LENGTH}  vae_test_dim: {VAE_TEST_DIM}")
    print(f" Device      : {args.device}")
    print(f"{'='*60}\n")

    # Load encoder
    encoder = load_eval_encoder(eval_ckpt, args.device)

    # Extract latents
    print("[INFO] Extracting GT latents...")
    gt_latents  = extract_latents(gt_paths,  encoder, args.device)
    print(f"       → {gt_latents.shape}")

    print("[INFO] Extracting generated latents...")
    res_latents = extract_latents(res_paths, encoder, args.device)
    print(f"       → {res_latents.shape}")

    # Compute FGD using SemTalk's own FIDCalculator (matches trainer exactly)
    fid_semtalk = FIDCalculator.frechet_distance(res_latents, gt_latents)

    # Cross-check with our own implementation
    gt_t  = torch.from_numpy(gt_latents).float()
    res_t = torch.from_numpy(res_latents).float()
    fid_ours = calculate_fgd(gt_t, res_t)

    print(f"\n{'='*60}")
    print(f"  FGD (SemTalk FIDCalculator) : {fid_semtalk:.4f}")
    print(f"  FGD (our implementation)    : {fid_ours:.4f}")
    print(f"\n  ⚠  These scores cover {len(gt_paths)}/265 test sequences")
    print(f"     (speaker 2 / Scott only).")
    print(f"     Run inference on all 265 sequences for paper-comparable results.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

