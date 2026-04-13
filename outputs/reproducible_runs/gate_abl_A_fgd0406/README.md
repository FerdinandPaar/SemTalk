# gate_abl_A: MoCLIP-only baseline — FGD 0.4056

**Date:** 2026-04-13
**Best FGD:** 0.4056 @ epoch 242 (BC=0.7649, DIV=12.426)
**Job:** 5174933 (4×A100 DDP)

## Checkpoint location

```
weights/best_gate_abl_A_fgd0406.bin          (654 MB, .gitignored)
outputs/custom/0413_074543_gate_abl_A_no_vib/ (full run dir)
```

## Reproduce

```bash
# From s01_base_r05 best_184 checkpoint:
python -m torch.distributed.run --nproc_per_node=4 \
    --master_addr=localhost --master_port=28600 \
    train.py \
    --config configs/gate_abl_A_no_vib.yaml \
    --ddp True \
    --load_ckpt outputs/custom/0311_102115_s01_base_r05_r05base_s01_base_r05_b0.010_fb0.30_pl0.08_tb0.50_tf0.10/best_184.bin \
    --start_epoch 184
```

## Key settings
- **MoCLIP:** ✓ (TMR encoder)
- **S-VIB:** ✗ (vib_enabled: false → simple linear gate)
- **Physics:** ✗ (phys_enabled: false)
- **gate_smooth:** 0.0
- **Epochs:** 184→264 (80 epochs fine-tune)
- **Gate behaviour:** fully collapsed ψ≈1.0 (all frames use semantic pathway)

## Significance
New SoTA FGD on 15-seq speaker-2 subset. Demonstrates that MoCLIP conditioning alone, without VIB or physics, produces the best distributional match.
