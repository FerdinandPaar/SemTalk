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
echo "Host: $(hostname)"
nvidia-smi -L

source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

cd /home/ferpaa/SemTalk

echo "--- Running inference ---"
/home/ferpaa/miniconda3/envs/semtalk/bin/python \
  -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29601 \
  train.py \
  --config configs/semtalk_moclip_sparse.yaml \
  --load_ckpt outputs/custom/0305_073649_semtalk_moclip_sparse/best_116.bin \
  --inference \
  --audio_infer_path demo/2_scott_0_1_1_test.wav

echo "--- Renaming output ---"
mv demo/2_scott_0_1_1_test.npz demo/2_scott_0_1_1_test_moclip_svib_phy_v1.npz

echo "=== Inference finished: $(date) ==="
echo "Output: demo/2_scott_0_1_1_test_moclip_svib_phy_v1.npz"
ls -lh demo/2_scott_0_1_1_test_moclip_svib_phy_v1.npz
