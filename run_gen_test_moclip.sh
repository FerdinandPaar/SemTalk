#!/bin/bash
# Run this script ON gridnode016 to regenerate TMR test data.
# Usage: qrsh -q mld.q@gridnode016 bash run_gen_test_moclip.sh
# OR: ssh gridnode016 -> bash run_gen_test_moclip.sh

set -e
cd /home/ferpaa/SemTalk

# Activate conda env
source /home/ferpaa/miniconda3/etc/profile.d/conda.sh
conda activate semtalk

echo "=== Generating TMR test pkl ==="
python dataloaders/save_test_dataset.py \
  --moclip_path weights/moclip_checkpoints/models/tmr_humanml3d_guoh3dfeats/last_weights/text_encoder.pt \
  --distilbert_path distilbert-base-uncased \
  --dst_pkl ./datasets/beat2_semtalk_test_moclip.pkl

echo "=== Verifying ==="
python -c "
import pickle, numpy as np
with open('datasets/beat2_semtalk_test_moclip.pkl', 'rb') as f:
    data = pickle.load(f)
s = data[0]
print('feat_clip_text:', s['feat_clip_text'].shape)
print('emo_clip_text:', s['emo_clip_text'].shape)
"

echo "=== Done. Now run training: ==="
echo "python train.py --config configs/semtalk_moclip_sparse.yaml"
