#!/bin/bash

# Test B: UNI2 features from each patch at its original size.
# Each array task encodes every 4th patch.
# Rerunning skips patches already saved.

#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=SAME_INPUT_B_embed
#SBATCH --output=/home/beh0765/Presentation3_base_code/logs/%x_%A_%a.out
#SBATCH --error=/home/beh0765/Presentation3_base_code/logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haowentan2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

export HF_HUB_CACHE="/home/beh0765/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1

X=/home/beh0765

cd "$X/Presentation3_base_code"

python3 precompute_pooled_features.py \
    --encoder uni2 --uni2_native_size \
    --patches_dir /projects/e32998/patch_images \
    --embeddings_dir "$X/embeddings_uni2_original_size" \
    --batch_size 64 --num_workers 2 \
    --log_every 20




