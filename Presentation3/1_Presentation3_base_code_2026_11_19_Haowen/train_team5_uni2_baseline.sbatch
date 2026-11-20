#!/bin/bash

#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=UNI2_baseline_512
#SBATCH --output=/home/beh0765/Presentation3_base_code/logs/%x_%j.out
#SBATCH --error=/home/beh0765/Presentation3_base_code/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haowentan2027@u.northwestern.edu

set -e

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

cd /home/beh0765/Presentation3_base_code

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python3 -u -c '
import os
import torch
from config import DATA_PATHS

if not torch.cuda.is_available():
    raise RuntimeError("GPU is not available")

DATA_PATHS["runs_dir"] = os.path.join(
    os.getcwd(),
    "team5_uni2_runs",
    "baseline_512_fold01_job_" + os.environ["SLURM_JOB_ID"]
)

from main import main
main()
' \
    --patches_dir /projects/e32998/patch_images \
    --embeddings_dir /projects/e32998/STAT390_Fall2026/Team5_TheWildcards/patch_embeddings_UNI2 \
    --load_splits team5_extended_splits/data_splits_new_01.npz \
    --pooled_dim 1536 \
    --embed_dim 512 \
    --epochs 30 \
    --stain_dropout_prob 0.0 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --num_workers 2 \
    --seed 42

echo "End time: $(date)"