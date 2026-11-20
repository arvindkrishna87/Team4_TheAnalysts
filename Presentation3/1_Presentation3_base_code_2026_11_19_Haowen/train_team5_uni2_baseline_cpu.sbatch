#!/bin/bash

#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=UNI2_CPU_128_SLICE30_NEW
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

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python3 -u -c '
import os
import torch
from config import DATA_PATHS

torch.set_num_threads(6)
torch.set_num_interop_threads(1)

DATA_PATHS["runs_dir"] = os.path.join(
    os.getcwd(),
    "team5_uni2_runs",
    "SLICE30_128_cpu_newsplit01_job_" + os.environ["SLURM_JOB_ID"])

from main import main
main()
' \
    --patches_dir /projects/e32998/patch_images \
    --embeddings_dir /projects/e32998/STAT390_Fall2026/Team5_TheWildcards/patch_embeddings_UNI2 \
    --load_splits team5_new_splits/data_splits_new_01.npz \
    --pooled_dim 1536 \
    --embed_dim 128 \
    --epochs 30 \
    --stain_dropout_prob 0.0 \
    --slice_dropout_prob 0.3 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --num_workers 2 \
    --seed 42

echo "End time: $(date)"