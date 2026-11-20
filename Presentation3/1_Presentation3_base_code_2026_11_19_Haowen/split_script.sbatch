#!/bin/bash

#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=UNI2_splits
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haowentan2027@u.northwestern.edu

set -e

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd ~/Presentation3_base_code

python3 -u make_splits.py \
    --labels_csv /projects/e32998/STAT390_Krish/Code/Code4_reduce_runtime/extra_benign_case_grade_match.csv \
    --patches_dir /projects/e32998/patch_images \
    --embeddings_dir /projects/e32998/STAT390_Fall2026/Team5_TheWildcards/patch_embeddings_UNI2 \
    --save_dir ./team5_new_splits \
    --seed 42 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --test_ratio 0.2

echo "End time: $(date)"