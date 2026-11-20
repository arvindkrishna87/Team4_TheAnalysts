#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=P3_UNI2_test
#SBATCH --output=p3_training_%j.out
#SBATCH --error=p3_training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mayaschwartz2027.1@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd /home/wtq4672/Presentation3_base_code

python main.py \
    --load_splits /home/wtq4672/Presentation3_base_code/repeated_splits/repeat_01/data_splits_new_01.npz \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --analyze_attention \
    --attention_top_n 8

echo "End time: $(date)"
