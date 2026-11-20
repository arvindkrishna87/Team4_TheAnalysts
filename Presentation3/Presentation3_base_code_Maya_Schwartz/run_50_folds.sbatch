#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=P3_50folds
#SBATCH --array=0-49%20
#SBATCH --output=/home/wtq4672/Presentation3_base_code/p3_array_%A_%a.out
#SBATCH --error=/home/wtq4672/Presentation3_base_code/p3_array_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mayaschwartz2027.1@u.northwestern.edu

module purge
module load python-anaconda3/2019.10

cd /home/wtq4672/Presentation3_base_code

# Convert array task ID into repetition and fold
REP=$((SLURM_ARRAY_TASK_ID / 5 + 1))
FOLD=$((SLURM_ARRAY_TASK_ID % 5 + 1))

printf -v REP_PAD "%02d" "$REP"
printf -v FOLD_PAD "%02d" "$FOLD"

SPLIT_FILE="/home/wtq4672/Presentation3_base_code/repeated_splits/repeat_${REP_PAD}/data_splits_new_${FOLD_PAD}.npz"

echo "Job ID: $SLURM_JOB_ID"
echo "Repetition: $REP"
echo "Fold: $FOLD"
echo "Split: $SPLIT_FILE"
echo "Start: $(date)"

# Stop if the split file is missing
if [ ! -f "$SPLIT_FILE" ]; then
    echo "ERROR: Split file not found!"
    exit 1
fi

/projects/e32998/env/bin/python main.py \
    --load_splits "$SPLIT_FILE" \
    --seed "$((100 + REP))" \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --analyze_attention \
    --attention_top_n 8

echo "End: $(date)"