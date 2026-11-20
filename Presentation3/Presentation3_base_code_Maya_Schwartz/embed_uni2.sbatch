#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=P3_embed_UNI2
#SBATCH --output=/home/wtq4672/Presentation3_base_code/p3_embed_%A_%a.out
#SBATCH --error=/home/wtq4672/Presentation3_base_code/p3_embed_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mayaschwartz2027.1@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

export HF_HUB_OFFLINE=1

echo "Job ID: $SLURM_JOB_ID"
echo "Shard: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

nvidia-smi -L

cd /home/wtq4672/Presentation3_base_code

python precompute_pooled_features.py \
    --encoder uni2 \
    --patches_dir /projects/e32998/patches_benign_split \
    --embeddings_dir /home/wtq4672/Presentation3_base_code/patch_embeddings_UNI2 \
    --batch_size 64 \
    --num_workers 6 \
    --num_shards 4 \
    --shard_index "$SLURM_ARRAY_TASK_ID" \
    --log_every 20

echo "End time: $(date)"
