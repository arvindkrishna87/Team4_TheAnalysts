#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=UNI2_smoke
#SBATCH --output=/home/beh0765/Presentation3_base_code/logs/uni2_smoke_%j.out
#SBATCH --error=/home/beh0765/Presentation3_base_code/logs/uni2_smoke_%j.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

unset HF_HUB_OFFLINE

cd /home/beh0765/Presentation3_base_code

/projects/e32998/env/bin/python precompute_pooled_features.py \
    --encoder uni2 \
    --uni2_native_size \
    --patches_dir /home/beh0765/Presentation3_base_code/uni2_smoke_input \
    --embeddings_dir /home/beh0765/Presentation3_base_code/uni2_smoke_embeddings \
    --batch_size 1 \
    --num_workers 1 \
    --num_shards 1 \
    --shard_index 0 \
    --log_every 1
