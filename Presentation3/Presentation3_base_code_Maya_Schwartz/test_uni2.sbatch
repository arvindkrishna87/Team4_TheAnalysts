#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --job-name=P3_UNI2_check
#SBATCH --output=/home/wtq4672/Presentation3_base_code/p3_uni2_check_%j.out
#SBATCH --error=/home/wtq4672/Presentation3_base_code/p3_uni2_check_%j.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

export HF_HUB_OFFLINE=1

cd /home/wtq4672/Presentation3_base_code

python test_uni2.py
