#!/bin/bash
#SBATCH --job-name=sift_case46
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --output=sift_case46_%j.out
#SBATCH --error=sift_case46_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /home/wtq4672/Presentation3_base_code

/projects/e32998/env/bin/python \
    patch_viz_on_slice/sift_method/top_patches_overlay.py \
    --run-dir /projects/e32998/MIL_training/final_runs/run_20261119_115429_845084_919520 \
    --cases 46 \
    --k 8 \
    --slice-root /home/wtq4672/Presentation3_base_code/full_slice_tifs