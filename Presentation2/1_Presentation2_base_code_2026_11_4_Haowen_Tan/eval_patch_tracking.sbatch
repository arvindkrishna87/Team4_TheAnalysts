#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=eval_patch_tracking
#SBATCH --output=/home/beh0765/eval_patch_tracking_%j.out
#SBATCH --error=/home/beh0765/eval_patch_tracking_%j.err


# Purpose: Reproduce and validate the changes on Quest without overwriting shared project results.

set -eo pipefail

# Load the existing Quest Python environment
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

# Use your personal code directory
cd /home/beh0765/Presentation2_base_code

# Create a personal output directory
mkdir -p /home/beh0765/MIL_patch_tracking_runs

# Run evaluation with personal output settings
python3 - <<'PY'
import sys
import config
import utils

# Redirect new run outputs to personal storage
config.DATA_PATHS["runs_dir"] = "/home/beh0765/MIL_patch_tracking_runs"

# Ensure create_run_directory uses the personal directory
# without modifying the original source files
original_create = utils.create_run_directory

def safe_create_run_directory(*args, **kwargs):
    import os
    from datetime import datetime

    base = "/home/beh0765/MIL_patch_tracking_runs"
    os.makedirs(base, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(base, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)

    return run_dir

utils.create_run_directory = safe_create_run_directory

# main.py imports create_run_directory directly
import main
main.create_run_directory = safe_create_run_directory

sys.argv = [
    "main.py",
    "--eval_only",
    "--resume",
    "/projects/e32998/MIL_training/final_runs/run_20261028_054746/checkpoints/best.pth",
    "--load_splits",
    "/home/beh0765/Presentation2_base_code/no_train_val_leakage_splits/data_splits_new_05.npz",
    "--analyze_attention",
    "--attention_top_n", "8",
    "--per_slice_cap", "500",
    "--max_slices_per_stain", "5",
    "--num_workers", "2",
]

main.main()
PY