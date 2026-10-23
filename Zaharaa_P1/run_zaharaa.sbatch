#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=zaharaa_P1
#SBATCH --output=zaharaa_out.out
#SBATCH --error=zaharaa_err.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alzaharaaalzaabi2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python3 -m pip install -r /projects/e32998/STAT390_Fall2026/Team4_TheAnalysts/Zaharaa_P1/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

cd /projects/e32998/STAT390_Fall2026/Team4_TheAnalysts/Zaharaa_P1

python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --seed 390
