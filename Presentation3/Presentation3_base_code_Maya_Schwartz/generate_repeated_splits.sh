#!/bin/bash
set -e

cd /home/wtq4672/Presentation3_base_code

for rep in $(seq 2 10); do
    seed=$((100 + rep))
    rep_dir=$(printf "repeat_%02d" "$rep")

    echo "Generating repetition $rep with seed $seed"

    /projects/e32998/env/bin/python make_splits.py \
        --seed "$seed" \
        --save_dir "/home/wtq4672/Presentation3_base_code/repeated_splits/$rep_dir" \
        > "/home/wtq4672/Presentation3_base_code/splits_${rep_dir}.log"
done

echo "All repetitions generated successfully!"
