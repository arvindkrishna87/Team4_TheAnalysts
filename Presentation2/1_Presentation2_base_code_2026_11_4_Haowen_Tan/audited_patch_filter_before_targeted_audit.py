# Purpose: Make missing embeddings visible, deterministic, and traceable before model evaluation.
# Added filter_missing_embeddings(split_name, data, embeddings_dir, output_dir), which checks that each PNG patch has its expected .pt embedding.
# Preserves the original per-stain, per-slice order of retained paths; removes unavailable entries before dataset loading.
# Writes separate missing_embeddings_train.csv, missing_embeddings_val.csv, and missing_embeddings_test.csv files recording excluded patches and their expected embedding locations.
# Raises an error if a case has no valid patches left; does not generate or modify any .pt embeddings.




import os
import csv
from collections import Counter


def filter_missing_embeddings(
    split_name,
    data,
    embeddings_dir,
    output_dir,
):
    """
    Remove patches with missing .pt embeddings before loading.

    Preserve case, stain, slice, and patch order.
    Save all missing-embedding exclusions to an audit CSV.
    Raise an error if a case has no valid patches.
    """
    case_dict, label_map = data

    filtered_cases = {}
    missing_records = []
    missing_by_case = Counter()

    total_patches = 0
    kept_patches = 0
    empty_slices_removed = 0

    for case_id, stain_map in case_dict.items():
        filtered_stains = {}
        valid_case_patches = 0

        for stain, slice_lists in stain_map.items():
            filtered_slices = []

            for slice_idx, patch_paths in enumerate(slice_lists):
                valid_paths = []

                for patch_path in patch_paths:
                    total_patches += 1

                    basename = os.path.splitext(
                        os.path.basename(patch_path)
                    )[0]

                    embedding_path = os.path.join(
                        embeddings_dir,
                        basename + ".pt"
                    )

                    if os.path.isfile(embedding_path):
                        valid_paths.append(patch_path)
                        kept_patches += 1
                        valid_case_patches += 1
                    else:
                        missing_by_case[case_id] += 1

                        missing_records.append({
                            "split": split_name,
                            "case_id": case_id,
                            "stain": stain,
                            "original_slice_idx": slice_idx,
                            "patch_path": patch_path,
                            "expected_embedding": embedding_path,
                            "reason": "missing_embedding",
                        })

                if valid_paths:
                    filtered_slices.append(valid_paths)
                elif patch_paths:
                    empty_slices_removed += 1

            filtered_stains[stain] = filtered_slices

        if valid_case_patches == 0:
            raise RuntimeError(
                f"Case {case_id} in {split_name} has "
                "no patches with available embeddings."
            )

        filtered_cases[case_id] = filtered_stains

    os.makedirs(output_dir, exist_ok=True)

    audit_path = os.path.join(
        output_dir,
        f"missing_embeddings_{split_name}.csv"
    )

    fieldnames = [
        "split",
        "case_id",
        "stain",
        "original_slice_idx",
        "patch_path",
        "expected_embedding",
        "reason",
    ]

    with open(audit_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_records)

    print(f"\nEMBEDDING AUDIT: {split_name}")
    print(f"Total patches: {total_patches}")
    print(f"Kept patches: {kept_patches}")
    print(f"Excluded patches: {len(missing_records)}")
    print(f"Empty slices removed: {empty_slices_removed}")
    print(f"Affected cases: {len(missing_by_case)}")
    print(f"Audit saved to: {audit_path}")

    return filtered_cases, label_map
