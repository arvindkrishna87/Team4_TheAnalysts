"""
Record missing embeddings encountered during actual dataset loading.

Only patches selected by the DataLoader are checked.
Multiple DataLoader workers can safely append to the audit CSV.
"""

import os
import csv
import fcntl


def log_missing_embedding(
    audit_dir,
    split_name,
    case_id,
    stain,
    slice_idx,
    patch_path,
    embedding_path,
):
    """Record an attempted embedding load whose file is missing."""

    if audit_dir is None:
        raise ValueError("audit_dir must be provided to log missing embeddings")

    os.makedirs(audit_dir, exist_ok=True)

    csv_path = os.path.join(
        audit_dir,
        f"missing_embeddings_{split_name}.csv"
    )

    fieldnames = [
        "split",
        "case_id",
        "stain",
        "slice_idx",
        "patch_path",
        "expected_embedding",
        "reason",
    ]

    record = {
        "split": split_name,
        "case_id": case_id,
        "stain": stain,
        "slice_idx": slice_idx,
        "patch_path": patch_path,
        "expected_embedding": embedding_path,
        "reason": "missing_embedding",
    }

    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        try:
            f.seek(0, os.SEEK_END)
            write_header = f.tell() == 0

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerow(record)
            f.flush()
            os.fsync(f.fileno())

        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return csv_path



def audit_selected_split(
    split_name,
    data,
    embeddings_dir,
    output_dir,
    per_slice_cap,
    max_slices_per_stain,
):
    """
    Audit patches selected by deterministic validation/test loading.

    Apply the same slice and patch caps as dataset.py.
    Record all selected patches with missing embeddings.
    Raise one summary error only after checking all cases.
    """

    if split_name not in ("val", "test"):
        raise ValueError(
            "This audit supports deterministic validation/test "
            "selection only."
        )

    case_dict, label_map = data
    missing_records = []
    total_selected = 0

    for case_id, stain_map in case_dict.items():

        if case_id not in label_map:
            continue

        for stain, slice_lists in stain_map.items():

            if max_slices_per_stain is not None:
                slice_lists = slice_lists[:max_slices_per_stain]

            for slice_idx, patch_paths in enumerate(slice_lists):

                if not patch_paths:
                    continue

                selected_paths = list(patch_paths)

                if per_slice_cap and len(selected_paths) > per_slice_cap:
                    selected_paths = selected_paths[:per_slice_cap]

                for patch_path in selected_paths:
                    total_selected += 1

                    basename = os.path.splitext(
                        os.path.basename(patch_path)
                    )[0]

                    embedding_path = os.path.join(
                        embeddings_dir,
                        basename + ".pt"
                    )

                    if not os.path.isfile(embedding_path):
                        missing_records.append({
                            "split": split_name,
                            "case_id": case_id,
                            "stain": stain,
                            "slice_idx": slice_idx,
                            "patch_path": patch_path,
                            "expected_embedding": embedding_path,
                            "reason": "missing_embedding",
                        })

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir,
        f"preflight_missing_embeddings_{split_name}.csv"
    )

    fieldnames = [
        "split",
        "case_id",
        "stain",
        "slice_idx",
        "patch_path",
        "expected_embedding",
        "reason",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_records)

    print(f"\nSELECTED PATCH AUDIT: {split_name}")
    print(f"Selected patches checked: {total_selected}")
    print(f"Missing embeddings: {len(missing_records)}")
    print(f"CSV saved to: {csv_path}")

    if missing_records:
        raise RuntimeError(
            f"{len(missing_records)} selected patches in "
            f"{split_name} have missing embeddings. "
            f"See: {csv_path}"
        )

    return csv_path
