
import csv
import os
import tempfile

from torch.utils.data import DataLoader

from dataset import (
    StainBagCasePooledFeatureDataset,
    case_collate_fn,
)

EMB_DIR = "/projects/e32998/patches_varsize_pooled4096"

with tempfile.TemporaryDirectory() as audit_dir:
    case_dict = {
        901: {"melan": [["case_901_test_melan_patch1.png"]]},
        902: {"melan": [["case_902_test_melan_patch1.png"]]},
        903: {"melan": [["case_903_test_melan_patch1.png"]]},
    }

    label_map = {901: 0, 902: 0, 903: 0}

    dataset = StainBagCasePooledFeatureDataset(
        case_dict,
        label_map,
        embeddings_dir=EMB_DIR,
        stains=("melan",),
        per_slice_cap=1,
        shuffle_patches=False,
        audit_dir=audit_dir,
        split_name="test",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True,
    )

    errors = 0

    iterator = iter(loader)

    for _ in range(len(dataset)):
        try:
            next(iterator)
        except FileNotFoundError:
            errors += 1

    csv_path = os.path.join(
        audit_dir, "missing_embeddings_test.csv"
    )

    assert os.path.isfile(csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert errors == 3, f"Expected 3 errors, got {errors}"
    assert len(rows) == 3, f"Expected 3 CSV rows, got {len(rows)}"
    assert {r["case_id"] for r in rows} == {
        "901", "902", "903"
    }

    print("PASS: All 3 missing embeddings raised errors")
    print("PASS: All 3 records were written to the CSV")
    print("PASS: Multiple workers wrote without CSV corruption")
    print("\nSUCCESS: Multi-worker audit integration test passed!")
