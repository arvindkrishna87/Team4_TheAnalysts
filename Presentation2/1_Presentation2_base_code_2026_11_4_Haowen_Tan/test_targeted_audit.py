
import os
import csv
import tempfile

from dataset import StainBagCasePooledFeatureDataset

# Use a temporary output directory.
with tempfile.TemporaryDirectory() as tmp:
    existing = "/projects/e32998/patches_benign_split/case_19_unmatched_2_melan_patch95.png"
    missing = "/projects/e32998/patches_benign_split/case_999_test_melan_patch999.png"

    case_dict = {
        999: {
            "melan": [[existing, missing]]
        }
    }
    labels = {999: 0}

    # TEST 1: The missing patch is outside the one-patch cap.
    ds = StainBagCasePooledFeatureDataset(
        case_dict, labels,
        embeddings_dir="/projects/e32998/patches_varsize_pooled4096",
        stains=("melan",),
        per_slice_cap=1,
        shuffle_patches=False,
        audit_dir=tmp,
        split_name="test",
    )

    ds[0]
    csv_path = os.path.join(tmp, "missing_embeddings_test.csv")
    assert not os.path.exists(csv_path)
    print("PASS: Unselected missing patch was not logged")

    # TEST 2: Select both patches.
    ds.per_slice_cap = 2

    try:
        ds[0]
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass

    assert os.path.isfile(csv_path)

    with open(csv_path, newline="") as f:
        records = list(csv.DictReader(f))

    assert len(records) == 1
    assert records[0]["patch_path"] == missing
    assert records[0]["case_id"] == "999"
    assert records[0]["split"] == "test"

    print("PASS: Selected missing patch was recorded in CSV")
    print("PASS: FileNotFoundError was raised")
    print("\nSUCCESS: Targeted missing-embedding audit passed!")
