import re
from pathlib import Path

import pandas as pd

# Paths
BASE_DIR = Path("/home/wtq4672/Presentation3_base_code")
JOB_ID = "9375219"
OUTPUT_DIR = BASE_DIR / "misclassification_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

all_predictions = []

# Load predictions from each of the 50 SLURM jobs
for task_id in range(50):
    log_file = BASE_DIR / f"p3_array_{JOB_ID}_{task_id}.out"
    log_text = log_file.read_text()

    # Find the exact results directory recorded in the job log
    match = re.search(r"Created run directory:\s*(\S+)", log_text)
    if not match:
        raise ValueError(f"Run directory not found in {log_file}")

    run_dir = Path(match.group(1))
    predictions_file = run_dir / "predictions_best.csv"

    if not predictions_file.exists():
        raise FileNotFoundError(predictions_file)

    df = pd.read_csv(predictions_file)

    # Recover repetition and fold from the SLURM array task ID
    df["repetition"] = task_id // 5 + 1
    df["fold"] = task_id % 5 + 1
    df["task_id"] = task_id

    all_predictions.append(df)

# Combine all 50 folds
predictions = pd.concat(all_predictions, ignore_index=True)

# Verify that every case has exactly 10 held-out predictions
counts = predictions.groupby("case_id").size()
assert len(counts) == 93, f"Expected 93 cases, found {len(counts)}"
assert (counts == 10).all(), "Some cases do not have exactly 10 predictions"

# Verify every case appears once per repetition
assert not predictions.duplicated(["case_id", "repetition"]).any(), \
    "A case appears more than once in a repetition"

# Verify true labels are consistent
assert (predictions.groupby("case_id")["true_label"].nunique() == 1).all(), \
    "Inconsistent true labels found"

# Calculate misclassification frequency
predictions["misclassified"] = (
    predictions["true_label"] != predictions["predicted_label"]
).astype(int)

summary = (
    predictions.groupby("case_id")
    .agg(
        true_label=("true_label", "first"),
        times_misclassified=("misclassified", "sum"),
        total_predictions=("misclassified", "count"),
        mean_prob_class1=("prob_class1", "mean"),
    )
    .reset_index()
)

summary["misclassification_rate"] = (
    summary["times_misclassified"] / summary["total_predictions"]
)

# Categorize cases
def categorize_case(n):
    if n >= 6:
        return "Frequently misclassified"
    elif n >= 1:
        return "Sometimes misclassified"
    return "Never misclassified"

summary["category"] = summary["times_misclassified"].apply(categorize_case)

summary = summary.sort_values(
    ["times_misclassified", "case_id"],
    ascending=[False, True]
)

# Save results
predictions.to_csv(OUTPUT_DIR / "all_930_predictions.csv", index=False)
summary.to_csv(OUTPUT_DIR / "case_misclassification_summary.csv", index=False)

# Display results
print("\nSUCCESS: All 50 runs loaded and validated!")
print(f"Total predictions: {len(predictions)}")
print(f"Unique cases: {len(summary)}")

print("\nMisclassification categories:")
print(summary["category"].value_counts().to_string())

print("\nTop 15 most frequently misclassified cases:")
print(
    summary[
        ["case_id", "true_label", "times_misclassified",
         "total_predictions", "category"]
    ].head(15).to_string(index=False)
)

print(f"\nResults saved to: {OUTPUT_DIR}")