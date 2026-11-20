
"""
STAT 390 - Presentation 3
Data Engineer Choice 2: Patch Features

Creates a patch-level CSV containing:
- Filename and case ID
- Diagnosis
- Slice identifier
- Stain
- Width and height
- Patch area in pixels squared

Also flags unusual filenames for later investigation.
"""

import os
import re
import csv
import argparse
from collections import Counter
from PIL import Image

PATCH_DIR = "/projects/e32998/patch_images"
LABELS_CSV = "../extra_benign_case_grade_match.csv"

VALID_CLASSES = {1.0, 3.0, 4.0}

# Matches case ID, match/unmatched slice, and patch number.
# The stain portion is extracted separately.
PATTERN = re.compile(
    r"^case_(\d+)_(match|unmatched)_?(\d+)_(.+)_patch(\d+)\.png$",
    re.IGNORECASE
)


def identify_stain(raw_stain):
    """Recognize standard stain names without guessing."""
    stain = raw_stain.lower()

    if stain in ("h&e", "he"):
        return "h&e"

    if stain in ("melan", "melan-a"):
        return "melan"

    if stain == "sox10":
        return "sox10"

    return "unrecognized"


def load_labels():
    """Read diagnosis labels from the original CSV."""
    labels = {}

    with open(LABELS_CSV, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            case_id = int(row["Case"])
            diagnosis = float(row["Class"])
            labels[case_id] = diagnosis

    return labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of eligible PNGs to inspect"
    )

    parser.add_argument(
        "--output",
        default="patch_features.csv"
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    # Prevent accidentally overwriting previous results.
    if os.path.exists(args.output):
        parser.error(
            "Output already exists: " + args.output
        )

    labels = load_labels()

    filenames = sorted(
        filename
        for filename in os.listdir(PATCH_DIR)
        if filename.lower().endswith(".png")
    )

    counts = Counter()
    cases = set()

    columns = [
        "filename",
        "case_id",
        "original_class",
        "grade",
        "slice_id",
        "stain",
        "raw_stain",
        "patch_number",
        "width_px",
        "height_px",
        "area_px2",
        "filename_status",
        "image_status"
    ]

    with open(args.output, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        for filename in filenames:

            # Read case ID even when the rest of the
            # filename has an unexpected format.
            case_match = re.match(
                r"^case_(\d+)_",
                filename,
                re.IGNORECASE
            )

            if not case_match:
                counts["unparsed_case_id"] += 1
                continue

            case_id = int(case_match.group(1))

            # Exclude cases outside the model's
            # configured diagnosis classes.
            if labels.get(case_id) not in VALID_CLASSES:
                counts["excluded_pngs"] += 1
                continue

            counts["eligible_pngs"] += 1
            cases.add(case_id)

            match = PATTERN.match(filename)

            slice_id = ""
            stain = "unrecognized"
            raw_stain = ""
            patch_number = ""
            filename_status = "unrecognized"

            if match:
                match_type = match.group(2).lower()
                slice_number = int(match.group(3))
                raw_stain = match.group(4).lower()
                patch_number = int(match.group(5))

                slice_id = (
                    f"{match_type}_{slice_number}"
                )

                stain = identify_stain(raw_stain)

                if stain != "unrecognized":
                    filename_status = "recognized"

            # Determine the diagnosis.
            original_class = labels[case_id]

            if original_class == 1.0:
                grade = "Benign"
            else:
                grade = "High-grade"

            # Read actual image dimensions.
            image_path = os.path.join(
                PATCH_DIR, filename
            )

            width = ""
            height = ""
            area = ""
            image_status = "ok"

            try:
                with Image.open(image_path) as img:
                    width, height = img.size

                area = width * height

            except (OSError, ValueError):
                image_status = "read_error"
                counts["read_errors"] += 1

            if filename_status == "unrecognized":
                counts["unrecognized_filenames"] += 1

            writer.writerow({
                "filename": filename,
                "case_id": case_id,
                "original_class": original_class,
                "grade": grade,
                "slice_id": slice_id,
                "stain": stain,
                "raw_stain": raw_stain,
                "patch_number": patch_number,
                "width_px": width,
                "height_px": height,
                "area_px2": area,
                "filename_status": filename_status,
                "image_status": image_status
            })

            counts["rows_written"] += 1

            if (
                args.limit is not None
                and counts["rows_written"] >= args.limit
            ):
                break

    print("\nPATCH FEATURE EXTRACTION COMPLETE")
    print("=" * 55)

    print("Eligible PNGs inspected:",
          counts["eligible_pngs"])

    print("Rows written:",
          counts["rows_written"])

    print("Distinct cases:",
          len(cases))

    print("Unrecognized filenames:",
          counts["unrecognized_filenames"])

    print("Image read errors:",
          counts["read_errors"])

    print("Output:", args.output)

    if args.limit is not None:
        print("NOTE: This was a limited test.")


if __name__ == "__main__":
    main()

