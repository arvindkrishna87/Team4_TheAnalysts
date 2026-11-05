# Purpose: Check correspondence independently of the full evaluation.

import os
import glob
import torch

from dataset import StainBagCasePooledFeatureDataset

PATCH_DIR = "/projects/e32998/patches_benign_split"
EMBED_DIR = "/projects/e32998/patches_varsize_pooled4096"

# Find patch images that have corresponding embeddings
patch_paths = []

for path in glob.glob(os.path.join(PATCH_DIR, "*.png")):
    basename = os.path.splitext(os.path.basename(path))[0]
    embedding_path = os.path.join(EMBED_DIR, basename + ".pt")

    if os.path.isfile(embedding_path):
        patch_paths.append(path)

    if len(patch_paths) == 5:
        break

assert len(patch_paths) == 5, "Could not find five matching patch/embedding pairs"

# Create a small dataset for testing
dataset = StainBagCasePooledFeatureDataset(
    case_dict={0: {"h&e": [patch_paths]}},
    label_map={0: 0},
    embeddings_dir=EMBED_DIR,
    stains=("h&e",),
    per_slice_cap=5,
    max_slices_per_stain=1,
    shuffle_patches=True,
)

# Load the data
case_data = dataset[0]

embeddings = case_data["stain_slices"]["h&e"][0]
filenames = case_data["stain_patch_paths"]["h&e"][0]

assert len(embeddings) == len(filenames) == 5

# Check each tensor against its actual embedding file
for i, patch_path in enumerate(filenames):
    basename = os.path.splitext(os.path.basename(patch_path))[0]
    embedding_path = os.path.join(EMBED_DIR, basename + ".pt")

    expected = torch.load(embedding_path, map_location="cpu")
    expected = expected.reshape(-1).to(embeddings.dtype)

    assert torch.equal(embeddings[i], expected), (
        f"Incorrect embedding for {patch_path}"
    )

    print(f"PASS: {os.path.basename(patch_path)} matches embedding at position {i}")

print("\nSUCCESS: All five patch filenames match their embeddings!")
