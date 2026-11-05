# Purpose: Check correspondence independently of the full evaluation.
import torch
from attention_analysis import compute_effective_patch_attention

# Three patches in model input order (not numerical order)
paths = {
    "h&e": [[
        "/test/patch_10.png",
        "/test/patch_2.png",
        "/test/patch_1.png",
    ]]
}

# Simulated attention weights in exactly the same order
attention_weights = {
    "case_weights": torch.tensor([1.0]),
    "stain_order": ["h&e"],
    "stain_weights": {
        "h&e": {
            "slice_weights": torch.tensor([1.0]),
            "patch_weights": [
                torch.tensor([0.2, 0.7, 0.1])
            ]
        }
    }
}

records = compute_effective_patch_attention(
    case_id=999,
    attention_weights=attention_weights,
    stain_patch_paths=paths
)

expected = {
    "patch_10.png": 0.2,
    "patch_2.png": 0.7,
    "patch_1.png": 0.1
}

for record in records:
    filename = record["patch_path"].split("/")[-1]
    score = record["patch_attn_weight"]

    assert abs(score - expected[filename]) < 1e-6
    print(f"PASS: {filename} -> attention {score:.2f}")

assert len(records) == 3
print("\nSUCCESS: All attention scores match their patch filenames!")
