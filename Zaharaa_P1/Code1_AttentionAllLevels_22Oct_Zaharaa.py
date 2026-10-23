"""
Code1_AttentionAllLevels_22Oct_Zaharaa.py

Extends attention_analysis.py so that every test case gets high- and low-attention
patches at all three levels the assignment asks for:
  - every slice   (ranked by patch weight within that slice)
  - every stain   (ranked by slice weight x patch weight within that stain)
  - the whole case (ranked by effective weight = stain x slice x patch)

It also saves:
  - predictions.csv: true label, predicted label, P(high-grade), outcome for every
    test case, so you can find the misclassified high-grade case quickly
  - mapping_check.csv: for every slice, compares the number of attention weights the
    model produced with the number of .pt patch files the image lookup finds. A
    mismatch means the picture shown may not be the patch that got the weight.
  - all_patch_weights.csv per case: every patch with all its weights

Weights are printed in scientific notation so small values do not show as 0.0000.

How to use: in the script that already calls analyze_attention_weights(...), add
right after that call:

    from Code1_AttentionAllLevels_22Oct_Zaharaa import analyze_all_levels
    analyze_all_levels(model, test_loader, output_dir, top_n=8)

Output goes to <output_dir>/attention_all_levels/
"""

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from config import IMAGE_CONFIG
from attention_analysis import (
    PATCH_PNG_DIR,
    PATCH_PT_DIR,
    _lookup_png_for_patch,
    _build_pt_index_for_case_stain,
    compute_effective_patch_attention,
    _plot_feature_heatmap,
)

# Check this against how labels are encoded in data_utils.py / config.py
LABEL_NAMES = {0: "benign", 1: "highgrade"}


# ---------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------
def _draw_patch(ax, case_id, stain, slice_idx, patch_idx, stain_slices):
    """Draw one patch: PNG via the existing lookup, else tensor fallback."""
    png = _lookup_png_for_patch(case_id, stain, slice_idx, patch_idx, PATCH_PNG_DIR)
    if png and os.path.exists(png):
        try:
            ax.imshow(Image.open(png).convert("RGB"))
            return
        except Exception:
            pass

    slice_list = stain_slices.get(stain, []) or []
    if slice_idx < len(slice_list) and isinstance(slice_list[slice_idx], torch.Tensor):
        sl = slice_list[slice_idx]
        try:
            if sl.dim() == 4 and patch_idx < sl.shape[0]:
                img = np.transpose(sl[patch_idx].detach().cpu().numpy(), (1, 2, 0))
                if img.shape[-1] == 3:
                    mean = np.array(IMAGE_CONFIG["normalize_mean"])
                    std = np.array(IMAGE_CONFIG["normalize_std"])
                    img = np.clip(img * std + mean, 0, 1)
                ax.imshow(img)
                return
            if sl.dim() == 2 and patch_idx < sl.shape[0]:
                vec = sl[patch_idx].detach().cpu().numpy().astype(np.float32)
                _plot_feature_heatmap(ax, vec)
                return
        except Exception:
            pass

    ax.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=9)


def _save_grid(entries, case_id, stain_slices, suptitle, filepath):
    """entries: list of dicts with stain, slice_idx, patch_idx, label."""
    n = len(entries)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, e in zip(axes, entries):
        _draw_patch(ax, case_id, e["stain"], e["slice_idx"], e["patch_idx"], stain_slices)
        ax.set_title(e["label"], fontsize=8)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _top_bottom(records, key, n):
    ranked = sorted(records, key=key)
    k = min(n, len(ranked))
    return list(reversed(ranked[-k:])), ranked[:k]


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def analyze_all_levels(model, test_loader, output_dir, top_n=8):
    root = os.path.join(output_dir, "attention_all_levels")
    os.makedirs(root, exist_ok=True)

    pred_rows = []
    map_rows = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            case_data = batch[0]
            case_id = case_data["case_id"]
            stain_slices = case_data["stain_slices"]

            logits, attn = model(stain_slices, return_attn_weights=True)

            # Prediction and probability of high-grade
            logits = logits.detach().view(1, -1)
            if logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1)[0]
                p_high = float(probs[1].item())
                pred = int(torch.argmax(probs).item())
            else:
                p_high = float(torch.sigmoid(logits)[0, 0].item())
                pred = int(p_high >= 0.5)

            lab = case_data.get("label")
            true = int(lab.view(-1)[0].item()) if isinstance(lab, torch.Tensor) else int(lab)

            outcome = ("correct" if true == pred else "incorrect") + "_" + LABEL_NAMES.get(true, str(true))
            pred_rows.append({
                "case_id": case_id,
                "true_label": LABEL_NAMES.get(true, true),
                "pred_label": LABEL_NAMES.get(pred, pred),
                "p_highgrade": f"{p_high:.4f}",
                "outcome": outcome,
            })

            records = compute_effective_patch_attention(case_id, attn)
            if not records:
                continue

            case_dir = os.path.join(root, f"case_{case_id}_{outcome}")
            os.makedirs(case_dir, exist_ok=True)

            # All patch weights for this case
            with open(os.path.join(case_dir, "all_patch_weights.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                w.writeheader()
                w.writerows(records)

            # 1) Case level: effective weight across all stains and slices
            eff = lambda r: r["effective_weight"]
            top, bottom = _top_bottom(records, eff, top_n)
            for name, entries in (("top", top), ("bottom", bottom)):
                _save_grid(
                    [{**r, "label": f"{r['stain']} s{r['slice_idx']} p{r['patch_idx']}\nEff={eff(r):.2e}"} for r in entries],
                    case_id, stain_slices,
                    f"Case {case_id} ({outcome}) - {name} {len(entries)} patches, whole case",
                    os.path.join(case_dir, "case_level", f"case_{case_id}_{name}.png"),
                )

            # 2) Stain level: slice weight x patch weight within each stain
            stain_w = lambda r: r["slice_attn_weight"] * r["patch_attn_weight"]
            for stain in sorted({r["stain"] for r in records}):
                srecs = [r for r in records if r["stain"] == stain]
                top, bottom = _top_bottom(srecs, stain_w, top_n)
                stain_attn = srecs[0]["stain_attn_weight"]
                for name, entries in (("top", top), ("bottom", bottom)):
                    _save_grid(
                        [{**r, "label": f"s{r['slice_idx']} p{r['patch_idx']}\nW={stain_w(r):.2e}"} for r in entries],
                        case_id, stain_slices,
                        f"Case {case_id} - {stain} (stain weight {stain_attn:.3f}) - {name} {len(entries)} patches",
                        os.path.join(case_dir, "stain_level", f"case_{case_id}_{stain}_{name}.png"),
                    )

            # 3) Slice level: patch weight within every slice
            patch_w = lambda r: r["patch_attn_weight"]
            for stain, s_idx in sorted({(r["stain"], r["slice_idx"]) for r in records}):
                srecs = [r for r in records if r["stain"] == stain and r["slice_idx"] == s_idx]
                top, bottom = _top_bottom(srecs, patch_w, top_n)
                slice_attn = srecs[0]["slice_attn_weight"]

                # Mapping check: weights from the model vs .pt files the lookup finds
                try:
                    slice_ids, slice_map = _build_pt_index_for_case_stain(case_id, stain, PATCH_PT_DIR)
                    slice_name = slice_ids[s_idx] if s_idx < len(slice_ids) else "NOT FOUND"
                    n_files = len(slice_map.get(slice_name, []))
                except Exception:
                    slice_name, n_files = "ERROR", -1
                map_rows.append({
                    "case_id": case_id,
                    "stain": stain,
                    "slice_idx": s_idx,
                    "slice_name_used": slice_name,
                    "n_weights_from_model": len(srecs),
                    "n_pt_files_found": n_files,
                    "counts_match": len(srecs) == n_files,
                })

                for name, entries in (("top", top), ("bottom", bottom)):
                    _save_grid(
                        [{**r, "label": f"p{r['patch_idx']}\nW={patch_w(r):.2e}"} for r in entries],
                        case_id, stain_slices,
                        f"Case {case_id} - {stain} slice {s_idx} ({slice_name}, slice weight {slice_attn:.3f}) - {name}",
                        os.path.join(case_dir, "slice_level", f"case_{case_id}_{stain}_s{s_idx}_{name}.png"),
                    )

    # Summary tables
    if pred_rows:
        with open(os.path.join(root, "predictions.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
            w.writeheader()
            w.writerows(pred_rows)
    if map_rows:
        with open(os.path.join(root, "mapping_check.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(map_rows[0].keys()))
            w.writeheader()
            w.writerows(map_rows)
        n_bad = sum(1 for r in map_rows if not r["counts_match"])
        print(f"Mapping check: {n_bad} of {len(map_rows)} slices have a weight/file count mismatch")

    print(f"All-level attention analysis saved to: {root}")
