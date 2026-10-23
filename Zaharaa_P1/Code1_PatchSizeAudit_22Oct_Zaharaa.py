"""
Code1_PatchSizeAudit_22Oct_Zaharaa.py
Consultant evidence: how small are the patches the model uses, and does attention favor them?

precompute_pooled_features.py feeds each patch to KimiaNet (DenseNet-121) at its ORIGINAL size
and keeps every patch >= 32 px (MIN_PATCH_SIZE, line 32). DenseNet shrinks images by 32x, so a
32-63 px patch becomes a single 1x1 cell before pooling. KimiaNet was trained on 1000 px patches
(about 31x31 cells). This script measures how many patches are that small.

Run: python3 Code1_PatchSizeAudit_22Oct_Zaharaa.py --cutoff 128 [--top_csv path/to/top_effective_patches_per_case_5.0pct.csv]
Outputs (./patch_size_audit/): patch_sizes.csv, summary.txt, patch_size_hist.png
"""

import os
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PT_DIR = "/projects/e32998/patches_varsize_pooled4096"
PNG_DIR = "/projects/e32998/patches_benign_split"
STRIDE = 32

CASE_RE = re.compile(r"case_(\d+)_", re.IGNORECASE)
STAINS = ("h&e", "melan", "sox10")


def _stain(name):
    low = name.lower()
    for s in STAINS:
        if s in low:
            return s
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=128)
    ap.add_argument("--top_csv", default=None)
    ap.add_argument("--out", default="patch_size_audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cut = args.cutoff

    pts = sorted(glob.glob(os.path.join(PT_DIR, "*.pt")))
    print(f"Found {len(pts)} .pt patches; reading matching PNG sizes...")

    rows, missing = [], 0
    for i, pt in enumerate(pts):
        base = os.path.basename(pt)[:-3]
        png = os.path.join(PNG_DIR, base + ".png")
        try:
            with Image.open(png) as im:
                w, h = im.size
        except Exception:
            missing += 1
            continue
        m = CASE_RE.search(base)
        rows.append({"file": base, "case": int(m.group(1)) if m else -1, "stain": _stain(base),
                     "w": w, "h": h, "short_side": min(w, h),
                     "feature_cells": (w // STRIDE) * (h // STRIDE)})
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1} done")

    with open(os.path.join(args.out, "patch_sizes.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    short = np.array([r["short_side"] for r in rows])
    cells = np.array([r["feature_cells"] for r in rows])
    L = []
    P = L.append
    P(f"Patches measured: {len(rows)}  (no PNG found: {missing})")
    P(f"Shorter side: median {np.median(short):.0f} px, 10th pct {np.percentile(short, 10):.0f}, "
      f"90th pct {np.percentile(short, 90):.0f}, min {short.min()}")
    for t in (64, 96, cut):
        P(f"  share with shorter side < {t} px: {100 * np.mean(short < t):.1f}%")
    P(f"Share that become a single 1x1 cell inside DenseNet: {100 * np.mean(cells <= 1):.1f}%")
    P(f"Median feature cells per patch: {np.median(cells):.0f}  (a 1000 px training patch has ~961)")

    P(f"\nBy stain (share < {cut} px):")
    by_stain = defaultdict(list)
    for r in rows:
        by_stain[r["stain"]].append(r["short_side"])
    for s, v in sorted(by_stain.items()):
        v = np.array(v)
        P(f"  {s}: n={len(v)}, median {np.median(v):.0f} px, {100 * np.mean(v < cut):.1f}% < {cut} px")

    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r["short_side"])
    lost = sorted(c for c, v in by_case.items() if max(v) < cut)
    P(f"\nCases: {len(by_case)}. Cases with NO patch >= {cut} px: {len(lost)} {lost[:20]}")
    P(f"Patches kept at {cut} px: {100 * np.mean(short >= cut):.1f}%")

    if args.top_csv and os.path.exists(args.top_csv):
        from attention_analysis import _lookup_png_for_patch
        sizes = []
        with open(args.top_csv) as f:
            for rec in csv.DictReader(f):
                png = _lookup_png_for_patch(int(rec["case_id"]), rec["stain"],
                                            int(rec["slice_idx"]), int(rec["patch_idx"]))
                if png and os.path.exists(png):
                    with Image.open(png) as im:
                        sizes.append(min(im.size))
        if sizes:
            sizes = np.array(sizes)
            P(f"\nTop-5% attention patches measured: {len(sizes)}")
            P(f"  share < {cut} px among TOP-attention patches: {100 * np.mean(sizes < cut):.1f}%")
            P(f"  share < {cut} px among ALL patches:           {100 * np.mean(short < cut):.1f}%")
            P(f"  median shorter side, top vs all: {np.median(sizes):.0f} px vs {np.median(short):.0f} px")

    text = "\n".join(L)
    print("\n" + text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")

    plt.figure(figsize=(8, 4.5))
    plt.hist(np.clip(short, 0, 600), bins=60, color="#6B7280")
    plt.axvline(32, color="#111827", linestyle=":", label="Current minimum 32 px")
    plt.axvline(cut, color="#DC2626", linestyle="--", label=f"Proposed minimum {cut} px")
    plt.xlabel("Shorter side of patch (px, clipped at 600)")
    plt.ylabel("Number of patches")
    plt.title("Patch sizes fed to KimiaNet")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "patch_size_hist.png"), dpi=200)
    print(f"\nSaved to ./{args.out}/")


if __name__ == "__main__":
    main()
