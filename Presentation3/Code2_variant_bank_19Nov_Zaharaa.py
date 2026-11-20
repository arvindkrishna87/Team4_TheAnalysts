"""
Code2_variant_bank_19Nov_Zaharaa.py  --  Team 4 (TheAnalysts), Presentation 3, Consultant: augmentation

Question: how do image augmentations reach a model that trains on PRECOMPUTED UNI2 embeddings?

The model never sees images during training: dataset.py loads one .pt file per patch from
embeddings_dir. So an augmentation only exists once the augmented image has been embedded.

Plan (checked against the Presentation 3 base code):
  1. Stain-specific policy: flips and 90-degree rotations for every stain; light color jitter
     for H&E only; no hue or saturation change on Melan-A or SOX10, where chromogen color
     is the signal.
  2. Variant bank, one folder per variant, same file names as now:
        <bank_root>/v0  -> the existing UNI2 embedding folder (symlink, nothing recomputed)
        <bank_root>/v1, v2, v3 -> embeddings of augmented copies
     Built in two steps per variant (see build_bank.sbatch text at the bottom):
        a. write_variant_images(): save augmented PNGs with the SAME file names
        b. run the team's precompute_pooled_features.py --encoder uni2 --uni2_native_size
           on that folder, so the UNI2 step is identical to the one used for v0.
  3. Strategy = which variant a patch reads in each epoch (Presentation 3 objectives, slide 47):
        none       v0 always (baseline)
        random     each patch draws v0..v{K-1}, new draw every epoch
        scheduled  every patch uses v_(epoch mod K)
        triggered  v0 until (val_loss - train_loss) > gap_on, then random, and stays on
     Validation and test always read v0.

Two details that matter in the Presentation 3 code:
  * dataset.py calls _patch_path_to_feat_path() twice per patch (existence check, then load).
    The pick below is a pure function of (seed, epoch, patch name), so both calls agree.
  * main.py builds the train loader with persistent_workers=True. Workers then keep a frozen
    copy of the selector and never see the epoch change. Set persistent_workers=False on the
    TRAIN loader (or num_workers=0); workers are then rebuilt each epoch with the current state.
"""
import hashlib
import os
import random
from typing import Optional

from PIL import Image
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# 1. Stain-specific policy
# ---------------------------------------------------------------------------
# Geometric for every stain: flips and 90-degree rotations via PIL transpose (pixels move, no
# interpolation, no padded corners). A 90-degree turn swaps width and height, so the
# precompute script's 32x32 minimum skips exactly the same patches as for v0.
# Color: H&E gets light jitter (Tellez et al. 2019). Melan-A and SOX10: brightness and contrast
# only, +-5%. No hue or saturation: the red or brown chromogen marks positive cells, and melanin
# is also brown, so a hue shift can blur that difference.
COLOR_POLICY = {
    "h&e":   dict(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),  # hue 0.02 = ~7 degrees
    "melan": dict(brightness=0.05, contrast=0.05, saturation=0.0, hue=0.0),
    "sox10": dict(brightness=0.05, contrast=0.05, saturation=0.0, hue=0.0),
}

_ROTATE = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]


def stain_of(fname: str) -> Optional[str]:
    """Same rule as data_utils.extract_stain_from_filename."""
    f = fname.lower()
    if "h&e" in f or "_he_" in f:
        return "h&e"
    if "melan" in f:
        return "melan"
    if "sox10" in f:
        return "sox10"
    return None


def _hash_int(*parts) -> int:
    return int(hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def make_variant(img: Image.Image, fname: str, k: int) -> Image.Image:
    """Variant k of one patch; k = 0 returns the original. Fixed seed per (patch, k)."""
    if k == 0:
        return img
    rng = random.Random(_hash_int(fname, k))
    p = COLOR_POLICY[stain_of(fname) or "h&e"]
    out = img
    if rng.random() < 0.5:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    r = rng.randint(0, 3)
    if r:
        out = out.transpose(_ROTATE[r])
    out = TF.adjust_brightness(out, 1 + rng.uniform(-p["brightness"], p["brightness"]))
    out = TF.adjust_contrast(out, 1 + rng.uniform(-p["contrast"], p["contrast"]))
    if p["saturation"]:
        out = TF.adjust_saturation(out, 1 + rng.uniform(-p["saturation"], p["saturation"]))
    if p["hue"]:
        out = TF.adjust_hue(out, rng.uniform(-p["hue"], p["hue"]))
    return out


# ---------------------------------------------------------------------------
# 2. Build the bank: write augmented PNGs, then embed them with the team's own script
# ---------------------------------------------------------------------------
def write_variant_images(patches_dir: str, out_dir: str, k: int,
                         num_shards: int = 1, shard_index: int = 0, overwrite: bool = False) -> int:
    """
    Save variant k of every PNG in patches_dir to out_dir under the SAME file name.
    Resumable (skips files that exist) and shardable like precompute_pooled_features.py.
    Returns the number of images written.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(patches_dir) if f.lower().endswith(".png"))
    written = 0
    for fname in names[shard_index::num_shards]:
        dst = os.path.join(out_dir, fname)
        if os.path.exists(dst) and not overwrite:
            continue
        try:
            img = Image.open(os.path.join(patches_dir, fname)).convert("RGB")
        except OSError:
            continue  # the precompute script also skips unreadable files
        make_variant(img, fname, k).save(dst)  # PNG is lossless
        written += 1
    return written


# ---------------------------------------------------------------------------
# 3. Strategy: which variant a patch uses in a given epoch
# ---------------------------------------------------------------------------
class VariantSelector:
    def __init__(self, strategy: str = "none", K: int = 4, gap_on: float = 0.10, seed: int = 0):
        assert strategy in ("none", "random", "scheduled", "triggered")
        self.strategy, self.K, self.gap_on, self.seed = strategy, K, gap_on, seed
        self.epoch = 0
        self.triggered_on = False

    def pick(self, patch_name: str) -> int:
        """
        Variant for this patch in the current epoch. A pure function of
        (seed, epoch, patch name): the same in every worker and on every call.
        """
        if self.strategy == "none":
            return 0
        if self.strategy == "scheduled":
            return self.epoch % self.K
        if self.strategy == "triggered" and not self.triggered_on:
            return 0
        return _hash_int(self.seed, self.epoch, patch_name) % self.K  # random, or triggered once on

    def end_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """Call after each epoch's validation (0-based epoch). Triggered stays on once it fires."""
        self.epoch = epoch + 1
        if self.strategy == "triggered" and (val_loss - train_loss) > self.gap_on:
            self.triggered_on = True


# ---------------------------------------------------------------------------
# 4. Hook into dataset.py: override ONE method
# ---------------------------------------------------------------------------
def make_bank_dataset(base_cls):
    """
    base_cls = dataset.StainBagCasePooledFeatureDataset (Presentation 3 version).
    Returns a subclass that reads <bank_root>/v<k>/<basename>.pt.

    Changes in main.py (create_data_loaders):
        sel = VariantSelector("triggered", K=4, gap_on=0.10, seed=args.seed)
        BankDS = make_bank_dataset(StainBagCasePooledFeatureDataset)
        train_ds = BankDS(train_case_dict, train_label_map, embeddings_dir=BANK_ROOT, selector=sel, ...)
        val_ds / test_ds: unchanged, embeddings_dir=BANK_ROOT + "/v0"
        train_loader = DataLoader(train_ds, ..., persistent_workers=False)   # so workers see the epoch
    Change in trainer.py (MILTrainer.train), right after val_loss is computed:
        if getattr(self, "selector", None): self.selector.end_epoch(epoch, train_loss, val_loss)
      and in main.py: trainer.selector = sel
    UNI2 embeddings are 1536 long, so create_model(..., pooled_dim=1536) as for any UNI2 run.
    """
    class BankDataset(base_cls):
        def __init__(self, *args, selector: VariantSelector, **kwargs):
            super().__init__(*args, **kwargs)
            self.selector = selector

        def _patch_path_to_feat_path(self, patch_path: str) -> str:
            stem = os.path.splitext(os.path.basename(patch_path))[0]
            k = self.selector.pick(stem)
            return os.path.join(self.embeddings_dir, f"v{k}", stem + ".pt")

    return BankDataset


# ---------------------------------------------------------------------------
# build_bank.sbatch (copy into its own file; one array task per variant k = 1, 2, 3)
# ---------------------------------------------------------------------------
BUILD_BANK_SBATCH = r"""#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=1-3
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --job-name=variant_bank
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
export HF_HUB_OFFLINE=1          # same as embed_testB_uni2_original.sbatch (UNI2 weights cached)

K=$SLURM_ARRAY_TASK_ID
PATCHES=/projects/e32998/patch_images
BANK=/projects/e32998/STAT390_Fall2026/Team4_TheAnalysts/variant_bank
TMP_PNG=$BANK/png_v$K
V0=<existing UNI2 native-size embedding folder for all 99 cases>

mkdir -p $BANK && ln -sfn $V0 $BANK/v0
python3 -c "from Code2_variant_bank_19Nov_Zaharaa import write_variant_images as w; print(w('$PATCHES', '$TMP_PNG', $K))"
python3 precompute_pooled_features.py --encoder uni2 --uni2_native_size \
    --patches_dir $TMP_PNG --embeddings_dir $BANK/v$K --batch_size 64 --num_workers 6 --log_every 20
rm -rf $TMP_PNG                  # the PNG copies are only needed for embedding
"""
