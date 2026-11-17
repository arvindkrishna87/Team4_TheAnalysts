"""
augment_transforms.py  --  Team 4 (TheAnalysts), Presentation 3, Consultant: augmentation

Part 1 (the recommendation): augmentations that keep tissue patterns, with exact amounts.
Part 2 (optional extra, not used on the slides): embedding-level noise / intra-slide mixing
and the Random / Scheduled / Triggered strategy controller.

Part 1 works on image patches (3 x H x W tensors) BEFORE embedding with UNI2.
"""
import csv
import random
from typing import Dict, List

import torch
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Part 1. Safe image augmentations (slide 3)
# ---------------------------------------------------------------------------
def rot90_random(x: torch.Tensor) -> torch.Tensor:
    """Rotate a (C, H, W) patch by 0, 90, 180 or 270 degrees (no interpolation, no blur)."""
    k = random.randint(0, 3)
    return torch.rot90(x, k, dims=(1, 2))


class Rot90:
    def __call__(self, x):
        return rot90_random(x)


def safe_augmentations(color: bool = True, stain_shift: bool = False) -> T.Compose:
    """
    Pipeline for PIL / tensor patches.
      flips          p = 0.5 each (horizontal and vertical)
      rotation       0 / 90 / 180 / 270 degrees
      brightness     +-10%      (ColorJitter brightness=0.1)
      contrast       +-10%      (ColorJitter contrast=0.1)
      stain_shift    saturation +-10%, hue +-0.02 -- OFF by default: stain color may carry
                     signal, so test it first (our reasoning, not a published result)
    Not used: resize, crop, scale, blur, noise, elastic, grayscale (they change cell size,
    texture or detail that the classifier may rely on).
    """
    steps = [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]
    if color:
        steps.append(T.ColorJitter(brightness=0.1, contrast=0.1))
    if stain_shift:
        steps.append(T.ColorJitter(saturation=0.1, hue=0.02))
    steps.append(T.ToTensor())
    steps.append(Rot90())
    return T.Compose(steps)


# Usage (before embedding with UNI2; each variation needs its own embedding pass):
#   from PIL import Image
#   aug = safe_augmentations()
#   x = aug(Image.open("patch.png").convert("RGB"))   # (3, H, W) tensor, then normalize + embed

# ---------------------------------------------------------------------------
# Part 2. Optional extra: embedding-level augmentation + strategies
# ---------------------------------------------------------------------------

STRATEGIES = ("none", "random", "scheduled", "triggered")


def add_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x: (N, D) patch embeddings of one slice."""
    if sigma <= 0:
        return x
    return x + sigma * x.std() * torch.randn_like(x)


def intra_slide_mix(x: torch.Tensor, max_lam: float) -> torch.Tensor:
    """Blend every patch with a random patch from the same slice, weight lam ~ U(0, max_lam)."""
    n = x.shape[0]
    if max_lam <= 0 or n < 2:
        return x
    j = torch.randperm(n, device=x.device)
    lam = torch.rand(n, 1, device=x.device) * max_lam
    return (1.0 - lam) * x + lam * x[j]


def augment_slice(x: torch.Tensor, sigma: float, max_lam: float) -> torch.Tensor:
    x = x.float()
    return add_noise(intra_slide_mix(x, max_lam), sigma)


class AugController:
    """
    strategy   none       never augment (baseline)
               random     each case, each step: augment at full strength with probability p
               scheduled  every case, strength ramps 0 -> 1 linearly over ramp_epochs
               triggered  off until the overfitting signal fires, then on at full strength;
                          signal = (val_loss - train_loss) > gap_on; turns off again when < gap_off
    sigma, mix are the FULL-strength noise level and maximum mixing weight.
    Call apply(stain_slices) for every training case, end_epoch(...) once per epoch.
    """

    def __init__(self, strategy="none", sigma=0.1, mix=0.3, p=0.5, ramp_epochs=20,
                 gap_on=0.10, gap_off=0.05, seed=0, log_path=None):
        assert strategy in STRATEGIES, f"strategy must be one of {STRATEGIES}"
        self.strategy, self.sigma, self.mix, self.p = strategy, sigma, mix, p
        self.ramp_epochs, self.gap_on, self.gap_off = ramp_epochs, gap_on, gap_off
        self.rng = random.Random(seed)
        self.epoch = 0
        self.on = False            # triggered state
        self.first_on_epoch = None
        self.log_path = log_path
        self.n_aug = 0
        self.n_seen = 0
        if log_path:
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "strategy", "strength", "frac_cases_augmented",
                                        "train_loss", "val_loss", "gap"])

    def strength(self) -> float:
        if self.strategy == "none":
            return 0.0
        if self.strategy == "random":
            return 1.0
        if self.strategy == "scheduled":
            return min(1.0, self.epoch / max(1, self.ramp_epochs))
        return 1.0 if self.on else 0.0           # triggered

    def apply(self, stain_slices: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        self.n_seen += 1
        s = self.strength()
        if s <= 0.0 or (self.strategy == "random" and self.rng.random() > self.p):
            return stain_slices
        self.n_aug += 1
        return {stain: [augment_slice(x, self.sigma * s, self.mix * s) for x in sl]
                for stain, sl in stain_slices.items()}

    def end_epoch(self, epoch: int, train_loss: float, val_loss: float):
        gap = float(val_loss) - float(train_loss)
        if self.log_path:
            frac = self.n_aug / max(1, self.n_seen)
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, self.strategy, round(self.strength(), 3), round(frac, 3),
                                        round(float(train_loss), 5), round(float(val_loss), 5), round(gap, 5)])
        self.n_aug = self.n_seen = 0
        self.epoch = epoch + 1                   # strength used in the NEXT epoch
        if self.strategy == "triggered":
            if not self.on and gap > self.gap_on:
                self.on = True
                if self.first_on_epoch is None:
                    self.first_on_epoch = epoch + 1
            elif self.on and gap < self.gap_off:
                self.on = False

