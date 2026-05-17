"""Minimal image-augmentation pipeline for FlowerVLA training.

Mirrors upstream lerobot's `ImageTransformsConfig` semantics on a flat YAML
schema. Each scalar in `color_jitter.*` and `sharpness` is the half-width of
a uniform jitter range — `s=0.2` ⇒ factor sampled in `[max(0,1-s), 1+s]`.
`random_crop_pct` is the min area-scale for `RandomResizedCrop` (0.95 ⇒
crop area in `[0.95, 1.0]`). `max_num_transforms` + `random_order` select a
subset each call (the `RandomSubsetApply` pattern).

Applied inside `FlowerSO101Dataset.__getitem__`, post-resize, on the
`(C=3, H, W) float32` tensor in `[0,1]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F


@dataclass
class ImageTransformsConfig:
    enabled: bool = False
    color_jitter: dict[str, float] | None = None
    sharpness: float | None = None
    random_crop_pct: float | None = None
    gaussian_blur: dict[str, Any] | float | None = None
    gaussian_noise: float | None = None
    max_num_transforms: int = 3
    random_order: bool = False


class _SharpnessJitter(torch.nn.Module):
    """Sample sharpness factor uniformly in `[max(0, 1-s), 1+s]` per call."""

    def __init__(self, sharpness: float) -> None:
        super().__init__()
        self.lo = max(0.0, 1.0 - float(sharpness))
        self.hi = 1.0 + float(sharpness)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        factor = self.lo + (self.hi - self.lo) * torch.rand(()).item()
        return F.adjust_sharpness(img, factor)


class _Sequential(torch.nn.Module):
    """Apply submodules in fixed order (leading look + the random subset)."""

    def __init__(self, mods: list[torch.nn.Module]) -> None:
        super().__init__()
        self.mods = torch.nn.ModuleList(mods)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        for m in self.mods:
            img = m(img)
        return img


class _RandomSubsetApply(torch.nn.Module):
    """Pick up to `n_subset` transforms uniformly without replacement and apply them."""

    def __init__(
        self,
        transforms: list[torch.nn.Module],
        max_num_transforms: int,
        random_order: bool,
    ) -> None:
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
        self.n_subset = min(int(max_num_transforms), len(transforms))
        self.random_order = bool(random_order)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.n_subset == 0:
            return img
        indices = torch.randperm(len(self.transforms))[: self.n_subset].tolist()
        if not self.random_order:
            indices.sort()
        for i in indices:
            img = self.transforms[i](img)
        return img


def build_image_transforms(
    cfg: dict[str, Any] | None, image_hw: int
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Build the aug pipeline from a YAML-style dict.

    Returns None when augmentations are disabled or no individual transform
    is configured — caller can skip the hook entirely.
    """
    if not cfg or not cfg.get("enabled", False):
        return None

    transforms: list[torch.nn.Module] = []

    cj = cfg.get("color_jitter") or {}
    if any(float(cj.get(k, 0.0)) > 0 for k in ("brightness", "contrast", "saturation")):
        transforms.append(
            T.ColorJitter(
                brightness=float(cj.get("brightness", 0.0)),
                contrast=float(cj.get("contrast", 0.0)),
                saturation=float(cj.get("saturation", 0.0)),
            )
        )

    sh = cfg.get("sharpness")
    if sh is not None and float(sh) > 0:
        transforms.append(_SharpnessJitter(float(sh)))

    rc = cfg.get("random_crop_pct")
    if rc is not None and float(rc) < 1.0:
        transforms.append(
            T.RandomResizedCrop(
                size=(int(image_hw), int(image_hw)),
                scale=(float(rc), 1.0),
                ratio=(1.0, 1.0),
                antialias=True,
            )
        )

    gb = cfg.get("gaussian_blur")
    if gb:
        if isinstance(gb, dict):
            ks = int(gb.get("kernel_size", 5))
            sig = gb.get("sigma", [0.1, 2.0])
            sigma = (
                (float(sig[0]), float(sig[1]))
                if isinstance(sig, (list, tuple))
                else (0.1, float(sig))
            )
        else:
            ks, sigma = 5, (0.1, float(gb))
        transforms.append(T.GaussianBlur(kernel_size=ks, sigma=sigma))

    # torchvision-native (v2 ≥ 0.18); clip=True keeps images in [0,1]. Keep
    # sigma small — large noise erodes the bowl-color signal the task needs.
    gn = cfg.get("gaussian_noise")
    if gn is not None and float(gn) > 0:
        transforms.append(T.GaussianNoise(mean=0.0, sigma=float(gn), clip=True))

    subset = (
        _RandomSubsetApply(
            transforms,
            max_num_transforms=int(cfg.get("max_num_transforms", len(transforms))),
            random_order=bool(cfg.get("random_order", False)),
        )
        if transforms
        else None
    )

    # Optional leading, always-applied per-call look picker. `original` is its
    # built-in no-op, so it is NOT a member of the random-subset pool.
    vv = cfg.get("visual_variants") or {}
    look = None
    if vv.get("enabled"):
        from src.data.visual_variants import LookJitter
        look = LookJitter(distribution=str(vv.get("distribution", "uniform")))

    if look is not None and subset is not None:
        return _Sequential([look, subset])
    return look or subset
