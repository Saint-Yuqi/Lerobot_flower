"""Per-__getitem__ photometric "look" augmentation (Task1-extra variants).

Reproduces, on-the-fly, the 6 lighting looks that `task1_20260509_plus` had
baked into its videos (via ffmpeg `eq`/`colorbalance` at dataset-build time),
so a policy trained on a clean dataset sees the same lighting distribution.

Looks are sampled per call via torch's global RNG (same mechanism as
torchvision ColorJitter) so each DataLoader worker gets torch's per-worker
seeded stream — no pickled-RNG pitfall. Train-only (wired through the
train-only image_transforms hook); val stays clean.

ffmpeg math is ported here, but the *binding* fidelity check is the
ffmpeg-reference fixture test (see the design spec): the recalled
`colorbalance` constants must be validated, not trusted.

Filter strings (exact, from the dataset's meta/*_metadata.jsonl):
  bright_high_contrast  eq=brightness=0.090:contrast=1.12:saturation=1.05
  dim_low_contrast      eq=brightness=-0.090:contrast=0.88:saturation=0.95
  warm_yellow_light     eq=brightness=0.035:contrast=1.07:saturation=1.12,
                         colorbalance=rs=0.060:gs=0.015:bs=-0.055
  cool_blue_light       eq=brightness=-0.025:contrast=1.05:saturation=1.08,
                         colorbalance=rs=-0.050:gs=-0.006:bs=0.065
  gamma_lifted_shadows  eq=gamma=0.82:brightness=0.020:contrast=1.08:saturation=1.02
  original              (identity)
"""
from __future__ import annotations

import torch

# name -> (brightness, contrast, saturation, gamma, (rs, gs, bs) | None).
# Insertion order is significant: it fixes the categorical index order so a
# given torch RNG state maps to a stable look. `original` first.
LOOKS: dict[str, tuple[float, float, float, float, tuple[float, float, float] | None]] = {
    "original":             (0.0,    1.0,  1.0,  1.0,  None),
    "bright_high_contrast": (0.090,  1.12, 1.05, 1.0,  None),
    "dim_low_contrast":     (-0.090, 0.88, 0.95, 1.0,  None),
    "warm_yellow_light":    (0.035,  1.07, 1.12, 1.0,  (0.060, 0.015, -0.055)),
    "cool_blue_light":      (-0.025, 1.05, 1.08, 1.0,  (-0.050, -0.006, 0.065)),
    "gamma_lifted_shadows": (0.020,  1.08, 1.02, 0.82, None),
}

# task1_20260509_plus per-episode counts (765 ep) from its metadata.
_EMPIRICAL = {
    "original": 165, "bright_high_contrast": 119, "dim_low_contrast": 123,
    "warm_yellow_light": 122, "cool_blue_light": 113, "gamma_lifted_shadows": 123,
}

# BT.601 luma (ffmpeg eq saturation operates on chroma in YUV; this is the
# documented RGB approximation, gated by the fixture test).
_LUMA = (0.299, 0.587, 0.114)


def apply_look(img: torch.Tensor, name: str) -> torch.Tensor:
    """Apply one named look to a CHW float32 [0,1] tensor. Out-of-place.

    `original` returns the input unchanged (bit-exact). Every other path
    allocates new tensors and never mutates `img` (the frame-cache path hands
    in a read-only memmap-derived array).
    """
    if name == "original":
        return img
    brightness, contrast, saturation, gamma, colorbal = LOOKS[name]

    v = img
    if brightness != 0.0 or contrast != 1.0 or gamma != 1.0 or saturation != 1.0:
        # ffmpeg `eq` operates in YUV: brightness/contrast/gamma on Y,
        # saturation on Cb/Cr. Replicate in BT.601 full-range YCbCr, not
        # per-RGB-channel (the latter shifts hue — fails the ffmpeg gate).
        r, g, b = v[0], v[1], v[2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        if contrast != 1.0 or brightness != 0.0:
            y = (y - 0.5) * contrast + 0.5 + brightness
        if gamma != 1.0:
            y = y.clamp(0.0, 1.0).pow(1.0 / gamma)
        if saturation != 1.0:
            cb = (cb - 0.5) * saturation + 0.5
            cr = (cr - 0.5) * saturation + 0.5
        cb0, cr0 = cb - 0.5, cr - 0.5
        v = torch.stack([
            y + 1.402 * cr0,
            y - 0.344136 * cb0 - 0.714136 * cr0,
            y + 1.772 * cb0,
        ], dim=0)

    if colorbal is not None:
        # ffmpeg colorbalance, shadows-only (rs/gs/bs; mid/high = 0).
        # l = per-pixel lightness; shadow weight ramps strong->0 dark->light.
        mx = v.amax(dim=0)
        mn = v.amin(dim=0)
        lightness = (mx + mn) * 0.5
        shadow_w = ((0.333 - lightness) * 4.0 + 0.5).clamp(0.0, 1.0) * 0.7
        shift = torch.tensor(colorbal, dtype=v.dtype, device=v.device).view(3, 1, 1)
        v = v + shift * shadow_w.unsqueeze(0)

    return v.clamp(0.0, 1.0)


def _weights(distribution: str) -> torch.Tensor:
    names = list(LOOKS)
    if distribution == "uniform":
        w = [1.0] * len(names)
    elif distribution == "task1plus_empirical":
        w = [float(_EMPIRICAL[n]) for n in names]
    else:
        raise ValueError(
            f"unknown distribution {distribution!r} "
            f"(expected 'uniform' or 'task1plus_empirical')"
        )
    t = torch.tensor(w, dtype=torch.float32)
    return t / t.sum()


class LookJitter(torch.nn.Module):
    """Pick one of the 6 looks per call (torch global RNG) and apply it.

    Always runs (`original` is its built-in no-op outcome) — it is a leading
    transform, not a member of the random-subset pool.
    """

    def __init__(self, distribution: str = "uniform") -> None:
        super().__init__()
        self.distribution = distribution
        self._names = list(LOOKS)
        self.register_buffer("_w", _weights(distribution), persistent=False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        idx = int(torch.multinomial(self._w, 1).item())
        return apply_look(img, self._names[idx])

    def __repr__(self) -> str:
        return f"LookJitter(distribution={self.distribution!r}, looks={self._names})"
