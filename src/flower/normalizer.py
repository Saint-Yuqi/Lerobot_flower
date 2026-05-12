"""Per-dim normalize / unnormalize using a dataset's `meta/stats.json` schema.

Stats schema (mirrors lerobot v3.0):
    stats[key]["min"]   -> ndarray of shape feature_shape
    stats[key]["max"]   ->     "      "
    stats[key]["mean"]  ->     "      "
    stats[key]["std"]   ->     "      "
    (q01, q10, q50, q90, q99 also present but unused here)

Two normalization modes per feature:
    "minmax"  : x -> 2 * (x - min) / (max - min) - 1   (maps to [-1, 1])
    "meanstd" : x -> (x - mean) / std

Inverse:
    "minmax"  : y -> (y + 1) / 2 * (max - min) + min
    "meanstd" : y -> y * std + mean

Saves to a single JSON file so checkpoints are self-contained for inference. No
torch.save'd buffers — we want load/save without env-specific tensor formats.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


NORM_MINMAX = "minmax"
NORM_MEANSTD = "meanstd"
NORM_IDENTITY = "identity"

_EPS = 1e-8


@dataclass
class FeatureNorm:
    """Per-feature normalization params + mode."""
    mode: str
    min: torch.Tensor | None = None
    max: torch.Tensor | None = None
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None

    def to(self, device) -> "FeatureNorm":
        for attr in ("min", "max", "mean", "std"):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v.to(device))
        return self


class FlowerNormalizer:
    """Per-feature normalize/unnormalize that mirrors lerobot's normalizer roles.

    Construct from a stats dict (e.g. ``FlowerSO101Dataset.stats``) plus a
    feature → mode dict, e.g.::

        norms = FlowerNormalizer.from_stats(ds.stats, {
            "observation.state": "meanstd",
            "action":            "minmax",
        })
        batch["action"] = norms.normalize("action", batch["action"])
        ...
        # at inference time:
        raw_action = norms.unnormalize("action", model_output)

    Image features are intentionally NOT normalized here — Florence-2 has its own
    preprocessor on top of the [0, 1] tensors we feed it.
    """

    def __init__(self, features: dict[str, FeatureNorm]) -> None:
        self.features = features

    @classmethod
    def from_stats(
        cls,
        stats: dict[str, dict[str, np.ndarray]],
        modes: dict[str, str],
    ) -> "FlowerNormalizer":
        out: dict[str, FeatureNorm] = {}
        for key, mode in modes.items():
            if mode == NORM_IDENTITY:
                out[key] = FeatureNorm(mode=NORM_IDENTITY)
                continue
            if key not in stats:
                raise KeyError(f"normalizer: stats missing for feature {key!r}")
            s = stats[key]
            if mode == NORM_MINMAX:
                out[key] = FeatureNorm(
                    mode=NORM_MINMAX,
                    min=torch.from_numpy(np.asarray(s["min"], dtype=np.float32)),
                    max=torch.from_numpy(np.asarray(s["max"], dtype=np.float32)),
                )
            elif mode == NORM_MEANSTD:
                out[key] = FeatureNorm(
                    mode=NORM_MEANSTD,
                    mean=torch.from_numpy(np.asarray(s["mean"], dtype=np.float32)),
                    std=torch.from_numpy(np.asarray(s["std"], dtype=np.float32)),
                )
            else:
                raise ValueError(f"unknown normalization mode {mode!r} for {key!r}")
        return cls(out)

    def to(self, device) -> "FlowerNormalizer":
        for fn in self.features.values():
            fn.to(device)
        return self

    def has(self, key: str) -> bool:
        return key in self.features

    # ----------------------------------------------------------- ops

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        fn = self.features.get(key)
        if fn is None or fn.mode == NORM_IDENTITY:
            return x
        if fn.mode == NORM_MINMAX:
            mn = fn.min.to(x.device).to(x.dtype)
            mx = fn.max.to(x.device).to(x.dtype)
            denom = (mx - mn).clamp(min=_EPS)
            return (x - mn) / denom * 2.0 - 1.0
        if fn.mode == NORM_MEANSTD:
            mean = fn.mean.to(x.device).to(x.dtype)
            std = fn.std.to(x.device).to(x.dtype).clamp(min=_EPS)
            return (x - mean) / std
        raise ValueError(f"unknown norm mode {fn.mode!r}")

    def unnormalize(self, key: str, y: torch.Tensor) -> torch.Tensor:
        fn = self.features.get(key)
        if fn is None or fn.mode == NORM_IDENTITY:
            return y
        if fn.mode == NORM_MINMAX:
            mn = fn.min.to(y.device).to(y.dtype)
            mx = fn.max.to(y.device).to(y.dtype)
            return (y + 1.0) / 2.0 * (mx - mn) + mn
        if fn.mode == NORM_MEANSTD:
            mean = fn.mean.to(y.device).to(y.dtype)
            std = fn.std.to(y.device).to(y.dtype)
            return y * std + mean
        raise ValueError(f"unknown norm mode {fn.mode!r}")

    # ----------------------------------------------------------- io

    def save(self, path: str | Path) -> None:
        """Save as a single JSON file so checkpoints are env-agnostic."""
        payload: dict[str, Any] = {"features": {}}
        for k, fn in self.features.items():
            entry: dict[str, Any] = {"mode": fn.mode}
            for attr in ("min", "max", "mean", "std"):
                v = getattr(fn, attr)
                if v is not None:
                    entry[attr] = v.cpu().numpy().tolist()
            payload["features"][k] = entry
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FlowerNormalizer":
        payload = json.loads(Path(path).read_text())
        out: dict[str, FeatureNorm] = {}
        for k, entry in payload["features"].items():
            mode = entry["mode"]
            kwargs: dict[str, Any] = {"mode": mode}
            for attr in ("min", "max", "mean", "std"):
                if attr in entry:
                    kwargs[attr] = torch.tensor(entry[attr], dtype=torch.float32)
            out[k] = FeatureNorm(**kwargs)
        return cls(out)


def default_so101_modes() -> dict[str, str]:
    """Sensible defaults for SO-101 FlowerVLA: state normalized, action [-1, 1]."""
    return {
        "observation.state": NORM_MEANSTD,
        "action":             NORM_MINMAX,
    }
