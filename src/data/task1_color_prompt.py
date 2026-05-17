"""Task1 color-keyed prompt randomization (per-`__getitem__`).

The existing prompt_aug module (`src/data/prompt_aug.py`) is arrangement-aware:
it needs an explicit `arrangements.json` mapping episodes to bowl layouts and
generates ordinal/relational/negation phrasings as well as direct ones. That
was the right tool when training on raw datasets where the only prompt baked
in was a single "Put the banana in the {color} colored bowl."

This module is a simpler tool for the case where the underlying dataset
already has color-keyed prompts per episode (e.g. `task1_20260509_plus`'s
prompt_assignment block, where every episode's `task` string contains exactly
one of {blue, red, green}). Per `__getitem__`:

  1. Detect which color the sample's existing `task` references.
  2. Replace `sample["task"]` with one of 20 user-supplied templates,
     formatted with the detected color.

The 20 templates duplicate `"Put the banana in the {color} colored bowl."`
eight times — the user's intentional weighting toward the canonical phrasing.
We preserve the duplicates as explicit list entries (NOT `[base] * 8`) so the
weighting is grep-able and obvious in code review.

Color detection reuses `_COLOR_RE` from `src/data/splits.py`. If the sample's
task string contains zero or more than one color (e.g. negation phrasings),
the wrapper passes through unchanged rather than guessing.

The wrapper composes with `LeRobotDataset` and `ConcatDataset` the same way
`PromptAugmentingDataset` does: `__len__` and unknown attributes are
forwarded to the base.
"""
from __future__ import annotations

import random
from typing import Any

import torch

from src.data.splits import COLORS, _COLOR_RE


# Order and duplicates are intentional — see module docstring.
TASK1_PROMPTS: tuple[str, ...] = (
    "Put the banana in the {color} colored bowl.",
    "Put the banana in the {color} colored bowl.",
    "Put the banana in the {color} colored bowl.",
    "Put the banana in the {color} colored bowl.",
    "Put the banana in the {color} colored bowl.",
    "Put the banana in the {color} colored bowl.",
    "Put the banana in the {color} colored bowl.",
    "Put the banana in the {color} colored bowl.",
    "Put the banana in the {color} bowl.",
    "Place the banana in the {color} bowl.",
    "Place the banana into the {color} colored bowl.",
    "Move the banana to the {color} colored bowl.",
    "Move the banana into the {color} bowl.",
    "Put the banana into the {color} bowl.",
    "Put the banana inside the {color} colored bowl.",
    "Please put the banana in the {color} colored bowl.",
    "Please place the banana in the {color} colored bowl.",
    "Pick up the banana and put it in the {color} bowl.",
    "Pick up the banana and place it into the {color} colored bowl.",
    "The banana should go in the {color} bowl.",
)


def _target_from_task(text: str) -> str | None:
    """Single-color match using the shared word-boundary regex.

    Returns None on zero or multiple matches so the caller can pass the
    sample through unchanged (e.g. negation/compositional prompts).
    """
    found = [c for c in COLORS if _COLOR_RE[c].search(text)]
    return found[0] if len(found) == 1 else None


class Task1ColorPromptDataset(torch.utils.data.Dataset):
    """Wraps a LeRobotDataset and rewrites `sample["task"]` per `__getitem__`.

    Each spawned dataloader worker re-imports this module and constructs its
    own RNG, so different workers diverge after the first call — the desired
    behavior for augmentation. Same `seed` across runs gives identical streams
    per worker, but workers within a run differ from each other after the
    fork because each draws independently.
    """

    def __init__(
        self,
        base: torch.utils.data.Dataset,
        seed: int = 42,
        prompts: tuple[str, ...] = TASK1_PROMPTS,
    ) -> None:
        self.base = base
        self._prompts = tuple(prompts)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]
        original = sample.get("task")
        if not isinstance(original, str):
            return sample
        target = _target_from_task(original)
        if target is None:
            return sample
        sample["task"] = self._rng.choice(self._prompts).format(color=target)
        return sample

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "base":
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError as e:
            raise AttributeError(name) from e
        return getattr(base, name)
