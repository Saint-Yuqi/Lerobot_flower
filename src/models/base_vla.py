"""Base interface for all VLA policies in this project.

The contract: given a current observation (images + robot state) and a
natural-language prompt, return an action chunk (sequence of next actions
to execute open-loop, then re-plan).

Both the end-to-end SmolVLA wrapper and the decoupled (VLM + policy)
implementation conform to this interface. Inference, eval, and the robot
runner code never need to know which is in use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Observation:
    """One control-step observation.

    Attributes:
        images: dict of camera_name -> uint8 array (H, W, 3) in RGB.
            For SO-101 with wrist cam only, this is {"wrist": img}.
        state: float32 array of joint positions (6,) for SO-101.
        prompt: natural-language instruction.
    """
    images: dict[str, np.ndarray]
    state: np.ndarray
    prompt: str


@dataclass
class ActionChunk:
    """A sequence of actions to execute before re-planning.

    Attributes:
        actions: float32 array (chunk_size, action_dim). For SO-101,
            action_dim is typically 6 (joint targets).
        chunk_size: number of actions in this chunk.
        meta: optional debug info (selected target bbox, VLM output, etc.)
    """
    actions: np.ndarray
    chunk_size: int
    meta: dict | None = None


class BaseVLA(ABC):
    """Abstract base class for all policies used in this project."""

    @abstractmethod
    def predict(self, obs: Observation) -> ActionChunk:
        """Return the next chunk of actions for this observation."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of every episode (clear any internal state)."""
        ...

    @property
    @abstractmethod
    def active_param_count(self) -> int:
        """Total active parameters used at inference time.

        For the project's bonus competition: count every parameter that
        gets a forward pass during one episode. For a frozen VLM used once
        per episode, this still counts. We track this so we can argue our
        bonus points clearly to the TAs.
        """
        ...

    @abstractmethod
    def to(self, device: str | torch.device) -> "BaseVLA":
        """Move the policy to a device."""
        ...

    @abstractmethod
    def eval(self) -> "BaseVLA":
        """Set to eval mode."""
        ...

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "BaseVLA":
        """Load a saved policy from disk."""
        ...
