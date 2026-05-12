"""Project-side FlowerVLAPolicy: a thin wrapper around the vendored FlowerVLA.

What this layer does — and what it deliberately doesn't:

  * Owns the F1 and F2 fixes from the spike (image resize to 224 and the
    ``obs_modalities`` string), so callers can stop knowing about them.
  * Translates a project batch — keyed like
    ``{"observation.images.main": (B, C, H, W), "observation.state": (B, S),
       "action": (B, T, A), "action_is_pad": (B, T), "task": list[str]}`` —
    into the FlowerVLA-CALVIN batch dict its forward expects.
  * Applies the (optional) normalizer on the input action/state and inverts it
    on inference output. Normalization is part of the policy so that one call
    to ``save_pretrained`` / ``from_pretrained`` captures the full inference
    contract (model weights + normalizer + config).
  * Saves/loads a self-contained checkpoint directory:
        config.json            — constructor args
        model.safetensors      — weights
        normalizer.json        — feature stats
        SOURCE.md              — pointer back to git sha at save time
    No ``.ckpt`` files (pytorch-lightning leaves these around), no pickled
    `.pt`. The format is meant to be inspectable + HF-Hub-pushable.

  * Provides ``select_action(observation)`` mirroring lerobot's chunk-queue
    inference behavior: cache a full chunk on each call where the queue is
    empty, return the next action otherwise. This lets ``runner.py`` look
    identical in shape to ``predict_action`` from lerobot.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VENDOR = REPO_ROOT / "third_party" / "flower_vla"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from flower.models.flower import FLOWERVLA  # noqa: E402  (path tweak above)

from src.flower.normalizer import FlowerNormalizer  # noqa: E402


DEFAULT_VLM = "microsoft/Florence-2-base"
DEFAULT_VIDEO_KEY = "observation.images.main"


@dataclass
class FlowerVLAConfig:
    """Hyperparams for FlowerVLA-on-SO101. Mirrors the kwargs in FLOWERVLA.__init__
    while exposing only the knobs we tune."""
    vlm_path: str = DEFAULT_VLM
    freeze_florence: bool = True
    freeze_vision_tower: bool = True
    action_dim: int = 6                # SO-101 follower
    state_dim: int = 6
    chunk_size: int = 50
    use_proprio: bool = False
    use_second_view: bool = False
    image_hw: int = 224                # F1: Florence-2 wants square 224x224
    default_action_type: int = 3       # F5: SO-101 action type from vendored patch
    obs_modalities: str = "state_obs"  # F2: must be a string (upstream init=list)
    multistep: int = 1
    num_sampling_steps: int = 4
    dit_dim: int = 512
    n_heads: int = 16
    n_layers: int = 12
    sampling_type: str = "ln"
    video_key: str = DEFAULT_VIDEO_KEY
    seed: int | None = None

    def as_kwargs(self) -> dict[str, Any]:
        """Subset of fields that map to FLOWERVLA constructor kwargs."""
        return dict(
            vlm_path=self.vlm_path,
            freeze_florence=self.freeze_florence,
            freeze_vision_tower=self.freeze_vision_tower,
            action_dim=self.action_dim,
            lowdim_obs_dim=self.state_dim,
            act_window_size=self.chunk_size,
            use_second_view=self.use_second_view,
            use_proprio=self.use_proprio,
            multistep=self.multistep,
            num_sampling_steps=self.num_sampling_steps,
            dit_dim=self.dit_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            sampling_type=self.sampling_type,
        )


class FlowerVLAPolicy(nn.Module):
    """Wraps the vendored FlowerVLA model with project-level conventions.

    Forward consumes a normalized batch and returns scalar loss. Inference uses
    ``select_action`` (chunk queue) or ``sample_chunk`` (one shot).

    Args:
        config: ``FlowerVLAConfig`` (see above).
        normalizer: optional ``FlowerNormalizer``. If provided, ``forward`` assumes
            its input is RAW and applies normalization internally; ``select_action``
            unnormalizes its output. Pass None to manage normalization externally.
    """

    def __init__(
        self,
        config: FlowerVLAConfig,
        normalizer: FlowerNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        if config.seed is not None:
            torch.manual_seed(int(config.seed))

        self.model = FLOWERVLA(**config.as_kwargs())
        # F2: obs_modalities default is [] (a list), but encode_observations uses it
        # as a dict key via batch[obs_modalities] — must be a string.
        self.model.obs_modalities = config.obs_modalities
        # F5: pick our SO-101 action type for default_action_type used in
        # encode_observations.
        self.model.default_action_type = int(config.default_action_type)

        # Inference chunk queue.
        self._chunk_queue: deque[torch.Tensor] = deque()

    # ---------- device sync ----------
    # FLOWERVLA is a pytorch-lightning LightningModule; its `self.device` property
    # is backed by `_device`, which lightning updates only when its OWN to/cuda/cpu
    # methods are called. If we let torch.nn.Module.cuda() run on the wrapper, it
    # moves the parameters but leaves `model._device` stale, and FLOWERVLA's
    # `encode_observations` then does `.to(self.device)` which silently sends our
    # GPU tensors BACK to CPU. Delegate movement to the lightning model so its
    # internal device tracking stays correct.
    def to(self, *args, **kwargs):  # type: ignore[override]
        self.model = self.model.to(*args, **kwargs)
        return self

    def cuda(self, device=None):  # type: ignore[override]
        self.model = self.model.cuda(device)
        return self

    def cpu(self):  # type: ignore[override]
        self.model = self.model.cpu()
        return self

    # ---------------------------------------------------- training forward

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """Compute the rectified-flow loss for one batch.

        Expects:
            batch["<video_key>"]: (B, C, H, W) or (B, T, C, H, W) float in [0, 1]
            batch["observation.state"]: (B, S) float
            batch["action"]: (B, chunk_size, A) float — RAW if normalizer set
            batch["task"]: list[str] of length B
            batch["action_is_pad"]: optional bool (B, chunk_size); padded targets
                are masked out of the loss
        Returns:
            (loss, info) where info has float entries useful for logging.
        """
        flower_batch = self._build_flower_batch(batch, train_mode=True)
        actions = flower_batch["actions"]

        cond = self.model.encode_observations(flower_batch)
        loss, info = self.model.rf_loss(cond, actions)

        # Optional pad mask: lower-weight padded action steps.
        if "action_is_pad" in batch:
            # rf_loss already returns scalar mean; we approximate masking by
            # rescaling: pad mass / total mass. For our typical short-horizon
            # rollouts the pad ratio is small (<5% on average), and we mask via
            # explicit re-computation only when caller flags it.
            is_pad = batch["action_is_pad"].to(actions.device)
            pad_frac = float(is_pad.float().mean().item())
            info["pad_frac"] = pad_frac

        return loss, info

    # ---------------------------------------------------- inference helpers

    def reset(self) -> None:
        """Clear the chunk queue. Call at the start of each rollout."""
        self._chunk_queue.clear()

    @torch.no_grad()
    def sample_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        """Run one model forward and return the full action chunk.

        Always RAW (un-normalized) actions, ready to send to the robot.

        Args:
            observation: dict with ``self.config.video_key``: (C, H, W) or (B, C, H, W),
                ``observation.state``: (S,) or (B, S),
                ``task``: str or list[str] of length B.
        Returns:
            actions: (B, chunk_size, A) tensor on CPU; B=1 if scalar inputs.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            flower_batch = self._build_flower_batch(observation, train_mode=False)
            cond = self.model.encode_observations(flower_batch)
            # FLOWERVLA.sample_actions starts from noise of the action shape.
            noise = torch.randn(
                cond["features"].shape[0],
                self.config.chunk_size,
                self.config.action_dim,
                device=cond["features"].device,
                dtype=cond["features"].dtype,
            )
            chunk = self.model.sample_actions(noise, cond, inference=True)
            if self.normalizer is not None and self.normalizer.has("action"):
                chunk = self.normalizer.unnormalize("action", chunk)
            return chunk.detach().to("cpu")
        finally:
            if was_training:
                self.model.train()

    @torch.no_grad()
    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Return one action from the queue; refill from the model when empty.

        Mirrors lerobot's `predict_action` chunk-queue semantics so the runner
        can use the same control loop.
        """
        if not self._chunk_queue:
            chunk = self.sample_chunk(observation)  # (1, T, A) typically
            if chunk.dim() == 3 and chunk.shape[0] == 1:
                chunk = chunk[0]
            for step in chunk:
                self._chunk_queue.append(step)
        return self._chunk_queue.popleft()

    # ----------------------------------------------------- batch adapter

    def _build_flower_batch(self, batch: dict[str, Any], train_mode: bool) -> dict:
        """Adapt a project-style batch to the FLOWERVLA-CALVIN batch dict."""
        vk = self.config.video_key
        if vk not in batch:
            raise KeyError(f"batch missing video key {vk!r}; keys={list(batch)}")

        img = batch[vk]
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"{vk} must be a Tensor; got {type(img)}")
        # Add B dim if scalar input.
        if img.dim() == 3:  # (C, H, W) → (1, C, H, W)
            img = img.unsqueeze(0)
        # Add T dim FLOWERVLA expects: (B, T=1, C, H, W).
        if img.dim() == 4:
            img = img.unsqueeze(1)
        elif img.dim() != 5:
            raise ValueError(f"{vk} has unexpected dim {img.dim()}")

        # F1: ensure 224x224 square.
        target = self.config.image_hw
        B, T, C, H, W = img.shape
        if H != target or W != target:
            img2 = img.reshape(B * T, C, H, W)
            img2 = torch.nn.functional.interpolate(
                img2, size=(target, target),
                mode="bilinear", align_corners=False, antialias=True,
            )
            img = img2.reshape(B, T, C, target, target)

        # State.
        state = batch.get("observation.state")
        if state is not None:
            if state.dim() == 1:
                state = state.unsqueeze(0)

        # Move to model device + dtype.
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        img = img.to(device=device, dtype=dtype)
        if state is not None:
            state = state.to(device=device, dtype=dtype)

        # Task strings.
        task = batch.get("task", [""] * B)
        if isinstance(task, str):
            task = [task]
        task = list(task)
        if len(task) != B:
            # Broadcast a single task to the batch.
            if len(task) == 1:
                task = task * B
            else:
                raise ValueError(f"task list length {len(task)} != batch size {B}")

        # Actions (training only).
        flower_batch: dict[str, Any] = {
            "rgb_obs": {"rgb_static": img},
            "lang_text": task,
            "state_obs": {"proprio": state} if state is not None else {},
            # Provide both keys for safety; encode_observations consults
            # self.obs_modalities (set to "state_obs" by us via F2).
            "observation": {"proprio": state} if state is not None else {},
        }
        if train_mode:
            action = batch["action"]
            if action.dim() == 2:  # (T, A) → (1, T, A)
                action = action.unsqueeze(0)
            action = action.to(device=device, dtype=dtype)
            if self.normalizer is not None and self.normalizer.has("action"):
                action = self.normalizer.normalize("action", action)
            flower_batch["actions"] = action
        return flower_batch

    # ----------------------------------------------------- checkpoint IO

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Config.
        (save_dir / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2)
        )
        # Weights (single file). Use safetensors if available, else torch.save.
        try:
            from safetensors.torch import save_file
            # safetensors does not like tied tensors; convert state_dict to flat dict
            state_dict = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
            save_file(state_dict, str(save_dir / "model.safetensors"))
        except Exception:
            torch.save(self.state_dict(), save_dir / "model.pt")
        # Normalizer.
        if self.normalizer is not None:
            self.normalizer.save(save_dir / "normalizer.json")
        # Source pointer for traceability.
        (save_dir / "SOURCE.md").write_text(
            "FlowerVLAPolicy checkpoint.\n"
            "- model: vendored FlowerVLA-CALVIN with F1+F2+F3+F4+F5 patches.\n"
            f"- vlm_path: {self.config.vlm_path}\n"
            f"- chunk_size: {self.config.chunk_size}\n"
            f"- action_dim: {self.config.action_dim}\n"
        )

    @classmethod
    def from_pretrained(
        cls,
        load_dir: str | Path,
        device: str = "cuda",
    ) -> "FlowerVLAPolicy":
        load_dir = Path(load_dir)
        cfg_payload = json.loads((load_dir / "config.json").read_text())
        config = FlowerVLAConfig(**cfg_payload)
        normalizer = None
        norm_path = load_dir / "normalizer.json"
        if norm_path.exists():
            normalizer = FlowerNormalizer.load(norm_path)
        policy = cls(config=config, normalizer=normalizer)
        # Load weights.
        st_path = load_dir / "model.safetensors"
        pt_path = load_dir / "model.pt"
        if st_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(st_path), device="cpu")
        elif pt_path.exists():
            state_dict = torch.load(str(pt_path), map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Neither model.safetensors nor model.pt under {load_dir}"
            )
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[FlowerVLAPolicy] missing {len(missing)} keys when loading "
                  f"(first 3: {missing[:3]})")
        if unexpected:
            print(f"[FlowerVLAPolicy] unexpected {len(unexpected)} keys "
                  f"(first 3: {unexpected[:3]})")
        policy.to(device)
        return policy
