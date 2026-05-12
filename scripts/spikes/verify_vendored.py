"""Sanity check the vendored flower_vla module.

Runs in the `flower` conda env. Loads the spike batch saved by `flower_spike.py --stage A`
and runs one forward + backward against `third_party/flower_vla/`, with the SO-101 path
(action_dim=6, default_action_type=3) — i.e. exercises F3+F4+F5 patches end-to-end.

This is the GO/NO-GO test for the vendored copy. Once green, we can build the policy
wrapper and training loop on top.

Run:
    /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/flower/bin/python \\
        scripts/spikes/verify_vendored.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "flower_vla"))

BATCH_PATH = Path("/tmp/flower_spike_batch.pt")


def main() -> int:
    if not BATCH_PATH.exists():
        print(f"[verify] no batch at {BATCH_PATH} — run flower_spike.py --stage A first")
        return 1

    import torch

    with BATCH_PATH.open("rb") as f:
        b = pickle.load(f)

    img    = b["images_main"]            # (B, C, H, W)
    state  = b["state"]                  # (B, 6)
    action = b["action"]                 # (B, chunk, 6 or 7 — pickle may be stale)
    task   = b["task"]                   # list[str]

    # If the pickle was saved by an older spike that padded action 6->7, drop the pad column.
    if action.shape[-1] == 7:
        action = action[..., :6]
        print("[verify] trimmed cached padded action 7 -> 6")

    # F1: Florence-2 wants square 224x224 input.
    target_hw = 224
    if img.shape[-1] != target_hw or img.shape[-2] != target_hw:
        img = torch.nn.functional.interpolate(
            img, size=(target_hw, target_hw),
            mode="bilinear", align_corners=False, antialias=True,
        )
    if img.dim() == 4:
        img = img.unsqueeze(1)           # (B, T=1, C, H, W)

    chunk = action.shape[1]
    flower_batch = {
        "rgb_obs": {"rgb_static": img},
        "lang_text": task,
        "actions": action,
        "state_obs": {"proprio": state},
    }
    print(f"[verify] batch B={img.shape[0]} T={img.shape[1]} "
          f"img={tuple(img.shape)} action={tuple(action.shape)} state={tuple(state.shape)}")

    from flower.models.flower import FLOWERVLA

    model = FLOWERVLA(
        vlm_path="microsoft/Florence-2-base",
        freeze_florence=True,
        freeze_vision_tower=True,
        action_dim=6,                 # F4: SO-101 6-DoF
        lowdim_obs_dim=6,
        act_window_size=chunk,
        use_second_view=False,
        use_proprio=False,            # keep off for first verify; will turn on after policy wrapper
        multistep=1,
        num_sampling_steps=4,
    )
    # F2: obs_modalities default is [], must be a string for batch[obs_modalities] indexing.
    model.obs_modalities = "state_obs"
    # F5: register SO-101 action type for this checkpoint.
    model.default_action_type = 3
    print(f"[verify] model built. action_dim={model.action_dim}  default_action_type={model.default_action_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    for k in ("rgb_obs", "state_obs"):
        if k in flower_batch:
            for sub_k, v in flower_batch[k].items():
                if isinstance(v, torch.Tensor):
                    flower_batch[k][sub_k] = v.to(device)
    flower_batch["actions"] = flower_batch["actions"].to(device)
    print(f"[verify] device={device}")

    model.train()
    cond = model.encode_observations(flower_batch)
    print(f"[verify] cond.action_type unique vals: {cond['action_type'].unique().tolist()}")
    print(f"[verify] cond.frequency_embeds shape: {tuple(cond['frequency_embeds'].shape)}")

    loss, info = model.rf_loss(cond, flower_batch["actions"])
    print(f"[verify] forward OK. loss={loss.item():.6f} info={info}")

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters()
                  if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()]
    if grad_norms:
        mean_gn = sum(grad_norms) / len(grad_norms)
        print(f"[verify] mean grad norm: {mean_gn:.4e}  finite_params={len(grad_norms)}")

    finite = torch.isfinite(loss).item()
    verdict = "GO" if finite and loss.item() > 0 and grad_norms else "NO-GO"
    print(f"[verify] VERDICT: {verdict}")
    return 0 if verdict == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
