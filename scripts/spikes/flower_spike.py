"""FlowerVLA feasibility spike — THROWAWAY.

Goal: prove FlowerVLA can consume one batch of `LeRobotDataset` data
and produce a finite action loss + non-degenerate grad-norm, before we
commit to writing the lerobot adapter in Phase 2.

**Two-env workflow** — lerobot needs torch>=2.7 (so does our dataset
loader); flower_calvin pins torch==2.2.2. They can't co-habit one env.
So this script runs in two stages:

  Stage A — in the lerobot conda env (torch 2.10):
    Loads one batch from `LeRobotDataset(task3)` and pickles it to
    /tmp/flower_spike_batch.pt. Also prints shape/dimension gaps so
    docs/flowervla_spike.md can record numbers.

  Stage B — in the flower conda env (torch 2.2.2 + flower_calvin deps):
    Loads the pickle, builds the FlowerVLA-CALVIN model, runs one
    `encode_observations` + `rf_loss` forward, prints loss + grad-norm,
    emits SPIKE_VERDICT.

Will be DELETED in the Phase 2 landing commit (called out in the message).
The `spikes/` subdir signals throwaway; do not import from this file.

Run:
    # Stage A (in lerobot env):
    /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/lerobot/bin/python \\
        scripts/spikes/flower_spike.py --stage A

    # Stage B (in flower env, expects /tmp/flower_calvin cloned):
    /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/flower/bin/python \\
        scripts/spikes/flower_spike.py --stage B
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

TASK3 = "ethrl2026/so101_pickup_20260509_185350_task3"
BATCH_PATH = Path("/tmp/flower_spike_batch.pt")


# ===================================================================== stage A

def stage_a(batch_size: int, chunk: int) -> None:
    """Load one batch from lerobot, save to disk, report gaps."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(TASK3)
    fps = ds_meta.fps
    delta = {"action": [i / fps for i in range(chunk)]}
    ds = LeRobotDataset(repo_id=TASK3, delta_timestamps=delta)
    print(f"[spike-A] dataset frames={len(ds)}  episodes={ds_meta.total_episodes}  fps={fps}")

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print("\n[spike-A] ============ lerobot batch ============")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}  dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)}  sample={v[0]!r}")

    # Save tensors + lang_text to disk.
    serializable = {
        "images_main": batch["observation.images.main"],
        "state":       batch["observation.state"],
        "action":      batch["action"],
        "task":        list(batch["task"]) if isinstance(batch["task"], (list, tuple)) else [batch["task"]],
    }
    BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BATCH_PATH.open("wb") as f:
        pickle.dump(serializable, f)
    print(f"\n[spike-A] saved batch to {BATCH_PATH}")
    print("[spike-A] now run: "
          "/shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/flower/bin/python "
          "scripts/spikes/flower_spike.py --stage B")


# ===================================================================== stage B

def stage_b(chunk: int) -> None:
    """Build FlowerVLA, run forward + grad-norm on the saved batch."""
    if not BATCH_PATH.exists():
        raise SystemExit(f"[spike-B] no batch at {BATCH_PATH} — run --stage A first")

    # flower_calvin clone
    flower_repo = Path("/tmp/flower_calvin")
    if not flower_repo.exists():
        raise SystemExit(f"[spike-B] missing {flower_repo}. "
                         "Clone first: git clone https://github.com/intuitive-robots/flower_vla_calvin.git /tmp/flower_calvin")
    sys.path.insert(0, str(flower_repo))

    import torch
    with BATCH_PATH.open("rb") as f:
        b = pickle.load(f)

    img    = b["images_main"]            # (B, C, H, W)
    state  = b["state"]                  # (B, 6)
    action = b["action"]                 # (B, chunk, 6)
    task   = b["task"]                   # list[str]

    # CALVIN's action_encoders are hardwired per ActionIndex action space:
    # joint_single=8, eef_delta=7, bimanual_nav=16. SO-101 is 6-DoF and
    # doesn't match any. For the spike, pad to 7 so we can get *one* forward
    # through. The Phase 2 adapter will need to extend ActionIndex with a
    # new 'so101' entry (dim=6) and rebuild action_encoders/decoders.
    if action.shape[-1] == 6:
        pad = action.new_zeros(action.shape[0], action.shape[1], 1)
        action = torch.cat([action, pad], dim=-1)
        print(f"[spike-B] padded action 6 -> 7 (CALVIN ActionIndex hardcode)")

    # Florence-2's _encode_image asserts square feature maps. SO-101 cameras
    # are 4:3 (480x640) — resize to 224 (Florence-2's training input). The
    # Phase-2 adapter resizes here; for the spike, 224 is a safe default.
    target_hw = 224
    if img.shape[-1] != img.shape[-2] or img.shape[-1] != target_hw:
        img = torch.nn.functional.interpolate(
            img, size=(target_hw, target_hw),
            mode="bilinear", align_corners=False, antialias=True,
        )

    # Adapt to FlowerVLA-CALVIN's batch dict.
    if img.dim() == 4:
        img = img.unsqueeze(1)           # (B, T=1, C, H, W)
    flower_batch = {
        "rgb_obs": {"rgb_static": img},
        "lang_text": task,
        "actions": action,
        # FlowerVLA-CALVIN reads proprio under batch[obs_modalities]["proprio"];
        # obs_modalities default is "state_obs" — try both for safety.
        "state_obs":   {"proprio": state},
        "observation": {"proprio": state},
    }
    print(f"[spike-B] loaded batch: B={img.shape[0]}  T={img.shape[1]}  "
          f"img={tuple(img.shape)}  action={tuple(action.shape)}  state={tuple(state.shape)}")

    try:
        from flower.models.flower import FLOWERVLA
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[spike-B] FAILED to import flower.models.flower: {e}")
        print(f"[spike-B] SPIKE_VERDICT: NO-GO (import)")
        return

    try:
        model = FLOWERVLA(
            vlm_path="microsoft/Florence-2-base",
            freeze_florence=True,
            freeze_vision_tower=True,
            action_dim=7,                 # match eef_delta entry in ActionIndex
            lowdim_obs_dim=6,
            act_window_size=chunk,
            use_second_view=False,
            use_proprio=False,            # CALVIN proprio path has a shape bug.
            multistep=1,
            num_sampling_steps=4,
        )
        model.obs_modalities = "state_obs"
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[spike-B] FAILED to build FLOWERVLA: {e}")
        print(f"[spike-B] SPIKE_VERDICT: NO-GO (build)")
        return

    # Move to CUDA if available; loss must be finite + grad must exist.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    for k in ("rgb_obs", "state_obs", "observation"):
        if k in flower_batch:
            for sub_k, v in flower_batch[k].items():
                if isinstance(v, torch.Tensor):
                    flower_batch[k][sub_k] = v.to(device)
    flower_batch["actions"] = flower_batch["actions"].to(device)
    print(f"[spike-B] device={device}")

    model.train()
    try:
        cond = model.encode_observations(flower_batch)
        loss, info = model.rf_loss(cond, flower_batch["actions"])
        print(f"\n[spike-B] forward OK. loss={loss.item():.6f}  info_keys={list(info.keys())}")
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters()
                      if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()]
        if grad_norms:
            mean_gn = sum(grad_norms) / len(grad_norms)
            print(f"[spike-B] mean grad norm: {mean_gn:.4e}  finite_params={len(grad_norms)}")
        finite = torch.isfinite(loss).item()
        verdict = "GO" if finite and loss.item() > 0 and grad_norms else "NO-GO"
        print(f"\n[spike-B] SPIKE_VERDICT: {verdict}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[spike-B] FORWARD FAILED: {e}")
        print(f"[spike-B] SPIKE_VERDICT: NO-GO (forward)")


# ===================================================================== main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B"], required=True,
                        help="A = dump batch from lerobot env; B = run forward in flower env")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=50)
    args = parser.parse_args()

    if args.stage == "A":
        stage_a(args.batch_size, args.chunk)
    else:
        stage_b(args.chunk)


if __name__ == "__main__":
    main()
