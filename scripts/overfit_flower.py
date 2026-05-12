"""FlowerVLA overfit smoke test — small + fast end-to-end gate.

Runs in the `flower` conda env. Loads one episode of an SO-101 dataset, trains
FlowerVLA for a few hundred steps with the project-side training utilities,
and asserts that loss drops below a configured threshold. Mirrors the spirit of
`scripts/overfit_test.py` (the SmolVLA equivalent) but uses the flower-env stack
(FlowerSO101Dataset, FlowerVLAPolicy, FlowerNormalizer).

Usage:
    /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/flower/bin/python \\
        scripts/overfit_flower.py --config configs/train/overfit_flower.yaml

The pass criteria are in cfg.train.pass_criteria.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"[overfit] config: {args.config}")
    print(f"[overfit] experiment: {cfg['experiment_name']}")

    # Lazy imports.
    import torch
    from torch.utils.data import DataLoader

    from src.flower.dataset import FlowerSO101Dataset
    from src.flower.normalizer import FlowerNormalizer, default_so101_modes
    from src.flower.policy import FlowerVLAConfig, FlowerVLAPolicy

    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]

    out_dir = Path(tcfg["output_dir"]) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[overfit] output_dir: {out_dir}")

    # ---- Dataset ----
    ds = FlowerSO101Dataset(
        repo_id=dcfg["repo_id"],
        root=dcfg.get("root"),
        episodes=dcfg.get("episodes"),
        chunk_size=int(mcfg["chunk_size"]),
        video_key=dcfg.get("video_key", "observation.images.main"),
        resize_hw=int(mcfg.get("image_hw", 224)),
    )
    print(f"[overfit] dataset: {ds}")
    if ds.stats is None:
        raise RuntimeError("dataset has no stats.json; FlowerVLA training needs it.")

    norms = FlowerNormalizer.from_stats(ds.stats, default_so101_modes())

    # Optional phase-weighted sampler — same wiring as scripts/train_flower.py.
    # On by default off; overfit_flower_phase.yaml flips it on for the smoke test.
    phase_cfg = tcfg.get("phase_sampling") or {}
    sampler = None
    if phase_cfg.get("enabled"):
        from src.data.phase_labels import compute_phase_labels, summarize
        from src.data.sampler import make_phase_weighted_sampler, assert_dataset_alignment
        phase_labels = compute_phase_labels(
            repo_id=dcfg["repo_id"], root=ds.root, episodes=dcfg.get("episodes"),
            open_frac=float(phase_cfg.get("open_frac", 0.6)),
            close_frac=float(phase_cfg.get("close_frac", 0.4)),
            min_amplitude=float(phase_cfg.get("min_amplitude", 5.0)),
            post_close_margin=int(phase_cfg.get("post_close_margin", 3)),
        )
        if len(phase_labels) != len(ds):
            raise RuntimeError(
                f"phase_sampling: labels length {len(phase_labels)} != dataset length "
                f"{len(ds)} — iteration order divergence."
            )
        assert_dataset_alignment(ds, phase_labels, n_check=8)
        print(f"[overfit] phase_sampling: {summarize(phase_labels, label=dcfg['repo_id'])}")
        sampler = make_phase_weighted_sampler(
            phase_labels,
            weight_pregrasp=float(phase_cfg.get("weight_pregrasp", 2.0)),
            replacement=bool(phase_cfg.get("replacement", True)),
            seed=int(cfg.get("seed", 42)),
        )

    loader = DataLoader(
        ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(tcfg.get("num_workers", 0)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(tcfg.get("num_workers", 0)) > 0,
    )

    # ---- Policy ----
    cfg_pol = FlowerVLAConfig(
        vlm_path=mcfg.get("vlm_path", "microsoft/Florence-2-base"),
        freeze_florence=bool(mcfg.get("freeze_florence", True)),
        freeze_vision_tower=bool(mcfg.get("freeze_vision_tower", True)),
        action_dim=int(ds.action_dim),
        state_dim=int(ds.state_dim),
        chunk_size=int(mcfg["chunk_size"]),
        use_proprio=bool(mcfg.get("use_proprio", False)),
        use_second_view=bool(mcfg.get("use_second_view", False)),
        image_hw=int(mcfg.get("image_hw", 224)),
        default_action_type=int(mcfg.get("default_action_type", 3)),
        num_sampling_steps=int(mcfg.get("num_sampling_steps", 4)),
        dit_dim=int(mcfg.get("dit_dim", 512)),
        n_heads=int(mcfg.get("n_heads", 16)),
        n_layers=int(mcfg.get("n_layers", 12)),
        video_key=dcfg.get("video_key", "observation.images.main"),
        seed=int(cfg.get("seed", 42)),
    )
    policy = FlowerVLAPolicy(cfg_pol, normalizer=norms).to(tcfg["device"])
    policy.train()
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[overfit] trainable params: {n_train:,}")

    # ---- Optim ----
    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    warmup = int(tcfg.get("warmup_steps", 0))
    total = int(tcfg["num_steps"])

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        import math
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # ---- Loop ----
    step = 0
    last_loss = float("inf")
    initial_loss = None
    loss_window: list[float] = []
    t0 = time.time()

    while step < total:
        for batch in loader:
            optim.zero_grad()
            loss, info = policy(batch)
            loss.backward()
            optim.step()
            sched.step()

            last_loss = float(loss.detach().cpu())
            if initial_loss is None:
                initial_loss = last_loss
            loss_window.append(last_loss)
            if len(loss_window) > 20:
                loss_window.pop(0)

            if step % int(tcfg["log_every"]) == 0:
                avg = sum(loss_window) / len(loss_window)
                lr = optim.param_groups[0]["lr"]
                elapsed = time.time() - t0
                steps_per_s = (step + 1) / max(elapsed, 1e-6)
                print(
                    f"[overfit] step={step:4d}/{total}  loss={last_loss:.4f}  "
                    f"avg={avg:.4f}  lr={lr:.2e}  {steps_per_s:.2f} steps/s"
                )

            step += 1
            if step >= total:
                break

    # ---- Save + assess ----
    save_dir = out_dir / "final"
    policy.save_pretrained(save_dir)
    print(f"[overfit] saved checkpoint -> {save_dir}")

    crit = tcfg.get("pass_criteria") or {}
    initial_min = float(crit.get("initial_loss_at_least", 0.0))
    final_max = float(crit.get("final_loss_at_most", float("inf")))
    final_avg = sum(loss_window) / max(1, len(loss_window))
    passed = (initial_loss is not None
              and initial_loss >= initial_min
              and final_avg <= final_max)
    print(f"[overfit] initial_loss={initial_loss}")
    print(f"[overfit] final_avg_loss={final_avg:.4f}")
    print(f"[overfit] pass_criteria: initial>={initial_min} final<={final_max}")
    print(f"[overfit] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
