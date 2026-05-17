"""FlowerVLA full-training entry point — flower-env equivalent of `scripts/train.py`.

Runs in the `flower` conda env. Mirrors the SmolVLA training loop's contract
(eval cadence decoupled from save cadence, best-val tracking, opt-in early stop,
HF Hub push at end, wandb_metadata sidecar) but trained on FlowerVLA instead.

Usage:
    /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/flower/bin/python \\
        scripts/train_flower.py --config configs/train/full_eval1_flower.yaml

Or via slurm:
    sbatch scripts/train_flower.slurm configs/train/full_eval1_flower.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_env(s):
    if isinstance(s, str):
        return os.path.expandvars(s)
    return s


def _push_to_hf(ckpt_dir: Path, hf_cfg: dict, ckpt_kind: str) -> None:
    """Best-effort HF Hub push. ckpt_kind is "best", "final", etc.

    Mirrors scripts/train.py's HF push semantics. Requires HF_TOKEN env var
    or a logged-in `huggingface-cli`.
    """
    try:
        from huggingface_hub import HfApi
        repo_id = expand_env(hf_cfg.get("repo_id"))
        if not repo_id:
            print(f"[train] hf.repo_id missing — skipping push of {ckpt_kind}")
            return
        private = bool(hf_cfg.get("private", True))
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(ckpt_dir), repo_id=repo_id, repo_type="model",
            commit_message=f"upload {ckpt_kind} from train_flower.py",
        )
        print(f"[train] pushed {ckpt_kind} to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[train] HF push of {ckpt_kind} failed: {e!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"[train] config: {args.config}")
    print(f"[train] experiment: {cfg['experiment_name']}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    if "SLURM_JOB_ID" in os.environ:
        run_id = f"{run_id}_job{os.environ['SLURM_JOB_ID']}"
    base_out = Path(cfg["train"]["output_dir"])
    out_dir = base_out / run_id
    cfg["train"]["output_dir"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run_id: {run_id}")
    print(f"[train] output_dir: {out_dir}")

    # ---- wandb ----
    log_cfg = cfg.get("logging") or {}
    use_wandb = bool(log_cfg.get("use_wandb", False)) and log_cfg.get("mode", "online") != "disabled"
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=log_cfg.get("project", "Lerobot"),
            entity=log_cfg.get("entity"),
            name=log_cfg.get("name") or f"{cfg['experiment_name']}-{run_id}",
            id=run_id,
            group=cfg["experiment_name"],
            tags=log_cfg.get("tags"),
            mode=log_cfg.get("mode", "online"),
            dir=str(out_dir),
            config=cfg,
        )
        print(f"[train] wandb: {wandb_run.url if wandb_run.url else '(offline)'}")

    # ---- Lazy imports ----
    import torch
    from torch.utils.data import DataLoader

    from src.data.splits import episodes_by_color, train_val_episode_split
    from src.flower.dataset import FlowerSO101Dataset
    from src.flower.normalizer import FlowerNormalizer, default_so101_modes
    from src.flower.policy import FlowerVLAConfig, FlowerVLAPolicy
    from src.utils.checkpoint_meta import write_checkpoint_meta
    from src.utils.run_metadata import git_sha as _git_sha_helper

    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]

    # ---- Compute episode split. ----
    repo_id = expand_env(dcfg["repo_id"])
    # We always build a "discovery" dataset first to learn the on-disk root, then
    # call episodes_by_color against that root to avoid the lerobot dep.
    ds_probe = FlowerSO101Dataset(
        repo_id=repo_id, root=dcfg.get("root"),
        chunk_size=int(mcfg["chunk_size"]),
        video_key=dcfg.get("video_key", "observation.images.main"),
        resize_hw=int(mcfg.get("image_hw", 224)),
    )
    print(f"[train] dataset root: {ds_probe.root}")
    print(f"[train] dataset frames: {len(ds_probe)} episodes: {len(ds_probe._episodes)}")

    train_eps = dcfg.get("episodes")
    val_eps: list[int] | None = None
    val_cfg = dcfg.get("val") or {}
    if train_eps is None and val_cfg:
        by = episodes_by_color(repo_id, root=ds_probe.root)
        if val_cfg.get("per_color") is not None:
            train_eps, val_eps = train_val_episode_split(
                by, per_color=int(val_cfg["per_color"]),
                min_train_per_color=int(val_cfg.get("min_train_per_color", 3)),
                seed=int(cfg.get("seed", 42)),
            )
        else:
            train_eps, val_eps = train_val_episode_split(
                by, fraction=float(val_cfg.get("fraction", 0.1)),
                min_train_per_color=int(val_cfg.get("min_train_per_color", 3)),
                seed=int(cfg.get("seed", 42)),
            )
        print(f"[train] color-stratified split: train_eps={len(train_eps)} val_eps={len(val_eps)} -> {val_eps}")

    # ---- Build train + val datasets (re-using probe's root for both). ----
    # Image augmentations are train-only (val stays clean for stable eval metrics).
    from src.data.image_transforms import build_image_transforms
    image_transforms = build_image_transforms(
        dcfg.get("augmentations"), image_hw=int(mcfg.get("image_hw", 224))
    )
    if image_transforms is not None:
        print(f"[train] image_transforms: enabled ({dcfg.get('augmentations')})")

    dataset = FlowerSO101Dataset(
        repo_id=repo_id, root=ds_probe.root, episodes=train_eps,
        chunk_size=int(mcfg["chunk_size"]),
        video_key=dcfg.get("video_key", "observation.images.main"),
        resize_hw=int(mcfg.get("image_hw", 224)),
        image_transforms=image_transforms,
    )
    val_dataset: FlowerSO101Dataset | None = None
    if val_eps:
        val_dataset = FlowerSO101Dataset(
            repo_id=repo_id, root=ds_probe.root, episodes=val_eps,
            chunk_size=int(mcfg["chunk_size"]),
            video_key=dcfg.get("video_key", "observation.images.main"),
            resize_hw=int(mcfg.get("image_hw", 224)),
        )
    print(f"[train] train frames: {len(dataset)}  val frames: {len(val_dataset) if val_dataset else 0}")

    # Optional per-getitem prompt augmentation (task1 only — color-keyed prompts).
    # Train-only; val keeps original episode prompts.
    prompt_aug_cfg = dcfg.get("prompt_augmentation") or {}
    if prompt_aug_cfg.get("enabled"):
        from src.data.task1_color_prompt import Task1ColorPromptDataset
        dataset = Task1ColorPromptDataset(base=dataset, seed=int(cfg.get("seed", 42)))
        print("[train] prompt_augmentation: enabled (Task1ColorPromptDataset)")

    # Stats from the FULL dataset (probe), not the filtered split — normalization
    # constants should be stable across splits.
    if ds_probe.stats is None:
        raise RuntimeError("dataset has no stats.json")
    norms = FlowerNormalizer.from_stats(ds_probe.stats, default_so101_modes())

    # Optional phase-weighted sampler (upweights pre-grasp frames). Off by default.
    # See `src/data/phase_labels.py` + plan flower-vla-smol-vla-flickering-puddle.md.
    phase_cfg = tcfg.get("phase_sampling") or {}
    sampler = None
    if phase_cfg.get("enabled"):
        from src.data.phase_labels import compute_phase_labels, summarize
        from src.data.sampler import make_phase_weighted_sampler, assert_dataset_alignment
        phase_labels = compute_phase_labels(
            repo_id=repo_id, root=ds_probe.root, episodes=train_eps,
            open_frac=float(phase_cfg.get("open_frac", 0.6)),
            close_frac=float(phase_cfg.get("close_frac", 0.4)),
            min_amplitude=float(phase_cfg.get("min_amplitude", 5.0)),
            post_close_margin=int(phase_cfg.get("post_close_margin", 3)),
        )
        if len(phase_labels) != len(dataset):
            raise RuntimeError(
                f"phase_sampling: labels length {len(phase_labels)} != dataset length "
                f"{len(dataset)} — iteration order divergence."
            )
        assert_dataset_alignment(dataset, phase_labels, n_check=16)
        print(f"[train] phase_sampling: {summarize(phase_labels, label=repo_id)}")
        sampler = make_phase_weighted_sampler(
            phase_labels,
            weight_pregrasp=float(phase_cfg.get("weight_pregrasp", 2.0)),
            replacement=bool(phase_cfg.get("replacement", True)),
            seed=int(cfg.get("seed", 42)),
        )

    loader = DataLoader(
        dataset,
        batch_size=int(tcfg["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(tcfg.get("num_workers", 4)),
        drop_last=True,
        pin_memory=True,
        persistent_workers=int(tcfg.get("num_workers", 4)) > 0,
    )
    val_loader = None
    val_subsample_fraction = float(tcfg.get("val_subsample_fraction", 1.0))
    if not (0.0 < val_subsample_fraction <= 1.0):
        raise ValueError(
            f"val_subsample_fraction must be in (0, 1], got {val_subsample_fraction}"
        )
    val_max_batches: int | None = None
    if val_dataset is not None:
        val_workers = min(2, int(tcfg.get("num_workers", 4)))
        # shuffle=True so successive eval passes see different subsets when subsampling;
        # over many evals this covers the full val set in expectation without paying
        # for the whole thing every cycle. Without subsample (fraction=1.0) it's
        # mathematically equivalent to shuffle=False but iterates fully.
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(tcfg["batch_size"]),
            shuffle=True,
            num_workers=val_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=val_workers > 0,
        )
        full_val_batches = len(val_loader)
        if val_subsample_fraction < 1.0:
            import math
            val_max_batches = max(1, math.ceil(full_val_batches * val_subsample_fraction))
            print(
                f"[train] val batches: {full_val_batches} (subsample "
                f"{val_subsample_fraction:.2f} -> {val_max_batches}/eval)"
            )
        else:
            print(f"[train] val batches: {full_val_batches}")

    # ---- Policy ----
    cfg_pol = FlowerVLAConfig(
        vlm_path=mcfg.get("vlm_path", "microsoft/Florence-2-base"),
        freeze_florence=bool(mcfg.get("freeze_florence", True)),
        freeze_vision_tower=bool(mcfg.get("freeze_vision_tower", True)),
        action_dim=int(ds_probe.action_dim),
        state_dim=int(ds_probe.state_dim),
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
    print(f"[train] params: {sum(p.numel() for p in policy.parameters()):,}")

    # ---- Optim + LR ----
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable, lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        fused=torch.cuda.is_available(),
    )
    warmup = int(tcfg.get("warmup_steps", 0))
    total_steps = int(tcfg["num_steps"])

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    grad_accum = int(tcfg.get("grad_accum_steps", 1))

    # ---- GPU sampler (system metrics) ----
    from src.utils.gpu_metrics import GpuSampler
    gpu_sampler = GpuSampler()

    _git_sha_str = _git_sha_helper()

    def run_eval() -> float | None:
        """Mean val loss over the (optionally subsampled) val set.

        When ``val_max_batches`` is set (i.e. ``val_subsample_fraction < 1.0``)
        we break after that many batches. The val_loader has shuffle=True so the
        subset drawn rotates between eval calls; over many evals the policy is
        evaluated on the entire val set in expectation."""
        if val_loader is None:
            return None
        policy.eval()
        losses: list[float] = []
        with torch.no_grad():
            for i, vbatch in enumerate(val_loader):
                vloss, _ = policy(vbatch)
                losses.append(float(vloss.detach().cpu()))
                if val_max_batches is not None and (i + 1) >= val_max_batches:
                    break
        policy.train()
        return sum(losses) / max(1, len(losses))

    eval_every = int(tcfg.get("eval_every", tcfg["save_every"]))

    es_cfg = tcfg.get("early_stop") or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 4))
    es_min_delta = float(es_cfg.get("min_delta", 0.005))
    es_min_steps = int(es_cfg.get("min_steps", 5000))
    if es_enabled:
        print(f"[train] early-stop: enabled patience={es_patience} "
              f"min_delta={es_min_delta} min_steps={es_min_steps} "
              f"eval_every={eval_every}")
    best_val_loss = float("inf")
    best_step = -1
    no_improve_evals = 0

    # ---- Loop ----
    step = 0
    last_loss = float("inf")
    loss_window: list[float] = []
    t0 = time.time()
    optim.zero_grad()
    should_stop = False

    while step < total_steps and not should_stop:
        for batch in loader:
            loss, _ = policy(batch)
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optim.step()
                sched.step()
                optim.zero_grad()

            last_loss = float(loss.detach().cpu())
            loss_window.append(last_loss)
            if len(loss_window) > 50:
                loss_window.pop(0)

            if step % int(tcfg["log_every"]) == 0:
                avg = sum(loss_window) / len(loss_window)
                lr = optim.param_groups[0]["lr"]
                elapsed = time.time() - t0
                steps_per_s = (step + 1) / max(elapsed, 1e-6)
                samples_per_s = steps_per_s * int(tcfg["batch_size"])
                eta_min = (total_steps - step) / max(steps_per_s, 1e-6) / 60
                print(
                    f"[train] step={step:6d}/{total_steps}  "
                    f"loss={last_loss:.4f}  avg50={avg:.4f}  "
                    f"lr={lr:.2e}  {steps_per_s:.2f} steps/s  ETA={eta_min:.1f}min"
                )
                if use_wandb:
                    payload = {
                        "train/loss": last_loss,
                        "train/loss_avg50": avg,
                        "train/lr": lr,
                        "train/steps_per_s": steps_per_s,
                        "train/samples_per_s": samples_per_s,
                    }
                    payload.update(gpu_sampler.sample())
                    import wandb
                    wandb.log(payload, step=step)

            if step > 0 and step % int(tcfg["save_every"]) == 0:
                ckpt_dir = out_dir / f"step_{step}"
                policy.save_pretrained(ckpt_dir)
                write_checkpoint_meta(ckpt_dir, wandb_run, cfg, step, _git_sha_str)

            if step > 0 and step % eval_every == 0:
                val_loss = run_eval()
                if val_loss is not None:
                    print(f"[train] step={step:6d}  eval/loss={val_loss:.4f}")
                    if use_wandb:
                        import wandb
                        wandb.log({"eval/loss": val_loss}, step=step)
                    improved = val_loss < best_val_loss
                    significant = val_loss < best_val_loss - es_min_delta
                    if improved:
                        best_val_loss = val_loss
                        best_step = step
                        best_dir = out_dir / "best"
                        policy.save_pretrained(best_dir)
                        (best_dir / "best_val_meta.json").write_text(json.dumps({
                            "val_loss": float(val_loss),
                            "step": int(step),
                            "experiment": cfg["experiment_name"],
                        }, indent=2))
                        write_checkpoint_meta(
                            best_dir, wandb_run, cfg, step, _git_sha_str,
                            extra={"val_loss": float(val_loss), "is_best": True},
                        )
                        print(f"[train] new best  val_loss={val_loss:.4f} @ step {step} -> {best_dir}")
                        if use_wandb:
                            import wandb
                            wandb.log({
                                "eval/best_loss": val_loss,
                                "eval/best_step": step,
                            }, step=step)
                    if significant:
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                        print(f"[train] no-improve evals: {no_improve_evals}/"
                              f"{es_patience} (best={best_val_loss:.4f} @ {best_step})")
                    if (es_enabled and no_improve_evals >= es_patience
                            and step >= es_min_steps):
                        print(f"[train] EARLY STOP at step {step}: "
                              f"{no_improve_evals} consecutive evals without "
                              f"≥{es_min_delta} improvement. "
                              f"Best val_loss={best_val_loss:.4f} @ step {best_step}.")
                        if use_wandb:
                            import wandb
                            wandb.summary["early_stop_step"] = step
                            wandb.summary["early_stop_best_step"] = best_step
                            wandb.summary["early_stop_best_loss"] = best_val_loss
                        should_stop = True

            step += 1
            if step >= total_steps or should_stop:
                break

    # ---- Final save + eval ----
    final_dir = out_dir / "final"
    try:
        policy.save_pretrained(final_dir)
        write_checkpoint_meta(
            final_dir, wandb_run, cfg, max(0, step - 1), _git_sha_str,
            extra={"is_final": True},
        )
        print(f"[train] saved final checkpoint -> {final_dir}")

        final_val = run_eval()
        if final_val is not None:
            print(f"[train] final eval/loss={final_val:.4f}")
            if use_wandb:
                import wandb
                wandb.log({"eval/loss": final_val}, step=max(0, step - 1))
                wandb.summary["final_eval_loss"] = final_val

        if use_wandb:
            import wandb
            avg = sum(loss_window) / max(1, len(loss_window))
            wandb.summary["final_loss"] = last_loss
            wandb.summary["final_loss_avg50"] = avg
    finally:
        gpu_sampler.shutdown()
        if use_wandb and wandb_run is not None:
            import wandb
            wandb.finish()

    # ---- HF push ----
    hf_cfg = cfg.get("hf") or {}
    if hf_cfg.get("push"):
        push_dir = out_dir / "best" if (out_dir / "best").exists() else final_dir
        _push_to_hf(push_dir, hf_cfg, "best" if push_dir.name == "best" else "final")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
