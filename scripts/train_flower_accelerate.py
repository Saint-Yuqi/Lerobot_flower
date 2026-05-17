"""Multi-GPU FlowerVLA training via HuggingFace `accelerate`.

Parallel to `scripts/train_flower.py` (single-GPU). Mirrors the SmolVLA
sibling repo's `scripts/train_accelerate.py` DDP design adapted for
FlowerVLA's stack (FlowerSO101Dataset + FlowerVLAPolicy + FlowerNormalizer).

Key differences vs. `train_flower.py`:
  * `Accelerator(gradient_accumulation_steps=...)` drives DDP.
  * `cfg.train.batch_size` is the **global** batch size; this script divides
    by world_size to get the per-process batch (errors if not divisible).
  * Saves, wandb logging, HF push run on rank 0 only.
  * Val loss is gathered across ranks via `accelerator.gather_for_metrics`.
  * Phase-weighted sampling is DDP-aware: each rank constructs its own
    `WeightedRandomSampler` seeded by `cfg.seed + rank` drawing
    `len(weights) // world_size` indices.
  * `find_unused_parameters=True`: FlowerVLA has `freeze_florence=True`,
    so most of the VLM is non-trainable and DDP must tolerate unused params.

Usage:
    accelerate launch --num_processes 4 --mixed_precision bf16 \\
        scripts/train_flower_accelerate.py --config configs/train/full_eval1_aug_mgpu.yaml
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
    """Rank-0 HF Hub push. Best-effort; logs failure but doesn't raise."""
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
            commit_message=f"upload {ckpt_kind} from train_flower_accelerate.py",
        )
        print(f"[train] pushed {ckpt_kind} to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[train] HF push of {ckpt_kind} failed: {e!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # ---- Lazy imports ----
    import torch
    # file_system sharing avoids "Too many open files" at 4 ranks × N workers.
    torch.multiprocessing.set_sharing_strategy("file_system")
    from torch.utils.data import DataLoader
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import broadcast_object_list

    from src.data.image_transforms import build_image_transforms
    from src.data.splits import episodes_by_color, train_val_episode_split
    from src.flower.dataset import FlowerSO101Dataset
    from src.flower.normalizer import FlowerNormalizer, default_so101_modes
    from src.flower.policy import FlowerVLAConfig, FlowerVLAPolicy
    from src.utils.checkpoint_meta import write_checkpoint_meta
    from src.utils.run_metadata import git_sha as _git_sha_helper

    tcfg = cfg["train"]
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    grad_accum = int(tcfg.get("grad_accum_steps", 1))

    # ---- Accelerator ----
    # find_unused_parameters=True: Florence-2 is frozen so its params don't
    # participate in backward; DDP must tolerate that without erroring.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        kwargs_handlers=[ddp_kwargs],
    )
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes

    def log(msg: str) -> None:
        if is_main:
            print(msg)

    log(f"[train] config: {args.config}")
    log(f"[train] experiment: {cfg['experiment_name']}")
    log(f"[train] world_size={world_size}  rank={accelerator.process_index}  device={accelerator.device}")

    # ---- Global batch -> per-process batch ----
    global_batch = int(tcfg["batch_size"])
    if global_batch % world_size != 0:
        raise SystemExit(
            f"cfg.train.batch_size={global_batch} is not divisible by world_size={world_size}. "
            f"Pick a batch size that is a multiple of {world_size}."
        )
    per_proc_batch = global_batch // world_size
    log(f"[train] batch: global={global_batch} per_proc={per_proc_batch} (world_size={world_size})")

    # ---- run_id (rank 0 mints, broadcast to others) ----
    if is_main:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        if "SLURM_JOB_ID" in os.environ:
            run_id = f"{run_id}_job{os.environ['SLURM_JOB_ID']}"
    else:
        run_id = None
    run_id = broadcast_object_list([run_id], from_process=0)[0]
    base_out = Path(tcfg["output_dir"])
    out_dir = base_out / run_id
    tcfg["output_dir"] = str(out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    log(f"[train] run_id: {run_id}")
    log(f"[train] output_dir: {out_dir}")

    # ---- wandb (rank 0 only) ----
    log_cfg = cfg.get("logging") or {}
    use_wandb = bool(log_cfg.get("use_wandb", False)) and log_cfg.get("mode", "online") != "disabled"
    wandb_run = None
    if use_wandb and is_main:
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
            config={**cfg, "world_size": world_size},
        )
        log(f"[train] wandb: {wandb_run.url if wandb_run.url else '(offline)'}")

    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # ---- Dataset ----
    repo_id = expand_env(dcfg["repo_id"])
    dataset_revision = str(dcfg.get("revision", "v3.0"))
    log(f"[train] dataset: {repo_id} @ revision={dataset_revision}")

    ds_probe = FlowerSO101Dataset(
        repo_id=repo_id, root=dcfg.get("root"),
        revision=dataset_revision,
        chunk_size=int(mcfg["chunk_size"]),
        video_key=dcfg.get("video_key", "observation.images.main"),
        resize_hw=int(mcfg.get("image_hw", 224)),
    )
    log(f"[train] dataset root: {ds_probe.root}")
    log(f"[train] dataset frames: {len(ds_probe)} episodes: {len(ds_probe._episodes)}")

    # Color-stratified train/val split.
    train_eps = dcfg.get("episodes")
    val_eps: list[int] | None = None
    val_cfg = dcfg.get("val") or {}
    if train_eps is None and val_cfg:
        by = episodes_by_color(repo_id, root=ds_probe.root)
        # Optional: drop whole color buckets before the split (e.g. exclude
        # 'green' to fine-tune on only blue/red). Removes the color from BOTH
        # train and val; the stratified split then runs over what remains.
        exclude_colors = [str(c).lower() for c in (dcfg.get("exclude_colors") or [])]
        if exclude_colors:
            dropped = {c: len(by.get(c, [])) for c in exclude_colors if c in by}
            for c in exclude_colors:
                by.pop(c, None)
            remaining = {k: len(v) for k, v in sorted(by.items())}
            log(f"[train] exclude_colors={exclude_colors} dropped={dropped} "
                f"remaining_buckets={remaining}")
            if not by:
                raise SystemExit(
                    f"[train] exclude_colors={exclude_colors} removed every "
                    f"episode bucket — nothing left to train on."
                )
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
        log(f"[train] color-stratified split: train_eps={len(train_eps)} val_eps={len(val_eps)} -> {val_eps}")

    # Image augmentations are train-only.
    image_transforms = build_image_transforms(
        dcfg.get("augmentations"), image_hw=int(mcfg.get("image_hw", 224))
    )
    if image_transforms is not None:
        log(f"[train] image_transforms: enabled ({dcfg.get('augmentations')})")

    frame_cache_dir = dcfg.get("frame_cache_dir")
    if frame_cache_dir:
        log(f"[train] frame_cache_dir: {frame_cache_dir}")

    dataset = FlowerSO101Dataset(
        repo_id=repo_id, root=ds_probe.root, episodes=train_eps,
        revision=dataset_revision,
        chunk_size=int(mcfg["chunk_size"]),
        video_key=dcfg.get("video_key", "observation.images.main"),
        resize_hw=int(mcfg.get("image_hw", 224)),
        image_transforms=image_transforms,
        frame_cache_dir=frame_cache_dir,
    )
    if frame_cache_dir and dataset._frame_cache is None:
        raise SystemExit(
            f"[train] frame_cache_dir set ({frame_cache_dir}) but no valid cache "
            f"found for {repo_id}@{dataset_revision}. Run scripts/build_frame_cache.py first."
        )
    val_dataset: FlowerSO101Dataset | None = None
    if val_eps:
        val_dataset = FlowerSO101Dataset(
            repo_id=repo_id, root=ds_probe.root, episodes=val_eps,
            revision=dataset_revision,
            chunk_size=int(mcfg["chunk_size"]),
            video_key=dcfg.get("video_key", "observation.images.main"),
            resize_hw=int(mcfg.get("image_hw", 224)),
            frame_cache_dir=frame_cache_dir,
        )
    log(f"[train] train frames: {len(dataset)}  val frames: {len(val_dataset) if val_dataset else 0}")

    # Optional per-getitem prompt augmentation (task1 only). Train-only.
    prompt_aug_cfg = dcfg.get("prompt_augmentation") or {}
    if prompt_aug_cfg.get("enabled"):
        from src.data.task1_color_prompt import Task1ColorPromptDataset
        dataset = Task1ColorPromptDataset(
            base=dataset,
            seed=int(cfg.get("seed", 42)) + accelerator.process_index,
        )
        log("[train] prompt_augmentation: enabled (Task1ColorPromptDataset, per-rank seed)")

    # Normalizer constants from the full probe dataset (stable across splits).
    if ds_probe.stats is None:
        raise RuntimeError("dataset has no stats.json")
    norms = FlowerNormalizer.from_stats(ds_probe.stats, default_so101_modes())

    # ---- Phase-weighted sampler (DDP-aware, per-rank) ----
    sampler = None
    phase_cfg = tcfg.get("phase_sampling") or {}
    if phase_cfg.get("enabled"):
        from src.data.phase_labels import compute_phase_labels, summarize
        from src.data.sampler import make_phase_weighted_sampler, assert_dataset_alignment
        # Rank 0 computes/caches phase labels first to avoid NPZ-cache races,
        # then everyone else reads from cache.
        if is_main:
            log("[train] phase_sampling: enabled (DDP, per-rank sampler)")
            compute_phase_labels(
                repo_id=repo_id, root=ds_probe.root, episodes=train_eps,
                open_frac=float(phase_cfg.get("open_frac", 0.6)),
                close_frac=float(phase_cfg.get("close_frac", 0.4)),
                min_amplitude=float(phase_cfg.get("min_amplitude", 5.0)),
                post_close_margin=int(phase_cfg.get("post_close_margin", 3)),
            )
        accelerator.wait_for_everyone()
        phase_labels = compute_phase_labels(
            repo_id=repo_id, root=ds_probe.root, episodes=train_eps,
            open_frac=float(phase_cfg.get("open_frac", 0.6)),
            close_frac=float(phase_cfg.get("close_frac", 0.4)),
            min_amplitude=float(phase_cfg.get("min_amplitude", 5.0)),
            post_close_margin=int(phase_cfg.get("post_close_margin", 3)),
        )
        if len(phase_labels) != len(dataset):
            raise RuntimeError(
                f"phase_sampling: labels length {len(phase_labels)} != dataset "
                f"length {len(dataset)} — iteration order divergence."
            )
        assert_dataset_alignment(dataset, phase_labels, n_check=16)
        if is_main:
            log(f"[train] phase_sampling: {summarize(phase_labels, label=repo_id)}")
        per_rank_num_samples = len(phase_labels) // world_size
        sampler = make_phase_weighted_sampler(
            phase_labels,
            weight_pregrasp=float(phase_cfg.get("weight_pregrasp", 2.0)),
            replacement=bool(phase_cfg.get("replacement", True)),
            seed=int(cfg.get("seed", 42)) + accelerator.process_index,
            num_samples=per_rank_num_samples,
        )
        log(f"[train] phase_sampling: per-rank num_samples={per_rank_num_samples} "
            f"(global={per_rank_num_samples * world_size}, was {len(phase_labels)})")

    # ---- DataLoaders ----
    # spawn context: fork workers under DDP with overlapping MP4 byte-range
    # reads can produce PyAV decoder crashes; spawn isolates worker state.
    nw = int(tcfg.get("num_workers", 4))
    pf = int(tcfg.get("prefetch_factor", 2))
    import multiprocessing as _mp
    mp_ctx = _mp.get_context("spawn") if nw > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=per_proc_batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        drop_last=True,
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=(pf if nw > 0 else None),
        multiprocessing_context=mp_ctx,
    )
    log(f"[train] train loader: num_workers={nw} prefetch_factor={pf}  "
        f"sampler={'phase-weighted (per-rank)' if sampler is not None else 'default-shuffle (DistributedSampler via prepare)'}")

    val_loader = None
    val_subsample_fraction = float(tcfg.get("val_subsample_fraction", 1.0))
    if not (0.0 < val_subsample_fraction <= 1.0):
        raise ValueError(
            f"val_subsample_fraction must be in (0, 1], got {val_subsample_fraction}"
        )
    val_max_batches: int | None = None
    if val_dataset is not None:
        val_workers = min(4, nw)
        val_mp_ctx = _mp.get_context("spawn") if val_workers > 0 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_proc_batch,
            shuffle=False,
            num_workers=val_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=val_workers > 0,
            prefetch_factor=(pf if val_workers > 0 else None),
            multiprocessing_context=val_mp_ctx,
        )

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
    # Optional warm-start: load a prior checkpoint's config + normalizer +
    # weights (true warm-start — keep the normalized space the weights were
    # learned in). `model.init_from` = HF model id or a local dir.
    init_from = mcfg.get("init_from")
    if init_from:
        if os.path.isdir(str(init_from)):
            ckpt_dir = str(init_from)
        else:
            from huggingface_hub import snapshot_download
            if is_main:
                snapshot_download(repo_id=init_from, repo_type="model")
            accelerator.wait_for_everyone()
            ckpt_dir = snapshot_download(repo_id=init_from, repo_type="model")
        policy = FlowerVLAPolicy.from_pretrained(ckpt_dir, device="cpu")
        ck = policy.config
        if (int(ck.action_dim) != int(ds_probe.action_dim)
                or int(ck.state_dim) != int(ds_probe.state_dim)):
            raise SystemExit(
                f"[train] init_from dim mismatch: ckpt "
                f"act/state={ck.action_dim}/{ck.state_dim} vs dataset "
                f"{ds_probe.action_dim}/{ds_probe.state_dim}"
            )
        log(f"[train] warm-start from {init_from} "
            f"(ckpt config + normalizer + weights; new-dataset stats NOT used)")
    else:
        policy = FlowerVLAPolicy(cfg_pol, normalizer=norms)
    policy.train()
    log(f"[train] params: {sum(p.numel() for p in policy.parameters()):,}")
    log(f"[train] trainable params: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")

    # ---- Optim + LR ----
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable, lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        fused=torch.cuda.is_available(),
    )
    warmup = int(tcfg.get("warmup_steps", 0))
    total_steps = int(tcfg["num_steps"])
    # Gradient clipping: opt-in via cfg.train.max_grad_norm (0/absent = off).
    max_grad_norm = float(tcfg.get("max_grad_norm", 0.0) or 0.0)
    if max_grad_norm > 0:
        log(f"[train] grad clip: max_norm={max_grad_norm}")

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # ---- accelerate.prepare ----
    # sched intentionally NOT passed: accelerate's AcceleratedScheduler
    # advances the LR scheduler world_size times per step, which silently
    # shortens warmup. Keep LambdaLR stepped manually once per loop iter.
    if val_loader is not None:
        policy, optim, loader, val_loader = accelerator.prepare(
            policy, optim, loader, val_loader
        )
    else:
        policy, optim, loader = accelerator.prepare(policy, optim, loader)

    if val_loader is not None and val_subsample_fraction < 1.0:
        full_val_batches = len(val_loader)
        val_max_batches = max(1, math.ceil(full_val_batches * val_subsample_fraction))
        log(f"[train] val batches: {full_val_batches} (subsample "
            f"{val_subsample_fraction:.2f} -> {val_max_batches}/eval)")

    # ---- GPU metrics + sidecar (rank 0 only) ----
    if is_main:
        from src.utils.gpu_metrics import GpuSampler
        gpu_sampler = GpuSampler()
    else:
        gpu_sampler = None
    _git_sha_str = _git_sha_helper() if is_main else ""

    def _save_checkpoint(ckpt_dir: Path) -> None:
        """Rank-0-only save. Unwraps DDP/compile wrappers before save_pretrained."""
        if not is_main:
            return
        unwrapped = accelerator.unwrap_model(policy)
        unwrapped.save_pretrained(ckpt_dir)

    def run_eval() -> float | None:
        """Cross-rank no-grad val pass; returns mean loss or None."""
        if val_loader is None:
            return None
        policy.eval()
        losses = []
        with torch.no_grad():
            for i, vbatch in enumerate(val_loader):
                vloss, _ = policy(vbatch)
                gathered = accelerator.gather_for_metrics(vloss.detach().unsqueeze(0))
                losses.append(gathered)
                if val_max_batches is not None and (i + 1) >= val_max_batches:
                    break
        policy.train()
        if not losses:
            return None
        return float(torch.cat(losses).mean().cpu())

    eval_every = int(tcfg.get("eval_every", tcfg["save_every"]))

    es_cfg = tcfg.get("early_stop") or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 4))
    es_min_delta = float(es_cfg.get("min_delta", 0.005))
    es_min_steps = int(es_cfg.get("min_steps", 5000))
    if es_enabled:
        log(f"[train] early-stop: enabled patience={es_patience} "
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
            with accelerator.accumulate(policy):
                loss, _ = policy(batch)
                accelerator.backward(loss)
                # Clip once per optimizer step (after grad sync/unscale).
                if max_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad()

            with torch.no_grad():
                loss_reduced = accelerator.gather_for_metrics(
                    loss.detach().unsqueeze(0)
                ).mean()
            last_loss = float(loss_reduced.cpu())
            loss_window.append(last_loss)
            if len(loss_window) > 50:
                loss_window.pop(0)

            if is_main and step % int(tcfg["log_every"]) == 0:
                avg = sum(loss_window) / len(loss_window)
                lr = optim.param_groups[0]["lr"]
                elapsed = time.time() - t0
                steps_per_s = (step + 1) / max(elapsed, 1e-6)
                samples_per_s = steps_per_s * global_batch
                eta_min = (total_steps - step) / max(steps_per_s, 1e-6) / 60
                print(
                    f"[train] step={step:6d}/{total_steps}  "
                    f"loss={last_loss:.4f}  avg50={avg:.4f}  "
                    f"lr={lr:.2e}  {steps_per_s:.2f} steps/s  ETA={eta_min:.1f}min"
                )
                if use_wandb:
                    import wandb
                    payload = {
                        "train/loss": last_loss,
                        "train/loss_avg50": avg,
                        "train/lr": lr,
                        "train/steps_per_s": steps_per_s,
                        "train/samples_per_s": samples_per_s,
                    }
                    if gpu_sampler is not None:
                        payload.update(gpu_sampler.sample())
                    wandb.log(payload, step=step)

            if step > 0 and step % int(tcfg["save_every"]) == 0:
                accelerator.wait_for_everyone()
                ckpt_dir = out_dir / f"step_{step}"
                _save_checkpoint(ckpt_dir)
                if is_main:
                    write_checkpoint_meta(ckpt_dir, wandb_run, cfg, step, _git_sha_str)

            if step > 0 and step % eval_every == 0:
                val_loss = run_eval()
                if val_loss is not None:
                    if is_main:
                        print(f"[train] step={step:6d}  eval/loss={val_loss:.4f}")
                        if use_wandb:
                            import wandb
                            wandb.log({"eval/loss": val_loss}, step=step)
                    improved = val_loss < best_val_loss
                    significant = val_loss < best_val_loss - es_min_delta
                    if improved:
                        best_val_loss = val_loss
                        best_step = step
                        accelerator.wait_for_everyone()
                        best_dir = out_dir / "best"
                        _save_checkpoint(best_dir)
                        if is_main:
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
                        log(f"[train] no-improve evals: {no_improve_evals}/"
                            f"{es_patience} (best={best_val_loss:.4f} @ {best_step})")
                    if (es_enabled and no_improve_evals >= es_patience
                            and step >= es_min_steps):
                        log(f"[train] EARLY STOP at step {step}: "
                            f"{no_improve_evals} consecutive evals without "
                            f"≥{es_min_delta} improvement. "
                            f"Best val_loss={best_val_loss:.4f} @ step {best_step}.")
                        if is_main and use_wandb:
                            import wandb
                            wandb.summary["early_stop_step"] = step
                            wandb.summary["early_stop_best_step"] = best_step
                            wandb.summary["early_stop_best_loss"] = best_val_loss
                        should_stop = True

            step += 1
            if step >= total_steps or should_stop:
                break

    # ---- Final save + eval ----
    accelerator.wait_for_everyone()
    final_dir = out_dir / "final"
    try:
        _save_checkpoint(final_dir)
        if is_main:
            write_checkpoint_meta(
                final_dir, wandb_run, cfg, max(0, step - 1), _git_sha_str,
                extra={"is_final": True},
            )
            print(f"[train] saved final checkpoint -> {final_dir}")

        final_val = run_eval()
        if final_val is not None and is_main:
            print(f"[train] final eval/loss={final_val:.4f}")
            if use_wandb:
                import wandb
                wandb.log({"eval/loss": final_val}, step=max(0, step - 1))
                wandb.summary["final_eval_loss"] = final_val

        if is_main and use_wandb:
            import wandb
            avg = sum(loss_window) / max(1, len(loss_window))
            wandb.summary["final_loss"] = last_loss
            wandb.summary["final_loss_avg50"] = avg
    finally:
        if gpu_sampler is not None:
            gpu_sampler.shutdown()
        if is_main and use_wandb and wandb_run is not None:
            import wandb
            wandb.finish()

    # ---- HF Hub push (rank 0 only) ----
    accelerator.wait_for_everyone()
    hf_cfg = cfg.get("hf") or {}
    if hf_cfg.get("push") and is_main:
        push_dir = out_dir / "best" if (out_dir / "best").exists() else final_dir
        _push_to_hf(push_dir, hf_cfg, "best" if push_dir.name == "best" else "final")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
