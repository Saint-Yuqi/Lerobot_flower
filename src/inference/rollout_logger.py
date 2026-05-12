"""Per-rollout telemetry writer.

Owns `steps.csv` (per-tick), `chunks.jsonl` + `chunks.npz` (per
`policy.predict()`), `episode.json` (computed metrics + termination
reason), `outcome.json` (operator verdict + notes), and the optional
`frames/*.jpg` + `episode.mp4` artifacts. Computes the closed-loop
quality summary at `close()` time and pushes the same metrics to
`wandb.summary` if a wandb run is attached.

Public surface (mirrored exactly by `NoopLogger`):
- `log_step` — one row per control tick
- `log_chunk` — one entry per `policy.predict()` call
- `note_event(kind, msg)` — append to `events_log` (errors, warnings,
  NaN triggers, operator notes)
- `maybe_save_frame(images, step)` — gated by `--frame-every`/`--video`,
  returns the saved JPEG's relative path or `""`
- `attach_wandb(run)` — idempotent; supplying `None` is a no-op
- `close(*, verdict, notes, reason)` — flush everything and finalise
  `episode.json` + `outcome.json`. Safe to call from a `finally` block.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.gpu_metrics import GpuSampler

LOG = logging.getLogger(__name__)


# ----- Helpers -----

# Per-column float formatting for steps.csv (saves a few KB per rollout
# vs. a blanket "{:.6f}" while keeping the file human-readable).
_COL_FMT: dict[str, str] = {
    "ts_mono_s":         "{:.6f}",
    "period_target_ms":  "{:.3f}",
    "period_actual_ms":  "{:.3f}",
    "gpu_util_pct":      "{:.1f}",
    "gpu_mem_pct":       "{:.1f}",
}


def _fmt_float(col: str, val: Any) -> str:
    """Format a float using the per-column rule (default: shortest repr)."""
    if val is None:
        return ""
    if isinstance(val, (bool, np.bool_)):
        return "1" if val else "0"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if not np.isfinite(f):
        return ""  # CSV-friendly NaN/Inf; matches "no value" sentinel
    fmt = _COL_FMT.get(col, "{:g}")
    return fmt.format(f)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _dhash(rgb_uint8: np.ndarray, size: int = 16) -> str:
    """64-bit difference hash of a 16x16 grayscale thumbnail.

    dHash is robust to 1-LSB sensor noise, unlike a sha1 of full bytes.
    Two consecutive frames whose dHashes differ by Hamming distance <= 4
    are treated as "stale" by the camera-staleness metric.
    """
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[-1] != 3:
        return "0" * 16
    h, w, _ = rgb_uint8.shape
    if h < 2 or w < 2:
        return "0" * 16
    # Average over color channels to grayscale.
    gray = rgb_uint8.mean(axis=-1)
    # Resample to (size, size+1) by simple nearest-neighbor index pick.
    ys = np.linspace(0, h - 1, size).astype(np.int64)
    xs = np.linspace(0, w - 1, size + 1).astype(np.int64)
    small = gray[np.ix_(ys, xs)]
    bits = (small[:, 1:] > small[:, :-1]).astype(np.uint8).flatten()
    # Pack into a 64-bit hex string (size*size = 64 bits when size == 8;
    # we use size=16 -> 240 bits -> 60 hex chars).
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    nbits = bits.size
    nhex = (nbits + 3) // 4
    return f"{val:0{nhex}x}"


def _hamming_hex(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 64  # treat as "very different" if lengths mismatch
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


def _blur_var_laplacian(rgb_uint8: np.ndarray) -> float:
    """Variance of the Laplacian on a downsampled grayscale frame.

    Caller MUST cast to float32 before calling scipy.ndimage.laplace,
    otherwise scipy returns a uint8 Laplacian that wraps on negative
    values and the variance is silent garbage.
    """
    try:
        import scipy.ndimage as ndi
    except Exception:
        return float("nan")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[-1] != 3:
        return float("nan")
    gray = np.asarray(rgb_uint8.mean(axis=-1), dtype=np.float32)
    gray_small = gray[::4, ::4]  # ~120x160 from a 480x640 input
    if gray_small.size == 0:
        return float("nan")
    lap = ndi.laplace(gray_small, mode="reflect")
    return float(lap.var())


def _percentile_or_none(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


# ----- Logger -----

class RolloutLogger:
    """Writer for one rollout's telemetry artifacts.

    All paths are anchored at `log_dir/<inference_run_id>/`. The
    constructor creates the directory and writes `meta.json` immediately
    so a crash between init and the first `log_step` still leaves a
    grep-able identity record on disk.
    """

    def __init__(
        self,
        log_dir: Path | str,
        inference_run_id: str,
        meta: dict,
        *,
        action_dim: int = 6,
        state_dim: int = 6,
        control_hz: float = 30.0,
        chunk_size: int = 50,
        frame_every: int = 10,
        video: bool = False,
        gpu_sampler: GpuSampler | None = None,
    ) -> None:
        self.run_dir = Path(log_dir) / inference_run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.inference_run_id = inference_run_id
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.control_hz = float(control_hz)
        self.period_target_ms = 1000.0 / max(self.control_hz, 1e-6)
        self.chunk_size = int(chunk_size)
        self.frame_every = int(frame_every)
        self.video_enabled = bool(video)

        # GPU sampler: owned by the logger so the runner is unaware of it.
        self._gpu_sampler = gpu_sampler if gpu_sampler is not None else GpuSampler()

        self._wandb_run = None  # set via attach_wandb

        # ---- meta.json (one-shot at start) ----
        self._meta = dict(meta)  # caller may mutate further; we snapshot now
        self._meta.setdefault("inference_run_id", inference_run_id)
        self._meta.setdefault("started_at", _utc_iso())
        self._t_rollout_start = time.monotonic()
        # perf_counter is what `chunks.jsonl` t0/t1 are derived from in the
        # runner. Capture the offset so request/response timestamps are
        # rollout-relative (just like ts_mono_s in steps.csv).
        self._t_rollout_start_perf = time.perf_counter()
        (self.run_dir / "meta.json").write_text(json.dumps(self._meta, indent=2, default=str))

        # ---- steps.csv ----
        self._csv_path = self.run_dir / "steps.csv"
        self._csv_columns = self._build_csv_columns()
        self._csv_fh = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=self._csv_columns)
        self._csv_writer.writeheader()
        self._csv_buffer: list[dict] = []
        self._csv_flush_every = 50

        # ---- chunks.jsonl + chunks.npz ----
        self._jsonl_path = self.run_dir / "chunks.jsonl"
        self._jsonl_fh = open(self._jsonl_path, "w")
        # Hold action_horizon arrays in memory; npz written at close().
        self._chunk_arrays: dict[str, np.ndarray] = {}
        # For chunk-seam metric.
        self._prev_chunk_actions: np.ndarray | None = None
        self._prev_chunk_last_executed: np.ndarray | None = None
        self._seam_discontinuities: list[float] = []
        self._latest_chunk_actions: np.ndarray | None = None  # current chunk
        self._latest_chunk_size: int = 0

        # Closed-loop quality accumulators.
        self._period_actuals_ms: list[float] = []
        self._infer_latencies_ms: list[float] = []
        self._action_history: list[np.ndarray] = []  # rolling last-3 for jerk
        self._jerk_values: list[float] = []          # rollout-wide accumulator
        self._state_track_errors: list[float] = []   # ||state_t - action_sent_{t-1}||
        self._prev_action_sent: np.ndarray | None = None
        self._safety_clamp_counts = np.zeros(self.action_dim, dtype=np.int64)
        self._n_clipped_dim_steps = 0
        self._n_total_dim_steps = 0
        self._camera_hashes_prev: dict[str, str] | None = None
        self._camera_staleness_count = 0

        # Frame saving / mp4 streaming.
        self._frames_dir = self.run_dir / "frames"
        self._mp4_writer = None
        self._mp4_path = self.run_dir / "episode.mp4"
        self._mp4_fps = max(1.0, self.control_hz / max(self.frame_every, 1))
        # Stream-to-mp4 only when video is on AND we'd otherwise dump a
        # frame every step (frame_every == 1) — keeps the disk-jpeg path
        # for the K=10 default which is cheap and easier to spot-check.
        self._stream_video = self.video_enabled and self.frame_every == 1
        if self._stream_video:
            try:
                import imageio.v2 as imageio
                self._mp4_writer = imageio.get_writer(
                    str(self._mp4_path), fps=self._mp4_fps, macro_block_size=1
                )
                LOG.info("mp4 streaming enabled at %.1f fps -> %s",
                         self._mp4_fps, self._mp4_path)
            except Exception as e:
                self.note_event(
                    "warn.imageio",
                    f"streaming mp4 unavailable ({e!r}); skipping video",
                )
                self._mp4_writer = None
                self.video_enabled = False
        if self.frame_every > 0 and not self._stream_video:
            self._frames_dir.mkdir(exist_ok=True)

        # Counters used by close().
        self._n_steps = 0
        self._n_chunks = 0
        self._events: list[dict] = []
        self._closed = False

    # ----- Public surface -----

    def attach_wandb(self, run) -> None:
        """Attach a wandb run handle; idempotent.

        Calling with `None` is a no-op. Calling twice with the same run
        is a no-op. Calling twice with two different non-None runs is a
        programmer error and raises.
        """
        if run is None:
            return
        if self._wandb_run is not None:
            if self._wandb_run is run:
                return
            raise RuntimeError(
                "RolloutLogger.attach_wandb: a different wandb run is "
                "already attached"
            )
        self._wandb_run = run

    def note_event(self, kind: str, msg: str) -> None:
        ts = time.monotonic() - self._t_rollout_start
        self._events.append({"ts_mono_s": float(ts), "kind": str(kind), "msg": str(msg)})

    def maybe_save_frame(self, images: dict, step: int) -> str:
        """Save a JPEG (or stream mp4 frame) per `--frame-every` rule.

        Returns the relative path to the saved JPEG, or "" if no file
        was written this step (e.g. frame_every==0, gated by step%K, or
        we're streaming directly to mp4).
        """
        if self.frame_every <= 0:
            return ""
        if step % self.frame_every != 0:
            return ""
        if not images:
            return ""

        # Pick the first camera image deterministically.
        cam_name = sorted(images.keys())[0]
        img = images[cam_name]
        if img is None:
            return ""

        if self._stream_video and self._mp4_writer is not None:
            try:
                self._mp4_writer.append_data(np.asarray(img))
            except Exception as e:
                self.note_event("warn.imageio", f"mp4 append failed: {e!r}")
            return ""

        try:
            from PIL import Image
        except Exception as e:
            self.note_event("warn.pil", f"PIL unavailable: {e!r}")
            return ""
        rel = f"frames/step_{step:04d}.jpg"
        out_path = self.run_dir / rel
        try:
            Image.fromarray(np.asarray(img)).save(out_path, format="JPEG", quality=85)
        except Exception as e:
            self.note_event("warn.frame_write", f"step={step} {e!r}")
            return ""
        return rel

    def log_chunk(
        self,
        *,
        chunk_idx: int,
        t0: float,
        t1: float,
        state: np.ndarray,
        prompt: str,
        images: dict,
        chunk,
    ) -> None:
        """Record one `policy.predict()` call.

        Called at the moment the action queue is refilled. Computes
        the chunk-seam discontinuity against the previously-active
        chunk, captures camera fingerprints, and stashes the action
        horizon array for the npz.
        """
        latency_ms = float((t1 - t0) * 1000.0)
        self._infer_latencies_ms.append(latency_ms)

        # Chunk-seam discontinuity: distance between this chunk's first
        # action and the last action that was actually executed from the
        # previous chunk. Skip the very first chunk.
        if (
            self._prev_chunk_last_executed is not None
            and chunk.actions is not None
            and len(chunk.actions) > 0
        ):
            try:
                seam = float(
                    np.linalg.norm(
                        np.asarray(chunk.actions[0], dtype=np.float64)
                        - np.asarray(self._prev_chunk_last_executed, dtype=np.float64)
                    )
                )
                self._seam_discontinuities.append(seam)
            except Exception as e:
                self.note_event("warn.seam", repr(e))

        # Per-camera fingerprints.
        cam_hashes: dict[str, str] = {}
        cam_blurs: dict[str, float] = {}
        for cam_name, img in (images or {}).items():
            if img is None:
                continue
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            cam_hashes[cam_name] = _dhash(arr)
            cam_blurs[cam_name] = _blur_var_laplacian(arr)

        # Camera staleness: hamming distance against the previous chunk's
        # fingerprint per camera. <=4 bits => "essentially identical".
        if self._camera_hashes_prev is not None:
            for cam, h in cam_hashes.items():
                prev = self._camera_hashes_prev.get(cam, "")
                if prev and _hamming_hex(prev, h) <= 4:
                    self._camera_staleness_count += 1
        self._camera_hashes_prev = cam_hashes if cam_hashes else self._camera_hashes_prev

        # JSONL record.
        actions_arr = np.asarray(chunk.actions, dtype=np.float32)
        npz_key = f"chunk_{chunk_idx:04d}"
        self._chunk_arrays[npz_key] = actions_arr
        cs = int(getattr(chunk, "chunk_size", actions_arr.shape[0]))
        record = {
            "chunk_idx": int(chunk_idx),
            "ts_request_mono_s": float(t0 - self._t_rollout_start_perf_offset()),
            "ts_response_mono_s": float(t1 - self._t_rollout_start_perf_offset()),
            "latency_ms": latency_ms,
            "chunk_size": cs,
            "state_at_request": _to_jsonable_list(state),
            "prompt": str(prompt),
            "camera_hash": cam_hashes,
            "camera_blur_var": cam_blurs,
            "meta": getattr(chunk, "meta", None),
            "action_horizon_npz_key": npz_key,
        }
        self._jsonl_fh.write(json.dumps(record, default=_json_default) + "\n")
        self._jsonl_fh.flush()

        self._latest_chunk_actions = actions_arr
        self._latest_chunk_size = actions_arr.shape[0]
        self._prev_chunk_actions = actions_arr
        # `last_executed` is captured by log_step on the NEXT refill: see
        # log_step's `inferred_this_step=True` branch below.

        self._n_chunks += 1

        if self._wandb_run is not None:
            try:
                self._wandb_run.log(
                    {"infer/latency_ms": latency_ms, "infer/chunk_idx": chunk_idx},
                )
            except Exception as e:
                self.note_event("warn.wandb_log", repr(e))

    def log_step(
        self,
        *,
        step: int,
        chunk_idx: int,
        chunk_step: int,
        inferred_this_step: bool,
        queue_depth_after: int,
        state: np.ndarray,
        action_raw: np.ndarray,
        action_sent: np.ndarray,
        clamped_mask: np.ndarray,
        period_actual_ms: float | None,
        frame_path: str,
    ) -> None:
        """Record one control tick."""
        # Capture last-executed action of the OUTGOING chunk at the moment
        # we just refilled — i.e. `chunk_step==0` AND `inferred_this_step`,
        # the action at chunk_step==0 of the new chunk has not yet executed
        # but the action just popped from the previous chunk has. The
        # cleanest mapping: `action_sent` at the previous step IS the last
        # executed of the previous chunk, captured below.
        if inferred_this_step and self._prev_action_sent is not None:
            self._prev_chunk_last_executed = np.asarray(
                self._prev_action_sent, dtype=np.float32
            ).copy()

        # State-tracking error against the previous step's commanded action.
        if self._prev_action_sent is not None:
            try:
                err = float(
                    np.linalg.norm(
                        np.asarray(state, dtype=np.float64)
                        - np.asarray(self._prev_action_sent, dtype=np.float64)
                    )
                )
                self._state_track_errors.append(err)
            except Exception as e:
                self.note_event("warn.tracking", repr(e))

        # Jerk: ||a_t - 2 a_{t-1} + a_{t-2}||. Accumulate one value per
        # step so the rollout-wide mean is exact; only the trailing 3
        # actions need to live in memory.
        a_now = np.asarray(action_sent, dtype=np.float32).copy()
        self._action_history.append(a_now)
        if len(self._action_history) > 3:
            self._action_history.pop(0)
        if len(self._action_history) == 3:
            j = float(np.linalg.norm(
                self._action_history[2]
                - 2.0 * self._action_history[1]
                + self._action_history[0]
            ))
            self._jerk_values.append(j)

        # Clamp accounting.
        mask = np.asarray(clamped_mask, dtype=np.uint8).reshape(-1)
        if mask.shape[0] != self.action_dim:
            # Renormalise to action_dim if caller passed a wrong-shape mask;
            # shouldn't happen but never crash on accounting.
            tmp = np.zeros(self.action_dim, dtype=np.uint8)
            n = min(mask.shape[0], self.action_dim)
            tmp[:n] = mask[:n]
            mask = tmp
        self._safety_clamp_counts += mask.astype(np.int64)
        self._n_clipped_dim_steps += int(mask.sum())
        self._n_total_dim_steps += self.action_dim

        # GPU snapshot.
        gpu = self._gpu_sampler.sample() or {}
        gpu_util = gpu.get("system/gpu_util_pct")
        gpu_mem = gpu.get("system/gpu_mem_pct")

        # NaN flag for the diagnostic column. The runner aborts on NaN
        # before send_action so this column is reserved for "did the
        # raw output ever show non-finite" — almost always 0.
        nan_in_action = bool(not np.isfinite(np.asarray(action_raw)).all())

        ts_mono = time.monotonic() - self._t_rollout_start
        if period_actual_ms is not None:
            self._period_actuals_ms.append(float(period_actual_ms))

        row: dict[str, Any] = {
            "step": int(step),
            "ts_mono_s": float(ts_mono),
            "ts_wall_iso": _utc_iso(),
            "period_target_ms": self.period_target_ms,
            "period_actual_ms": period_actual_ms,
            "inferred_this_step": int(bool(inferred_this_step)),
            "chunk_idx": int(chunk_idx),
            "chunk_step": int(chunk_step),
            "queue_depth_after": int(queue_depth_after),
        }
        s_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        a_raw = np.asarray(action_raw, dtype=np.float32).reshape(-1)
        a_sent = np.asarray(action_sent, dtype=np.float32).reshape(-1)
        for i in range(self.state_dim):
            row[f"state_{i}"] = float(s_arr[i]) if i < s_arr.size else None
        for i in range(self.action_dim):
            row[f"action_raw_{i}"] = float(a_raw[i]) if i < a_raw.size else None
            row[f"action_sent_{i}"] = float(a_sent[i]) if i < a_sent.size else None

        # Bitfield of clamped dims (low bit = dim 0).
        bitfield = 0
        for i in range(self.action_dim):
            if i < mask.shape[0] and mask[i]:
                bitfield |= (1 << i)
        row["clamped_mask"] = int(bitfield)
        row["nan_in_action"] = int(nan_in_action)
        row["prompt_id"] = 0  # reserved for mid-run prompt switching
        row["frame_path"] = str(frame_path or "")
        row["gpu_util_pct"] = gpu_util
        row["gpu_mem_pct"] = gpu_mem

        # Format and buffer.
        formatted = {k: (_fmt_float(k, v) if isinstance(v, float)
                         or (isinstance(v, (int,)) and k in _COL_FMT)
                         else "" if v is None else v)
                     for k, v in row.items()}
        self._csv_buffer.append(formatted)
        if len(self._csv_buffer) >= self._csv_flush_every:
            self._flush_csv()

        # Wandb per-step.
        if self._wandb_run is not None:
            try:
                payload = {
                    "infer/period_actual_ms": period_actual_ms,
                    "infer/queue_depth": int(queue_depth_after),
                    "infer/inferred_this_step": int(bool(inferred_this_step)),
                }
                if gpu:
                    payload.update({
                        "system/gpu_util_pct": gpu.get("system/gpu_util_pct"),
                        "system/gpu_mem_pct": gpu.get("system/gpu_mem_pct"),
                    })
                self._wandb_run.log(payload, step=int(step))
            except Exception as e:
                self.note_event("warn.wandb_log", repr(e))

        self._prev_action_sent = a_sent.copy()
        self._n_steps += 1

    def close(self, *, verdict: str, notes: str, reason: str) -> None:
        """Flush all artifacts and finalise episode.json + outcome.json.

        Safe to call multiple times (subsequent calls are no-ops). Safe
        to call from a `finally` block; no I/O failure here propagates.
        """
        if self._closed:
            return
        self._closed = True

        try:
            self._flush_csv()
        except Exception as e:
            LOG.warning("flush_csv failed: %r", e)
        try:
            self._csv_fh.close()
        except Exception:
            pass
        try:
            self._jsonl_fh.close()
        except Exception:
            pass

        # chunks.npz — single shot at close.
        if self._chunk_arrays:
            try:
                np.savez_compressed(self.run_dir / "chunks.npz", **self._chunk_arrays)
            except Exception as e:
                LOG.warning("savez_compressed failed: %r", e)
                self._events.append({
                    "ts_mono_s": float(time.monotonic() - self._t_rollout_start),
                    "kind": "warn.npz",
                    "msg": repr(e),
                })
        else:
            # Empty npz: still write so a directory listing is consistent.
            try:
                np.savez_compressed(self.run_dir / "chunks.npz")
            except Exception:
                pass

        if self._mp4_writer is not None:
            try:
                self._mp4_writer.close()
            except Exception as e:
                LOG.warning("mp4 close failed: %r", e)

        episode = self._build_episode_json(reason=reason)
        try:
            (self.run_dir / "episode.json").write_text(json.dumps(episode, indent=2, default=_json_default))
        except Exception as e:
            LOG.warning("episode.json write failed: %r", e)

        outcome = {
            "verdict": str(verdict),
            "notes": str(notes or ""),
            "tags": [],
            "label_history": [
                {
                    "ts_wall_iso": _utc_iso(),
                    "by": "runner",
                    "verdict": str(verdict),
                    "notes": str(notes or ""),
                }
            ],
        }
        try:
            (self.run_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
        except Exception as e:
            LOG.warning("outcome.json write failed: %r", e)

        # Wandb summary + artifact.
        if self._wandb_run is not None:
            try:
                summary = {f"summary/{k}": v for k, v in episode.items()
                           if isinstance(v, (int, float, str, bool)) or v is None}
                summary["summary/verdict"] = verdict
                self._wandb_run.summary.update({k.replace("summary/", ""): v
                                                for k, v in summary.items()})
            except Exception as e:
                LOG.warning("wandb summary update failed: %r", e)

            try:
                import wandb
                art = wandb.Artifact(
                    name=f"rollout-{self.inference_run_id}",
                    type="rollout",
                )
                for fname in ("meta.json", "steps.csv", "chunks.jsonl",
                              "chunks.npz", "episode.json", "outcome.json"):
                    p = self.run_dir / fname
                    if p.exists():
                        art.add_file(str(p), name=fname)
                if self._mp4_path.exists():
                    art.add_file(str(self._mp4_path), name="episode.mp4")
                self._wandb_run.log_artifact(art)
            except Exception as e:
                LOG.warning("wandb artifact upload failed: %r", e)

        # GPU sampler shutdown — release NVML handle.
        try:
            self._gpu_sampler.shutdown()
        except Exception:
            pass

    # ----- Internals -----

    def _t_rollout_start_perf_offset(self) -> float:
        return self._t_rollout_start_perf

    def _build_csv_columns(self) -> list[str]:
        cols = [
            "step", "ts_mono_s", "ts_wall_iso", "period_target_ms",
            "period_actual_ms", "inferred_this_step", "chunk_idx",
            "chunk_step", "queue_depth_after",
        ]
        cols += [f"state_{i}" for i in range(self.state_dim)]
        cols += [f"action_raw_{i}" for i in range(self.action_dim)]
        cols += [f"action_sent_{i}" for i in range(self.action_dim)]
        cols += ["clamped_mask", "nan_in_action", "prompt_id",
                 "frame_path", "gpu_util_pct", "gpu_mem_pct"]
        return cols

    def _flush_csv(self) -> None:
        if not self._csv_buffer:
            return
        # Convert any leftover floats / Nones to display strings consistently.
        for row in self._csv_buffer:
            for k in list(row.keys()):
                v = row[k]
                if isinstance(v, float):
                    row[k] = _fmt_float(k, v)
                elif v is None:
                    row[k] = ""
        self._csv_writer.writerows(self._csv_buffer)
        self._csv_buffer.clear()
        self._csv_fh.flush()

    def _build_episode_json(self, *, reason: str) -> dict:
        """Compute the closed-loop quality summary."""
        clamp_effective = bool(self._meta.get("clamp_effective", True))

        period_jitter_p95 = _percentile_or_none(self._period_actuals_ms, 95.0)
        infer_p95 = _percentile_or_none(self._infer_latencies_ms, 95.0)
        infer_mean = _mean_or_none(self._infer_latencies_ms)
        denom = max(self.chunk_size * self.period_target_ms, 1e-6)
        infer_budget_p95 = (infer_p95 / denom) if infer_p95 is not None else None
        infer_budget_mean = (infer_mean / denom) if infer_mean is not None else None

        jerk_mean = _mean_or_none(self._jerk_values)

        clip_rate_pct = None
        if clamp_effective and self._n_total_dim_steps > 0:
            clip_rate_pct = 100.0 * self._n_clipped_dim_steps / self._n_total_dim_steps

        seam_mean = _mean_or_none(self._seam_discontinuities)
        track_mean = _mean_or_none(self._state_track_errors)

        duration_s = float(time.monotonic() - self._t_rollout_start)

        return {
            "inference_run_id": self.inference_run_id,
            "started_at": self._meta.get("started_at"),
            "ended_at": _utc_iso(),
            "duration_s": duration_s,
            "n_steps": int(self._n_steps),
            "n_chunks": int(self._n_chunks),
            "termination_reason": str(reason),
            "period_jitter_p95_ms": period_jitter_p95,
            "infer_latency_budget_ratio": infer_budget_p95,
            "infer_latency_budget_ratio_mean": infer_budget_mean,
            "action_jerk_mean": jerk_mean,
            "action_clip_rate_pct": clip_rate_pct,
            "chunk_seam_discontinuity_mean": seam_mean,
            "state_tracking_error_mean": track_mean,
            "camera_staleness_count": int(self._camera_staleness_count),
            "safety_clamp_counts_per_dim": self._safety_clamp_counts.tolist(),
            "clamp_effective": clamp_effective,
            "events_log": list(self._events),
        }


# ----- NoopLogger (mirrors public surface) -----

class NoopLogger:
    """Drop-in stand-in for `RolloutLogger` when `--no-log` is set.

    The runner has zero `if logger is not None` branches; both classes
    share a public surface, and `NoopLogger` returns benign defaults.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Accept and discard any constructor args so call sites can be
        # branchless.
        self._wandb_run = None

    def attach_wandb(self, run) -> None:
        if run is None:
            return
        if self._wandb_run is not None and self._wandb_run is not run:
            raise RuntimeError(
                "NoopLogger.attach_wandb: a different wandb run is "
                "already attached"
            )
        self._wandb_run = run

    def note_event(self, kind: str, msg: str) -> None:  # noqa: D401
        return

    def maybe_save_frame(self, images: dict, step: int) -> str:
        return ""

    def log_chunk(self, **kwargs) -> None:  # noqa: D401
        return

    def log_step(self, **kwargs) -> None:  # noqa: D401
        return

    def close(self, *, verdict: str, notes: str, reason: str) -> None:  # noqa: D401
        return


# ----- JSON encoder helpers -----

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Not JSON serialisable: {type(o)!r}")


def _to_jsonable_list(arr) -> list:
    a = np.asarray(arr)
    return [None if not np.isfinite(float(x)) else float(x) for x in a.reshape(-1)]
