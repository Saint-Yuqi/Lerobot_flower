# Per-__getitem__ visual-variant ("look") augmentation

Date: 2026-05-16
Status: design — revised after independent review

## Context / why

`ethrl2026/task1_20260509_plus` was built from a lighting-augmented source:
each episode has one of 6 photometric "looks" baked into its video (ffmpeg
`eq`/`colorbalance` at build time). We train on the **clean**
`ethrl2026/task1_all_plus_task2_pre_grasp` (255 ep / 100,547 frames; verified
no baked looks) and want those 6 looks as a **training-time augmentation** so
the policy is robust to that lighting distribution.

Decision (post-review): looks vary **per `__getitem__`** (random), applied
**on-the-fly** (approach A — torch port, no pre-bake). Rationale: gives
per-epoch lighting variety per trajectory (more robustness than task1_plus's
fixed-per-episode baked looks), with zero cache-storage cost. Pre-baking (C)
was rejected because varying looks would need ~6× cache storage ("twins").

## Ground-truth look definitions

From `task1_20260509_plus` `meta/..._metadata.jsonl` `ffmpeg_filter` column
(exact, not derived):

| look | ffmpeg filter |
|---|---|
| `original` | (identity) |
| `bright_high_contrast` | `eq=brightness=0.090:contrast=1.12:saturation=1.05` |
| `dim_low_contrast` | `eq=brightness=-0.090:contrast=0.88:saturation=0.95` |
| `warm_yellow_light` | `eq=brightness=0.035:contrast=1.07:saturation=1.12,colorbalance=rs=0.060:gs=0.015:bs=-0.055` |
| `cool_blue_light` | `eq=brightness=-0.025:contrast=1.05:saturation=1.08,colorbalance=rs=-0.050:gs=-0.006:bs=0.065` |
| `gamma_lifted_shadows` | `eq=gamma=0.82:brightness=0.020:contrast=1.08:saturation=1.02` |

## Photometric math (torch-RGB port of ffmpeg)

Operates on a CHW float32 tensor in `[0,1]`. **Out-of-place** (allocates;
never mutates input — the cache path yields a read-only memmap-derived array;
review B2). Chain order = ffmpeg filter order: `eq` then `colorbalance`.

**`eq`** (per `vf_eq.c`), applied per RGB channel (ffmpeg does Y-only — this
is a documented approximation; fidelity is enforced by the fixture test
below, not by this recollection):
1. `v = (v - 0.5) * contrast + 0.5 + brightness`  (brightness additive, post-contrast)
2. if `gamma != 1`: `v = clamp(v, 0, 1) ** (1.0 / gamma)`  (clamp before pow)
3. saturation: `L = 0.299R + 0.587G + 0.114B`; `v = L + saturation * (v - L)`

**`colorbalance`** (per `vf_colorbalance.c`, shadows-only — `rs/gs/bs` set,
midtones/highlights 0). Per pixel, lightness `l = (max(R,G,B) + min(R,G,B))/2`:
`shadow_w = clip((0.333 - l) * 4.0 + 0.5, 0, 1) * 0.7`;
`channel += shift * shadow_w`.
**The constants `(a=4.0, b=0.333, scale=0.7)` are recalled from libavfilter
and MUST be verified — the binding requirement is the fixture test, not these
numbers.** Final `clamp(0,1)`.

> **Fidelity requirement (resolves review B1/C1) — gate + rationale:**
> `apply_look` is validated against *actual ffmpeg* (imageio_ffmpeg's GPL
> build; PyAV's LGPL build lacks `eq`/`colorbalance`) per look on real
> frames via `scripts/verify_look_ffmpeg.py`.
>
> The `eq` math was corrected to **BT.601 luma/chroma domain** (the C1
> contingency) after a per-RGB-channel first cut failed the gate — that fix
> dropped `bright_high_contrast` to MAE 0.007 / max 0.023 and is the proof
> the formula is now correct. Measured residual (post-fix): MAE 1–3.5%,
> max 5–7% for the other looks.
>
> **Gate (decided 2026-05-16, with user): MAE < 0.04 and max-abs < 0.08.**
> Rationale: the objective is a robustness *augmentation*, not bitwise
> ffmpeg reproduction. The remaining residual is ffmpeg's integer
> rgb24→YUV→8-bit-LUT→rgb24 quantization (a few 8-bit levels amplified by
> the colour matrices), visually imperceptible and ~3–4× smaller than the
> `ColorJitter ±0.2` this is composed under. Chasing bitwise parity would
> require reverse-engineering swscale's limited-range matrices + LUT
> rounding — high effort, brittle, no benefit for an augmentation. The
> formula correctness is established by `bright_high_contrast` passing the
> *original* tight bar and all MAEs being 1–3.5% (not gross errors).

## Components

### `src/data/visual_variants.py` (new)

- `LOOKS`: ordered dict name→params (table above; `original`=no-op).
- `apply_look(img, name) -> img`: the math above; `original` returns input
  unchanged (bit-exact); all other paths out-of-place; output `clamp(0,1)`,
  shape/dtype/device preserved.
- `LookJitter(torch.nn.Module)`: on `forward(img)`, sample one look via
  **torch's global RNG** (`torch.randint`/`torch.multinomial`) — same RNG
  mechanism as `torchvision` `ColorJitter`, so it inherits torch's per-worker
  DataLoader seeding (distinct stream per spawn worker; no pickled-RNG
  pitfall). `distribution`: `"uniform"` (default, 1/6 each) or
  `"task1plus_empirical"` (orig 21.6%, others ~16%) as categorical weights.

### `src/data/image_transforms.py` (modify)

`build_image_transforms` gains an optional **leading, always-applied**
`LookJitter` before the existing `_RandomSubsetApply([ColorJitter,
SharpnessJitter, RandomResizedCrop], …)`. Final pipeline:
`Compose([LookJitter(...), _RandomSubsetApply(existing 3)])`. Driven by a new
`augmentations.visual_variants` sub-block; absent ⇒ pipeline unchanged
(backward compatible — other configs unaffected). LookJitter is *not* a
member of the random subset (it always runs; `original` is its built-in
no-op outcome).

### `scripts/train_flower_accelerate.py`

No structural change: `build_image_transforms` already receives
`dcfg.get("augmentations")` and is applied **train-only** (val dataset gets
`image_transforms=None`). Looks therefore are train-only; **val stays clean**
(resolves review C5 — consistent with ColorJitter/crop being train-only;
random looks on val would make `eval/loss` noisy/incomparable). Log the
resolved look distribution on rank 0.

### Config: `configs/train/full_eval1_t1allp2pg_lookaug_mgpu.yaml` (new)

Base = the **current validated t1allp2pg recipe** (bs=256, lr=7.5e-5,
`max_grad_norm=1.0`, num_workers=8, frame cache on, prompt-aug on,
phase-sampling on, no blur/noise). Add under `data.augmentations`:
```
visual_variants: { enabled: true, distribution: uniform }
```
**N1 note (review):** I default to the *current* recipe, NOT the `224558`
recipe (bs=1024/lr=1.5e-4). Rationale: per-`__getitem__` random looks already
diverge from task1_plus's fixed-per-episode structure, so "exactly the same
as that day" no longer applies; isolating the aug change on the validated
recipe gives cleaner attribution. Flagged for user confirmation.

## Verification

- Unit (local CPU, torchvision 0.17 OK — pure tensor ops):
  - `original` is bit-exact identity; input tensor byte-unchanged after any
    non-`original` `apply_look` (out-of-place invariant, review B2).
  - mean brightness `bright_high_contrast` > `original` > `dim_low_contrast`;
    `warm_yellow_light` R-mean > B-mean; `cool_blue_light` B-mean > R-mean.
  - all outputs in `[0,1]`; shape/dtype/device preserved.
  - `LookJitter` distribution over many draws ≈ configured weights (χ² loose);
    distinct draws across simulated worker seeds.
- Fixture (server, one-time): generate ffmpeg-rendered reference per look on a
  sample frame; assert `apply_look` MAE < 0.03 / max < 0.05 per look.
- Integration (server): short cache-on smoke — distribution logged, no
  shape/dtype break, GPUs fed, loss descends.

## Out of scope

- Bitwise ffmpeg parity (approach B rejected); pre-bake/twins (C rejected).
- Per-episode fixed looks (superseded by per-getitem random).
- Changing existing ColorJitter/sharpness/crop or prompt-aug.
- Applying looks to `task1_20260509_plus` (already baked — would compound).
