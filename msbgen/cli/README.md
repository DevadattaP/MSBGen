# `msbgen.cli`

Command-line tools, run as modules from the project root:

```shell
python -m msbgen.cli.<tool> --help
```

| Tool | Purpose |
| --- | --- |
| [`analyze`](analyze.py) | Dataset-level bit-plane analysis → CSV, JSON and plots |
| [`rate`](rate.py) | Lossless adaptive range-coded top-k planes → actual bpp report |
| [`prepare_data`](prepare_data.py) | Download and convert datasets (DIV2K, Kodak) to grayscale or RGB `.npy` |
| [`train`](train.py) | Train a baseline, diffusion/flow-matching model, or conditional normalizing flow |
| [`evaluate`](evaluate.py) | Score the fills (zero / mid / random) and trained checkpoints on a test set |
| [`bitdepth`](bitdepth.py) | Train, evaluate, and plot the sequential bit-plane recovery baseline |
| [`plot_runs`](plot_runs.py) | Plot training histories (loss, PSNR, SSIM, LR) and export them as CSV |
| [`plot_eval`](plot_eval.py) | Plot evaluation results: all metrics per method, perception vs distortion, paired per-image comparison |
| [`plot_steps`](plot_steps.py) | Plot generative models against their number of sampling steps (curves and perception–distortion trade-off) |

The baseline workflow, end to end:

```shell
python -m msbgen.cli.prepare_data div2k kodak           # once: ~4 GB download
python -m msbgen.cli.train --preset unet-res            # → runs/unet-res/best.pt
python -m msbgen.cli.plot_runs                          # training curves → outputs/plots/
python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips
uvicorn msbgen.main:app                                 # models appear in the web UI
```

The generative model (Milestone 3) uses the same tools:

```shell
python -m msbgen.cli.train --preset diffusion           # → runs/diffusion/best.pt
python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips --dists --kid --samples 4
python -m msbgen.cli.plot_eval                          # evaluation plots → outputs/plots/
```

### RGB reconstruction workflow

The default preparation and training workflow remains grayscale. RGB preparation writes
three-channel uint8 arrays under `<root>/processed/<name>-rgb/`; the grayscale folders are
unchanged. For RGB, `k` means k MSB planes **per channel** (3k input planes per pixel).

```shell
# Prepare RGB DIV2K and Kodak once
python -m msbgen.cli.prepare_data div2k kodak --color rgb

# Train any preset on RGB arrays; auto (the default) reads the channel count from --train
python -m msbgen.cli.train --preset unet-res --name unet-res-rgb-k2 --k 2 \
  --train data/processed/div2k-train-rgb --val data/processed/div2k-valid-rgb

# Evaluate checkpoints against prepared RGB Kodak arrays
python -m msbgen.cli.evaluate data/processed/kodak-rgb --k 2 \
  --models runs/unet-res-rgb-k2/best.pt --lpips --dists --kid

# Sequential bit-depth recovery uses one RGB checkpoint per missing bitplane
for b in 1 2 3 4 5 6 7; do
  python -m msbgen.cli.bitdepth train --input-bits "$b" \
    --train data/processed/div2k-train-rgb --val data/processed/div2k-valid-rgb \
    --out runs/bitdepth-rgb --augment --device cuda
done
python -m msbgen.cli.bitdepth evaluate --models runs/bitdepth-rgb \
  --test data/processed/kodak-rgb --start-bits 1 2 3 --save-images
```

Use the same channel mode for training and evaluation. `train --color auto` infers it from
the training arrays; `--color gray|rgb` can select it explicitly. Evaluation reads channel
counts from `.npy` arrays; `evaluate --color rgb` selects RGB conversion for ordinary image
files. Checkpoint channel mismatches are skipped by the evaluator.

Every tool is a thin layer over [`msbgen.core`](../core/README.md). The numbers
match what the web app shows for a single image.

## `analyze`: dataset-level bit-plane analysis

Runs [`analyze_image`](../core/README.md#analysis-analysispy) on every image in a
folder and averages the results. This produces the Milestone "bit-plane analysis"
numbers: entropy per plane, plane correlations, and quality of top-k
reconstructions.

```shell
# 8-bit grayscale study, as in PLAN.md, with plots
python -m msbgen.cli.analyze data/kodak --gray --plots

# RGB as-is, sub-folders included, quick pass without SSIM
python -m msbgen.cli.analyze data/div2k -r --no-ssim --limit 100 --out outputs/analysis/div2k_quick
```

## `rate`: actual top-plane coding rate

`rate` encodes the transmitted top-k planes with an adaptive binary range coder
using the same four causal neighbours as `context_entropy`, then decodes the
stream and verifies the top-bit image. It reports the theoretical raw rate, the
context-entropy estimate, the arithmetic-coded payload rate, and the framed stream
rate including four-byte per-plane length fields. Image dimensions and k are
assumed to be known by the surrounding container and are not counted.

```shell
python -m msbgen.cli.rate data/processed/kodak --gray --ks 1 2 3 \
  --out outputs/rate/kodak
```

Outputs are `per_image.csv` and `summary.json`. `roundtrip` must be true for
every row. The framed bpp is the measured lossless stream rate for this coder;
it excludes the image/container metadata and any model side information.

### Rate options

| Option | Default | Meaning |
| --- | --- | --- |
| `input` | required | Folder of images |
| `--ks` | `1 2 3` | Top-plane counts to encode |
| `--gray` | off | Convert RGB images to grayscale |
| `-r`, `--recursive` | off | Search subfolders |
| `--limit N` | all | Process at most N images |
| `--out DIR` | `outputs/rate/<input folder name>` | Output folder |

### Analyze options

| Option | Default | Meaning |
| --- | --- | --- |
| `input` | required | Folder of images (`.png .jpg .jpeg .bmp .tif .tiff .webp .ppm .pgm`) |
| `--out DIR` | `outputs/analysis/<input folder name>` | Where to write results (`outputs/` is git-ignored) |
| `--gray` | off | Convert RGB to grayscale (BT.601 luma, identical to Pillow's `L`) before analysis |
| `-r`, `--recursive` | off | Also search sub-folders |
| `--limit N` | all | Analyse only the first N images (sorted by path) |
| `--no-ssim` | off | Skip SSIM, which is the slowest part of the analysis |
| `--plots` | off | Also save PNG plots (needs `matplotlib`) |
| `--workers N` | half the CPU cores | Number of parallel processes; each process handles whole images |

### How images are processed

1. Each file is decoded with `load_image`, exactly as the web app does: grayscale
   or RGB at the native bit depth, with alpha dropped and palette/CMYK/… converted
   to RGB.
2. With `--gray`, RGB images are converted to luma. The bit depth is kept, so
   16-bit RGB becomes 16-bit grayscale.
3. Images are grouped by **colour mode and bit depth**: `L-8`, `RGB-8`, `L-16`,
   `RGB-16`, `L-1`. Statistics with different numbers of planes or channels can't
   be averaged, so each group is summarised separately. A normal 8-bit dataset
   gives one group.
4. Files that fail to decode are skipped, reported on stderr, and listed in
   `summary.json` under `meta.skipped`. The exit code is 1 only if *no* image could
   be analysed.

### Outputs

```bash
outputs/analysis/kodak/
├── planes.csv              one row per image × channel × plane
├── reconstruction.csv      one row per image × k
├── summary.json            per-group means + mean correlation matrix
├── L-8_entropy.png         ┐
├── L-8_psnr.png            │ with --plots, one set per group
├── L-8_ssim.png            │ (no SSIM plot with --no-ssim)
└── L-8_correlation.png     ┘
```

A summary table (entropy per plane, and quality per k) is also printed to stdout.
Progress goes to stderr, so `> summary.txt` captures just the table.

#### `planes.csv`

| Column | Meaning |
| --- | --- |
| `file` | Path relative to the input folder |
| `group` | e.g. `L-8` |
| `channel`, `bit` | `L` / `R` / `G` / `B`, and bit index k (n−1 = MSB) |
| `ones_ratio` | Fraction of pixels with $B_k = 1$ |
| `entropy` | i.i.d. entropy, bits/pixel |
| `context_entropy` | Entropy given the W, NW, N, NE neighbours, bits/pixel |
| `spatial_correlation` | Mean neighbour correlation; empty for a constant plane |

#### `reconstruction.csv`

| Column | Meaning |
| --- | --- |
| `file`, `group` | As above |
| `k` | Number of most-significant planes kept |
| `bits_per_pixel` | Raw cost, k × channels |
| `coded_bpp` | Sum of the kept planes' context entropies (an estimate, see [core](../core/README.md#context-entropy-context_entropyplane)) |
| `psnr_zero`, `psnr_mid`, `psnr_random` | PSNR (dB) for zero / midpoint / random fill; `inf` when lossless (k = n) |
| `ssim_zero`, `ssim_mid`, `ssim_random` | SSIM; empty with `--no-ssim` or for images below 11×11 |
| `mae_zero`, `mae_mid`, `mae_random` | Mean absolute error in intensity levels |

These are long-format tables, so they load straight into pandas for your own
aggregation:

```python
import pandas as pd
rec = pd.read_csv("outputs/analysis/kodak/reconstruction.csv")
rec.groupby("k")[["psnr_mid", "ssim_mid", "coded_bpp"]].agg(["mean", "std"])
```

#### `summary.json`

```json
{
  "meta": { "input": "data/kodak", "gray": true, "images": 24, "skipped": [] },
  "groups": {
    "L-8": {
      "images": 24, "bit_depth": 8, "channels": ["L"], "mean_pixels": 393216.0,
      "planes": [
        { "channel": "L", "bit": 7, "ones_ratio": 0.45, "entropy": 0.98,
          "context_entropy": 0.12, "spatial_correlation": 0.89 }
      ],
      "reconstructions": [
        { "k": 1, "bits_per_pixel": 1, "coded_bpp": 0.12,
          "psnr_zero": 10.4, "psnr_mid": 17.0, "psnr_random": 13.8,
          "ssim_zero": 0.238, "ssim_mid": 0.579, "ssim_random": 0.129,
          "mae_zero": 69.1, "mae_mid": 31.8, "mae_random": 42.7, "lossless_images": 0 }
      ],
      "correlation": { "labels": ["B7", "B6", "..."], "matrix": [[1.0, 0.21, "..."]] }
    }
  }
}
```

(The numbers above are illustrative.)

How the averages are computed:

- Every value is the **mean over images** of the per-image value. PSNR is
  therefore the mean of per-image PSNR, the usual convention in papers, not the
  PSNR of the mean MSE.
- Infinite PSNR (lossless) is excluded from the mean. If every image is lossless
  at that k (always true at k = n), the value is `null`. `lossless_images` counts
  how many images were lossless at that k.
- The correlation matrix is the element-wise mean over images, ignoring images
  where a pair is undefined. It is `null` only if the plane is constant in every
  image.

### Performance notes

- Each worker holds one whole image at a time, together with its planes and a few
  float64 full-size buffers for the metrics. Lower `--workers` if very large
  images run out of memory.
- SSIM runs 3 × n times per image (all three fills, every k), so `--no-ssim` makes the
  run several times faster. Use it for large sweeps, then run SSIM on the final
  set.
- With several workers, output order and results are deterministic. Files are
  processed in sorted order and nothing is sampled at random.

### Programmatic use

```python
from msbgen.cli.analyze import main, analyse_file, summarise

main(["data/kodak", "--gray", "--no-ssim", "--workers", "1"])   # same as the CLI
```

`analyse_file(path, root, gray, with_ssim)` returns the per-image rows, and
`summarise(results)` builds the `groups` part of `summary.json`.

## `prepare_data`: download and prepare datasets

```shell
python -m msbgen.cli.prepare_data --list            # available datasets
python -m msbgen.cli.prepare_data div2k kodak       # default when no names are given
python -m msbgen.cli.prepare_data div2k kodak --color rgb
```

| Option | Default | Meaning |
| --- | --- | --- |
| `datasets` | `div2k kodak` | Names or groups: `div2k-train`, `div2k-valid`, `kodak`; `div2k` = both DIV2K splits |
| `--root DIR` | `data` | Data folder (git-ignored) |
| `--color` | `gray` | `gray`: grayscale BT.601 arrays in `processed/<name>`; `rgb`: three-channel RGB arrays in `processed/<name>-rgb` |
| `--delete-raw` | off | Delete downloads and extracted originals after conversion |
| `--list` | | List datasets with sizes and exit |

It downloads to `data/raw/` (resumable: just re-run after an interruption),
extracts, and writes `data/processed/<name>/*.npy` plus a `manifest.json`. Gray mode stores
8-bit `(H, W)` BT.601 luma arrays; RGB mode stores `(H, W, 3)` RGB arrays in the separate
`<name>-rgb` folders. Finished steps are skipped, so re-running is instant. Details are in
[`msbgen/data/README.md`](../data/README.md).

## `train`: train a model

```shell
python -m msbgen.cli.train --preset unet-res                 # PLAN baseline 4
python -m msbgen.cli.train --preset cnn --steps 10000        # override any option
python -m msbgen.cli.train --preset diffusion                # Milestone 3
python -m msbgen.cli.train --preset flow                     # Milestone 4
python -m msbgen.cli.train --preset normalizing-flow         # Milestone 5
python -m msbgen.cli.train --resume runs/unet-res/last.pt --preset unet-res
python -m msbgen.cli.train --preset unet-res --color rgb --k 2 \
  --train data/processed/div2k-train-rgb --val data/processed/div2k-valid-rgb
```

**Presets:**

| Preset | Model | Output | PLAN.md |
| --- | --- | --- | --- |
| `cnn` | `SimpleCNN` width 64, depth 8 (0.22 M params) | direct: predicts I | Baseline 2: pixelwise regression |
| `unet` | `UNet` width 24, 4 levels (4.4 M params) | direct: predicts I | Baseline 3: U-Net |
| `unet-res` | same U-Net | residual: predicts R = I − 128·B7 | Baseline 4: residual prediction |
| `diffusion` | `MSBDiffusion`: `TimeUNet` width 32, 4 levels, 1 block (4.0 M params); EMA 0.999, 40k steps, validation every 2k | samples R ~ p(R \| B7) | Milestone 3: conditional diffusion |
| `flow` | `MSBFlow`: the same network and training as `diffusion`; velocity loss on straight paths | samples R ~ p(R \| B7) | Milestone 4: conditional flow matching |
| `normalizing-flow` | `MSBNormalizingFlow`: conditional Glow, 4 scales × 4 coupling blocks, 128-pixel patches; 40k steps, likelihood checkpoint selection | samples R ~ p(R \| B7), no ODE steps | Milestone 5: conditional normalizing flow |
| `hybrid` | `MSBHybrid`: residual U-Net → flow correction → diffusion refinement; EMA 0.999, 40k steps (`--init-unet`, `--freeze-unet`) | samples R | Milestone 6: hybrid |
| `vit` | `ViT` patch 8, dim 192, depth 6, 6 heads (2.9 M params); 40k steps | residual: predicts R | ViT experiment |
| `vit-hybrid` | `MSBHybrid` with a residual ViT as stage 1 (`unet_arch="vit"`, 10.9 M params); EMA 0.999, 40k steps | samples R | ViT experiment |

**`--resume` rebuilds the configuration from the command line**, so repeat the same
preset and options (and the same `--steps`: the learning-rate schedule depends on it).

**Options:**

| Option | Default | Meaning |
| --- | --- | --- |
| `--preset` | | One of the presets above |
| `--name` | the preset name | Run name; output goes to `<out-dir>/<name>/` |
| `--kind` | from the preset | `regression`, `diffusion`, `flow`, `normalizing_flow` or `hybrid` |
| `--arch`, `--output` | from the preset | Regression only: `cnn`/`unet`/`vit`, `direct`/`residual` |
| `--width`, `--depth`, `--levels`, `--blocks` | from the preset | Model size (`--depth`: CNN layers / ViT blocks; `--blocks`: ResBlocks per resolution, generative models) |
| `--patch`, `--dim`, `--heads` | 8, 192, 6 | ViT patch size, token width, attention heads (also for `--unet-arch vit`) |
| `--unet-arch` | `unet` | Hybrid: stage-1 backbone, `unet` or `vit` |
| `--crop-size`, `--scales`, `--blocks-per-scale` | 128, 4, 4 | Normalizing flow: fixed patch size, squeeze scales, and Glow blocks per scale |
| `--condition-width`, `--hidden-width` | 32, 64 | Normalizing flow: B7 feature width and coupling subnet width |
| `--k` | 1 | Transmitted planes per channel (the model input is B7 … B(8−k) for each channel) |
| `--train`, `--val` | DIV2K train / valid | Folders of prepared `.npy` files (several allowed) |
| `--color` | `auto` | `auto` infers 1 or 3 channels from training arrays; `gray` / `rgb` sets the model channel count explicitly |
| `--steps` | 20000 (generative presets 40000) | Optimiser steps |
| `--batch-size`, `--crop` | 16, 128 | Batch of random crops |
| `--lr` | 2e-4 | Adam learning rate; 500-step warm-up, then cosine decay to 1e-6 |
| `--loss` | mse | Regression only: `mse` (as in PLAN.md) or `l1` |
| `--val-every`, `--val-limit` | 1000 (generative presets 2000), all | Validation interval / number of validation images |
| `--ema` | 0 (generative presets 0.999) | Decay of an exponential moving average of the weights; validation and the saved inference weights use it. 0 = off |
| `--sample-steps` | 3 | Diffusion / flow matching: DDIM / ODE sampling steps; not used by the normalizing flow |
| `--val-sample-steps` | `--sample-steps` | Diffusion / flow matching: sampling steps for validation samples |
| `--solver` | euler | Flow matching only: default ODE solver (`euler` / `heun`), used in validation and stored in the checkpoint |
| `--log-every` | 100 | Training-loss log interval |
| `--workers` | 4 | DataLoader processes |
| `--seed` | 0 | Makes the crop stream and initialisation reproducible |
| `--device` | auto | `cuda` if available, else `cpu` |
| `--resume` | | Continue from a `last.pt` (same crop stream, optimiser and LR schedule) |
| `--gpu-max-temp`, `--gpu-resume-temp` | 80, 72 | Thermal guard: pause at ≥ max °C, resume at ≤ resume °C |
| `--cpu-max-temp`, `--cpu-resume-temp` | 90, 80 | The same for the CPU package temperature |
| `--no-thermal-guard` | off | Disable the guard (desktops / servers with proper cooling) |

**Thermal guard.** The development laptop shut itself down at about 95 °C during a
training run. Its MX350 can't be power-capped (`nvidia-smi -pl` isn't supported), so
training protects the machine itself:

- Every 5 s it reads the GPU temperature (`nvidia-smi`) and the CPU package
  temperature (`psutil`).
- When either reaches its max, it **saves `last.pt`**, then pauses until *both* have
  cooled to their resume temperature.
- It is also checked inside validation, after every batch and every denoising step.
  Diffusion validation samples every crop, which is minutes of full load: the
  laptop once shut down during an unguarded one.
- The gap between the two thresholds (hysteresis) stops it flapping on and off.
- Every training log line shows the current temperatures. `log.jsonl` records
  `gpu_temp`, `cpu_temp` and the total `paused_s`, and `it_per_s` excludes the pauses.
- Without sensors (no NVIDIA GPU, no psutil) the guard does nothing.

The code is in [`msbgen/utils`](../utils/): `gpu_guard.py`, `cpu_guard.py` and
`thermal.py`. If training pauses very often, lower the load with a smaller
`--batch-size` or fewer `--workers`, or raise the limits a little. The GPU's own
slowdown point is 100 °C. The diffusion model runs hotter than the baselines: on the
MX350 it pauses for roughly 15–20 s after every ~20 s of training, which makes a run
about 1.8× longer than its compute time.

**What a run writes** (`runs/<name>/`, git-ignored):

| File | Content |
| --- | --- |
| `config.json` | Full training and model configuration |
| `log.jsonl` | The full training history (see below) |
| `last.pt` | Latest checkpoint, written at every validation, before every thermal pause, and on **Ctrl-C** |
| `best.pt` | Best checkpoint so far: highest validation PSNR (regression) or lowest validation loss (generative models). This is what `evaluate` and the web app use |

With EMA (generative models), a checkpoint's `state_dict` holds the EMA weights and
`train_state_dict` the raw weights that `--resume` continues from.

**Training history (`log.jsonl`)** has one JSON object per line, of two kinds:

| Record | When | Fields |
| --- | --- | --- |
| training | every `--log-every` steps | `step`, `epoch`, `loss` (mean over those steps), `train_psnr` (PSNR of that mean MSE; `null` for generative models), `lr`, `it_per_s`, `gpu_temp`, `cpu_temp`, `paused_s` |
| validation | step 0, then every `--val-every` steps | `step`, `epoch`, `val_loss`, `val_psnr`, `val_psnr_mid`, `val_psnr_random`, `val_gain_db`, `val_ssim`, `val_ssim_mid`, `val_ssim_random`, `val_msb_consistency`, `best_val_psnr` (regression) or `best_val_loss` (generative models) |

- Training counts steps, not epochs. `epoch` = pixels seen / pixels in the training
  set, so it is fractional: 20k steps × 16 crops of 128² ≈ 2.4 epochs of DIV2K.
- The step-0 validation scores the untrained model, so every curve starts from the
  beginning.
- On `--resume`, records after the checkpoint's step are dropped before training
  continues, so a crash never leaves duplicate steps.
- Plot the history with [`plot_runs`](#plot_runs-training-curves).

**Validation** uses one 256×256 centre crop per DIV2K-valid image. It reports the
model's mean PSNR, the midpoint and random fills' PSNR and SSIM on the same crops
(the random fill uses a fixed seed, so it is identical at every validation), and
the **gain in dB** over the midpoint fill, which is the number that matters for
regression. If the gain isn't clearly positive, the model has learned nothing beyond
the constant fill.

**Diffusion validation** is different in two ways:

- `val_loss` is the v-prediction loss with fixed t and noise per crop, so it is
  comparable between validations. It selects `best.pt`.
- `val_psnr` / `val_ssim` are of **one sample** per crop (`--val-sample-steps` DDIM
  steps, fixed noise). A good generative model should end up about 3 dB *below* the
  regression U-Net on PSNR, near the random fill or above it, with a negative
  `val_gain_db`. See [`msbgen/models/README.md`](../models/README.md#why-a-sample-scores-lower-psnr).
  Watch SSIM and the loss instead.

On the MX350 laptop GPU, 20k steps take about 1–1.5 h for the U-Net and about
1.7 h for the CNN. The diffusion preset's 40k steps take about 5 h of compute, plus
validations and thermal pauses: plan for 9–10 h. See
[`msbgen/models/README.md`](../models/README.md) for measured speeds.

## `plot_runs`: training curves

```shell
python -m msbgen.cli.plot_runs                                   # every runs/*/ with a log.jsonl
python -m msbgen.cli.plot_runs runs/unet-res runs/unet --x epoch --out outputs/plots
```

| Option | Default | Meaning |
| --- | --- | --- |
| `runs` | all of `runs/*/` | Run folders to include |
| `--out DIR` | `outputs/plots` | Output folder |
| `--x` | `step` | x axis: `step` or `epoch` |
| `--smooth N` | 5 | Moving-average window for the (noisy) training curves |

**Outputs:**

| File | Content |
| --- | --- |
| `<run>_history.csv` | Training and validation records merged into one row per step, for your own plots or tables |
| `<run>_overview.png` | Per run, 4 panels: train vs validation loss, train vs validation PSNR (with the midpoint- and random-fill lines), validation SSIM, learning rate |
| `loss.png` | All runs: training loss (faint, smoothed) and validation loss (markers), log scale |
| `val_psnr.png`, `val_ssim.png` | All runs, with the midpoint-fill (dashed) and random-fill (dotted) baselines. Runs logged before the random fill existed have no random-fill values, so that line is omitted for them. |
| `val_gain.png` | All runs: PSNR gain over midpoint fill, the main "did it learn anything" curve |
| `lr.png` | Learning-rate schedules |

How to read them: validation loss rising while training loss keeps falling means
overfitting. A gain curve that flattens means more steps won't help, and a bigger
model or more context might.

## `plot_eval`: evaluation plots

```shell
python -m msbgen.cli.plot_eval                                     # every outputs/eval/*/ with a summary.json
python -m msbgen.cli.plot_eval outputs/eval/kodak-s8 --pair diffusion unet-res
```

| Option | Default | Meaning |
| --- | --- | --- |
| `evals` | all of `outputs/eval/*/` | Evaluation folders (written by [`evaluate`](#evaluate-score-methods-on-a-test-set)) |
| `--out DIR` | `outputs/plots` | Output folder |
| `--pair A B` | the generative model vs the best-PSNR regression model | Methods for the paired plot |

**Outputs** (per evaluation folder `<name>`):

| File | Content |
| --- | --- |
| `eval_<name>_metrics.png` | One panel per metric (PSNR, SSIM, LPIPS, DISTS, KID), the same method order in each, coloured by kind (fill, regression, generative); the best value is labelled. Hollow circles in the PSNR panel: PSNR of the mean of the samples. |
| `eval_<name>_perception.png` | Perception vs distortion: PSNR against DISTS and LPIPS. Better is towards the bottom right; a generative model is expected to sit lower and further left than regression. |
| `eval_<name>_paired.png` | One point per image, A against B on each per-image metric, with the y = x line and how many images A wins |

Whiskers on per-image metrics are 95 % intervals of the mean over images. They are
unpaired, so two methods can differ reliably even when their whiskers overlap: the
paired plot shows that directly. KID whiskers are the std over KID subsets; summaries
written before that was fixed show none ("no spread: rerun evaluate").

## `plot_steps`: generative models against sampling steps

```shell
# one evaluate run per step count first, e.g. --sample-steps $s --out outputs/eval/kodak-gen-s$s
python -m msbgen.cli.plot_steps outputs/eval/kodak-gen-s* outputs/eval/kodak-fm
```

Collects every generative method from the folders, keyed by its `sample_steps`. Each
model is one series; a non-default sampler gets its own series, e.g. `flow (heun)` or
`diffusion (η = 1)`. The regression models and the midpoint fill are reference points.
All folders must be on the same data and k. If a model and step count appears in
several folders, the most recently written summary wins.

| Option | Default | Meaning |
| --- | --- | --- |
| `evals` | all of `outputs/eval/*/` | Evaluation folders |
| `--out DIR` | `outputs/plots` | Output folder |
| `--name` | `<data folder>-k<k>` | Name in the output files |

| File | Content |
| --- | --- |
| `steps_<name>_curves.png` | PSNR, DISTS, KID and diversity against steps (log axis), one line per series with a 95 % band (KID: ± std), and the best-PSNR regression model as a dashed line |
| `steps_<name>_tradeoff.png` | PSNR against DISTS, each series' step counts joined in order and labelled, with the regression models and midpoint fill as points (needs `--dists`) |

## `evaluate`: score methods on a test set

```shell
python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips
```

Every method gets **only the top-k bits of each channel** (through the
`Reconstructor` interface) and is scored against the original. Prepared `.npy` images retain
their stored channels; ordinary image files are loaded as grayscale by default or RGB with
`--color rgb`. The zero, midpoint
and random fills are included as baselines by default.

**Stochastic methods** (random fill, diffusion, flow matching) are scored on their first sample
(seed 0). With `--samples N` they are drawn N times (seeds 0 … N−1), which adds
two metrics:

- `psnr_avg`: the PSNR of the **average** of the N samples. For a well-calibrated
  generative model it rises towards the regression U-Net's PSNR as N grows, because
  the average of many samples approximates the MMSE estimate. For the random fill
  it approaches the midpoint fill.
- `diversity`: the mean per-pixel standard deviation across the samples, in
  intensity levels. 0 means every sample is identical.

The thermal guard is checked between images and after every sampling step of a
generative model.

| Option | Default | Meaning |
| --- | --- | --- |
| `data` | required | Folder of prepared `.npy` files or ordinary images |
| `--color` | `gray` | Conversion for ordinary image files: `gray` or `rgb`; `.npy` channels are retained |
| `--models` | none | Checkpoints (regression, diffusion, flow, normalizing flow, or hybrid); each is named after its run folder |
| `--bitdepth-models` | none | Folder(s) of sequential bit-plane CNN checkpoints, trained for the same channel count as the test images |
| `--samples N` | 1 | Draws per image for stochastic methods |
| `--sample-steps` | checkpoint's (3; 50 in runs trained before the step sweep) | Generative models: DDIM / ODE steps |
| `--eta` | checkpoint's (0) | Diffusion: DDIM eta; 0 = deterministic given the starting noise, 1 = fresh noise every step |
| `--solver` | checkpoint's (euler) | Flow matching: ODE solver, `euler` or `heun` |
| `--k` | 1 | Transmitted planes; models trained for another k are skipped |
| `--fills` | `zero mid random` | Fill baselines to include |
| `--lpips` | off | Also compute LPIPS (downloads AlexNet weights once) |
| `--dists` | off | Also compute DISTS (needs `piq`; downloads VGG16 + DISTS weights once) |
| `--kid` | off | Also compute patch KID per method against the originals (needs `piq`; downloads the FID Inception once) |
| `--kid-patch PX` | 128 | KID patch size: a non-overlapping grid plus the grid shifted by PX/2 (39 patches per Kodak image) |
| `--no-ssim` | off | Skip SSIM |
| `--limit N` | all | First N images only |
| `--save-images N` | 4 | Save reconstructions and a comparison strip for the first N images |
| `--out DIR` | `outputs/eval/<data folder name>` | Output folder |
| `--device` | auto | `cuda` / `cpu` |
| `--no-thermal-guard` | off | Don't pause when the GPU / CPU is hot |

**Outputs:**

| File | Content |
| --- | --- |
| `per_image.csv` | `image, method, psnr, ssim, lpips, dists, mae, msb_consistency, psnr_avg, diversity, time_s` (`time_s`: seconds for one reconstruction) |
| `summary.json` | Per method: mean and std of each metric over images, plus checkpoint path, step, model kind/config and sampler settings (`sample_steps`/`eta`, `sample_steps`/`solver`, or normalizing-flow `patch_size`). With `--kid`, `kid` holds the mean and std over random KID subsets, patch counts, and subset size |
| `images/<image>_<method>.png` | Rounded 8-bit reconstructions (sample 0 for stochastic methods) |
| `images/<image>_compare.png` | Original and every method side by side, labelled with PSNR |
| `images/<image>_<method>_samples.png` | With `--samples` > 1: original, every sample, their mean and their per-pixel std × 4 |

KID is a set-level score, so it is not in `per_image.csv`. It compares Inception
features of patches from a method's outputs (every sample, for a stochastic
method) with those of the original images' patches, and asks only whether the
outputs *look like* real images, regardless of pixel alignment. 24 Kodak images
are far too few for FID, but their ~900 patches are enough for KID, which is
unbiased at small sizes. The table shows it × 1000.

A summary table is printed as well. `msb_consistency` is the fraction of pixels
whose reconstruction keeps the transmitted bits. It is always 1 for the fills and
residual models, and can be lower for direct models.

## Adding a tool

Create `msbgen/cli/<name>.py` with a `main(argv: list[str] | None = None) -> int`
that parses its own arguments, and end it with
`if __name__ == "__main__": raise SystemExit(main())`. Put the logic in
`msbgen.core` and keep the tool itself to I/O and argument handling. Add a test
that calls `main([...])` with a `tmp_path`, like
[`tests/test_cli.py`](../../tests/test_cli.py), and a row in the table above.
