# `msbgen.cli`

Command-line tools, run as modules from the project root:

```shell
python -m msbgen.cli.<tool> --help
```

| Tool | Purpose |
| --- | --- |
| [`analyze`](analyze.py) | Dataset-level bit-plane analysis → CSV, JSON and plots |
| [`prepare_data`](prepare_data.py) | Download and convert datasets (DIV2K, Kodak) to grayscale `.npy` |
| [`train`](train.py) | Train a deterministic baseline (CNN / U-Net) |
| [`evaluate`](evaluate.py) | Score constant fills and trained checkpoints on a test set |
| [`plot_runs`](plot_runs.py) | Plot training histories (loss, PSNR, SSIM, LR) and export them as CSV |

The baseline workflow, end to end:

```shell
python -m msbgen.cli.prepare_data div2k kodak           # once: ~4 GB download
python -m msbgen.cli.train --preset unet-res            # → runs/unet-res/best.pt
python -m msbgen.cli.plot_runs                          # training curves → outputs/plots/
python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips
uvicorn msbgen.main:app                                 # models appear in the web UI
```

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

### Options

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
| `psnr_zero`, `psnr_mid` | PSNR (dB) for zero / midpoint fill; `inf` when lossless (k = n) |
| `ssim_zero`, `ssim_mid` | SSIM; empty with `--no-ssim` or for images below 11×11 |
| `mae_zero`, `mae_mid` | Mean absolute error in intensity levels |

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
          "psnr_zero": 11.9, "psnr_mid": 16.1, "ssim_zero": 0.42, "ssim_mid": 0.55,
          "mae_zero": 52.1, "mae_mid": 32.8, "lossless_images": 0 }
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
- SSIM runs 2 × n times per image (both fills, every k), so `--no-ssim` makes the
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
```

| Option | Default | Meaning |
| --- | --- | --- |
| `datasets` | `div2k kodak` | Names or groups: `div2k-train`, `div2k-valid`, `kodak`; `div2k` = both DIV2K splits |
| `--root DIR` | `data` | Data folder (git-ignored) |
| `--delete-raw` | off | Delete downloads and extracted originals after conversion |
| `--list` | | List datasets with sizes and exit |

It downloads to `data/raw/` (resumable: just re-run after an interruption),
extracts, and writes `data/processed/<name>/*.npy` (8-bit grayscale, BT.601) plus a
`manifest.json`. Finished steps are skipped, so re-running is instant. Details are
in [`msbgen/data/README.md`](../data/README.md).

## `train`: train a baseline

```shell
python -m msbgen.cli.train --preset unet-res                 # PLAN baseline 4
python -m msbgen.cli.train --preset cnn --steps 10000        # override any option
python -m msbgen.cli.train --resume runs/unet-res/last.pt --preset unet-res
```

**Presets** (the PLAN.md baselines):

| Preset | Model | Output | PLAN.md |
| --- | --- | --- | --- |
| `cnn` | `SimpleCNN` width 64, depth 8 (0.22 M params) | direct: predicts I | Baseline 2: pixelwise regression |
| `unet` | `UNet` width 24, 4 levels (4.4 M params) | direct: predicts I | Baseline 3: U-Net |
| `unet-res` | same U-Net | residual: predicts R = I − 128·B7 | Baseline 4: residual prediction |

**Options:**

| Option | Default | Meaning |
| --- | --- | --- |
| `--preset` | | One of the presets above |
| `--name` | the preset name | Run name; output goes to `<out-dir>/<name>/` |
| `--arch`, `--output` | from the preset | `cnn`/`unet`, `direct`/`residual` |
| `--width`, `--depth`, `--levels` | from the preset | Model size |
| `--k` | 1 | Transmitted planes (the model input is B7 … B(8−k)) |
| `--train`, `--val` | DIV2K train / valid | Folders of prepared `.npy` files (several allowed) |
| `--steps` | 20000 | Optimiser steps |
| `--batch-size`, `--crop` | 16, 128 | Batch of random crops |
| `--lr` | 2e-4 | Adam learning rate; 500-step warm-up, then cosine decay to 1e-6 |
| `--loss` | mse | `mse` (as in PLAN.md) or `l1` |
| `--val-every`, `--val-limit` | 1000, all | Validation interval / number of validation images |
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
- The gap between the two thresholds (hysteresis) stops it flapping on and off.
- Every training log line shows the current temperatures. `log.jsonl` records
  `gpu_temp`, `cpu_temp` and the total `paused_s`, and `it_per_s` excludes the pauses.
- Without sensors (no NVIDIA GPU, no psutil) the guard does nothing.

The code is in [`msbgen/utils`](../utils/): `gpu_guard.py`, `cpu_guard.py` and
`thermal.py`. If training pauses very often, lower the load with a smaller
`--batch-size` or fewer `--workers`, or raise the limits a little. The GPU's own
slowdown point is 100 °C.

**What a run writes** (`runs/<name>/`, git-ignored):

| File | Content |
| --- | --- |
| `config.json` | Full training and model configuration |
| `log.jsonl` | The full training history (see below) |
| `last.pt` | Latest checkpoint, written at every validation, before every thermal pause, and on **Ctrl-C** |
| `best.pt` | Checkpoint with the best validation PSNR; this is what `evaluate` and the web app use |

**Training history (`log.jsonl`)** has one JSON object per line, of two kinds:

| Record | When | Fields |
| --- | --- | --- |
| training | every `--log-every` steps | `step`, `epoch`, `loss` (mean over those steps), `train_psnr` (PSNR of that mean MSE), `lr`, `it_per_s`, `gpu_temp`, `cpu_temp`, `paused_s` |
| validation | step 0, then every `--val-every` steps | `step`, `epoch`, `val_loss`, `val_psnr`, `val_psnr_mid`, `val_gain_db`, `val_ssim`, `val_ssim_mid`, `val_msb_consistency`, `best_val_psnr` |

- Training counts steps, not epochs. `epoch` = pixels seen / pixels in the training
  set, so it is fractional: 20k steps × 16 crops of 128² ≈ 2.4 epochs of DIV2K.
- The step-0 validation scores the untrained model, so every curve starts from the
  beginning.
- On `--resume`, records after the checkpoint's step are dropped before training
  continues, so a crash never leaves duplicate steps.
- Plot the history with [`plot_runs`](#plot_runs-training-curves).

**Validation** uses one 256×256 centre crop per DIV2K-valid image. It reports the
model's mean PSNR, the midpoint fill's PSNR on the same crops, and the **gain in
dB**, which is the number that matters. If the gain isn't clearly positive, the
model has learned nothing beyond the constant fill.

On the MX350 laptop GPU, 20k steps take about 1–1.5 h for the U-Net and about
1.7 h for the CNN. See [`msbgen/models/README.md`](../models/README.md) for
measured speeds.

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
| `<run>_overview.png` | Per run, 4 panels: train vs validation loss, train vs validation PSNR (with the midpoint-fill line), validation SSIM, learning rate |
| `loss.png` | All runs: training loss (faint, smoothed) and validation loss (markers), log scale |
| `val_psnr.png`, `val_ssim.png` | All runs, with the midpoint-fill baseline as a dashed line |
| `val_gain.png` | All runs: PSNR gain over midpoint fill, the main "did it learn anything" curve |
| `lr.png` | Learning-rate schedules |

How to read them: validation loss rising while training loss keeps falling means
overfitting. A gain curve that flattens means more steps won't help, and a bigger
model or more context might.

## `evaluate`: score methods on a test set

```shell
python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips
```

Every method gets **only the top-k bits** of each image (through the
`Reconstructor` interface) and is scored against the original. The zero and
midpoint fills are always included as baselines.

| Option | Default | Meaning |
| --- | --- | --- |
| `data` | required | Folder of prepared `.npy` files or ordinary images (converted to grayscale) |
| `--models` | none | Checkpoints; each is named after its run folder |
| `--k` | 1 | Transmitted planes; models trained for another k are skipped |
| `--fills` | `zero mid` | Constant baselines to include |
| `--lpips` | off | Also compute LPIPS (downloads AlexNet weights once) |
| `--no-ssim` | off | Skip SSIM |
| `--limit N` | all | First N images only |
| `--save-images N` | 4 | Save reconstructions and a comparison strip for the first N images |
| `--out DIR` | `outputs/eval/<data folder name>` | Output folder |
| `--device` | auto | `cuda` / `cpu` |

**Outputs:**

| File | Content |
| --- | --- |
| `per_image.csv` | `image, method, psnr, ssim, lpips, mae, msb_consistency` |
| `summary.json` | Per method: mean and std of each metric over images, plus checkpoint path, step and model config |
| `images/<image>_<method>.png` | Rounded 8-bit reconstructions |
| `images/<image>_compare.png` | Original and every method side by side, labelled with PSNR |

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
