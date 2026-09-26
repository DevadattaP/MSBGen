# `msbgen.cli`

Command-line tools, run as modules from the project root:

```shell
python -m msbgen.cli.<tool> --help
```

| Tool | Purpose |
| --- | --- |
| [`analyze`](analyze.py) | Dataset-level bit-plane analysis → CSV, JSON and plots |

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

## Adding a tool

Create `msbgen/cli/<name>.py` with a `main(argv: list[str] | None = None) -> int`
that parses its own arguments, and end it with
`if __name__ == "__main__": raise SystemExit(main())`. Put the logic in
`msbgen.core` and keep the tool itself to I/O and argument handling. Add a test
that calls `main([...])` with a `tmp_path`, like
[`tests/test_cli.py`](../../tests/test_cli.py), and a row in the table above.
