# MSBGen

## Generative Image Reconstruction from Most-Significant Bitplanes

MSBGen studies how much of an 8-bit image can be recovered from only its most-significant bit-plane(s). The long-term goal is an ultra-low-bitrate perceptual image codec: transmit just the MSB plane $B_7$ (≈1 bit/pixel before entropy coding), then reconstruct the full image with a conditional generative model (diffusion, flow matching, normalizing flows).

$$
I = \sum_{k=0}^{7} 2^k B_k = 128\,B_7 + R, \qquad R \in [0, 127]
$$

$$
\hat I = 128\,B_7 + \hat R, \qquad \hat R \sim p_\theta(R \mid B_7)
$$

A single MSB plane matches many possible images, so recovering the image is a one-to-many problem. That is why a **generative** model makes sense here rather than a single regression. See [PLAN.md](PLAN.md) for the full research plan, research questions and milestones.

### RGB model results

The RGB experiments train on RGB DIV2K and evaluate on all 24 RGB Kodak images. Here `k` counts retained MSB bitplanes **per channel**, so the RGB input carries `3k` binary planes per pixel before entropy coding. The sequential bitplane CNN leads PSNR and SSIM at k=2 and k=3; the best perceptual metrics vary by k and metric, and no single model wins across all measures. These are single-run results, not a multi-seed benchmark.

| k | Highest PSNR | Highest SSIM | Lowest LPIPS | Lowest DISTS | Lowest KID × 10³ |
| ---: | --- | --- | --- | --- | --- |
| 1 | ViT hybrid stage 1 (18.82 dB) | Bit-depth recovery (0.6660) | Hybrid (0.4223) | Hybrid (0.2399) | ViT hybrid stages 1+2 (45.06) |
| 2 | Bit-depth recovery (27.00 dB) | Bit-depth recovery (0.8468) | Hybrid (0.1898) | Hybrid stages 1+2 (0.1189) | Hybrid (18.13) |
| 3 | Bit-depth recovery (33.89 dB) | Bit-depth recovery (0.9358) | Hybrid (0.0772) | Hybrid (0.0566) | ViT hybrid (5.33) |

The [RGB Kodak results in REPORT.md](REPORT.md#rgb-kodak-results) contain the complete comparisons for every method, including MAE, MSB consistency, sample-mean PSNR, diversity and inference time.

### Bit-plane explorer

A FastAPI web app where you upload an image and see every bit-plane:

Every image is sliced as either **grayscale** or **RGB**, and keeps its **bit depth**. Each channel gives one plane per bit:

| Input | Sliced as | Planes |
| --- | --- | ---: |
| Grayscale, 8-bit (gray + alpha: alpha dropped) | Grayscale (L) | 8 |
| RGB, RGBA, palette, CMYK, YCbCr, LAB, HSV, 8-bit | RGB | 24 |
| Grayscale, 16-bit (gray + alpha: alpha dropped) | Grayscale (L) | 16 (B15 … B0) |
| RGB / RGBA, 16-bit | RGB | 48 |
| Bilevel (1-bit) | Grayscale (L) | 1 |

- For each plane: its place value $2^k$, the fraction of pixels set to 1, and its binary entropy (an i.i.d. upper bound in bits/pixel)
- For RGB images, an extra **Combined RGB** row stacks bit $k$ of R, G and B into one colour image for each $k$ (each channel 255 where its bit is set, else 0). The result has 8 possible colours: yellow means R and G are set, white means all three, black means none. The MSB tile is exactly the 1-bit-per-channel image $\lfloor I/128 \rfloor \cdot 255$, i.e. what $B_7$ alone "sees" in colour.
- Click a plane to enlarge it or download it as a 1-bit PNG
- Accepts drag-and-drop or pasting from the clipboard. Works in light and dark themes and on phone-sized screens.

Input handling:

- **Alpha** (transparency) is dropped. The colour values stay exactly as stored; they are not blended onto a background.
- **Palette** (indexed) images become RGB. They have one channel, but its values are indices into a colour table, not gray levels.
- **CMYK, YCbCr, LAB and HSV** are converted to RGB with Pillow.
- EXIF rotation is applied. The UI reports every conversion when it happens.
- Pillow can't hold 16-bit colour, so 16-bit RGB/RGBA PNG and TIFF files are decoded with OpenCV. The true bit depth is read from the file header first, so nothing is silently truncated to 8 bits.
- The preview image is an 8-bit rendering used **for display only**. The planes always come from the full-bit-depth data.

### Bit-plane analysis

How much information does each plane carry, and how good is an image made from only its top planes? The same analysis runs in the web app (section 3, computed after slicing) and over whole datasets from the command line.

**Per plane**

| Measure | Meaning |
| --- | --- |
| $H$ (i.i.d.) | Binary entropy of the plane, ignoring all structure. Close to 1 bit/pixel for almost every plane. |
| $H \mid \text{ctx}$ | Empirical entropy of $B_k$ given its 4 causal neighbours (W, NW, N, NE) in the same plane: roughly what a JBIG-style context-model coder would pay for that plane on its own. |
| $\rho_\leftrightarrow$ | Correlation between a pixel and its right and lower neighbours. Near 1 for structured planes, near 0 for noise-like ones. |
| Plane correlation | Pearson (phi) correlation between every pair of planes, including across R, G and B. |

**Top-k reconstruction.** Keep the $k$ most-significant planes and fill in the missing $n-k$ bits in one of three ways:

$$
\hat I_\text{zero} = \sum_{j=n-k}^{n-1} 2^j B_j, \qquad
\hat I_\text{mid} = \hat I_\text{zero} + \frac{2^{n-k}-1}{2}, \qquad
\hat I_\text{random} = \hat I_\text{zero} + U\{0, \dots, 2^{n-k}-1\}
$$

For $n = 8$, $k = 1$ the midpoint fill is the constant baseline $128\,B_7 + 63.5$ from the plan, and it is MSE-optimal when the missing bits are uniform. The random fill draws the missing bits uniformly (fixed seed, so results are reproducible): it is a *sample* from a uniform prior, the no-model counterpart of the generative models. For uniform pixels its MSE is exactly twice the midpoint fill's, i.e. 3.01 dB lower PSNR, the same gap expected between a perfect posterior sample and the best deterministic (MMSE) prediction. For each $k$ the analysis reports PSNR, SSIM and MAE for all three fills, the raw cost ($k \times$ channels bpp) and an estimated coded cost (sum of the kept planes' context entropies). Metrics are computed on the full-bit-depth, unrounded reconstruction. SSIM uses the standard Gaussian window (σ = 1.5), implemented with OpenCV for speed and tested against scikit-image.

First observations on a handful of scikit-image sample photos (not yet a standard dataset):

- $B_7$ costs about 1 bit/pixel i.i.d. but only **≈ 0.1–0.15 bit/pixel** with a 4-neighbour context, so it is highly compressible. The lower planes approach 1 bit/pixel even with context: they look like noise.
- The midpoint fill $128\,B_7 + 63.5$ gives **≈ 16–17 dB PSNR** (15.8 dB mean on the grayscale samples), and each extra plane adds ≈ 6 dB.
- Midpoint fill always wins on PSNR/MAE, but zero fill often has *higher* SSIM for $k \ge 2$. This is one reason not to rely on a single metric.
- Random fill sits ≈ 3 dB below midpoint on PSNR (13.8 vs 17.0 dB on Kodak at $k = 1$) and has by far the lowest SSIM (0.13): independent per-pixel noise destroys local structure. A generative model must beat it on both to show it has learned something beyond "any value in the range".
- For RGB, the MSB planes of R, G and B are clearly correlated with each other (ρ ≈ 0.3–0.7). This drops quickly for lower bits and is ≈ 0 by $B_3$.

**Dataset CLI**

```shell
# Analyse every image in a folder; --gray converts RGB to 8-bit luma first
python -m msbgen.cli.analyze data/kodak --gray --plots
```

| Option | Meaning |
| --- | --- |
| `--out DIR` | Output folder (default `outputs/analysis/<folder name>`) |
| `--gray` | Convert RGB to grayscale (BT.601 luma, as Pillow's `L`) |
| `-r`, `--recursive` | Include sub-folders |
| `--limit N` | Analyse at most N images |
| `--no-ssim` | Skip SSIM (faster) |
| `--plots` | Save entropy, PSNR/SSIM-vs-k and correlation plots |
| `--workers N` | Parallel processes (default: half the CPU cores) |

Images are decoded exactly as in the web app and grouped by colour mode and bit depth (`L-8`, `RGB-8`, `L-16`, …), since statistics of different shapes can't be averaged together. Files that can't be decoded are skipped and listed. Outputs:

- `planes.csv`: one row per image × channel × plane
- `reconstruction.csv`: one row per image × k
- `summary.json`: per group, means over images (PSNR is the mean of per-image PSNR; lossless $k = n$ is `null`) and the mean correlation matrix
- `<group>_{entropy,psnr,ssim,correlation}.png` with `--plots`

A summary table is also printed to the terminal. See [msbgen/cli/README.md](msbgen/cli/README.md) for the column definitions and how
the averages are computed.

## Project structure

```shell
MSBGen/
├── msbgen/
│   ├── main.py            # FastAPI app factory; serves UI + API
│   ├── config.py          # Settings (env vars: MSBGEN_*)
│   ├── core/              # Framework-independent logic (see core/README.md)
│   │   ├── bitplanes.py   # slice_bitplanes, reconstruct_from_planes, binary_entropy
│   │   ├── analysis.py    # context entropy, correlations, top-k reconstruction, analyze_image
│   │   ├── reconstruction.py  # Reconstructor interface, fills (zero/mid/random), evaluate()
│   │   ├── metrics.py     # PSNR, SSIM, MAE
│   │   └── image_io.py    # decode uploads → grayscale/RGB uint8/uint16 arrays, PNG encoding
│   ├── data/              # Dataset download/preparation + training crops (see data/README.md)
│   ├── models/            # CNN / U-Net / ViT, diffusion, flow matching, normalizing flow, hybrids, checkpoints and metrics
│   ├── training.py        # training loop: regression + generative models, validation, checkpoints, resume
│   ├── utils/             # gpu_guard / cpu_guard / thermal: pause training when too hot
│   ├── cli/               # Command-line tools (see cli/README.md)
│   │   ├── analyze.py     # dataset-level analysis → CSV / JSON / plots
│   │   ├── prepare_data.py  # download + convert DIV2K / Kodak
│   │   ├── train.py       # train a baseline or generative model
│   │   ├── plot_runs.py   # training curves + history CSV
│   │   ├── plot_eval.py   # evaluation plots: metrics, perception vs distortion, paired
│   │   ├── plot_steps.py  # generative models against their number of sampling steps
│   │   └── evaluate.py    # score fills + checkpoints on a test set
│   ├── api/
│   │   ├── routes.py      # /api/health, bitplanes, analysis, models, reconstruct, evaluations
│   │   ├── model_registry.py  # which models run, thermal guard, GPU memory release
│   │   └── schemas.py     # Pydantic response models
│   └── web/static/        # Front-end (plain HTML / CSS / JS, no build step) + README.md: config and API
│       ├── index.html
│       ├── css/styles.css
│       └── js/app.js
├── tests/                 # pytest suite (core, models, API, CLI)
├── data/                  # datasets (git-ignored; created by prepare_data)
├── runs/                  # training runs and checkpoints (git-ignored)
├── outputs/               # analysis / evaluation outputs (git-ignored)
├── PLAN.md                # Research plan
├── requirements.txt
└── README.md
```

`msbgen.core` has no web or torch dependencies, so the same bit-plane, metric and
reconstruction code is shared by the web app, the CLIs and training. Module-level
docs:

- [msbgen/core/README.md](msbgen/core/README.md): bit-planes, analysis,
  reconstructors, metrics
- [msbgen/models/README.md](msbgen/models/README.md): networks, the diffusion and flow-matching models, checkpoints, LPIPS / DISTS / KID
- [msbgen/data/README.md](msbgen/data/README.md): datasets and preparation
- [msbgen/cli/README.md](msbgen/cli/README.md): every command, its options and
  output formats

## Getting started

Requires Python 3.10+.

```bash
# 1. Activate your environment (or create one: python -m venv ~/venv)
source ~/venv/bin/activate

# 2. Install dependencies (for a CUDA build of PyTorch, install torch first
#    following https://pytorch.org/get-started/)
pip install -r requirements.txt

# 3. Run the app (from the project root)
uvicorn msbgen.main:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API docs are at
<http://127.0.0.1:8000/docs>.

To make it reachable from other machines on your network:
`uvicorn msbgen.main:app --host 0.0.0.0 --port 8000`.

### Configuration

Environment variables (`MSBGEN_*`), model selection, devices and the API are documented in
[msbgen/web/static/README.md](msbgen/web/static/README.md).

### Tests

```bash
python -m pytest
```

## Using the core library

```python
import numpy as np
from PIL import Image
from msbgen.core import slice_bitplanes, reconstruct_from_planes

img = np.asarray(Image.open("photo.png").convert("L"))   # (H, W) uint8
planes = slice_bitplanes(img)                            # (1, 8, H, W), values {0,1}

msb_only = reconstruct_from_planes(planes, bits=[7])     # 128·B7
top3     = reconstruct_from_planes(planes, bits=[7, 6, 5])
assert (reconstruct_from_planes(planes)[..., 0] == img).all()  # lossless
```

Analysis and metrics:

```python
from msbgen.core import analyze_image, msb_reconstruction, psnr, ssim

a = analyze_image(img)                      # img: (H, W) or (H, W, C), uint8/uint16
a.planes[0].context_entropy                 # MSB cost with a 4-neighbour context
a.correlation                               # (C·n, C·n) plane correlation matrix
a.reconstructions[0].psnr["mid"]            # PSNR of 128·B7 + 63.5
a.reconstructions[0].psnr["random"]         # PSNR of 128·B7 + uniform random low bits

est = msb_reconstruction(img, k=2, fill="mid")   # float64, unrounded
psnr(img, est, data_range=255), ssim(img, est, data_range=255)
```

The bit depth comes from the dtype (`bool` → 1, `uint8` → 8, `uint16` → 16), and
any number of channels is accepted. For example, a 16-bit RGB array gives
`(3, 16, H, W)`. Use `msbgen.core.load_image(bytes)` to decode a file into
grayscale or RGB the same way the web app does.
