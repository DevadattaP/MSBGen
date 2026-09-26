# MSBGen

## Generative Image Reconstruction from Most-Significant Bitplanes

MSBGen studies how much of an 8-bit image can be recovered from only its
most-significant bit-plane(s). The long-term goal is an ultra-low-bitrate
perceptual image codec: transmit just the MSB plane $B_7$ (≈1 bit/pixel before
entropy coding), then reconstruct the full image with a conditional generative
model (diffusion, flow matching, normalizing flows).

$$
I = \sum_{k=0}^{7} 2^k B_k = 128\,B_7 + R, \qquad R \in [0, 127]
$$

$$
\hat I = 128\,B_7 + \hat R, \qquad \hat R \sim p_\theta(R \mid B_7)
$$

A single MSB plane matches many possible images, so recovering the image is a
one-to-many problem. That is why a **generative** model makes sense here rather
than a single regression. See [PLAN.md](PLAN.md) for the full research plan,
research questions and milestones.

## Current status

| Milestone | Status |
| --- | --- |
| Bit-plane explorer web app (slice & visualise planes) | ✅ done |
| Bit-plane analysis (entropy, correlations, k-MSB reconstruction) | ✅ done |
| Deterministic baselines (CNN / U-Net, residual prediction) | 🚧 code done, training runs in progress |
| Conditional diffusion / flow matching / normalizing flow | ⏳ |
| Multi-MSB experiments, entropy coding, final codec | ⏳ |

### Bit-plane explorer

A FastAPI web app where you upload an image and see every bit-plane:

Every image is sliced as either **grayscale** or **RGB**, and keeps its **bit
depth**. Each channel gives one plane per bit:

| Input | Sliced as | Planes |
| --- | --- | ---: |
| Grayscale, 8-bit (gray + alpha: alpha dropped) | Grayscale (L) | 8 |
| RGB, RGBA, palette, CMYK, YCbCr, LAB, HSV, 8-bit | RGB | 24 |
| Grayscale, 16-bit (gray + alpha: alpha dropped) | Grayscale (L) | 16 (B15 … B0) |
| RGB / RGBA, 16-bit | RGB | 48 |
| Bilevel (1-bit) | Grayscale (L) | 1 |

- For each plane: its place value $2^k$, the fraction of pixels set to 1, and its
  binary entropy (an i.i.d. upper bound in bits/pixel)
- For RGB images, an extra **Combined RGB** row stacks bit $k$ of R, G and B
  into one colour image for each $k$ (each channel 255 where its bit is set, else
  0). The result has 8 possible colours: yellow means R and G are set, white
  means all three, black means none. The MSB tile is exactly the 1-bit-per-channel
  image $\lfloor I/128 \rfloor \cdot 255$, i.e. what $B_7$ alone "sees" in colour.
- Click a plane to enlarge it or download it as a 1-bit PNG
- Accepts drag-and-drop or pasting from the clipboard. Works in light and dark
  themes and on phone-sized screens.

Input handling:

- **Alpha** (transparency) is dropped. The colour values stay exactly as stored;
  they are not blended onto a background.
- **Palette** (indexed) images become RGB. They have one channel, but its values
  are indices into a colour table, not gray levels.
- **CMYK, YCbCr, LAB and HSV** are converted to RGB with Pillow.
- EXIF rotation is applied. The UI reports every conversion when it happens.
- Pillow can't hold 16-bit colour, so 16-bit RGB/RGBA PNG and TIFF files are
  decoded with OpenCV. The true bit depth is read from the file header first, so
  nothing is silently truncated to 8 bits.
- The preview image is an 8-bit rendering used **for display only**. The planes
  always come from the full-bit-depth data.

### Bit-plane analysis

How much information does each plane carry, and how good is an image made from
only its top planes? The same analysis runs in the web app (section 3, computed
after slicing) and over whole datasets from the command line.

**Per plane**

| Measure | Meaning |
| --- | --- |
| $H$ (i.i.d.) | Binary entropy of the plane, ignoring all structure. Close to 1 bit/pixel for almost every plane. |
| $H \mid \text{ctx}$ | Empirical entropy of $B_k$ given its 4 causal neighbours (W, NW, N, NE) in the same plane: roughly what a JBIG-style context-model coder would pay for that plane on its own. |
| $\rho_\leftrightarrow$ | Correlation between a pixel and its right and lower neighbours. Near 1 for structured planes, near 0 for noise-like ones. |
| Plane correlation | Pearson (phi) correlation between every pair of planes, including across R, G and B. |

**Top-k reconstruction.** Keep the $k$ most-significant planes and fill in the
missing $n-k$ bits in one of two ways:

$$
\hat I_\text{zero} = \sum_{j=n-k}^{n-1} 2^j B_j, \qquad
\hat I_\text{mid} = \hat I_\text{zero} + \frac{2^{n-k}-1}{2}
$$

For $n = 8$, $k = 1$ the midpoint fill is the constant baseline $128\,B_7 + 63.5$
from the plan, and it is MSE-optimal when the missing bits are uniform. For each
$k$ the analysis reports PSNR, SSIM and MAE for both fills, the raw cost
($k \times$ channels bpp) and an estimated coded cost (sum of the kept planes'
context entropies). Metrics are computed on the full-bit-depth, unrounded
reconstruction. SSIM uses the standard Gaussian window (σ = 1.5), implemented with
OpenCV for speed and tested against scikit-image.

First observations on a handful of scikit-image sample photos (not yet a standard
dataset):

- $B_7$ costs about 1 bit/pixel i.i.d. but only **≈ 0.1–0.15 bit/pixel** with a
  4-neighbour context, so it is highly compressible. The lower planes approach
  1 bit/pixel even with context: they look like noise.
- The midpoint fill $128\,B_7 + 63.5$ gives **≈ 16–17 dB PSNR** (15.8 dB mean on the grayscale samples), and each extra
  plane adds ≈ 6 dB.
- Midpoint fill always wins on PSNR/MAE, but zero fill often has *higher* SSIM
  for $k \ge 2$. This is one reason not to rely on a single metric.
- For RGB, the MSB planes of R, G and B are clearly correlated with each other
  (ρ ≈ 0.3–0.7). This drops quickly for lower bits and is ≈ 0 by $B_3$.

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

Images are decoded exactly as in the web app and grouped by colour mode and bit
depth (`L-8`, `RGB-8`, `L-16`, …), since statistics of different shapes can't be
averaged together. Files that can't be decoded are skipped and listed. Outputs:

- `planes.csv`: one row per image × channel × plane
- `reconstruction.csv`: one row per image × k
- `summary.json`: per group, means over images (PSNR is the mean of per-image
  PSNR; lossless $k = n$ is `null`) and the mean correlation matrix
- `<group>_{entropy,psnr,ssim,correlation}.png` with `--plots`

A summary table is also printed to the terminal. See
[msbgen/cli/README.md](msbgen/cli/README.md) for the column definitions and how
the averages are computed.

### Deterministic baselines

This milestone asks how much of the image a plain regression network can recover
from $B_7$ alone, before trying anything generative. It covers PLAN.md baselines
1–4 on **8-bit grayscale** images:

| # | Method | What it predicts | How to get it |
| --- | --- | --- | --- |
| 1 | Midpoint fill $128\,B_7 + 63.5$ | constant (no model) | always included |
| 2 | Small CNN (8 conv layers, 17 px receptive field) | $I$ | `--preset cnn` |
| 3 | U-Net (4 levels, 4.4 M params) | $I$ | `--preset unet` |
| 4 | U-Net, residual | $R = I - 128\,B_7 \in [0, 127]$, then $\hat I = 128\,B_7 + \hat R$ | `--preset unet-res` |

All networks are trained with MSE on random 128×128 crops of **DIV2K** (800
train images), validated on DIV2K-valid, and tested on **Kodak** (24 images).
Every method, the fills included, gets **only the transmitted plane** through the
same `Reconstructor` interface. Each is scored with PSNR, SSIM, LPIPS, MAE and
**MSB consistency**, the fraction of pixels where Î doesn't contradict the
transmitted $B_7$. Residual models are consistent by construction.

```shell
# 1. Data (once; ~4 GB download, resumable, reproducible via a manifest)
python -m msbgen.cli.prepare_data div2k kodak

# 2. Train (each run → runs/<name>/best.pt; Ctrl-C saves, --resume continues)
python -m msbgen.cli.train --preset cnn
python -m msbgen.cli.train --preset unet
python -m msbgen.cli.train --preset unet-res

# 3. Training curves (loss, PSNR, SSIM, LR) → outputs/plots/
python -m msbgen.cli.plot_runs

# 4. Evaluate on Kodak (tables, per-image CSV, comparison images)
python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips
```

Training logs its full history to `runs/<name>/log.jsonl`: training loss and PSNR,
plus validation loss, PSNR, SSIM and gain over midpoint fill, starting at step 0.
It also runs a **thermal guard**. When the GPU reaches 80 °C or the CPU 90 °C, it
saves a checkpoint and pauses until both have cooled, which keeps a laptop from
overheating and shutting down mid-run. See
[msbgen/cli/README.md](msbgen/cli/README.md#train-train-a-baseline) for all options.

Trained models show up automatically in the web app, in section **4 · Model
reconstruction**, which sits next to the midpoint baseline. It has a toggle
between reconstructions and error maps ($|I - \hat I| \times 4$), and the
PSNR/SSIM/MAE/consistency for the uploaded image. Colour uploads are converted to
grayscale for the models.

Each training run takes roughly 1–1.5 h on the development laptop (GeForce MX350,
2 GB) with the default sizes. Measured speeds and memory are in
[msbgen/models/README.md](msbgen/models/README.md).

**Results:** to be filled in from the Kodak evaluation once the training runs
finish.

## Project structure

```shell
MSBGen/
├── msbgen/
│   ├── main.py            # FastAPI app factory; serves UI + API
│   ├── config.py          # Settings (env vars: MSBGEN_*)
│   ├── core/              # Framework-independent logic (see core/README.md)
│   │   ├── bitplanes.py   # slice_bitplanes, reconstruct_from_planes, binary_entropy
│   │   ├── analysis.py    # context entropy, correlations, top-k reconstruction, analyze_image
│   │   ├── reconstruction.py  # Reconstructor interface, constant fills, evaluate()
│   │   ├── metrics.py     # PSNR, SSIM, MAE
│   │   └── image_io.py    # decode uploads → grayscale/RGB uint8/uint16 arrays, PNG encoding
│   ├── data/              # Dataset download/preparation + training crops (see data/README.md)
│   ├── models/            # CNN / U-Net, checkpoints, ModelReconstructor, LPIPS (see models/README.md)
│   ├── training.py        # training loop (validation vs midpoint fill, checkpoints, resume)
│   ├── utils/             # gpu_guard / cpu_guard / thermal: pause training when too hot
│   ├── cli/               # Command-line tools (see cli/README.md)
│   │   ├── analyze.py     # dataset-level analysis → CSV / JSON / plots
│   │   ├── prepare_data.py  # download + convert DIV2K / Kodak
│   │   ├── train.py       # train a baseline
│   │   ├── plot_runs.py   # training curves + history CSV
│   │   └── evaluate.py    # score fills + checkpoints on a test set
│   ├── api/
│   │   ├── routes.py      # /api/health, bitplanes, analysis, models, reconstruct
│   │   ├── model_registry.py  # discovers runs/*/best.pt for the web app
│   │   └── schemas.py     # Pydantic response models
│   └── web/static/        # Front-end (plain HTML / CSS / JS, no build step)
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
- [msbgen/models/README.md](msbgen/models/README.md): networks, checkpoints, LPIPS
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

| Variable | Default | Meaning |
| --- | --- | --- |
| `MSBGEN_MAX_UPLOAD_MB` | `25` | Maximum upload size |
| `MSBGEN_MAX_PIXELS` | `16777216` (4096²) | Maximum image area |
| `MSBGEN_MODELS_DIR` | `runs` | Where the web app looks for `<run>/best.pt` checkpoints |
| `MSBGEN_DEVICE` | auto | Device for model inference (`cuda`, `cpu`, …) |

### Tests

```bash
python -m pytest
```

## API

### `POST /api/bitplanes`

Multipart form upload with a single field `file`.

```bash
curl -F "file=@photo.jpg" http://127.0.0.1:8000/api/bitplanes
```

Response (abridged):

```json
{
  "filename": "photo.jpg",
  "width": 800, "height": 600,
  "original_mode": "RGBA", "original_format": "PNG",
  "color_mode": "RGB", "bit_depth": 8,
  "channels": [
    { "name": "R", "label": "Red" }, { "name": "G", "label": "Green" },
    { "name": "B", "label": "Blue" }
  ],
  "notes": ["Dropped alpha channel."],
  "preview": "data:image/png;base64,...",
  "planes": [
    { "channel": "R", "channel_index": 0, "bit": 7, "weight": 128,
      "ones_ratio": 0.498, "entropy": 0.99999,
      "image": "data:image/png;base64,..." }
  ]
}
```

`planes` are ordered by channel, then from MSB ($B_{n-1}$, where n = `bit_depth`)
down to LSB ($B_0$). Each plane image is a 1-bit PNG (white = 1, black = 0).

For RGB images the response also has `combined`, a list of
`{ "bit", "weight", "image" }` from MSB to LSB. Each `image` is an 8-colour PNG
of bit $k$ across R, G and B. For grayscale images the list is empty.

Errors: `400` for an empty or undecodable file or one over the pixel limit, and
`413` for a file over the size limit.

### `POST /api/analysis`

Same upload as `/api/bitplanes`. Returns the same image fields (`filename`,
`width`, `color_mode`, `bit_depth`, `channels`, `notes`, …) plus:

```json
{
  "planes": [
    { "channel": "R", "channel_index": 0, "bit": 7, "ones_ratio": 0.635,
      "entropy": 0.947, "context_entropy": 0.098, "spatial_correlation": 0.935 }
  ],
  "correlation": { "labels": ["R·B7", "R·B6", "..."], "matrix": [[1.0, 0.26, "..."]] },
  "reconstructions": [
    { "k": 1, "bits": [7], "bits_per_pixel": 3, "coded_bpp": 0.288,
      "psnr": { "zero": 11.85, "mid": 16.27 }, "ssim": { "zero": 0.468, "mid": 0.505 },
      "mae": { "zero": 53.15, "mid": 33.14 }, "image": "data:image/png;base64,..." }
  ]
}
```

`reconstructions` runs from k = 1 to `bit_depth`, and each `image` is the
midpoint-fill reconstruction as an 8-bit PNG. JSON has no infinity, so the
lossless PSNR (k = `bit_depth`) is `null`. SSIM is `null` for images smaller than
11×11, and correlations involving a constant plane are `null`.

### `GET /api/models`

The trained checkpoints found in `MSBGEN_MODELS_DIR`:
`{"torch_available", "models_dir", "models": [{"name", "label", "arch", "output", "k", "bit_depth", "channels", "parameters", "step", "val_psnr", "val_gain_db"}]}`.
The list is re-scanned on every call, so new runs appear without a restart.

### `POST /api/reconstruct`

Multipart `file`, plus optional form fields `k` (default 1) and `models`
(comma-separated names; default: every compatible model). The image must be
8-bit; colour images are converted to grayscale, with a note. Returns the image
fields plus:

```json
{
  "k": 1,
  "original": "data:image/png;base64,...",
  "results": [
    { "name": "mid", "label": "Midpoint fill", "kind": "fill",
      "psnr": 17.29, "ssim": 0.635, "mae": 30.66, "msb_consistency": 1.0,
      "psnr_gain": 0.0, "image": "data:...", "error_map": "data:..." },
    { "name": "unet-res", "label": "U-Net (residual)", "kind": "model", "...": "..." }
  ]
}
```

Fills come first, then models. `psnr_gain` is relative to midpoint fill, and
`error_map` is $|I - \hat I| \times 4$. Errors: `400` for non-8-bit input or an
invalid `k`, and `404` for an unknown model name.

### `GET /api/health`

Returns `{"status": "ok", "version": "..."}`.

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

est = msb_reconstruction(img, k=2, fill="mid")   # float64, unrounded
psnr(img, est, data_range=255), ssim(img, est, data_range=255)
```

The bit depth comes from the dtype (`bool` → 1, `uint8` → 8, `uint16` → 16), and
any number of channels is accepted. For example, a 16-bit RGB array gives
`(3, 16, H, W)`. Use `msbgen.core.load_image(bytes)` to decode a file into
grayscale or RGB the same way the web app does.
