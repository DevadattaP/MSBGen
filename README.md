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
missing $n-k$ bits in one of three ways:

$$
\hat I_\text{zero} = \sum_{j=n-k}^{n-1} 2^j B_j, \qquad
\hat I_\text{mid} = \hat I_\text{zero} + \frac{2^{n-k}-1}{2}, \qquad
\hat I_\text{random} = \hat I_\text{zero} + U\{0, \dots, 2^{n-k}-1\}
$$

For $n = 8$, $k = 1$ the midpoint fill is the constant baseline $128\,B_7 + 63.5$
from the plan, and it is MSE-optimal when the missing bits are uniform. The random
fill draws the missing bits uniformly (fixed seed, so results are reproducible): it
is a *sample* from a uniform prior, the no-model counterpart of the generative
models. For uniform pixels its MSE is exactly twice the midpoint fill's, i.e.
3.01 dB lower PSNR, the same gap expected between a perfect posterior sample and
the best deterministic (MMSE) prediction. For each
$k$ the analysis reports PSNR, SSIM and MAE for all three fills, the raw cost
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
- Random fill sits ≈ 3 dB below midpoint on PSNR (13.8 vs 17.0 dB on Kodak at
  $k = 1$) and has by far the lowest SSIM (0.13): independent per-pixel noise
  destroys local structure. A generative model must beat it on both to show it
  has learned something beyond "any value in the range".
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
| 1 | Midpoint fill $128\,B_7 + 63.5$ (plus zero and random fill for reference) | no model | always included |
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

# 4. Evaluate on Kodak (tables, per-image CSV, comparison images), then plot the results
python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips --dists --kid
python -m msbgen.cli.plot_eval
```

Training logs its full history to `runs/<name>/log.jsonl`: training loss and PSNR,
plus validation loss, PSNR, SSIM, the midpoint- and random-fill baselines on the
same crops, and the gain over midpoint fill, starting at step 0.
It also runs a **thermal guard**. When the GPU reaches 80 °C or the CPU 90 °C, it
saves a checkpoint and pauses until both have cooled, which keeps a laptop from
overheating and shutting down mid-run. See
[msbgen/cli/README.md](msbgen/cli/README.md#train-train-a-baseline) for all options.

Trained models show up automatically in the web app, in section **4 · Model
reconstruction**, which sits next to the zero, midpoint and random fills. It has a toggle
between reconstructions and error maps ($|I - \hat I| \times 4$), and the
PSNR/SSIM/LPIPS/DISTS/MAE/consistency for the uploaded image. Colour uploads are converted to
grayscale for the models. Saved `evaluate` results (including the set-level KID,
PSNR of the sample mean and diversity) appear in the **Test-set evaluation**
section, with a switch between evaluation folders.

Each training run takes roughly 1–1.5 h on the development laptop (GeForce MX350,
2 GB) with the default sizes. Measured speeds and memory are in
[msbgen/models/README.md](msbgen/models/README.md).

#### Results (Kodak)

All 24 Kodak images, 8-bit grayscale, full resolution, $k = 1$ (only $B_7$
transmitted). The table shows means over images; PSNR also shows ± the standard
deviation over images. Each model is its `best.pt` (highest validation PSNR).

| Method | PSNR (dB) ↑ | Δ vs midpoint | SSIM ↑ | LPIPS ↓ | MAE ↓ | MSB consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero fill | 10.37 ± 0.92 | −6.64 | 0.238 | 0.583 | 69.1 | 1.000 |
| Midpoint fill | 17.01 ± 1.53 | 0.00 | 0.579 | 0.517 | 31.8 | 1.000 |
| Random fill | 13.82 ± 0.68 | −3.19 | 0.129 | 0.924 | 42.7 | 1.000 |
| CNN (direct) | 18.04 ± 1.54 | +1.03 | 0.617 | **0.449** | 27.3 | 0.9995 |
| U-Net (direct) | 18.55 ± 1.29 | +1.55 | 0.637 | 0.456 | 25.5 | 0.9986 |
| U-Net (residual) | **18.60 ± 1.23** | **+1.59** | **0.637** | 0.455 | **25.3** | **1.000** |

Training (DIV2K, 20k steps of 16 random 128×128 crops ≈ 2.35 epochs; validation on
256×256 centre crops of DIV2K-valid):

| Run | Params | Best step | Val PSNR (dB) | Val gain vs midpoint | Speed (MX350) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cnn` | 0.22 M | 17k | 19.05 | +1.67 dB | 3.2 it/s |
| `unet` | 4.4 M | 17k | 19.25 | +1.86 dB | 5.1 it/s |
| `unet-res` | 4.4 M | 17k | 19.30 | +1.92 dB | 4.9 it/s |

What the numbers show:

- **A network recovers about 1.6 dB beyond the best constant guess.** The residual
  U-Net beats the midpoint fill on PSNR for 21 of the 24 images, and every network
  beats every fill on PSNR, SSIM, LPIPS and MAE on average. The gain is real but
  small: $B_7$ alone carries limited information about the missing 7 bits.
- **Global context helps fidelity.** The U-Net gains 0.5 dB PSNR and 0.02 SSIM over
  the CNN, whose receptive field is only 17 px. The U-Net is also 20× larger, so
  this comparison doesn't separate context from capacity.
- **Residual vs direct is a tie on quality.** On PSNR the residual U-Net is ahead
  by only 0.05 dB and wins on 13 of 24 images. Its real advantage is that it can
  never contradict the transmitted bit (consistency 1.000 vs 0.9986), so it is
  the default from here on.
- **Perceptual quality barely improves.** LPIPS goes from 0.517 (midpoint) to only
  ≈ 0.45, and the small CNN has the *best* LPIPS, beating the residual U-Net on 16
  of 24 images. MSE training predicts the average of all images consistent with
  $B_7$, which is smooth and blurry. Extra PSNR from the bigger model doesn't make
  the output look more realistic. This is the gap the generative models (next
  milestone) are meant to close.
- **Noise is not texture.** Random fill scores 3.19 dB below the midpoint fill,
  close to the theoretical 3.01 dB, and it has the worst LPIPS of all (0.924).
  Filling the missing bits with plausible values *independently per pixel* looks
  worse than a flat fill. A generative model has to produce detail that is
  correlated across neighbouring pixels, and it should beat random fill clearly
  on SSIM and LPIPS, not only on PSNR.
- **Validation and test agree.** The gain over midpoint fill is +1.9 dB on
  DIV2K-valid crops and +1.6 dB on Kodak. Training and validation PSNR are about
  equal, so the models are not overfitting. The validation curves are flat after
  ≈ 12k steps.

**Failure case: brightness is ambiguous.** On `kodim02` (a dark door) all three
networks are ≈ 2.5 dB *worse* than the midpoint fill. 98 % of its pixels have
$B_7 = 0$ and they average 77.5, but the networks predict ≈ 55 there, because in
DIV2K large $B_7 = 0$ regions are usually deep shadow. The midpoint's 63.5 happens
to be closer. $B_7$ cannot tell a dim scene from a dark one, which is the
one-to-many problem in its simplest form. A few bits of side information (e.g. one
mean intensity per block) could remove this ambiguity cheaply; that is a candidate
for the codec stage.

The per-image scores and comparison strips are in `outputs/eval/kodak/`
(`per_image.csv`, `summary.json`, `images/`); reproduce them with step 4 above.

### Conditional diffusion

The baselines predict the *average* of all images consistent with $B_7$, which is
blurry. This milestone learns the *distribution* instead: a conditional diffusion
model of the residual,

$$
\hat I = 128\,B_7 + \hat R, \qquad \hat R \sim p_\theta(R \mid B_7)
$$

so every sample keeps the transmitted bit exactly (MSB consistency 1), and
different noise seeds give different plausible images.

| | |
| --- | --- |
| Network | Time-conditioned U-Net (ResBlocks, GroupNorm, no attention), width 32, 4 levels: 4.0 M params. Input: noisy residual + $B_7$ planes |
| Diffusion | Continuous-time cosine schedule, **v-prediction**, DDIM sampler (3 steps by default, chosen by the [step sweep](#flow-matching-results-kodak); `eta` 0 = deterministic given the noise, 1 = fresh noise every step) |
| Training | Same DIV2K crops, optimiser and LR schedule as the baselines; 40k steps, EMA 0.999; `best.pt` = lowest validation loss |
| Evaluation | `--samples N`: PSNR of the average of N samples (`psnr_avg`) and per-pixel spread across samples (`diversity`) |

```shell
# Train (≈ 5 h of compute on the MX350; with validations and thermal pauses plan for 9–10 h)
python -m msbgen.cli.train --preset diffusion

# If it stops (Ctrl-C, shutdown), resume with the same options
python -m msbgen.cli.train --preset diffusion --resume runs/diffusion/last.pt

# Evaluate against the baselines, 4 samples per image
python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips --dists --kid --samples 4
```

**What to expect.** A sample from the true posterior has *twice* the MSE of the
posterior mean, so even a perfect model scores about **3 dB below** the regression
U-Net on PSNR (≈ 15.6 dB on Kodak), below the midpoint fill. The random fill shows
the same effect with a uniform prior (−3.19 dB). The diffusion model has to win
elsewhere:

- lower LPIPS, DISTS and patch KID than the regression models (DISTS and KID
  don't require texture to be pixel-aligned, so they are the fairer test)
- SSIM and LPIPS far better than the random fill
- `psnr_avg` climbing back towards the U-Net's PSNR as the number of samples grows
- non-zero `diversity`

The derivation and details are in
[msbgen/models/README.md](msbgen/models/README.md#conditional-diffusion-diffusionpy).

**Thermal guard.** Sampling is minutes of sustained full GPU load, and the laptop
shut down once during an unguarded diffusion validation. The guard is now checked
after every denoising step in training validation, `evaluate` and the web app.

#### Diffusion results (Kodak)

Same 24 images and $k = 1$ as the baselines, with 8 samples per image for the
stochastic methods (`outputs/eval/kodak-s8`; diffusion `best.pt` at step 40k,
DDIM 50 steps, `eta` 0). Stochastic methods are scored on sample 0; **PSNR avg** is
the PSNR of the mean of the 8 samples. KID is patch KID × 10³ (128 px patches,
936 per method, every sample for stochastic ones).

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | KID ↓ | PSNR avg | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Midpoint fill | 17.01 | 0.579 | 0.517 | 0.364 | 81.6 | | |
| Random fill | 13.82 | 0.129 | 0.924 | 0.385 | 119.6 | 16.44 | 33.9 |
| CNN (direct) | 18.04 | 0.617 | **0.449** | 0.281 | 93.6 | | |
| U-Net (residual) | **18.60** | **0.637** | 0.455 | 0.293 | 79.6 | | |
| Diffusion (residual) | 16.33 | 0.480 | 0.461 | **0.269** | **77.7** | 17.85 | 16.6 |

What the numbers show:

- **The sampler is sound.** One sample is 2.3 dB below the residual U-Net (an ideal
  posterior sampler: 3 dB). The mean of 8 samples is only 0.75 dB below it, against
  0.51 dB expected for 8 ideal samples, so the model's posterior mean is close to
  the regression estimate. The samples are somewhat under-dispersed, as is usual
  for deterministic DDIM.
- **DISTS is where it wins.** Diffusion has the best DISTS of all methods and beats
  the residual U-Net on 20 of 24 images (`eval_kodak-s8_paired.png`), while staying
  far ahead of the random fill. LPIPS, which compares features at the same
  position, is a tie (11/24).
- **KID can't separate the methods at this size.** Diffusion is lowest, but the
  midpoint fill beats the CNN and the plain U-Net, and every method is far from
  real images. This summary predates the KID error-bar fix, so it has no spreads.
- **The win is small because $B_7$ says little.** With one plane the posterior is
  very wide; multi-plane runs ($k = 2$) are the test of whether the generative
  advantage grows with information.

Plots: `python -m msbgen.cli.plot_eval` (metrics per method, perception vs
distortion, paired per-image comparison). The web app shows the same table in its
**Test-set evaluation** section.

### Conditional flow matching

Milestone 4 learns the same distribution $p(R \mid B_7)$ with **flow matching**
instead of diffusion. Everything else is held fixed (residual, $B_7$ conditioning,
the 4.0 M-parameter `TimeUNet`, DIV2K crops, 40k steps, EMA 0.999), so the comparison
isolates the generative process:

| | Diffusion | Flow matching |
| --- | --- | --- |
| Path from data to noise | $x_t = \alpha_t r + \sigma_t \varepsilon$ (cosine schedule) | $x_t = (1 - t)\, r + t\, \varepsilon$ (straight line) |
| Network predicts | $v = \alpha_t \varepsilon - \sigma_t r$ | velocity $\varepsilon - r$ |
| Sampler | DDIM | ODE, Euler (or Heun) |

Both samplers default to 3 steps, chosen by the step sweep below. Checkpoints
trained before the sweep store 50; pass `--sample-steps` to `evaluate` to override.

```shell
# Train (same budget as the diffusion run: ≈ 5 h of compute on the MX350, 9–10 h with validations and thermal pauses)
python -m msbgen.cli.train --preset flow

# Evaluate next to diffusion and the baselines, then plot
python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips --dists --kid --samples 8 --out outputs/eval/kodak-fm
python -m msbgen.cli.plot_eval outputs/eval/kodak-fm --pair flow diffusion

# Step sweep: both generative models at each step count, then the step plots
for s in 1 2 3 5 10 20 50; do
  python -m msbgen.cli.evaluate data/processed/kodak --models runs/diffusion/best.pt runs/flow/best.pt \
      --fills --dists --kid --samples 8 --sample-steps $s --out outputs/eval/kodak-gen-s$s
done
python -m msbgen.cli.plot_steps outputs/eval/kodak-gen-s* outputs/eval/kodak-fm
```

Details in [msbgen/models/README.md](msbgen/models/README.md#conditional-flow-matching-flowpy).

#### Flow matching results (Kodak)

Same 24 images, $k = 1$, 8 samples per image (seeds 0–7) for every generative row.
Flow `best.pt` is at step 38k (lowest validation loss), trained on an H100. Every
step count is a separate `evaluate` run (`outputs/eval/kodak-gen-s<N>`; 50 steps is
`outputs/eval/kodak-fm`). KID is patch KID × 10³, with a spread of about ± 2–3.5.

| Steps | Flow PSNR (dB) ↑ | Flow DISTS ↓ | Flow KID ↓ | Flow diversity | Diffusion PSNR | Diffusion DISTS | Diffusion KID |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **18.26** | 0.275 | 73.8 | 3.6 | 17.88 | 0.277 | 80.4 |
| 2 | 18.21 | 0.259 | **52.9** | 4.8 | 17.83 | 0.264 | 53.4 |
| **3** | 18.11 | 0.252 | 54.9 | 6.2 | 17.75 | 0.257 | 54.6 |
| 5 | 17.88 | **0.248** | 60.7 | 8.4 | 17.58 | 0.253 | 62.5 |
| 10 | 17.41 | 0.251 | 69.8 | 12.0 | 17.23 | 0.256 | 72.6 |
| 20 | 16.90 | 0.258 | 75.7 | 15.1 | 16.80 | 0.262 | 76.7 |
| 50 | 16.45 | 0.264 | 79.1 | 17.6 | 16.33 | 0.269 | 77.1 |
| *U-Net (residual)* | *18.60* | *0.293* | *79.8* | | | | |

What the numbers show:

- **Few steps beat many on every metric.** From 50 steps down to 3, PSNR rises by
  1.7 dB, DISTS and KID both improve, and sampling is 17× cheaper. Diversity falls
  steadily: fewer Euler / DDIM steps pull each sample towards the posterior mean.
  One flow Euler step from pure noise lands exactly on the model's estimate of that
  mean. The step count therefore moves the output along the perception–distortion
  curve, from a regression-like estimate (1 step) to a full sample (50), and the
  perceptual metrics are best in between: DISTS at 3–5 steps, KID at 2–3.
- **At 2–5 steps, flow matching beats the best regression model perceptually.** At
  3 steps it has better DISTS than the residual U-Net on **24 of 24** images
  (−0.041 ± 0.007, paired 95 % interval) and KID 55 against 80, at a cost of 0.49 dB
  PSNR. At 50 steps it gave up 2.15 dB and won DISTS on 20 of 24 images.
- **Flow matching is ahead of diffusion at every step count, most at few steps.**
  Its PSNR lead grows from +0.12 dB at 50 steps to +0.36–0.38 dB at 1–3 steps (22 of
  24 images). Its DISTS lead is a steady −0.005, with the paired interval clear of 0
  at every step count except 1. KID is a tie except at 1 step. Its mean of 8 samples
  is about 0.3 dB closer to the original at every step count, and at 10–50 steps its
  samples are slightly more diverse (about +1 level, 23–24 of 24 images).
- **Caveat.** At 3 steps the samples have a third of the 50-step diversity: they sit
  between the posterior mean and true posterior samples. For a codec that is a
  legitimate operating point, but it is not faithful posterior sampling.
  Full-resolution crops show 3–50 step samples of one seed with the same structure.
  Their visible errors (shutters of the wrong brightness, invented window panes)
  come from how little $B_7$ says, not from the sampler.

Default sampler from here on: **3 steps** (Euler for flow matching, DDIM `eta` 0 for
diffusion), DISTS-optimal is 5. Plots: `python -m msbgen.cli.plot_steps`
(`steps_kodak-k1_curves.png`: PSNR, DISTS, KID and diversity against steps;
`steps_kodak-k1_tradeoff.png`: PSNR against DISTS, step counts joined in order).

### Conditional normalizing flow

Milestone 5 adds a Glow-style model for the dequantized residual $R=I-128B_7$.
Four FrEIA coupling blocks at each of four squeeze scales receive features from a
learned, multi-resolution B7 encoder. The model uses 128×128 patches because the
invertible blocks have fixed spatial dimensions; patches are sampled independently.
Its dequantized continuous likelihood estimates residual coding cost, but does not
replace discrete entropy coding or establish actual lossless bpp.

Install FrEIA in the HPC environment, then train, evaluate against the existing
k=1 models, and generate plots:

```shell
python -m pip install 'FrEIA>=0.2'
pytest -q tests/test_normalizing_flow.py
python -m msbgen.cli.train --preset normalizing-flow --name normalizing-flow \
  --k 1 --steps 40000 --batch-size 16 --crop 128 --crop-size 128 \
  --scales 4 --blocks-per-scale 4 --condition-width 32 --hidden-width 64 \
  --device cuda --no-thermal-guard
python -m msbgen.cli.plot_runs runs/normalizing-flow --out outputs/plots/normalizing-flow
python -m msbgen.cli.evaluate data/processed/kodak \
  --models runs/cnn/best.pt runs/unet/best.pt runs/unet-res/best.pt \
  runs/diffusion/best.pt runs/flow/best.pt runs/normalizing-flow/best.pt \
  --lpips --dists --kid --samples 8 --sample-steps 3 \
  --out outputs/eval/kodak-normalizing-flow
python -m msbgen.cli.plot_eval outputs/eval/kodak-normalizing-flow \
  --pair normalizing-flow unet-res --out outputs/plots/normalizing-flow
```

The evaluation uses one draw per deterministic regressor and eight seeded draws
per generative model. For review, copy `runs/normalizing-flow/config.json` and
`log.jsonl`, `outputs/eval/kodak-normalizing-flow/summary.json` and
`per_image.csv`, and the generated PNGs under `outputs/plots/normalizing-flow/`.

### Normalizing-flow results (Kodak)

The k=1 Glow-style model completed 40k steps on DIV2K. It models dequantized residuals in 128×128 patches. Best validation NLL was 4.2647 bits per residual dimension; validation PSNR at step 40k was 16.21 dB, below midpoint fill (17.38 dB on the same validation crops).

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | KID × 10³ ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Midpoint fill | 17.01 | 0.579 | 0.517 | 0.364 | 81.8 |
| Residual U-Net | 18.60 | 0.637 | 0.455 | 0.293 | 79.8 |
| Diffusion, 3 steps | 17.75 | 0.610 | 0.424 | 0.257 | 54.6 |
| Flow matching, 3 steps | 18.11 | 0.615 | 0.419 | 0.252 | 54.9 |
| Normalizing flow, one sample | 15.77 | 0.410 | 0.577 | 0.316 | 120.0 |
| Normalizing flow, 256-pixel patches | 15.85 | 0.423 | 0.568 | 0.320 | 124.9 |

All test metrics use the same 24 Kodak images; stochastic-model KID uses eight samples per image. Averaging eight 128-pixel-patch flow samples raises PSNR to 17.69 dB, but diversity is 23.94 intensity levels versus about 6.1 for diffusion/flow matching. Moving to 256-pixel patches changes single-sample PSNR by only +0.08 dB and sample-mean PSNR by +0.02 dB; SSIM/LPIPS improve slightly, but DISTS/KID worsen slightly. It is about 22% faster per image and shows fewer obvious tile boundaries in the inspected example, but remains behind residual U-Net on PSNR, SSIM, and LPIPS for all 24 images. The outputs retain noisy texture and soft structure.

The dequantized NLL is a continuous-density estimate, not measured lossless bpp. Keep the normalizing flow as a negative k=1 baseline. The k=1/2/3 comparison and follow-up three-stage hybrid experiments are now complete; hybrid results follow below.

### Multi-MSB results (Kodak)

The corrected k=3 evaluation uses the same 24 images and eight samples for each stochastic model. The best residual U-Net gives the highest PSNR, while diffusion and flow matching give the best perceptual scores:

| k | Midpoint PSNR | Residual U-Net PSNR | Diffusion PSNR / DISTS / KID | Flow PSNR / DISTS / KID |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17.01 | 18.60 | 17.75 / 0.257 / 54.6 | 18.11 / 0.252 / 54.9 |
| 2 | 22.63 | 26.14 | 25.63 / 0.129 / 26.1 | 25.60 / 0.130 / 25.3 |
| 3 | 28.74 | **32.00** | 31.58 / **0.072** / **8.6** | 31.51 / 0.072 / 9.0 |

At k=3, eight-sample mean PSNR is 32.37 dB for diffusion and 32.34 dB for flow matching, above the residual U-Net, with diversity around 2.4 intensity levels. Each extra transmitted plane adds roughly 6 dB for the midpoint baseline and about 5.9-7.9 dB for the learned/generative models. These are 24-image Kodak results; the reference coder's measured rates are reported below.

The k=3 step sweep again places one step at the fidelity/latency point and three-to-five steps in the perceptual operating region. The reference coder now measures Kodak rates of 0.165, 0.501 and 1.021 bpp at k=1, 2 and 3. Next priorities are multi-seed and cross-dataset evaluation, plus production codec refinements.

The first Milestone 8 reference coder is available as:

```shell
python -m msbgen.cli.rate data/processed/kodak --gray --ks 1 2 3 \
  --out outputs/rate/kodak
```

It adaptively range-codes each top plane, verifies lossless decoding, and reports
raw, context-estimated, arithmetic payload, and framed stream bpp.

The first Kodak measurement is complete. All 24 images round-tripped losslessly:

| k | Raw bpp | Context estimate | Framed stream bpp |
| ---: | ---: | ---: | ---: |
| 1 | 1.000 | 0.165 | **0.165** |
| 2 | 2.000 | 0.500 | **0.501** |
| 3 | 3.000 | 1.020 | **1.021** |

These are actual rates for the reference coder, excluding image/container
metadata. At those rates, the reconstruction points are 18.60 dB for residual
U-Net at k=1, 26.14 dB at k=2, and 32.00 dB at k=3. The hybrid results below add
perceptual reconstruction points at the same transmitted rates.

### Hybrid results (Kodak)

The hybrid uses a matching residual U-Net checkpoint for stage 1 (frozen in these
runs), flow matching for a residual correction, and truncated diffusion refinement.
All stages generate only missing low bits, preserving MSB consistency exactly. Runs
used 40k steps, batch size 16, 128×128 DIV2K crops and EMA 0.999; the best k=2
checkpoint was selected at step 38k. Evaluation used 24 grayscale Kodak images,
eight samples for stochastic models, three flow steps, two DDIM refinement steps
(`refine_t=0.5`, `eta=0`), and patch KID × 1000.

| k | Method | PSNR ↑ | DISTS ↓ | KID × 10³ ↓ | PSNR avg ↑ | Diversity | Time / image (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Residual U-Net | **18.60** | 0.2931 | 79.75 | — | — | 0.006 |
| 1 | Sequential bitplane CNN (D4) | 18.56 | 0.2872 | 93.87 | — | — | 0.091 |
| 1 | Diffusion | 17.75 | 0.2567 | 54.61 | 18.20 | 6.08 | 0.088 |
| 1 | Flow matching | 18.11 | **0.2519** | 54.92 | 18.54 | 6.15 | 0.059 |
| 1 | Hybrid | 18.22 | 0.2594 | **46.89** | 18.61 | 5.81 | 0.101 |
| 2 | Residual U-Net | **26.14** | 0.1554 | 41.76 | — | — | 0.006 |
| 2 | Sequential bitplane CNN (D4) | 25.65 | 0.1657 | 54.33 | — | — | 0.062 |
| 2 | Diffusion | 25.63 | 0.1292 | 26.06 | 26.24 | 3.97 | 0.111 |
| 2 | Flow matching | 25.60 | 0.1302 | 25.34 | 26.23 | 4.03 | 0.059 |
| 2 | Hybrid | 25.99 | **0.1249** | **17.25** | **26.60** | 3.85 | 0.101 |
| 3 | Residual U-Net | 32.00 | 0.0852 | 19.04 | — | — | 0.006 |
| 3 | Sequential bitplane CNN (D4) | **32.04** | 0.0943 | 21.15 | — | — | 0.052 |
| 3 | Diffusion | 31.58 | 0.0717 | 8.56 | 32.37 | 2.35 | 0.088 |
| 3 | Flow matching | 31.51 | 0.0719 | 9.01 | 32.34 | 2.42 | 0.059 |
| 3 | Hybrid | 31.65 | **0.0681** | **5.93** | **32.60** | 2.54 | 0.102 |

The sequential bitplane CNN (D4) remains the single-sample distortion target:
18.56 / 25.65 / 32.04 dB at k=1/2/3 (SSIM 0.6328 / 0.7976 / 0.9015; LPIPS
0.4718 / 0.2807 / 0.1425; DISTS 0.2872 / 0.1657 / 0.0943; MAE 25.03 / 10.49 /
4.94). Its MSB consistency is 1.0000 at each k. It leads PSNR at k=3, narrowly
above residual U-Net (32.00 dB) and the hybrid (31.65 dB), while the hybrid has
lower LPIPS, DISTS and KID there. `PSNR avg` and diversity are not applicable to
this deterministic model; the common evaluator now measures the other metrics.

At k=2 and k=3 the hybrid improves DISTS and KID over both standalone generative
models. It also gives the highest eight-sample mean PSNR at those bit depths, while
single-sample PSNR remains below residual U-Net. At k=1 it gives the best KID, but
flow matching has better DISTS. The k=3 ablation isolates refinement: stages 1+2
score 31.69 dB / 0.0701 DISTS / 7.38 KID, versus 31.65 dB / 0.0681 / 5.93 for the
full model. Diffusion refinement therefore trades a small amount of single-sample
PSNR for better perceptual metrics in this setting. This is one training run on 24
Kodak images; multiple seeds and broader datasets are needed before general claims.
Timings are measurements from this evaluation hardware, not portable latency claims.

### ViT results (Kodak)

Two Vision Transformer variants were added (`--preset vit`, `--preset vit-hybrid`) and trained for k=1/2/3
with the same recipe (DIV2K, 40k steps, batch 16, 128×128 crops, lr 2e-4). `ViT` is a residual regressor:
8×8 patch tokens (192-d), 6 transformer blocks with 6 heads, fixed 2D sin-cos position embeddings, a linear
unpatchify head and a 3×3 conv fusion branch (2.9 M parameters, no EMA). `ViT hybrid` uses it as stage 1 of
the hybrid (10.9 M parameters, EMA 0.999). Unlike the U-Net hybrid, the ViT hybrid trained its stage 1 jointly
from scratch (not initialized or frozen). Best checkpoints: ViT at step 10k/22k/30k, ViT hybrid at 36k/40k/40k.
Evaluation protocol is the same as above.

| k | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | KID × 10³ ↓ | PSNR avg ↑ | Diversity | Time / image (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Residual U-Net | 18.60 | **0.6373** | 0.4550 | 0.2931 | 79.75 | — | — | 0.006 |
| 1 | Hybrid (U-Net) | 18.22 | 0.6242 | **0.4237** | **0.2594** | **46.89** | 18.61 | 5.81 | 0.102 |
| 1 | ViT | 18.35 | 0.6206 | 0.4662 | 0.2932 | 89.81 | — | — | 0.018 |
| 1 | ViT hybrid | **18.64** | 0.6230 | 0.4293 | 0.2618 | 49.17 | **18.83** | 5.07 | 0.114 |
| 2 | Residual U-Net | **26.14** | **0.8044** | 0.2443 | 0.1554 | 41.76 | — | — | 0.006 |
| 2 | Hybrid (U-Net) | 25.99 | 0.7936 | **0.2051** | **0.1249** | **17.25** | **26.60** | 3.85 | 0.102 |
| 2 | ViT | 24.51 | 0.7777 | 0.2676 | 0.1644 | 48.19 | — | — | 0.017 |
| 2 | ViT hybrid | 24.71 | 0.7698 | 0.2251 | 0.1314 | 19.46 | 25.30 | 4.27 | 0.114 |
| 3 | Residual U-Net | 32.00 | **0.9015** | 0.1171 | 0.0852 | 19.04 | — | — | 0.006 |
| 3 | Hybrid (U-Net) | 31.65 | 0.8930 | **0.0905** | **0.0681** | **5.93** | **32.60** | 2.54 | 0.102 |
| 3 | ViT | 30.78 | 0.8845 | 0.1487 | 0.0976 | 23.40 | — | — | 0.018 |
| 3 | ViT hybrid | 31.21 | 0.8896 | 0.0951 | 0.0700 | 7.12 | 32.10 | 2.59 | 0.115 |

The standalone ViT is below the residual U-Net at every k (−0.25 / −1.63 / −1.22 dB) and below the plain CNN
at k=2 and k=3: global attention did not make up for the missing convolutional locality prior at this size
and budget. Flow matching and diffusion improve it substantially on perceptual metrics (k=3 versus the ViT
hybrid's own stage 1: LPIPS 0.1496 → 0.0951, DISTS 0.0975 → 0.0700, KID 22.90 → 7.12). At k=1 the ViT hybrid
reaches the best single-sample PSNR of all models (18.64 dB) and the best eight-sample mean (18.83 dB), with
DISTS/KID slightly behind the U-Net hybrid; at k=2 and k=3 it trails the U-Net hybrid on every metric. Stage
ablations (`vit-hybrid-unet`, `vit-hybrid-flow`) are in `outputs/eval/kodak-vit-hybrid-k*/`. Single runs, 24
images, untuned ViT hyperparameters, and different stage-1 protocols: treat the comparison as indicative.

```shell
python -m msbgen.cli.train --preset vit --name vit-k2 --k 2 --device cuda
python -m msbgen.cli.train --preset vit-hybrid --name vit-hybrid-k2 --k 2 --device cuda
```

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

| Variable | Default | Meaning |
| --- | --- | --- |
| `MSBGEN_MAX_UPLOAD_MB` | `25` | Maximum upload size |
| `MSBGEN_MAX_PIXELS` | `16777216` (4096²) | Maximum image area |
| `MSBGEN_MODELS_DIR` | `runs` | Where the web app looks for `<run>/best.pt` checkpoints |
| `MSBGEN_DEVICE` | auto | Device for model inference (`cuda`, `cpu`, …) |
| `MSBGEN_EVAL_DIR` | `outputs/eval` | Where the web app looks for saved evaluations (`<name>/summary.json`) |
| `MSBGEN_PERCEPTUAL` | `1` | `0` turns off LPIPS / DISTS per reconstruction |
| `MSBGEN_PERCEPTUAL_MAX_PIXELS` | `4194304` (2048²) | Larger images skip LPIPS / DISTS (GPU memory) |

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
      "psnr": { "zero": 11.85, "mid": 16.27, "random": 13.50 },
      "ssim": { "zero": 0.468, "mid": 0.505, "random": 0.121 },
      "mae": { "zero": 53.15, "mid": 33.14, "random": 44.00 }, "image": "data:image/png;base64,..." }
  ]
}
```

`reconstructions` runs from k = 1 to `bit_depth`, and each `image` is the
midpoint-fill reconstruction as an 8-bit PNG. JSON has no infinity, so the
lossless PSNR (k = `bit_depth`) is `null`. SSIM is `null` for images smaller than
11×11, and correlations involving a constant plane are `null`.

### `GET /api/models`

The trained checkpoints found in `MSBGEN_MODELS_DIR`:
`{"torch_available", "models_dir", "models": [{"name", "label", "kind", "arch", "output", "k", "bit_depth", "channels", "parameters", "step", "val_psnr", "val_gain_db"}]}`.
`kind` is `regression` or `diffusion` (`arch` `tunet`); for diffusion, `val_psnr` is of one sample.
The list is re-scanned on every call, so new runs appear without a restart.

### `POST /api/reconstruct`

Multipart `file`, plus optional form fields `k` (default 1) and `models`
(comma-separated names; default: every compatible model). The image must be
8-bit; colour images are converted to grayscale, with a note. Diffusion and flow
models return their seed-0 sample with the checkpoint's sampler (3 steps by default,
about 1–2 s for a Kodak-sized image on the MX350; the request waits if the thermal
guard pauses). Returns the image
fields plus:

```json
{
  "k": 1,
  "original": "data:image/png;base64,...",
  "results": [
    { "name": "mid", "label": "Midpoint fill", "kind": "fill",
      "psnr": 17.29, "ssim": 0.635, "mae": 30.66, "msb_consistency": 1.0,
      "lpips": 0.383, "dists": 0.265,
      "psnr_gain": 0.0, "image": "data:...", "error_map": "data:..." },
    { "name": "unet-res", "label": "U-Net (residual)", "kind": "model", "...": "..." }
  ],
  "perceptual": ["lpips", "dists"]
}
```

`lpips` and `dists` are `null` (and missing from `perceptual`) when `lpips` / `piq`
or their weights are unavailable, when `MSBGEN_PERCEPTUAL=0`, or for images larger
than `MSBGEN_PERCEPTUAL_MAX_PIXELS` (with a note). The networks load on the first
request (the first ever use downloads their weights).

Fills come first (`zero`, `mid`, `random`), then models. `psnr_gain` is relative to midpoint fill, and
`error_map` is $|I - \hat I| \times 4$. Errors: `400` for non-8-bit input or an
invalid `k`, and `404` for an unknown model name.

### `GET /api/evaluations`

Saved test-set evaluations (`msbgen.cli.evaluate` output) found in
`MSBGEN_EVAL_DIR`, newest first:
`{"eval_dir", "evaluations": [{"name", "data", "images", "k", "samples", "kid_patch", "modified", "methods": [{"name", "label", "kind", "metrics": {"psnr": {"mean", "std"}, "...": null}}]}]}`.
`kind` is `fill`, `regression` or `generative`; `metrics` has `psnr`, `ssim`,
`lpips`, `dists`, `mae`, `msb_consistency`, `psnr_avg`, `diversity`, `time_s` and
`kid` (std over images, or over KID subsets for `kid`; `null` if not computed).
Set-level metrics such as KID exist only here, not per uploaded image. The web
app shows these in its **Test-set evaluation** section.

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
a.reconstructions[0].psnr["random"]         # PSNR of 128·B7 + uniform random low bits

est = msb_reconstruction(img, k=2, fill="mid")   # float64, unrounded
psnr(img, est, data_range=255), ssim(img, est, data_range=255)
```

The bit depth comes from the dtype (`bool` → 1, `uint8` → 8, `uint16` → 16), and
any number of channels is accepted. For example, a 16-bit RGB array gives
`(3, 16, H, W)`. Use `msbgen.core.load_image(bytes)` to decode a file into
grayscale or RGB the same way the web app does.
