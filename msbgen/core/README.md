# `msbgen.core`

Framework-independent image and bit-plane logic. Nothing here imports FastAPI, so
the web app, the CLI and future training / dataset scripts all share the same code.
Everything important is re-exported from the package:

```python
from msbgen.core import analyze_image, msb_reconstruction, psnr, ssim, load_image
```

| Module | What it does |
| --- | --- |
| [`bitplanes.py`](bitplanes.py) | Split images into bit-planes and put them back together |
| [`analysis.py`](analysis.py) | Information content of planes: entropy, correlation, top-k reconstruction |
| [`reconstruction.py`](reconstruction.py) | The `Reconstructor` interface, fills (zero / mid / random), `evaluate` (scores any Î) |
| [`metrics.py`](metrics.py) | Full-reference quality metrics: PSNR, SSIM, MAE, MSE |
| [`image_io.py`](image_io.py) | Decode files to grayscale / RGB arrays, PNG encoding helpers |

## Conventions

- **Images** are numpy arrays of shape `(H, W)` or `(H, W, C)` with dtype `bool`
  (1-bit), `uint8` (8-bit) or `uint16` (16-bit). The bit depth `n` comes from the
  dtype unless you pass `bit_depth` explicitly.
- **Planes** have shape `(C, n, H, W)` with values in {0, 1}. Index `[c, k]` is
  $B_k$ of channel `c`, so `k = n - 1` is the MSB.
- **Plane order in results** (stats lists, correlation matrix, labels) is by
  channel, then **MSB → LSB**, e.g. `R·B7, R·B6, …, R·B0, G·B7, …`.
- **`data_range`** for metrics is $2^n - 1$: 255 for 8-bit, 65535 for 16-bit.

## Bit-planes (`bitplanes.py`)

$$I = \sum_{k=0}^{n-1} 2^k B_k$$

| Function | Returns |
| --- | --- |
| `slice_bitplanes(image, bit_depth=None)` | `(C, n, H, W)` uint8 planes |
| `reconstruct_from_planes(planes, bits=None)` | `(H, W, C)` image from the selected bits (all by default); uint8 if n ≤ 8, else uint16 |
| `combine_channel_planes(planes, bit)` | `(H, W, C)` {0,1}: bit `k` of every channel stacked (the "Combined RGB" tiles) |
| `binary_entropy(p)` | Entropy in bits of a Bernoulli(p) source |
| `infer_bit_depth(image)` | 1, 8 or 16 from the dtype; `ValueError` for anything else |

```python
planes = slice_bitplanes(img)                        # (1, 8, H, W) for 8-bit gray
msb_only = reconstruct_from_planes(planes, bits=[7])  # 128·B7
```

## Analysis (`analysis.py`)

### Why analyse bit-planes?

MSBGen's premise is: send only the top plane(s), cheaply, and let a model
recreate the rest. Before training any model, we need to know three things. Each
measure in this module answers one of them:

| Question | Measure | Why it matters |
| --- | --- | --- |
| How expensive is it to send a plane? | `entropy`, `context_entropy`, `coded_bpp` | The rate side of the codec. If $B_7$ compresses to ~0.1 bpp, sending it is very cheap. If the lower planes cost ~1 bpp each, they are expensive to send and worth predicting instead. |
| How much structure does a plane have? | `spatial_correlation` | Structured planes (edges, shapes) are what a model can learn from. Noise-like planes can't be predicted from neighbours, so a model can only guess them statistically. |
| How much do planes tell us about each other? | `plane_correlation` | Correlation between $B_7$ and lower planes, or between R/G/B, is information a model or coder could exploit. It also warns when the "one plane" input is less independent than it looks. |
| How good is an image with no model at all? | top-k reconstruction + [metrics](#metrics-metricspy) | The baseline (e.g. $128\,B_7 + 63.5$) that every learned model has to beat, and a rate–distortion curve to compare against. |

### `analyze_image(image, bit_depth=None, with_ssim=True) -> ImageAnalysis`

Runs everything below on one image and returns plain dataclasses:

```python
a = analyze_image(img)          # img: (H, W) or (H, W, C)

a.bit_depth, a.channels         # n, C
a.planes                        # list[PlaneStats], C·n entries, MSB first per channel
a.correlation                   # (C·n, C·n) float array, NaN where undefined
a.reconstructions               # list[ReconstructionStats], k = 1 … n
```

| `PlaneStats` field | Meaning |
| --- | --- |
| `channel_index`, `bit` | Which plane |
| `ones_ratio` | Fraction of pixels where $B_k = 1$ |
| `entropy` | i.i.d. entropy $H(B_k)$, bits/pixel |
| `context_entropy` | $H(B_k \mid W, NW, N, NE)$, bits/pixel (see below) |
| `spatial_correlation` | Mean neighbour correlation, or `None` for a constant plane |

| `ReconstructionStats` field | Meaning |
| --- | --- |
| `k` | Number of most-significant planes kept |
| `bits_per_pixel` | Raw cost, `k × C` |
| `coded_bpp` | Sum of `context_entropy` over the kept planes and channels |
| `psnr`, `ssim`, `mae` | Dicts keyed by fill, `{"zero": …, "mid": …, "random": …}` |

`psnr` is `inf` for a lossless reconstruction (always the case at `k = n`). `ssim`
is `None` when `with_ssim=False` or the image is smaller than 11×11. SSIM is by far
the most expensive step, so turn it off for quick sweeps.

### Context entropy: `context_entropy(plane)`

The empirical conditional entropy of each bit given its 4 **causal** neighbours in
the same plane (the pixels a raster-order decoder has already seen):

```text
NW  N  NE
W   x
```

The 4 neighbour bits make 16 contexts. For each context $c$ we count zeros and
ones and compute
$H = \sum_c p(c)\, h\big(p(x{=}1 \mid c)\big)$. Neighbours outside the image
count as 0.

How to read it:

- It is the rate an ideal adaptive binary coder with this context model would
  approach (JBIG uses the same idea with a larger template). The i.i.d. `entropy`
  ignores all structure and is almost always ≈ 1.
- Each plane is treated **on its own**: no information from other planes or
  channels is used, so real coders that exploit those could do better.
- It is a plug-in estimate with no model cost, so it is slightly optimistic for
  very small images. With 16 contexts that doesn't matter for normal image sizes.

So `coded_bpp` is an *estimate* of what transmitting the top k planes would cost,
not a real code length. It is enough to compare planes and images; the actual
lossless stream is measured by `msbgen.cli.rate` in Milestone 8.

### Actual lossless top-plane coding

`encode_top_planes(image, k)` uses an adaptive binary range coder with the same
four causal neighbours as `context_entropy`. `decode_top_planes` reverses it and
the rate CLI verifies that decoding reproduces the top-k image exactly. The stream
has a four-byte length prefix per plane, so the report distinguishes arithmetic
payload bpp from framed stream bpp. This is a simple reference coder, not yet a
production entropy coder: its adaptive model resets for each plane and its
metadata/container overhead is reported separately.

### Spatial correlation: `spatial_correlation(plane)`

The Pearson (phi) correlation between each pixel and its right neighbour, and
between each pixel and the one below, averaged. It is about 0.9 for the MSB of a
natural image and about 0 for the LSB, which looks like noise. It returns `None`
when the plane is constant, because the correlation is undefined.

### Plane correlation: `plane_correlation(planes)`

The Pearson correlation between every pair of planes, pixel by pixel, over all
channels. It takes the `(C, n, H, W)` array from `slice_bitplanes` and returns a
`(C·n, C·n)` matrix in the result order (channel, then MSB → LSB). The joint
counts are accumulated exactly in chunks of $2^{20}$ pixels, so memory stays
bounded for large images. Entries involving a constant plane are `NaN`.

### Top-k reconstruction: `msb_reconstruction(image, k, fill="mid", bit_depth=None)`

Keeps the top `k` planes and fills in the missing $n-k$ bits:

| `fill` | Formula | Notes |
| --- | --- | --- |
| `"zero"` | $\sum_{j=n-k}^{n-1} 2^j B_j$ | Truncation; always too dark on average |
| `"mid"` | zero fill $+ \frac{2^{n-k}-1}{2}$ | Centre of the missing range. MSE-optimal if the missing bits are uniform. For n = 8, k = 1 this is the baseline $128\,B_7 + 63.5$. |
| `"random"` | zero fill $+ U\{0, \dots, 2^{n-k}-1\}$ | Missing bits drawn uniformly per pixel (fixed seed 0, so repeatable). A sample from a uniform prior: the no-model counterpart of a generative reconstruction. |

It returns **float64 without rounding**, so the metrics score exactly
$128\,B_7 + 63.5$. Round only for display, as the API does with round-half-up. It
raises `ValueError` for `k` outside `1…n` or an unknown `fill`.

```python
est = msb_reconstruction(img, k=1)             # 128·B7 + 63.5 for 8-bit
psnr(img, est, 255)
```

For uniformly distributed pixels the midpoint fill has
$\text{MSE} = (4^{n-k}-1)/12$. A test checks this, and it gives a useful point of
reference: each extra plane adds about 6 dB.

The random fill is an independent draw from the same range, so its error is the
difference of two independent uniforms: $\text{MSE} = (4^{n-k}-1)/6$, exactly
twice the midpoint fill's, or **3.01 dB lower PSNR** (also tested). The same
factor of 2 holds in general: a sample from any posterior has twice the MSE of
that posterior's mean. That is why a good generative model is expected to score
about 3 dB *below* the best deterministic model on PSNR, and why the random fill
is the right no-model reference for it.

## Reconstructors (`reconstruction.py`)

Everything that produces Î goes through one interface, so the constant baselines
and every trained model are scored identically:

```python
class Reconstructor(ABC):
    name: str      # short id: "zero", "mid", "random", or a run name such as "unet-res"
    label: str     # "Midpoint fill", "U-Net (residual)", …
    def supports(self, k, bit_depth, channels) -> bool: ...
    def reconstruct(self, top, k, bit_depth) -> np.ndarray: ...   # float64 Î
```

**The receiver rule.** `reconstruct` gets `top`, the image with every bit below the
top k set to zero, i.e. $\sum_{\text{top }k} 2^j B_j$. It never gets the original.
This makes it impossible for a method to use information the receiver wouldn't
have (a test checks it).

| Name | Function | Returns |
| --- | --- | --- |
| | `keep_top_bits(image, k, bit_depth=None)` | What the receiver has: `image` with the lower n−k bits zeroed (same dtype) |
| `zero` | `ZeroFill` | `top` as float (truncation) |
| `mid` | `MidpointFill` | `top` + (2^(n−k) − 1)/2, e.g. 128·B7 + 63.5 |
| `random` | `RandomFill(seed=0)` | `top` + uniform random integers in [0, 2^(n−k) − 1]; same input → same output |
| | `FILL_RECONSTRUCTORS` | `{"zero": ZeroFill(), "mid": MidpointFill(), "random": RandomFill()}` |
| | `evaluate(reference, estimate, k, bit_depth, with_ssim=True)` | `{"psnr", "ssim", "mae", "msb_consistency"}` |
| | `reconstruct_and_evaluate(reconstructor, image, k, …)` | Applies the receiver rule, runs the method, scores it: `(estimate, scores)` |
| | `msb_consistency(reference, estimate, k, bit_depth)` | Fraction of pixels whose *rounded* estimate keeps the transmitted top k bits |
| | `round_estimate(estimate, bit_depth)` | Round half up and clip; what you would store or display |

```python
from msbgen.core import FILL_RECONSTRUCTORS, reconstruct_and_evaluate

estimate, scores = reconstruct_and_evaluate(FILL_RECONSTRUCTORS["mid"], gray, k=1)
scores   # {"psnr": 17.0, "ssim": 0.58, "mae": 31.8, "msb_consistency": 1.0}
```

**MSB consistency** checks something the quality metrics don't: whether the
reconstruction contradicts the data the receiver actually got. If $B_7 = 1$ (value
≥ 128) but the model outputs 120, the output is inconsistent with the transmission.
The fills and residual-mode models are consistent by construction (1.0), while
models that predict the whole image can fall below 1.

Trained networks implement the same interface in
[`msbgen.models.ModelReconstructor`](../models/README.md#modelreconstructor).
`msb_reconstruction` (above) is a shortcut for the three fills.

## Metrics (`metrics.py`)

### Why we measure quality at all

MSBGen is about a trade-off between two quantities:

- **Rate**: how many bits we send. This is the `bits_per_pixel` / `coded_bpp`
  side, measured by the entropies above.
- **Distortion**: how different the reconstruction $\hat I$ is from the original
  $I$.

A codec is only interesting if it gets lower distortion for the same rate, or the
same distortion for a lower rate. The metrics below measure the distortion side.
They are *full-reference* metrics: they compare $\hat I$ with the original $I$,
which we have during research but a real receiver would not.

In this milestone they give the **floor** every later model must beat: how good
an image is if you just fill in the missing bits with a constant. When the U-Net,
diffusion and flow models arrive (PLAN.md Milestones 2–5), their outputs are
scored with exactly the same functions, so the numbers can be compared directly.
The plan asks for PSNR, SSIM and LPIPS together because each one catches
something the others miss.

### MAE: mean absolute error

$$\text{MAE} = \frac{1}{N}\sum_i |I_i - \hat I_i|$$

- **What it says:** how far off a pixel is on average, in intensity levels. For
  8-bit images this is on the 0–255 scale, so MAE = 33 means "off by 33 gray
  levels on average". It is the easiest metric to read.
- **Range:** 0 (identical) upwards; lower is better.
- **Behaviour:** every error counts in proportion to its size, so a few large
  mistakes don't dominate the score.
- **Why here:** it gives an intuitive check on the other numbers. With only $B_7$
  and midpoint fill, the true value lies in a range 128 levels wide. The worst
  possible error is 64 levels, and a typical image averages about 30–35.

### MSE and PSNR: peak signal-to-noise ratio

$$\text{MSE} = \frac{1}{N}\sum_i (I_i - \hat I_i)^2, \qquad
\text{PSNR} = 10 \log_{10} \frac{\text{MAX}^2}{\text{MSE}} \;\text{dB}$$

where $\text{MAX}$ = `data_range` = $2^n - 1$ (255 for 8-bit).

- **What it says:** the error energy relative to the largest possible signal, on a
  log scale. It is the standard metric in image compression, so papers report it
  and we must too.
- **Range:** higher is better, and identical images give $\infty$. For 8-bit
  images, a rough guide:

  | PSNR | Typical impression |
  | --- | --- |
  | < 20 dB | Obviously wrong: large blotches or banding |
  | 20–30 dB | Recognisable but clearly degraded |
  | 30–40 dB | Good; artefacts are visible only on close inspection |
  | > 40 dB | Very hard to tell from the original |

  This is only a guide; how visible an error is depends on image content.
- **Log scale:** +3 dB means half the MSE, and +6 dB means a quarter of the MSE
  (half the typical error size). Each extra bit-plane halves the range of the
  missing bits, which is why every added plane gives about **+6 dB**. For uniform
  pixels, midpoint fill has exactly $\text{MSE} = (4^{n-k}-1)/12$.
- **Behaviour:** errors are squared, so large mistakes are penalised heavily. PSNR
  is also blind to *where* the errors are: an image shifted by one pixel, or one
  with realistic but invented texture, can score worse than a blurry one.
- **Why here:** midpoint fill is the reconstruction with the lowest MSE when
  nothing is known about the missing bits. PSNR therefore tells us how much a
  learned model improves on "no knowledge".

### SSIM: structural similarity

SSIM compares images the way the eye tends to: by local brightness, local contrast
and local structure (patterns of light and dark), not by individual pixels. For
each small window it compares the means $\mu$, variances $\sigma^2$ and covariance
$\sigma_{xy}$ of the two images:

```math
\text{SSIM}(x, y) =
\underbrace{\frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}}_{\text{luminance}}
\cdot
\underbrace{\frac{2\sigma_{xy} + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}}_{\text{contrast} \times \text{structure}}
```

The window slides over the whole image and the values are averaged. $C_1$ and
$C_2$ are small constants that keep the division stable in flat regions.

- **What it says:** whether the image *looks* like the original: edges in the same
  places, similar textures, similar local contrast.
- **Range:** up to 1 (identical); higher is better. About 0.95 or more is usually
  hard to tell apart, and below about 0.5 looks very different. It can go
  negative, but not for reconstructions like ours.
- **Behaviour:** it forgives uniform brightness shifts much more than PSNR does,
  and it punishes lost edges and texture more.
- **Why here:** the whole idea of MSBGen is that a generative model can restore
  *plausible structure* from $B_7$. It may not recover the exact pixel values, so
  PSNR alone would undervalue it. SSIM is the standard structural complement.

### How they disagree, with this project's numbers

For the scikit-image "astronaut" photo (RGB, 8-bit):

| Top k | Fill | PSNR (dB) | SSIM | MAE |
| --- | --- | ---: | ---: | ---: |
| 1 | midpoint | **16.3** | **0.505** | **33.1** |
| 1 | random | 13.5 | 0.121 | 44.0 |
| 1 | zero | 11.9 | 0.468 | 53.2 |
| 4 | midpoint | **34.1** | 0.823 | **4.4** |
| 4 | random | 31.5 | 0.739 | 5.6 |
| 4 | zero | 29.9 | **0.886** | 6.5 |

- Midpoint fill always wins on PSNR and MAE, because it minimises the error
  magnitude.
- At k = 4, zero fill wins on SSIM. Adding a constant changes only SSIM's
  luminance term, and that term is evaluated window by window. Inside a small
  window the missing bits are often nearly constant rather than spread uniformly,
  so one global "midpoint" offset is not the best *local* guess.
- Random fill lands between the two on PSNR and MAE, but has by far the worst SSIM
  at k = 1. Per-pixel noise has no structure, so SSIM's structure term collapses.
  Realistic *texture* is not the same as random values: a generative model has to
  produce structure that is correlated across neighbouring pixels.

The metrics measure different things, and a single number can mislead. This is
why the analysis reports all three for every fill, and why PLAN.md warns against
relying on PSNR alone. Generative models make this worse, because sharp invented
detail typically lowers PSNR while raising perceived quality. LPIPS (a learned
perceptual distance) and FID (realism over a whole dataset) will cover that. They
need pretrained networks, so they will be added with the deterministic baselines.

### Functions

Every function takes `(reference, estimate)` of shape `(H, W)` or `(H, W, C)`,
with any numeric dtype, and computes in float64. Pass the original image first.

| Function | Returns |
| --- | --- |
| `mse(reference, estimate)` | Mean squared error |
| `mae(reference, estimate)` | Mean absolute error, in intensity levels |
| `psnr(reference, estimate, data_range)` | PSNR in dB; `inf` if identical |
| `ssim(reference, estimate, data_range)` | Mean SSIM averaged over channels; `None` if `min(H, W) < 11` |

**SSIM implementation.** This is the standard Wang et al. (2004) formulation:

- an 11×11 Gaussian window with σ = 1.5, $K_1 = 0.01$, $K_2 = 0.03$ (so
  $C_1 = (K_1 \cdot \text{MAX})^2$ and $C_2 = (K_2 \cdot \text{MAX})^2$)
- population (not sample) covariance
- the 5-pixel border, where the window hangs off the image, is excluded from the mean
- computed per channel and averaged

It matches
`skimage.metrics.structural_similarity(..., gaussian_weights=True, sigma=1.5, use_sample_covariance=False)`
to about 1e-9 (tested). It is implemented with `cv2.GaussianBlur` because that is
much faster than scikit-image on large images.

Things to keep in mind:

- **Always pass the right `data_range`.** Using 255 for a 16-bit image makes PSNR
  about 48 dB too high, and it also breaks SSIM's constants.
- Metrics are computed on the **unrounded** float reconstruction. Rounding to
  integers first changes the results slightly, especially at large k.
- Colour images: PSNR and MAE pool all channels, while SSIM is averaged over
  channels. Neither is weighted for perception. If a comparison needs it, compute
  them on luma (`to_grayscale`).

## Image I/O (`image_io.py`)

| Function | What it does |
| --- | --- |
| `load_image(data, max_pixels=None) -> LoadedImage` | Decode file bytes to grayscale `(H, W, 1)` or RGB `(H, W, 3)`, keeping the bit depth. Alpha is dropped; palette, CMYK, LAB, … become RGB; 16-bit colour goes through OpenCV. Raises `ImageDecodeError`. |
| `to_grayscale(array)` | RGB → BT.601 luma `(H, W, 1)`, same dtype (identical to Pillow's `L`). Grayscale input is returned unchanged. |
| `to_preview(array, bit_depth)` | 8-bit version for display only (16-bit >> 8, 1-bit × 255) |
| `encode_png(array, binary=False)` | PNG bytes; `binary=True` writes a 1-bit PNG for a {0,1} plane |
| `encode_rgb_bits_png(bits)` | `(H, W, 3)` {0,1} → 8-colour palette PNG |

A `LoadedImage` has `array`, `color_mode` (`"L"` / `"RGB"`), `bit_depth`,
`preview`, `original_mode`, `original_format`, `notes` (conversions applied) and
the helpers `width`, `height`, `channel_names` and `channel_labels`.

## Tests

- [`tests/test_bitplanes.py`](../../tests/test_bitplanes.py): slicing and
  reconstruction
- [`tests/test_analysis.py`](../../tests/test_analysis.py): metrics (including the
  scikit-image SSIM check), entropy and correlation on known patterns, fills, and
  the uniform-MSE theory
- [`tests/test_reconstruction.py`](../../tests/test_reconstruction.py): the receiver
  rule, fills, rounding, MSB consistency and `evaluate`
