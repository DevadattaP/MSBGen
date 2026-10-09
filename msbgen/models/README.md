# `msbgen.models`

Neural reconstructors, i.e. the trained counterparts of the fills in
[`msbgen.core.reconstruction`](../core/README.md#reconstructors-reconstructionpy).
This package needs PyTorch. `msbgen.core` never imports it, so the web app and the
analysis tools still work without torch installed.

For model architectures, input/output paths, targets, losses, training, and inference,
see the [model guide](MODEL_GUIDE.md).

| Module | What it does |
| --- | --- |
| [`nets.py`](nets.py) | Backbones: `SimpleCNN`, `UNet` and the time-conditioned `TimeUNet` |
| [`bitdepth_recovery.py`](bitdepth_recovery.py) | Paper-style one-plane CNN, patch dataset, trainer, and sequential recovery |
| [`msb_net.py`](msb_net.py) | `MSBNet`: backbone + input encoding + output mode; `ModelConfig` (regression) |
| [`generative.py`](generative.py) | `ResidualGenerator`: what the generative models share (residual scaling, $B_7$ conditioning, `TimeUNet`) |
| [`diffusion.py`](diffusion.py) | `MSBDiffusion`: conditional diffusion on the residual, p(R \| B7); `DiffusionConfig` |
| [`flow.py`](flow.py) | `MSBFlow`: conditional flow matching on the residual, p(R \| B7); `FlowConfig` |
| [`normalizing_flow.py`](normalizing_flow.py) | `MSBNormalizingFlow`: patchwise conditional Glow likelihood and inverse sampling; `NormalizingFlowConfig` |
| [`checkpoint.py`](checkpoint.py) | `save_checkpoint`, `load_model`, `read_checkpoint`, `build_model`; `GENERATIVE_KINDS` |
| [`reconstructor.py`](reconstructor.py) | Checkpoint-backed regression and generative `Reconstructor`s, `load_reconstructor`, `discover_checkpoints` |
| [`perceptual.py`](perceptual.py) | Deep-feature metrics: `LPIPS`, `DISTS`, and `InceptionFeatures` for KID |

The training loop is in [`msbgen/training.py`](../training.py), and the
command-line entry points are in [`msbgen/cli`](../cli/README.md).

## The task

The receiver has only the top k planes. For the first baselines k = 1, n = 8,
grayscale:

$$
B_7 \;\longrightarrow\; \text{network} \;\longrightarrow\; \hat I \approx I
$$

The networks are trained with MSE, exactly like the plan's baselines 2–4. Their
job is to beat the no-model **midpoint fill** $128\,B_7 + 63.5$. The training log
reports this directly as `val_gain_db`, the PSNR gain over midpoint fill on the
same validation crops.

## `MSBNet`: input and output conventions

```python
from msbgen.models import ModelConfig, MSBNet

model = MSBNet(ModelConfig(arch="unet", output="residual", k=1, arch_args={"width": 24, "levels": 4}))
est = model(top)       # top: (B, 1, H, W) image with only the top k bits set, any numeric dtype
                       # est: (B, 1, H, W) in [0, 1] = Î / 255
```

**Input.** The model receives `top` (the top-k image) and splits it into planes
itself: $B_{n-1}, \dots, B_{n-k}$, stacked as channels and mapped to ±0.5. The
same data pipeline therefore works for any k, and the model can't see bits the
receiver doesn't have.

**Output modes** (`ModelConfig.output`):

| Mode | Formula | PLAN.md |
| --- | --- | --- |
| `"direct"` | $\hat I = \sigma(z)$ | Baselines 2 and 3: predict the image |
| `"residual"` | $\hat I = \big(\text{top} + \sigma(z)\,(2^{n-k}-1)\big) / (2^n - 1)$ | Baseline 4: predict only $R = I - 128\,B_7 \in [0, 127]$ |

The two modes differ in one important way:

- **Residual** reproduces the transmitted planes exactly, by construction: MSB
  consistency is always 1. The network only has to model the missing range.
- **Direct** can output a value whose MSB contradicts $B_7$, e.g. 120 where
  $B_7 = 1$ says the value is at least 128. Evaluation reports this as
  `msb_consistency` < 1.

**Padding.** The U-Net needs H and W divisible by $2^\text{levels}$, so inputs are
reflect-padded and the output is cropped back. Any image size works.

## Backbones (`nets.py`)

| | `SimpleCNN` (`arch="cnn"`) | `UNet` (`arch="unet"`) |
| --- | --- | --- |
| Structure | `depth` 3×3 convs at constant `width`, ReLU | Encoder–decoder with skip connections, `levels` ×2 downsamplings, channels double per level |
| Receptive field | $2\cdot\text{depth}+1$ px (17 px by default) | Large (≈ 200 px for 4 levels) |
| Can use | Local neighbourhood only | Scene-level context: shapes, smooth gradients, objects |
| Default size | width 64, depth 8: 0.22 M params | width 24, levels 4: 4.4 M params |
| PLAN.md | Baseline 2: pixelwise regression | Baselines 3 and 4: strong deterministic baseline |

The CNN answers "how much does *local* context recover?", and the U-Net answers
"how much does *global* context add?". Comparing them tells us how far the
reconstruction depends on large-scale structure, which is the kind of thing a
generative prior should be good at.

**Default sizes and the laptop GPU.** On the development machine (GeForce MX350,
2 GB) with batch 16 of 128×128 crops, measured in fp32:

| Model | Params | Speed | Peak memory |
| --- | ---: | ---: | ---: |
| CNN 64×8 | 0.22 M | 3.3 it/s | 1.0 GB |
| U-Net w24 L4 | 4.4 M | 5.5 it/s | 1.3 GB |
| U-Net w32 L4 | 7.8 M | 3.9 it/s | 1.8 GB |
| TimeUNet w32 L4 b1 (diffusion default) | 4.0 M | 2.3 it/s | 1.2 GB |
| TimeUNet w32 L3 b1 | 2.9 M | 2.3 it/s | 1.2 GB |
| TimeUNet w24 L3 b2 | 2.4 M | 1.6 it/s | 1.4 GB |
| TimeUNet w32 L3 b2 | 4.3 M | out of memory | > 2 GB |

The `TimeUNet` is compute-bound at about 0.6 Mpixel/s of training data whatever its
size, and 2 GB rules out two blocks per resolution at width 32. The default keeps
the regression U-Net's 4 levels (its receptive field) and its batch of 16 × 128².
One full-resolution Kodak forward pass (768×512) takes 0.22 s, so a 50-step sample
takes about 11 s. A 1024×1024 pass needs 0.78 GB.

Mixed precision (AMP) was *slower* on this Pascal-generation GPU, so training runs
in fp32. On a modern GPU, larger widths and batches are fine: pass `--width` and
`--batch-size`.

## Conditional diffusion (`diffusion.py`)

PLAN.md Milestone 3: learn the *distribution* of the missing bits instead of their
average. Like the residual U-Net, the model only produces the residual, so it can
never contradict $B_7$ (MSB consistency is 1 by construction):

$$
R = I - 128\,B_7 \in [0, 127], \qquad r = \frac{2R}{127} - 1 \in [-1, 1], \qquad
\hat I = 128\,B_7 + \hat R, \quad \hat R \sim p_\theta(R \mid B_7)
$$

**Forward process.** Continuous time $t \in [0, 1]$ and the cosine schedule
(Nichol & Dhariwal 2021), with $\alpha_t^2 + \sigma_t^2 = 1$, $\alpha_0 = 1$ and $\alpha_1 = 0$:

$$
x_t = \alpha_t\, r + \sigma_t\, \varepsilon, \qquad \varepsilon \sim \mathcal N(0, I)
$$

**Network and loss.** A `TimeUNet` gets $[x_t, B_7 \text{ planes as } \pm 0.5]$ as
input channels and $t$ through a sinusoidal embedding added in every ResBlock. It
predicts $v = \alpha_t \varepsilon - \sigma_t r$ (*v-prediction*, which PLAN.md suggests
investigating; it behaves well at both ends of the schedule, unlike $\varepsilon$-prediction
at $t = 1$). The loss is $\lVert \hat v - v \rVert^2$ with $t \sim U(0, 1)$.

**Sampling (DDIM).** Start from $x_1 = z \sim \mathcal N(0, I)$ and step
$t = 1 \to 0$ in `sample_steps` equal steps (default 3; runs trained before the step sweep
store 50, see [the step sweep](../../README.md#flow-matching-results-kodak)). At each step,
$\hat r = \text{clip}(\alpha_t x_t - \sigma_t \hat v, -1, 1)$ and
$\hat\varepsilon = (x_t - \alpha_t \hat r) / \sigma_t$. With `eta` = 0 the sampler is
deterministic given $z$. With `eta` = 1 it adds fresh noise every step (DDPM-like).
Different $z$ give different plausible residuals, which is the point.

```python
from msbgen.models import DiffusionConfig, MSBDiffusion

model = MSBDiffusion(DiffusionConfig(arch_args={"width": 32, "levels": 4, "blocks": 1}))
loss = model.loss(x)                                    # x: (B, 1, H, W) uint8 originals
est = model.sample(top, steps=50, generator=torch.Generator().manual_seed(0))   # Î / 255
```

**Training details** (`--preset diffusion`, see [`msbgen/cli`](../cli/README.md#train-train-a-model)):
the same DIV2K crops, optimiser and LR schedule as the baselines, for 40k steps,
with an **EMA** of the weights (decay 0.999) that is used for validation and
inference. `best.pt` is chosen by validation *loss* with fixed $t$ and noise, not by
PSNR, for the reason below.

A test runs the sampler with an *oracle* that returns the true $v$: DDIM must then
reproduce the original image exactly, for `eta` = 0 and 1.

### Why a sample scores lower PSNR

For any estimator, $\text{MSE} = \text{bias}^2 + \text{variance}$. The regression
networks estimate the posterior **mean** $E[R \mid B_7]$ (MMSE). A **sample** from the
true posterior is an independent draw, so its error is the posterior variance twice
over:

$$
E\lVert \hat R_\text{sample} - R \rVert^2 = 2\, E\lVert E[R \mid B_7] - R \rVert^2
$$

That is **3.01 dB below the MMSE PSNR**, even for a perfect model. The random fill is
the same effect with a uniform prior: it is 3.19 dB below the midpoint fill on
Kodak. So for the diffusion model:

- **Single-sample PSNR** should land near 18.6 − 3 ≈ 15.6 dB on Kodak, *below* the
  midpoint fill. That is not a failure.
- **The average of N samples** (`evaluate --samples N`, `psnr_avg`) should climb back
  towards the U-Net's PSNR as N grows. If it doesn't, the samples are biased, not
  just diverse.
- **Success** means lower LPIPS than the regression models (sharper, realistic
  detail), SSIM and LPIPS clearly better than the random fill (structure, not
  noise), and non-zero `diversity`.

## Checkpoints (`checkpoint.py`)

A checkpoint is a plain dict, loadable with `torch.load(weights_only=True)`, so
there is no arbitrary pickle code:

```python
{"format": "msbgen-checkpoint-v1",
 "model_kind": "regression",     # or "diffusion"; older checkpoints without it are regression
 "model_config": {...},          # enough to rebuild the model: arch, output, k, arch_args
 "state_dict": {...},            # the weights to infer with (the EMA weights when EMA is on)
 "step": 20000,
 "metrics": {"val_psnr": ..., "val_gain_db": ..., "best_val_psnr": ...},   # diffusion: best_val_loss
 "train_config": {...},          # full TrainConfig, for provenance
 "optimizer": {...}, "scheduler": {...},   # for --resume
 "train_state_dict": {...}}      # only with EMA: the raw weights --resume continues from
```

```python
from msbgen.models import load_model
model, ckpt = load_model("runs/unet-res/best.pt", device="cuda")   # MSBNet or MSBDiffusion
```

Writes are atomic (a temporary file, then rename), so an interrupted run never
leaves a corrupt checkpoint.

## Conditional flow matching (`flow.py`)

PLAN.md Milestone 4: the same task as the diffusion model, $\hat R \sim p_\theta(R \mid B_7)$,
with the same residual, conditioning, `TimeUNet` and training recipe (both subclass
`ResidualGenerator` in [`generative.py`](generative.py)). Only the way noise becomes a
residual differs, so a comparison between the two isolates that choice.

**Probability path.** A straight line between the residual and Gaussian noise
(rectified flow, the conditional optimal-transport path), with $t = 0$ the data and
$t = 1$ the noise, as in the diffusion model:

$$
x_t = (1 - t)\, r + t\, \varepsilon, \qquad
\frac{dx_t}{dt} = \varepsilon - r
$$

**Network and loss.** The `TimeUNet` gets $[x_t, B_7 \text{ planes}]$ and $t$ and
predicts the velocity: $\lVert \hat v - (\varepsilon - r) \rVert^2$, $t \sim U(0, 1)$.
Each training pair has a known straight path, but at a given $x_t$ many pairs
cross, so the network learns their *average* velocity. Integrating that field
carries the noise distribution onto $p(R \mid B_7)$. The target has variance
$1 + \operatorname{Var}(r) \approx 1.33$, so the loss's floor is well above 0 and its
values are not comparable with the diffusion loss.

**Sampling (ODE).** Start from $x_1 = z \sim \mathcal N(0, I)$ and integrate
$dx/dt = v_\theta(x, t, B_7)$ to $t = 0$ in `sample_steps` equal steps (default 3, as for
diffusion):

| `solver` | Step | Network evaluations |
| --- | --- | --- |
| `euler` (default) | $x \leftarrow x + (s - t)\, v(x, t)$ | `steps` |
| `heun` | Euler step, then the average of the velocities at both ends | 2 · `steps` − 1 (plain Euler on the last step, which ends at $t = 0$) |

Like DDIM with `eta` 0, a sample is deterministic given $z$. Straight paths can
be followed with few steps, which is the speed claim to test against diffusion
(`evaluate --sample-steps`).

```python
from msbgen.models import FlowConfig, MSBFlow

model = MSBFlow(FlowConfig(arch_args={"width": 32, "levels": 4, "blocks": 1}))
loss = model.loss(x)                                    # x: (B, 1, H, W) uint8 originals
est = model.sample(top, steps=20, solver="heun", generator=torch.Generator().manual_seed(0))   # Î / 255
```

**Training** (`--preset flow`): identical to `--preset diffusion` (40k steps, EMA
0.999, `best.pt` by validation loss). `--solver` sets the default solver, used in
validation and stored in the checkpoint. The PSNR argument of
[why a sample scores lower PSNR](#why-a-sample-scores-lower-psnr) applies unchanged.

A test runs the sampler with an *oracle* velocity $(x_t - r) / t$: with straight
paths, both solvers must land exactly on the original image for any number of steps.

## Conditional normalizing flow (`normalizing_flow.py`)

Milestone 5 models the dequantized missing residual with a conditional Glow-style
normalizing flow. This is distinct from `MSBFlow`: there is no time variable or ODE.
The forward map sends residuals to standard-normal latents and accumulates an exact
change-of-variables Jacobian for the *continuous, dequantized* data:

$$
\log p(y \mid B_7) = \log p(z) + \log\left|\det \frac{\partial z}{\partial y}\right|.
$$

**Data transform.** For each pixel, let $R=I-128B_7\in\{0,\ldots,127\}$.
Training draws $u\sim U[0,1)$, forms $q=(R+u)/128$, then uses
$y=\operatorname{logit}(q)$. The objective includes both the logit Jacobian and
the 128-level dequantization correction, and is reported in bits per residual
dimension. Validation fixes the random stream for repeatable checkpoint selection.

**Architecture and condition.** The default event is a 128×128 single-channel
residual patch. Four `pixel_unshuffle` scales feed four FrEIA `AllInOneBlock`
Glow blocks per scale. Each block combines a conditional affine coupling, channel
permutation, and global affine transform; its coupling subnet is a small CNN. A
separate CNN turns the encoded B7 plane (the existing ±0.5 bit-plane encoding)
into a matching 64, 32, 16, and 8 pixel feature pyramid. Each coupling receives
the corresponding B7 feature map as its condition. Only the residual is
transformed, so the transmitted bit is preserved when reconstructing.

FrEIA blocks are built for fixed spatial dimensions. Larger images are split into
non-overlapping 128×128 patches and modeled independently; edge padding is cropped
away after sampling. This is an explicit limitation: neighboring patches do not
share residual context, and the model likelihood assumes patchwise factorization.

**Likelihood caveat.** Uniform dequantization gives a continuous-density estimate
of discrete residual coding cost. It is not, by itself, a lossless coded bitrate.
Exact bpp requires integrated discrete-bin probabilities and an entropy coder,
which remains Milestone 8.

Sampling draws Gaussian latents once and applies the inverse flow conditioned on
B7. It has no sampler-step setting. Evaluation records the patch size and can draw
multiple seeded samples for PSNR-of-the-mean, diversity, LPIPS, DISTS, and patch KID.

## Reconstructors (`reconstructor.py`)

They wrap a checkpoint so it can be used anywhere a `Reconstructor` is expected: the
evaluation CLI, the web app and `reconstruct_and_evaluate`. `load_reconstructor`
picks the class from the checkpoint's `model_kind`.

```python
from msbgen.core import reconstruct_and_evaluate
from msbgen.models import load_reconstructor

unet = load_reconstructor("runs/unet-res/best.pt")   # ModelReconstructor; name = run folder
estimate, scores = reconstruct_and_evaluate(unet, gray_image, k=1)
scores   # {"psnr": ..., "ssim": ..., "mae": ..., "msb_consistency": ...}

diff = load_reconstructor("runs/diffusion/best.pt", steps=50, eta=0.0)   # DiffusionReconstructor
diff.sample(top, k=1, bit_depth=8, seed=3)   # a different plausible image per seed
fm = load_reconstructor("runs/flow/best.pt", steps=20, solver="heun")      # FlowReconstructor
```

Sampler options that don't apply to a checkpoint's kind are ignored (`eta` for flow
matching, `solver` for diffusion, both for regression), so callers can pass them
all. The normalizing flow has no step-count option and records its patch size.
`sampler` holds the settings in use; `evaluate` records it in `summary.json`.

**`DiffusionReconstructor`**, **`FlowReconstructor`**, and
**`NormalizingFlowReconstructor`** are `stochastic`. `reconstruct` returns the sample for
`seed` (default 0), and `sample(..., seed)` draws any other. The noise comes from a
CPU generator, so a seed gives the same image on CPU and GPU. `on_step` is called
after every network evaluation: `evaluate` and the web app pass their **thermal guard**
check there, because sampling is sustained full GPU load. Its default `tile` is
1024, not 512. The network's GroupNorm layers normalise over their whole input, so
tiles would see different statistics than a full pass. 1024 keeps Kodak-sized
images in one piece; only larger ones are tiled, at every denoising step.

**`ModelReconstructor`** (regression):

- `supports(k, bit_depth, channels)` is true only for what the model was trained
  on (here k = 1, 8-bit, grayscale). The UI and CLI skip it for anything else.
- **Tiled inference:** images larger than `tile` (512 px) are processed in tiles,
  each with `margin` (64 px) of extra context that is discarded afterwards. This
  keeps GPU memory bounded, so a 16 MP photo fits in 2 GB. The margin is smaller
  than the U-Net's receptive field, so tiled output can differ from a full-image
  pass by a tiny amount near tile borders. For the CNN it is exact (tested).
- `discover_checkpoints(root)` finds `root/*/best.pt`, newest first. The web app
  uses it on every request, so a finished run shows up without a restart.

## LPIPS (`perceptual.py`)

```python
from msbgen.models.perceptual import LPIPS
lpips = LPIPS()                      # AlexNet variant; downloads weights (~230 MB) on first use
lpips(reference, estimate, 255.0)    # lower is better
```

LPIPS compares deep network features, not pixels, so it reflects blur and missing
texture the way people judge them. Grayscale is replicated to 3 channels.

We need it because MSE-trained networks are expected to improve PSNR but look
**blurry**: they predict the average of all plausible images. LPIPS makes that
visible, and it motivates the generative models (Milestones 3–5), which should
trade some PSNR for sharper, more realistic detail. See
[why a sample scores lower PSNR](#why-a-sample-scores-lower-psnr).

## DISTS and KID (`perceptual.py`, via `piq`)

```python
from msbgen.models.perceptual import DISTS, InceptionFeatures
from msbgen.core import image_patches, kid

dists = DISTS()                         # downloads VGG16 + DISTS weights on first use
dists(reference, estimate, 255.0)       # lower is better, in [0, 1]

features = InceptionFeatures()          # FID Inception, 2048-d pool features
kid(features(image_patches(fakes, 128), 255.0), features(image_patches(reals, 128), 255.0))  # (mean, std)
```

LPIPS still compares features *at the same position*, so a sample with the right
kind of texture in slightly the wrong place is penalised much like a blurry one.
DISTS compares feature statistics as well as structure, and so it tolerates
plausible texture that is not pixel-aligned. KID drops the pairing altogether: it
compares the distribution of output patches with that of real patches. These two
metrics are where a generative model is expected to beat regression. Keep the
random fill in the table as a sanity check: its noise is "texture" too, and it
should still lose clearly.

## Tests

[`tests/test_diffusion.py`](../../tests/test_diffusion.py) covers the schedule, the
residual round trip, a repeatable loss, the DDIM sampler against an oracle, seeds and
MSB consistency, the thermal-guard hook in sampling and validation, checkpoints and
`load_reconstructor`, a tiny training run through the CLI with resume, `evaluate
--samples` and the API.

[`tests/test_models.py`](../../tests/test_models.py) covers:

- plane encoding
- arbitrary input sizes
- residual mode keeping the transmitted bits
- config validation
- the checkpoint round trip
- tiled vs full inference
- a tiny training run with resume
- the evaluate CLI and data preparation

These tests are skipped if torch isn't installed.
