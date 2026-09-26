# `msbgen.models`

Neural reconstructors, i.e. the trained counterparts of the constant fills in
[`msbgen.core.reconstruction`](../core/README.md#reconstructors-reconstructionpy).
This package needs PyTorch. `msbgen.core` never imports it, so the web app and the
analysis tools still work without torch installed.

| Module | What it does |
| --- | --- |
| [`nets.py`](nets.py) | Backbones: `SimpleCNN` and `UNet` |
| [`msb_net.py`](msb_net.py) | `MSBNet`: backbone + input encoding + output mode; `ModelConfig` |
| [`checkpoint.py`](checkpoint.py) | `save_checkpoint`, `load_model`, `read_checkpoint` |
| [`reconstructor.py`](reconstructor.py) | `ModelReconstructor` (a checkpoint as a `Reconstructor`, tiled inference), `discover_checkpoints` |
| [`perceptual.py`](perceptual.py) | `LPIPS` perceptual metric wrapper |

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

Mixed precision (AMP) was *slower* on this Pascal-generation GPU, so training runs
in fp32. On a modern GPU, larger widths and batches are fine: pass `--width` and
`--batch-size`.

## Checkpoints (`checkpoint.py`)

A checkpoint is a plain dict, loadable with `torch.load(weights_only=True)`, so
there is no arbitrary pickle code:

```python
{"format": "msbgen-checkpoint-v1",
 "model_config": {...},          # enough to rebuild the model: arch, output, k, arch_args
 "state_dict": {...},
 "step": 20000,
 "metrics": {"val_psnr": ..., "val_gain_db": ..., "best_val_psnr": ...},
 "train_config": {...},          # full TrainConfig, for provenance
 "optimizer": {...}, "scheduler": {...}}   # for --resume
```

```python
from msbgen.models import load_model
model, ckpt = load_model("runs/unet-res/best.pt", device="cuda")
```

Writes are atomic (a temporary file, then rename), so an interrupted run never
leaves a corrupt checkpoint.

## `ModelReconstructor`

It wraps a checkpoint so it can be used anywhere a `Reconstructor` is expected: the
evaluation CLI, the web app and `reconstruct_and_evaluate`.

```python
from msbgen.core import reconstruct_and_evaluate
from msbgen.models import ModelReconstructor

unet = ModelReconstructor("runs/unet-res/best.pt")   # name = run folder ("unet-res")
estimate, scores = reconstruct_and_evaluate(unet, gray_image, k=1)
scores   # {"psnr": ..., "ssim": ..., "mae": ..., "msb_consistency": ...}
```

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
trade some PSNR for sharper, more realistic detail.

## Tests

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
