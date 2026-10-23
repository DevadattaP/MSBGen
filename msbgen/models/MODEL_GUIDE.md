# MSBGen Model Guide

This guide describes how the models in this package turn a partially observed image into a reconstruction, what each model learns, and how inference works. The model implementations are in this directory; the shared training loop is in [`../training.py`](../training.py), and command-line interfaces are in [`../cli`](../cli/README.md).

## 1. The Reconstruction Problem

Let `x` be an 8-bit image and `k` the number of transmitted most-significant bits. The observed image is the original with its lower `8-k` bits cleared:

```text
 top = floor(x / 2^(8-k)) * 2^(8-k)
 residual = x - top
```

For grayscale, each pixel is a value in `[0, 255]`; RGB uses the same operation per channel. A model can only use `top`, never the discarded bits. Most MSBGen models predict the complete image or its missing residual in one pass/sample. The bitplane-wise model is different: it predicts one missing binary plane at a time, in order.

Two scales appear in the code:

- Regression `MSBNet` accepts `top` in integer image levels and returns an image normalized to `[0, 1]`.
- Generative models operate internally on a residual normalized to `[-1, 1]`, then return the reconstructed image normalized to `[0, 1]`.

## 2. Model Map

| Model | Input to backbone | Backbone output / target | Training objective | Inference |
| --- | --- | --- | --- | --- |
| `MSBNet` with `SimpleCNN` | Encoded top-k bitplanes | Full image or residual estimate | MSE by default; optional L1 | One deterministic forward pass |
| `MSBNet` with `UNet` | Encoded top-k bitplanes | Full image or residual estimate | MSE by default; optional L1 | One deterministic forward pass |
| `MSBDiffusion` | Noisy residual, top-k planes, time | Diffusion velocity `v` | Velocity MSE | DDIM denoising from Gaussian noise |
| `MSBFlow` | Interpolated residual/noise, top-k planes, time | Flow velocity | Velocity MSE | Integrate learned ODE, Euler or Heun |
| `MSBHybrid` | Residual estimate, noisy correction/residual, top-k planes and stage estimates | Residual; flow correction velocity; diffusion velocity | Optional residual-U-Net MSE + flow velocity MSE + diffusion velocity MSE | Residual U-Net -> flow correction -> truncated DDIM refinement |
| `MSBNormalizingFlow` | Dequantized residual and top-k condition | Density / likelihood | Negative log likelihood in bits per dimension | Draw latent noise and invert the flow |
| `BitplaneResidualCNN` | Current partial image | Logit for the next binary plane | Binary cross entropy with logits | Predict and add each missing plane sequentially |

`SimpleCNN`, `UNet`, and `TimeUNet` are backbones rather than full image-task models. The objective and meaning of their outputs are set by the model that wraps them.

## 3. Deterministic Regression: CNN and U-Net

### Input and encoding

`MSBNet.forward(top)` first extracts each transmitted bitplane, from most to least significant, and stacks them as channels. Bit values `0/1` are mapped to `-0.5/+0.5`. For grayscale with `k=1`, the backbone receives one channel; with `k=3`, it receives three. RGB receives `3*k` channels when configured with three channels. The model API supports this, while the repository's standard crop-training loader uses grayscale `.npy` data. A U-Net input is reflect-padded (or replicate-padded for very small dimensions) to a multiple of `2**levels`, then cropped back after inference.

### Backbones

- **SimpleCNN (`arch="cnn"`)**: 3x3 convolution, ReLU, repeated 3x3 convolution/ReLU layers at a constant width, then a 3x3 output convolution. The default width/depth is 64/8. It is fully convolutional and uses only local spatial context.
- **UNet (`arch="unet"`)**: each encoder/decoder block contains two 3x3 convolutions with ReLU. Max pooling downsamples; channel counts double at each level. Transposed convolutions upsample, encoder features are concatenated as skip connections, and a final 1x1 convolution produces the output. The default is width 24 and four downsampling levels.

Neither backbone has a sigmoid itself; `MSBNet` applies the selected output mapping.

### Output, target, and loss

For **direct** output, the backbone emits logits `z` and `sigmoid(z)` estimates the full image `x / 255`. The target is the original image divided by 255. This can predict values inconsistent with the transmitted MSBs.

For **residual** output, the same sigmoid estimates the missing residual fraction. The image estimate is:

```text
estimate = (top + sigmoid(z) * (2^(8-k) - 1)) / 255
```

The target remains the full normalized image, not a separate residual tensor. The mapping constrains the result to the interval compatible with `top`, so transmitted bits are preserved. The shared trainer minimizes mean squared error by default; `--loss l1` selects mean absolute error instead.

### Training and prediction

The standard trainer samples random 128x128 crops from prepared DIV2K by default, with random flips and right-angle rotations. It uses Adam (default learning rate `2e-4`), 16 crops per batch, gradient clipping, a short learning-rate warmup, and cosine decay. Regression checkpoints are selected by validation PSNR. At inference, quantize the input to its top `k` bits, call the model once, multiply the `[0,1]` output by 255, and round/clip if an integer image is needed.

Example:

```bash
python -m msbgen.cli.train --preset unet-res --name unet-res-k1 --k 1
```

## 4. Shared Generative Backbone: TimeUNet

Diffusion and flow matching use `TimeUNet`, not the regression U-Net. Its input channels concatenate the current one-channel residual state `x_t` with the `k` encoded bitplanes. A sinusoidal embedding of continuous time `t` is passed through an MLP. At every resolution, residual blocks apply GroupNorm, SiLU, a 3x3 convolution, add a projected time embedding, then apply another GroupNorm/SiLU/3x3 convolution and a skip connection. Strided convolutions downsample; nearest-neighbor upsampling and skip concatenation restore resolution. The final normalized activation and 3x3 convolution produce a one-channel prediction. The final convolution is initialized to zero.

`TimeUNet` has no attention. Its time and output are interpreted differently by diffusion and flow matching below.

## 5. Conditional Diffusion (`MSBDiffusion`)

### Flow Path and Target

The known top bits are held fixed. The missing residual is scaled to `r` in `[-1,1]`. For each image, training samples time `t` uniformly and Gaussian noise `epsilon`, then uses the cosine schedule to form:

```text
x_t = alpha_t * r + sigma_t * epsilon
v_target = alpha_t * epsilon - sigma_t * r
```

`TimeUNet` receives `[x_t, encoded top planes]` and `t`; its output is an estimate of `v_target`. The loss is mean squared error between predicted and target velocity, averaged over pixels and batch. This is **v-prediction**, not direct image or epsilon prediction.

### Flow Training and Sampling

The diffusion preset uses the shared DIV2K crop pipeline, 40,000 optimization steps, and an exponential moving average (EMA) with decay 0.999. The standard optimizer and schedule defaults are described above. Validation uses a fixed random stream; the best generative checkpoint is selected by validation loss, and EMA weights are saved for inference.

Sampling begins with Gaussian residual noise at `t=1`. DDIM steps move from `t=1` to `t=0`; at each step, the network predicts velocity, which is converted to a clean residual estimate. `eta=0` is deterministic for a fixed initial noise seed; positive `eta` adds noise during intermediate steps. The final residual is clipped to its legal range and added to `top`. Therefore the output always respects the transmitted bits.

```bash
python -m msbgen.cli.train --preset diffusion --name diffusion-k1 --k 1
```

## 6. Conditional Flow Matching (`MSBFlow`)

### Likelihood Target and Architecture

Flow matching uses the same scaled residual, top-plane conditioning, and `TimeUNet`. It samples Gaussian noise `epsilon` and uniform time `t`, and constructs a straight interpolation from data at `t=0` to noise at `t=1`:

```text
x_t = (1 - t) * r + t * epsilon
v_target = epsilon - r
```

The network receives `[x_t, encoded top planes]` and `t`; the loss is pixelwise/batch mean squared error against the constant path velocity. The network learns the conditional average velocity field over training pairs.

### Normalizing-Flow Training and Sampling

The flow preset uses the same crop/optimizer/step/EMA recipe as diffusion; its best checkpoint is selected by validation loss. Inference samples Gaussian noise at `t=1` and integrates `dx/dt = v(x,t,top)` toward `t=0`. Euler uses one network evaluation per step. Heun corrects an Euler step using the velocity at both ends (except for the final step). The resulting residual is clipped and added to `top`; transmitted bits remain consistent.

```bash
python -m msbgen.cli.train --preset flow --name flow-k1 --k 1
```

## 7. Three-stage hybrid (`MSBHybrid`)

The hybrid composes the distortion-first residual U-Net with conditional flow matching
and a diffusion refiner. It generates only the missing residual, so transmitted top
bits remain exact by construction:

1. The residual U-Net estimates `r0` in `[-1, 1]`.
2. Conditional flow matching models the correction `delta = r - r0`, conditioned on
	 the transmitted planes and `r0`. Its ODE starts from Gaussian noise and uses a few
	 Euler steps; adding the correction to `r0` produces `r1`.
3. A conditional diffusion refiner sees the planes, `r0`, and `r1`. It noises `r1`
	 up to `refine_t` and uses truncated DDIM to produce the final residual.

The flow loss uses the straight interpolation target `epsilon - delta`. The refiner
uses cosine-schedule v-prediction, trained on coarse `r1` samples generated by the
flow stage. The three losses are summed. With a pretrained frozen U-Net the first
stage contributes no training loss; with joint training its residual MSE is included
with weight `unet_weight`.

>- Residual U-Net: Makes a direct, deterministic estimate of the missing lower bits from the known top bits. “Residual” means it predicts the missing part relative to the known image, rather than rebuilding the whole image.
>- Flow matching correction: Adds plausible detail to that estimate. It starts with random noise and uses a learned flow to transform it into a correction that fits the image.
>- Diffusion refinement: Takes that corrected result, adds some noise, then learns to denoise it into a cleaner, more realistic reconstruction.
>
>In short: U-Net gives the stable base; flow adds plausible detail; diffusion polishes it.

The HPC runs initialize each k-specific U-Net from the corresponding `unet-res`
checkpoint and freeze it. The flow and diffusion stages train for 40k steps with EMA
0.999; checkpoints are selected by the summed validation loss, which is not comparable
with standalone-model losses. Default inference uses three flow Euler steps followed
by two DDIM steps from `t=0.5`.

```bash
python -m msbgen.cli.train --preset hybrid --name hybrid --k 1 \
	--init-unet runs/unet-res/best.pt --freeze-unet --steps 40000 --device cuda
python -m msbgen.cli.evaluate data/processed/kodak --k 1 \
	--models runs/unet-res/best.pt runs/diffusion/best.pt runs/flow/best.pt runs/hybrid/best.pt \
	--hybrid-ablation --lpips --dists --kid --samples 8 --sample-steps 3
```

`--hybrid-ablation` evaluates the complete model and two partial paths: `hybrid-unet`
(stage 1 only) and `hybrid-flow` (stages 1+2). Hybrid inference currently runs on the
whole image rather than using tiles; very large images may require more GPU memory.

### Hybrid results (Kodak)

These are the 24-image grayscale Kodak evaluations, with eight samples per stochastic
method, three flow steps, two refiner steps, `refine_t=0.5`, and DDIM `eta=0`. KID is
patch KID × 1000. Best k=2 checkpoint was selected at step 38k; k=1 and k=3 at 40k.

| k | Method | PSNR ↑ | LPIPS ↓ | DISTS ↓ | KID × 10³ ↓ | PSNR of 8-sample mean ↑ | Diversity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Residual U-Net | 18.60 | 0.4550 | 0.2931 | 79.75 | — | — |
| 1 | Sequential bitplane CNN (D4) | 18.56 | 0.4718 | 0.2872 | 93.87 | — | — |
| 1 | Flow matching | 18.11 | 0.4190 | 0.2519 | 54.92 | 18.54 | 6.15 |
| 1 | Hybrid | 18.22 | 0.4237 | 0.2594 | **46.89** | 18.61 | 5.81 |
| 2 | Residual U-Net | **26.14** | 0.2443 | 0.1554 | 41.76 | — | — |
| 2 | Sequential bitplane CNN (D4) | 25.65 | 0.2807 | 0.1657 | 54.33 | — | — |
| 2 | Flow matching | 25.60 | 0.2141 | 0.1302 | 25.34 | 26.23 | 4.03 |
| 2 | Hybrid | 25.99 | **0.2051** | **0.1249** | **17.25** | **26.60** | 3.85 |
| 3 | Residual U-Net | 32.00 | 0.1171 | 0.0852 | 19.04 | — | — |
| 3 | Sequential bitplane CNN (D4) | **32.04** | 0.1425 | 0.0943 | 21.15 | — | — |
| 3 | Flow matching | 31.51 | 0.0953 | 0.0719 | 9.01 | 32.34 | 2.42 |
| 3 | Hybrid | 31.65 | **0.0905** | **0.0681** | **5.93** | **32.60** | 2.54 |

The deterministic sequential model has no multi-sample PSNR or diversity measurement.
Its MAE is 25.03 / 10.49 / 4.94 and MSB consistency is 1.0000 at k=1/2/3. It
leads single-sample PSNR at k=3 by 0.04 dB over residual U-Net and 0.39 dB over the
hybrid, while the hybrid has better LPIPS, DISTS and KID.

At k=1 the hybrid improves KID over the standalone generative models, but not DISTS
over flow matching. At k=2 and k=3 it improves LPIPS, DISTS and KID over both
standalone generative baselines; the eight-sample mean also has the highest PSNR.
Single-sample PSNR remains below residual U-Net. At k=3, the flow-only ablation has
31.69 dB / 0.0701 DISTS / 7.38 KID, while full refinement has 31.65 dB / 0.0681 /
5.93: a modest perceptual improvement for a small PSNR cost. This is a single set of
training runs on 24 Kodak images; multi-seed and broader-dataset confirmation remains
necessary.

## 8. Conditional Normalizing Flow (`MSBNormalizingFlow`)

### Training input and target

This model learns a conditional density for the missing residual, rather than a denoising or velocity function. For discrete residual `R` with `L = 2^(8-k)` possible values, training draws uniform dequantization noise `u` and transforms:

```text
q = (R + u) / L
 y = logit(q)
```

The encoded top planes are processed by a convolutional condition pyramid. The transformed residual is passed through a multi-scale invertible Glow-style flow. At each scale, `pixel_unshuffle` trades spatial resolution for channels; a sequence of FrEIA `AllInOneBlock`s applies conditional affine coupling and learned mixing/affine transforms. Coupling subnetworks use 3x3 convolutions and SiLU activations. Intermediate latent channels are factored out; the final latent has a standard normal prior.

The loss is the negative log likelihood of the dequantized residual, including the standard-normal latent density, flow log-determinants, logit-transform Jacobian, and the discrete-level correction. It is reported in bits per residual dimension. This is a density-based training objective; it is not itself a lossless coded bitrate.

### Training and inference

The normalizing-flow preset uses the shared trainer and selects checkpoints by validation negative log likelihood. The flow requires fixed spatial dimensions, so images are split into non-overlapping patches (default 128x128); patches are conditionally independent and do not exchange context. For inference, draw standard-normal latents, invert the flow conditioned on the top-bit features, map from logit space back to the residual range, and add it to the top-bit image.

```bash
python -m msbgen.cli.train --preset normalizing-flow --name nf-k1 --k 1
```

## 9. Sequential Bitplane CNN (`BitplaneResidualCNN`)

This is the paper-style bit-depth recovery path. It differs from MSBNet: each network predicts **one binary plane**, and several separately trained networks are chained to recover the full image.

### Training input, target, and architecture

For a training image, the input is quantized to `input_bits` while retaining the full-depth numeric scale. The target is the next lower bitplane as a binary tensor. For example, an 8-bit input quantized to 3 bits is paired with the `B4` target; the next model takes the reconstructed 4-bit image and targets `B3`.

The model starts with a 3x3 convolution from image channels to 64 features. Each D-depth residual block applies BatchNorm, 3x3 convolution, BatchNorm, ReLU, 3x3 convolution, then adds the block input. A final BatchNorm and 3x3 convolution produce one logit per output channel/pixel. `sigmoid(logits)` gives the plane probability.

Training uses `BCEWithLogitsLoss` against the binary target. Defaults are Adam at `1e-3`, batch size 128, 48x48 non-overlapping patches, and 30 epochs; the learning rate is divided by five after epoch 16. `--augment` enables random horizontal flips and rotations by multiples of 90 degrees. The trainer can use a validation set and thermal guard.

### Sequential prediction

Start with the quantized top-bit image. Load the model for that input bit count, apply sigmoid, and add its plane probability weighted by that bit's place value. Use the updated image as input to the model for the next lower plane, continuing until the desired bit depth is reached. The implementation keeps soft probabilities rather than thresholding each prediction to 0/1. Each step therefore contributes an expected plane value and the final image is clipped to `[0,1]` after normalization.

Train each required input depth separately. For recovery from `k` bits to 8 bits, checkpoints are needed for input depths `k` through 7:

```bash
python -m msbgen.cli.bitdepth train --input-bits 1 --train data/processed/div2k-train --out runs/bitdepth-d4 --augment
python -m msbgen.cli.bitdepth evaluate --models runs/bitdepth-d4 --test data/processed/kodak --start-bits 1 2 3 --save-images
```

## 10. Shared Training, Checkpoints, and Inference

The standard training CLI is `python -m msbgen.cli.train`. Its regression presets are `cnn`, `unet`, and `unet-res`; generative presets are `diffusion`, `flow`, `normalizing-flow`, and `hybrid`. Prepared training/validation arrays default to `data/processed/div2k-train` and `data/processed/div2k-valid`. A run directory contains configuration, JSONL history, `last.pt` for resume, and `best.pt` for inference.

Standard checkpoints record model kind, configuration, inference weights, step, metrics, and training configuration. With EMA, they also retain raw training weights for resuming. `load_model()` reconstructs the PyTorch module directly; `load_reconstructor()` wraps it for the evaluation CLI, API, or web application and handles image scaling, sampling seeds, and tiling for large inputs.

The bitplane CLI has a separate checkpoint format and saves one `bit_XX.pt` per input depth plus epoch history JSON. Its `recover_bitplanes()` function performs the sequential inference described above.

## 11. Evaluation Networks (Not Trained MSBGen Models)

`perceptual.py` contains frozen pretrained networks used only for scoring outputs, not for producing reconstructions:

- **LPIPS** compares deep AlexNet features; lower is more perceptually similar.
- **DISTS** compares VGG feature statistics and structure; lower is better.
- **InceptionFeatures** extracts features for set-level KID; these features are not a per-image reconstruction loss.

These metrics are opt-in in evaluation. LPIPS requires the `lpips` package; DISTS and Inception/KID use `piq`.

## In short

Not all of them use only convolutions. Also, the models don’t need to enlarge the image to construct it: they receive a full-resolution top-bit image and predict an output at the same height and width. The convolutions use padding and stride 1, so each output pixel is estimated from nearby input/features.

- **CNN regression:** no downsampling or upsampling. Same-resolution convolutions produce a full-resolution image estimate (or missing residual), pixel by pixel.
- **U-Net regression:** does downsample with max pooling, then upsamples with ConvTranspose2d. Skip connections bring fine-resolution features back into the decoder.
- **Diffusion/flow TimeUNet:** also has a down/up path, but upsamples with nearest-neighbor interpolation followed by convolution, rather than transposed convolution. Its output is a full-resolution residual prediction, velocity, or noise-related estimate; sampling turns that into the reconstructed image.
- **Sequential bitplane CNN:** keeps the image dimensions unchanged. It predicts the next bitplane at each pixel, scales that plane by its bit significance, and adds it before predicting the next plane.
- **Normalizing flow:** reconstructs residual patches by inverting learned transformations; it uses internal pixel shuffle/unshuffle operations, not a U-Net-style decoder.

So, there is learned upsampling in the regression U-Net, and interpolation-based upsampling in TimeUNet. Plain CNN and bitplane CNN models predict directly at the input’s resolution.

The missing information is in the pixel values, not in empty pixel locations. The input is still a complete image grid; its lower bit values have been cleared.

The model processes that grid through learned filters and produces one estimate per pixel (and channel) at the original resolution. Training compares its reconstructed image with the original target and adjusts the filters to reduce the error. The CNN mostly uses nearby pixels; the U-Net also combines broader context through downsampling, upsampling, and skip connections.

There are two regression output modes:

- Direct: the final output is the estimated full image, compared with the original image.
- Residual: the output estimates only the missing value range; the known top bits are added back, and that reconstructed full image is compared with the original.

Diffusion, flow matching, and the sequential bitplane model differ: they train against noise/velocity or a binary bitplane target rather than directly comparing a full-image output at every training step.
