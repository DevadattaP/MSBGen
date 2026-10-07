# Bit-Depth Recovery : Initial report

## 1. Scope

This report records two related experiments:

1. A reproduction of **A Little Bit More: Bitplane-Wise Bit-Depth Recovery** by Punnappurath and Brown, using the authors' released TensorFlow/Keras checkpoints converted to PyTorch.
2. The MSBGen grayscale experiments, which study reconstruction from only the most significant bitplanes using deterministic regressors and stochastic generative models.

The paper and the MSBGen project address different operating points. The paper reconstructs an image after keeping several **top bitplanes** and predicts the remaining lower planes. MSBGen additionally studies the much more ambiguous one-plane case, especially reconstruction from only `B7`.

## 2. Reference Paper

The reference is [A Little Bit More: Bitplane-Wise Bit-Depth Recovery](https://arxiv.org/abs/2005.01091), with the official implementation at [a-little-bit-more](https://github.com/abhijithpunnappurath/a-little-bit-more).

Given an `N`-bit image and a `q`-bit quantized input, the method trains one independent CNN for each missing bitplane:

```text
I_q -> predict B_(N-q-1) -> add its bit value
    -> predict the next lower bitplane -> ... -> reconstruct I_N
```

Each network receives the currently reconstructed image and predicts one binary RGB plane. Training uses binary cross-entropy. At inference, the predicted plane is weighted by its place value and added before the next network is applied.

### Published network and training setup

- RGB input and RGB output
- 64 convolution filters throughout
- 3x3 convolutions with same padding
- D4 or D16 residual-block depth
- Residual block: BatchNorm, convolution, BatchNorm, ReLU, convolution, skip addition
- Final BatchNorm, convolution, and sigmoid
- Independent model for each missing bitplane
- 48x48 non-overlapping training patches
- Random flips and 90-degree rotations
- Adam, learning rate `1e-3`, batch size `128`
- 30 epochs
- Learning rate divided by 5 after epoch 16
- Training data: 1,000 Sintel images plus 1,000 MIT-Adobe FiveK images

The paper reports PSNR and SSIM over Sintel, MIT-Adobe FiveK, TESTIMAGES, Kodak, and ESPL v2.

## 3. PyTorch Reproduction

The original implementation is in TensorFlow/Keras. The authors released pretrained checkpoints for 4->8 through 16->8 RGB recovery. The checkpoints are in HDF5 format and contain Conv2D kernels, BatchNorm parameters, and optimizer state.

I have ported this to PyTorch. It provides:

- the paper-style residual CNN;
- quantization and binary-plane target generation;
- 48x48 patch datasets;
- BCE-with-logits training;
- sequential bitplane recovery;
- GPU training with configurable DataLoader workers and thermal protection.

To start with I created evaluation wrapper around existing pretrained models. It:

1. Reads the authors' `model_030.hdf5` checkpoints.
2. Converts Keras Conv2D kernels from `H x W x Cin x Cout` to PyTorch `Cout x Cin x H x W`.
3. Converts Keras BatchNorm gamma, beta, moving mean, and moving variance.
4. Saves converted PyTorch checkpoints beside the HDF5 files.
5. Runs RGB Kodak evaluation.
6. Writes per-image metrics, aggregate JSON, reconstructed images, and a plot.

The released checkpoints contain `bitpos_04` through `bitpos_16`. For 8-bit Kodak evaluation, the available published models support 3->8 and higher input depths. The experiments below use 3->8, 4->8, 5->8, 6->8, and 7->8.

The evaluation used all 24 RGB Kodak images. Metrics are computed against the original 8-bit RGB images. `MSB consistency` measures whether rounding the estimate preserves the transmitted top bits.

## 4. Pretrained RGB Kodak Results

Values are means over 24 images. Lower MAE is better. D4 and D16 are the number of residual blocks in each independent plane model.

| Model | Recovery | PSNR (dB) | SSIM | MAE | MSB consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| D4 | 3->8 | 33.126 | 0.93289 | 4.216 | 1.0000 |
| D16 | 3->8 | **33.476** | **0.93490** | **4.024** | 1.0000 |
| D4 | 4->8 | 39.337 | 0.9710 | 2.097 | 1.0000 |
| D16 | 4->8 | **39.529** | **0.9723** | **2.044** | 1.0000 |
| D4 | 5->8 | 44.457 | 0.98755 | 1.179 | 1.0000 |
| D16 | 5->8 | **44.651** | **0.98802** | **1.151** | 1.0000 |
| D4 | 6->8 | 49.213 | 0.99534 | 0.709 | 1.0000 |
| D16 | 6->8 | **49.291** | **0.99545** | **0.699** | 1.0000 |
| D4 | 7->8 | 54.927 | 0.99865 | 0.400 | 1.0000 |
| D16 | 7->8 | **54.989** | **0.99867** | **0.393** | 1.0000 |

The D16 improvement over D4 is small on Kodak, approximately `0.06-0.19 dB` depending on the input depth. The largest effect comes from transmitting more bitplanes: moving from 5->8 to 7->8 improves PSNR by approximately 10.3 dB for both model depths.

## 5. Simple MSBGen Grayscale Experiments

The simple MSBGen experiments use 8-bit grayscale Kodak images and compare:

- zero fill;
- midpoint fill;
- random fill;
- direct CNN regression;
- direct U-Net regression;
- residual U-Net regression;
- conditional diffusion;
- conditional flow matching;
- conditional normalizing flow.

The grayscale deterministic and generative models are trained on DIV2K grayscale crops and evaluated on the 24-image grayscale Kodak set. The generative models predict a distribution of missing residual values rather than a single deterministic estimate.

### 5.1 One transmitted plane: grayscale `k=1`

The input is only `B7`; seven lower bits are missing.

| Method | PSNR (dB) | SSIM | MAE | DISTS | MSB consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero fill | 10.369 | 0.2381 | 69.104 | 0.3710 | 1.0000 |
| Midpoint fill | 17.007 | 0.5793 | 31.764 | 0.3635 | 1.0000 |
| Random fill | 13.816 | 0.1289 | 42.684 | 0.3852 | 1.0000 |
| CNN | 18.039 | 0.6167 | 27.321 | 0.2809 | 0.9995 |
| U-Net | 18.553 | 0.6371 | 25.474 | 0.2966 | 0.9986 |
| Residual U-Net | **18.599** | **0.6373** | **25.339** | 0.2931 | 1.0000 |
| Diffusion | 16.326 | 0.4801 | 32.192 | **0.2689** | 1.0000 |
| Flow matching | 16.451 | 0.4986 | 31.364 | **0.2638** | 1.0000 |
| Normalizing flow | 15.765 | 0.4096 | 33.612 | 0.3163 | 1.0000 |

For stochastic models, the sample mean is more informative for distortion metrics. With eight diffusion samples, the mean PSNR was `17.853 dB`; with eight flow samples it was `18.148 dB`.

The one-plane result demonstrates the core MSBGen problem: many grayscale images are compatible with the same MSB plane. Deterministic models maximize average fidelity and therefore produce smooth reconstructions, while generative models can produce plausible variation but may score lower on single-sample PSNR.

### 5.2 Two transmitted planes: grayscale `k=2`

The input contains `B7` and `B6`; six lower bits are missing.

| Method | PSNR (dB) | SSIM | MAE | DISTS | MSB consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Midpoint fill | 22.630 | 0.7033 | 16.462 | 0.2154 | 1.0000 |
| CNN | 25.132 | 0.7846 | 11.546 | 0.1612 | 0.9970 |
| U-Net | 25.903 | 0.7983 | 10.315 | 0.1587 | 0.9935 |
| Residual U-Net | **26.141** | **0.8044** | **10.099** | 0.1554 | 1.0000 |
| Diffusion | 25.627 | 0.7885 | 10.540 | **0.1292** | 1.0000 |
| Flow matching | 25.602 | 0.7902 | 10.592 | 0.1302 | 1.0000 |

The generative models have lower DISTS than the deterministic regressors, while the residual U-Net remains strongest on single-sample PSNR and SSIM. The eight-sample PSNR averages are approximately `26.237 dB` for diffusion and `26.233 dB` for flow matching.

### 5.3 Three transmitted planes: grayscale `k=3`

The input contains `B7`, `B6`, and `B5`; five lower bits are missing.

| Method | PSNR (dB) | SSIM | MAE | DISTS | MSB consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Midpoint fill | 28.740 | 0.8237 | 8.126 | 0.1286 | 1.0000 |
| CNN | 31.104 | 0.8915 | 5.784 | 0.0940 | 0.9914 |
| U-Net | 30.872 | 0.8849 | 5.810 | 0.0992 | 0.9857 |
| Residual U-Net | **31.998** | **0.9015** | **5.099** | 0.0852 | 1.0000 |
| Diffusion | 31.578 | 0.8951 | 5.238 | **0.0717** | 1.0000 |
| Flow matching | 31.513 | 0.8944 | 5.278 | 0.0719 | 1.0000 |

At `k=3`, the conditional generative models become competitive with deterministic regression on PSNR and SSIM while achieving substantially better DISTS. The eight-sample PSNR averages are `32.366 dB` for diffusion and `32.342 dB` for flow matching.

## 6. Grayscale Bitplane-Wise Training Status

I also implemented a grayscale PyTorch version of the paper's independent bitplane training procedure. The execution is in progress.

This experiment is separate from the authors' RGB pretrained checkpoints. It uses the local DIV2K grayscale preparation and trains one model for each input depth. For an 8-bit target, the planned models are:

```text
input_bits=1 -> predicts the next missing plane
input_bits=2 -> predicts the next missing plane
...
input_bits=7 -> predicts B0
```

## 7. Main Findings

1. The published RGB bitplane-wise method performs very strongly when five to seven top bits are already available. On Kodak, D16 reaches `54.989 dB` for 7->8 recovery.
2. D16 is consistently, but only slightly, better than D4 on Kodak. The number of transmitted bitplanes has a much larger effect than increasing residual depth.
3. For grayscale MSB-only reconstruction, deterministic residual U-Net gives the best single-sample PSNR and perfect transmitted-bit consistency.
4. Diffusion and flow matching are not expected to win single-sample PSNR in a one-to-many problem. Their strongest evidence is lower DISTS, nonzero sample diversity, and improving PSNR of the sample mean.
5. As more MSB planes are transmitted, the generative models become more competitive in both distortion and perceptual metrics.

## 8. Next Steps

1. Complete the grayscale bitplane-wise training and evaluation.
2. Study the effect of residual depth on grayscale bitplane-wise recovery vs MSB-only recovery.
3. Design conditional generative model using diffusion and flow matching for bitplane-wise recovery.
