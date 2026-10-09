# Bit-Depth Recovery and MSBGen: Short Report

## Previous Work

The reference method, [A Little Bit More: Bitplane-Wise Bit-Depth Recovery](https://arxiv.org/abs/2005.01091), with the official implementation at [GitHub](https://github.com/abhijithpunnappurath/a-little-bit-more), predicts missing lower bitplanes sequentially: reconstruct one plane, add it to the image, then predict the next. We ported its residual-CNN approach to PyTorch and converted the authors' released D4/D16 RGB checkpoints from Keras to evaluate them on Kodak.

MSBGen studies a related but harder task: reconstructing a complete grayscale image from only its top `k` bits (`k=1,2,3`). It compares direct CNN/U-Net regressors and conditional diffusion/flow models. The paper-style sequential model and the MSBGen full-image models are distinct experiments.

## RGB Pretrained Results

Mean over 24 RGB Kodak images. These are evaluations of the authors' pretrained checkpoints, not models retrained here.

| Input bits -> 8 | D4 PSNR | D16 PSNR | D16 SSIM |
| --- | ---: | ---: | ---: |
| 3 -> 8 | 33.13 | 33.48 | 0.9349 |
| 4 -> 8 | 39.34 | 39.53 | 0.9723 |
| 5 -> 8 | 44.46 | 44.65 | 0.9880 |
| 6 -> 8 | 49.21 | 49.29 | 0.9954 |
| 7 -> 8 | 54.93 | 54.99 | 0.9987 |

All cases preserved the transmitted top bits. More input bits had a much larger effect than increasing depth from D4 to D16.

## Grayscale Kodak Results

Mean PSNR (dB) over 24 images; each model reconstructs the full 8-bit image from the indicated number of top bits. CNN/U-Net/Res U-Net are deterministic. Diffusion and flow columns are single-sample results.

| Method | k=1 | k=2 | k=3 |
| --- | ---: | ---: | ---: |
| Midpoint fill | 17.01 | 22.63 | 28.74 |
| CNN | 18.04 | 25.13 | 31.10 |
| U-Net | 18.55 | 25.90 | 30.87 |
| Residual U-Net | **18.60** | **26.14** | 32.00 |
| Sequential bitplane CNN (D4) | 18.56 | 25.65 | **32.04** |
| Diffusion | 17.75 | 25.63 | 31.58 |
| Flow matching | 18.11 | 25.60 | 31.51 |

With four generated samples, diffusion/flow PSNR of the sample mean was 18.12/18.47 dB at k=1, 26.16/26.16 at k=2, and 32.24/32.22 at k=3. Residual U-Net is strongest or competitive on single-sample distortion; the generative sample means become competitive as more bits are given. The sequential bitplane CNN is close to the best regressors and reaches the highest single-result PSNR at k=3.

## Why Metrics Are Empty

- The general evaluator computes LPIPS and DISTS only when run with `--lpips --dists` (requiring the `lpips` and `piq` packages). The `kodak-all-k1/2/3` summaries have both set to `null`, so those metrics were not measured in these runs. KID is also opt-in via `--kid`.
- PSNR avg and diversity need multiple stochastic samples; they are intentionally blank for deterministic models. Here they are present for stochastic methods because the run used four samples.
- The bitplane CLI writes only PSNR, SSIM, and MAE to `outputs/bitdepth/kodak/evaluation.csv`; it does not currently calculate LPIPS, DISTS, KID, or MSB consistency.

## Gaps and Next Steps

- The pretrained RGB benchmark uses released weights on Kodak; it is not a full reproduction of the paper's training data or protocol. The grayscale experiments use DIV2K and should not be presented as directly comparable RGB results.
- CNN/U-Net grayscale regressors do not always preserve the transmitted bits exactly (MSB consistency is below 1); residual U-Net and the generative methods reported 1.0.
- Re-run grayscale evaluation with LPIPS/DISTS (and KID if useful); add corresponding optional metrics to the bitplane evaluator for a fairer comparison.
- Locate or restore the sequential model checkpoints and training histories; they are not currently alongside the evaluation CSV/images. Then repeat across seeds and report uncertainty. Test a hard MSB-consistency projection for direct regressors.
- Complete controlled comparisons of sequential bitplane recovery against full-image reconstruction, keeping RGB pretrained and grayscale trained results clearly separated.
