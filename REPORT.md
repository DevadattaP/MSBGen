# Bit-Depth Recovery and MSBGen: Short Report

## Previous Work

The reference method, [A Little Bit More: Bitplane-Wise Bit-Depth Recovery](https://arxiv.org/abs/2005.01091), with the official implementation at [GitHub](https://github.com/abhijithpunnappurath/a-little-bit-more), predicts missing lower bitplanes sequentially: reconstruct one plane, add it to the image, then predict the next. We ported its residual-CNN approach to PyTorch and converted the authors' released D4/D16 RGB checkpoints from Keras to evaluate them on Kodak.

MSBGen studies a related but harder task: reconstructing a complete grayscale image from only its top `k` bits (`k=1,2,3`). It compares CNN/U-Net regressors, conditional diffusion and flow matching, a three-stage residual U-Net + flow matching + diffusion hybrid, and Vision Transformer (ViT) variants of the regressor and the hybrid. The paper-style sequential model and the MSBGen full-image models are distinct experiments.

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

## Hybrid Results

The hybrid runs initialize stage 1 from the matching residual U-Net checkpoint and
freeze it. Conditional flow matching models a residual correction, and a truncated
diffusion stage refines the coarse result. The checkpoints trained for 40k steps
(best k=2 checkpoint selected at step 38k), with batch size 16, 128×128 DIV2K crops
and EMA 0.999. Evaluation uses the same 24 grayscale Kodak images, eight stochastic
samples, three flow steps, two DDIM refinement steps, `refine_t=0.5`, and `eta=0`.
KID is patch KID × 1000.

| k | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | KID × 10³ ↓ | PSNR avg ↑ | Diversity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Residual U-Net | 18.60 | 0.6373 | 0.4550 | 0.2931 | 79.75 | — | — |
| 1 | Sequential bitplane CNN (D4) | 18.56 | 0.6328 | 0.4718 | 0.2872 | 93.87 | — | — |
| 1 | Diffusion | 17.75 | 0.6101 | 0.4245 | 0.2567 | 54.61 | 18.20 | 6.08 |
| 1 | Flow matching | 18.11 | 0.6151 | 0.4190 | **0.2519** | 54.92 | 18.54 | 6.15 |
| 1 | Hybrid | 18.22 | 0.6242 | 0.4237 | 0.2594 | **46.89** | 18.61 | 5.81 |
| 2 | Residual U-Net | **26.14** | **0.8044** | 0.2443 | 0.1554 | 41.76 | — | — |
| 2 | Sequential bitplane CNN (D4) | 25.65 | 0.7976 | 0.2807 | 0.1657 | 54.33 | — | — |
| 2 | Diffusion | 25.63 | 0.7885 | 0.2134 | 0.1292 | 26.06 | 26.24 | 3.97 |
| 2 | Flow matching | 25.60 | 0.7902 | 0.2141 | 0.1302 | 25.34 | 26.23 | 4.03 |
| 2 | Hybrid | 25.99 | 0.7936 | **0.2051** | **0.1249** | **17.25** | **26.60** | 3.85 |
| 3 | Residual U-Net | 32.00 | **0.9015** | 0.1171 | 0.0852 | 19.04 | — | — |
| 3 | Sequential bitplane CNN (D4) | **32.04** | **0.9015** | 0.1425 | 0.0943 | 21.15 | — | — |
| 3 | Diffusion | 31.58 | 0.8951 | 0.0941 | 0.0717 | 8.56 | 32.37 | 2.35 |
| 3 | Flow matching | 31.51 | 0.8944 | 0.0953 | 0.0719 | 9.01 | 32.34 | 2.42 |
| 3 | Hybrid | 31.65 | 0.8930 | **0.0905** | **0.0681** | **5.93** | **32.60** | 2.54 |

The sequential bitplane CNN has MAE 25.03 / 10.49 / 4.94 and MSB consistency 1.0000
at k=1/2/3. Its k=3 PSNR (32.04 dB) remains slightly above the hybrid (31.65 dB)
and residual U-Net (32.00 dB), but the hybrid has better LPIPS, DISTS and KID.
`PSNR avg` and diversity are not applicable to deterministic methods.

At k=2 and k=3, the hybrid improves LPIPS, DISTS and KID over the standalone
generative models; its eight-sample mean PSNR is also highest. The single-sample
PSNR remains below residual U-Net. At k=1 the hybrid has the best KID but flow
matching has better DISTS. The k=3 flow-only ablation (stages 1+2) gives 31.69 dB,
0.0701 DISTS and 7.38 KID; the full hybrid gives 31.65 dB, 0.0681 and 5.93. This
supports a modest perceptual gain from refinement with a small PSNR trade-off.
These are single-run results on 24 Kodak images and do not establish generalization.
Evaluation artifacts are in `outputs/eval/kodak-hybrid-k1/`, `...-k2/`, and
`...-k3/`; paired and metric plots are in `outputs/plots/hybrid-k1/` through
`hybrid-k3/`.

## ViT Results

Two ViT variants were trained for k=1/2/3 under the same protocol (DIV2K, 40k steps, batch 16, 128×128 crops, same Kodak evaluation). `ViT` is a residual ViT regressor (8×8 patches, 192-d tokens, 6 blocks, 6 heads, conv fusion head, 2.9 M parameters). `ViT hybrid` replaces the residual U-Net in stage 1 by this ViT (10.9 M parameters). The ViT hybrid's stage 1 was trained jointly from scratch, not initialized from `vit` and not frozen, unlike the U-Net hybrid. Best checkpoints: ViT at step 10k/22k/30k, ViT hybrid at 36k/40k/40k (k=1/2/3).

| k | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | KID × 10³ ↓ | PSNR avg ↑ | Diversity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Residual U-Net | 18.60 | **0.6373** | 0.4550 | 0.2931 | 79.75 | — | — |
| 1 | Hybrid (U-Net) | 18.22 | 0.6242 | **0.4237** | **0.2594** | **46.89** | 18.61 | 5.81 |
| 1 | ViT | 18.35 | 0.6206 | 0.4662 | 0.2932 | 89.81 | — | — |
| 1 | ViT hybrid | **18.64** | 0.6230 | 0.4293 | 0.2618 | 49.17 | **18.83** | 5.07 |
| 2 | Residual U-Net | **26.14** | **0.8044** | 0.2443 | 0.1554 | 41.76 | — | — |
| 2 | Hybrid (U-Net) | 25.99 | 0.7936 | **0.2051** | **0.1249** | **17.25** | **26.60** | 3.85 |
| 2 | ViT | 24.51 | 0.7777 | 0.2676 | 0.1644 | 48.19 | — | — |
| 2 | ViT hybrid | 24.71 | 0.7698 | 0.2251 | 0.1314 | 19.46 | 25.30 | 4.27 |
| 3 | Residual U-Net | 32.00 | **0.9015** | 0.1171 | 0.0852 | 19.04 | — | — |
| 3 | Hybrid (U-Net) | 31.65 | 0.8930 | **0.0905** | **0.0681** | **5.93** | **32.60** | 2.54 |
| 3 | ViT | 30.78 | 0.8845 | 0.1487 | 0.0976 | 23.40 | — | — |
| 3 | ViT hybrid | 31.21 | 0.8896 | 0.0951 | 0.0700 | 7.12 | 32.10 | 2.59 |

The standalone ViT is below the residual U-Net at every k (−0.25 / −1.63 / −1.22 dB) and below the plain CNN at k=2 and k=3. Adding flow matching and diffusion helps it a lot perceptually (at k=3, LPIPS 0.1496 → 0.0951, DISTS 0.0975 → 0.0700, KID 22.90 → 7.12 versus the ViT hybrid's own stage 1). At k=1 the ViT hybrid has the highest single-sample PSNR of all methods (18.64 dB) and highest eight-sample-mean PSNR (18.83 dB), but slightly worse DISTS/KID than the U-Net hybrid. At k=2 and k=3 it trails the U-Net hybrid on all metrics. Inference takes about 0.018 s for ViT and 0.114 s for the ViT hybrid (residual U-Net 0.006 s, U-Net hybrid 0.102 s). These are single runs on 24 images with untuned ViT hyperparameters and different stage-1 training protocols, so the comparison with the U-Net hybrid is indicative only.
Artifacts: `outputs/eval/kodak-vit-k1/` … `kodak-vit-hybrid-k3/`; plots in `outputs/plots/vit-k1/` … `vit-hybrid-k3/`.

## Metric Notes

- The earlier `kodak-all-k1/2/3` summaries did not measure LPIPS, DISTS or KID. The final common-evaluator run measured them for D4 and the other models.
- PSNR avg and diversity need multiple stochastic samples; they are intentionally blank for deterministic models. The final evaluation used eight samples for stochastic methods.
- The standalone bitplane CLI still writes PSNR, SSIM and MAE; the shared `msbgen.cli.evaluate` path now evaluates D4 with the same LPIPS, DISTS, KID and MSB-consistency metrics as the other methods.

## Gaps and Next Steps

- The pretrained RGB benchmark uses released weights on Kodak; it is not a full reproduction of the paper's training data or protocol. The grayscale experiments use DIV2K and should not be presented as directly comparable RGB results.
- CNN/U-Net grayscale regressors do not always preserve the transmitted bits exactly (MSB consistency is below 1); residual U-Net and the generative methods reported 1.0.
- The hybrid and baseline comparisons are single training runs on Kodak. Repeat multiple seeds, report paired uncertainty, and validate on additional image classes/datasets before general claims.
- ViT hyperparameters (patch size, width, depth, learning rate, EMA, warm-starting the ViT hybrid from a trained ViT) were not tuned; try a frozen/warm-started ViT stage 1 for a like-for-like comparison with the U-Net hybrid.
- Hybrid inference currently runs on the whole image rather than using tiled inference; assess memory and latency at larger resolutions.
- Direct CNN/U-Net regressors have small MSB-consistency errors; test hard projection and whether it affects image metrics.
- Complete controlled comparisons of sequential bitplane recovery against full-image reconstruction, keeping RGB pretrained and grayscale trained results clearly separated.
- The entropy coder is a correctness/reference implementation. Improve context modeling and include container/side-information costs before claiming production codec rates.
