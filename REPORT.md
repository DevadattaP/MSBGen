# Bit-Depth Recovery and MSBGen: Short Report

## Previous Work

The reference method, [A Little Bit More: Bitplane-Wise Bit-Depth Recovery](https://arxiv.org/abs/2005.01091), with the official implementation at [GitHub](https://github.com/abhijithpunnappurath/a-little-bit-more), predicts missing lower bitplanes sequentially: reconstruct one plane, add it to the image, then predict the next. We ported its residual-CNN approach to PyTorch and converted the authors' released D4/D16 RGB checkpoints from Keras to evaluate them on Kodak.

MSBGen studies a related but harder task: reconstructing a complete grayscale or RGB image from only its top `k` bits per channel (`k=1,2,3`). It compares CNN/U-Net regressors, conditional diffusion and flow matching, conditional normalizing flows, a three-stage residual U-Net + flow matching + diffusion hybrid, and Vision Transformer (ViT) variants of the regressor and the hybrid. The paper-style sequential model and the MSBGen full-image models are distinct experiments.

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

## RGB Kodak Results

These are full-resolution reconstructions evaluated on all 24 RGB Kodak images. The models were trained on the RGB DIV2K training split and validated on RGB DIV2K validation images. Each `k` is the number of most-significant bitplanes retained **per channel** (three channels, so the raw input contains `3k` binary planes per pixel). The tables report means over images; KID is patch KID × 1,000. `MSB ok` is the fraction of pixels whose reconstruction preserves the transmitted top bits. Stochastic methods use eight samples for `PSNR avg` and diversity; their first sample supplies the other per-image metrics. Sampling used three steps. `—` means not applicable to deterministic methods. The `hybrid-*` rows are ablations; `*-unet` is the hybrid's stage 1 and `*-flow` is stages 1+2.

### k = 1

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | MAE ↓ | MSB ok | KID × 10³ ↓ | PSNR avg ↑ | Diversity | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero fill | 10.79 | 0.2384 | 0.5720 | 0.3252 | 64.82 | 1.0000 | 82.60 | — | — | 0.001 |
| Midpoint fill | 16.74 | 0.5612 | 0.5330 | 0.3175 | 32.58 | 1.0000 | 86.93 | — | — | 0.001 |
| Random fill | 13.73 | 0.1233 | 0.8907 | 0.3508 | 43.01 | 1.0000 | 85.45 | 16.23 | 33.92 | 0.004 |
| CNN | 18.06 | 0.6415 | 0.4496 | 0.2508 | 26.81 | 0.9956 | 104.91 | — | — | 0.011 |
| U-Net | 18.43 | 0.6508 | 0.4523 | 0.2564 | 25.78 | 0.9923 | 92.47 | — | — | 0.013 |
| Residual U-Net | 18.39 | 0.6354 | 0.4563 | 0.2601 | 25.96 | 1.0000 | 71.99 | — | — | 0.009 |
| ViT | 18.57 | 0.6451 | 0.4526 | 0.2593 | 25.24 | 1.0000 | 69.55 | — | — | 0.021 |
| Normalizing flow | 14.89 | 0.3778 | 0.6394 | 0.3334 | 37.24 | 1.0000 | 173.26 | 16.59 | 24.92 | 0.018 |
| Flow matching | 18.06 | 0.6321 | 0.4255 | 0.2435 | 26.38 | 1.0000 | 51.45 | 18.33 | 6.52 | 0.065 |
| Diffusion | 17.97 | 0.6282 | 0.4274 | 0.2462 | 26.70 | 1.0000 | 52.47 | 18.21 | 6.41 | 0.064 |
| Hybrid | 18.56 | 0.6480 | 0.4223 | 0.2399 | 24.82 | 1.0000 | 53.35 | 18.81 | 6.00 | 0.105 |
| Hybrid, stage 1 | 18.46 | 0.6437 | 0.4565 | 0.2545 | 25.55 | 1.0000 | 73.70 | — | — | 0.007 |
| Hybrid, stages 1+2 | 18.51 | 0.6398 | 0.4225 | 0.2403 | 24.98 | 1.0000 | 53.70 | 18.77 | 6.19 | 0.064 |
| ViT hybrid | 18.72 | 0.6463 | 0.4445 | 0.2505 | 24.61 | 1.0000 | 48.01 | 18.96 | 5.33 | 0.116 |
| ViT hybrid, stage 1 | 18.82 | 0.6425 | 0.4735 | 0.2630 | 24.47 | 1.0000 | 78.01 | — | — | 0.020 |
| ViT hybrid, stages 1+2 | 18.70 | 0.6344 | 0.4436 | 0.2502 | 24.70 | 1.0000 | 45.06 | 18.95 | 5.63 | 0.077 |
| Sequential bitplane CNN (D4) | 18.40 | 0.6660 | 0.4481 | 0.2486 | 24.93 | 1.0000 | 87.81 | — | — | 0.043 |

### k = 2

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | MAE ↓ | MSB ok | KID × 10³ ↓ | PSNR avg ↑ | Diversity | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero fill | 16.66 | 0.5437 | 0.3475 | 0.1869 | 32.72 | 1.0000 | 27.53 | — | — | 0.001 |
| Midpoint fill | 22.76 | 0.7005 | 0.3294 | 0.1810 | 16.05 | 1.0000 | 29.91 | — | — | 0.001 |
| Random fill | 19.76 | 0.3209 | 0.5280 | 0.2120 | 21.43 | 1.0000 | 39.10 | 22.25 | 16.96 | 0.005 |
| CNN | 25.62 | 0.8306 | 0.2478 | 0.1419 | 10.80 | 0.9796 | 48.47 | — | — | 0.018 |
| U-Net | 25.88 | 0.8272 | 0.2439 | 0.1429 | 10.42 | 0.9723 | 40.67 | — | — | 0.020 |
| Residual U-Net | 25.34 | 0.7932 | 0.2581 | 0.1426 | 11.19 | 1.0000 | 24.26 | — | — | 0.009 |
| ViT | 26.25 | 0.8187 | 0.2473 | 0.1377 | 9.78 | 1.0000 | 25.11 | — | — | 0.024 |
| Normalizing flow | 20.51 | 0.5537 | 0.4433 | 0.2133 | 19.09 | 1.0000 | 77.32 | 22.65 | 14.35 | 0.029 |
| Flow matching | 26.19 | 0.8332 | 0.1927 | 0.1203 | 10.00 | 1.0000 | 19.18 | 26.64 | 3.69 | 0.070 |
| Diffusion | 26.18 | 0.8330 | 0.1933 | 0.1211 | 9.95 | 1.0000 | 19.71 | 26.67 | 3.76 | 0.066 |
| Hybrid | 26.48 | 0.8402 | 0.1898 | 0.1202 | 9.55 | 1.0000 | 18.13 | 26.90 | 3.37 | 0.105 |
| Hybrid, stage 1 | 26.25 | 0.8308 | 0.2293 | 0.1338 | 9.90 | 1.0000 | 25.64 | — | — | 0.007 |
| Hybrid, stages 1+2 | 26.41 | 0.8376 | 0.1899 | 0.1189 | 9.64 | 1.0000 | 18.72 | 26.82 | 3.38 | 0.064 |
| ViT hybrid | 26.65 | 0.8342 | 0.1973 | 0.1263 | 9.35 | 1.0000 | 21.11 | 27.02 | 3.01 | 0.117 |
| ViT hybrid, stage 1 | 26.26 | 0.8152 | 0.2504 | 0.1412 | 9.76 | 1.0000 | 25.23 | — | — | 0.020 |
| ViT hybrid, stages 1+2 | 26.51 | 0.8287 | 0.2024 | 0.1283 | 9.50 | 1.0000 | 22.11 | 26.87 | 3.04 | 0.078 |
| Sequential bitplane CNN (D4) | 27.00 | 0.8468 | 0.2367 | 0.1379 | 8.66 | 1.0000 | 50.51 | — | — | 0.041 |

### k = 3

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | MAE ↓ | MSB ok | KID × 10³ ↓ | PSNR avg ↑ | Diversity | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero fill | 22.78 | 0.7634 | 0.1759 | 0.1021 | 16.00 | 1.0000 | 8.48 | — | — | 0.001 |
| Midpoint fill | 28.61 | 0.8178 | 0.1683 | 0.0981 | 8.24 | 1.0000 | 8.53 | — | — | 0.001 |
| Random fill | 25.70 | 0.5846 | 0.2576 | 0.1191 | 10.81 | 1.0000 | 17.38 | 28.12 | 8.48 | 0.004 |
| CNN | 31.42 | 0.9162 | 0.1214 | 0.0799 | 5.55 | 0.9612 | 17.12 | — | — | 0.011 |
| U-Net | 31.03 | 0.9077 | 0.1276 | 0.0812 | 5.68 | 0.9625 | 15.21 | — | — | 0.012 |
| Residual U-Net | 30.85 | 0.8827 | 0.1327 | 0.0796 | 5.96 | 1.0000 | 7.00 | — | — | 0.009 |
| ViT | 32.05 | 0.9108 | 0.1173 | 0.0707 | 4.88 | 1.0000 | 6.63 | — | — | 0.021 |
| Normalizing flow | 25.96 | 0.7144 | 0.2455 | 0.1266 | 10.27 | 1.0000 | 21.30 | 28.16 | 7.78 | 0.023 |
| Flow matching | 32.29 | 0.9186 | 0.0804 | 0.0590 | 4.89 | 1.0000 | 6.07 | 32.94 | 2.13 | 0.067 |
| Diffusion | 32.32 | 0.9199 | 0.0791 | 0.0587 | 4.87 | 1.0000 | 6.08 | 32.96 | 2.10 | 0.064 |
| Hybrid | 32.52 | 0.9238 | 0.0772 | 0.0566 | 4.72 | 1.0000 | 6.12 | 33.14 | 2.00 | 0.105 |
| Hybrid, stage 1 | 31.84 | 0.9063 | 0.1158 | 0.0736 | 5.22 | 1.0000 | 9.21 | — | — | 0.007 |
| Hybrid, stages 1+2 | 32.45 | 0.9233 | 0.0795 | 0.0572 | 4.77 | 1.0000 | 6.72 | 33.06 | 2.01 | 0.064 |
| ViT hybrid | 32.58 | 0.9222 | 0.0798 | 0.0608 | 4.63 | 1.0000 | 5.33 | 33.19 | 1.94 | 0.117 |
| ViT hybrid, stage 1 | 31.89 | 0.9080 | 0.1199 | 0.0718 | 4.96 | 1.0000 | 6.78 | — | — | 0.020 |
| ViT hybrid, stages 1+2 | 32.50 | 0.9224 | 0.0845 | 0.0645 | 4.67 | 1.0000 | 6.93 | 32.99 | 1.79 | 0.078 |
| Sequential bitplane CNN (D4) | 33.89 | 0.9358 | 0.0964 | 0.0648 | 3.88 | 1.0000 | 21.43 | — | — | 0.031 |

The sequential bitplane CNN has the highest PSNR and SSIM at k=2 and k=3; the ViT-hybrid stage-1 ablation has the highest k=1 PSNR, while bit-depth recovery has the highest k=1 SSIM. The full hybrids are competitive on perceptual distances: the U-Net hybrid has the lowest LPIPS at k=1/2/3 and the lowest DISTS at k=1 and k=3; at k=2 the stages 1+2 ablation has the lowest DISTS. KID's lowest result is the ViT-hybrid stages 1+2 ablation at k=1, the U-Net hybrid at k=2, and the ViT hybrid at k=3. Thus there is no single winner across distortion and perceptual metrics. Direct CNN and U-Net outputs have MSB consistency below 1.0; the residual, generative, hybrid, fill and sequential bitplane methods preserve the transmitted planes exactly.

These results are single training runs on one 24-image test set. Inference times are from the attached evaluation logs and depend on the H100 environment. Artifacts are under `outputs/rgb/eval/kodak-k1/` through `kodak-k3/`, plots under `outputs/rgb/plots/`, and the sequential model's CSV/plot under `outputs/rgb/bitdepth/kodak/`.

## Metric Notes

- The earlier `kodak-all-k1/2/3` summaries did not measure LPIPS, DISTS or KID. The final common-evaluator run measured them for D4 and the other models.
- PSNR avg and diversity need multiple stochastic samples; they are intentionally blank for deterministic models. The final evaluation used eight samples for stochastic methods.
- The standalone bitplane CLI still writes PSNR, SSIM and MAE; the shared `msbgen.cli.evaluate` path now evaluates D4 with the same LPIPS, DISTS, KID and MSB-consistency metrics as the other methods.

## Gaps and Next Steps

- The pretrained RGB benchmark uses released weights on Kodak; it is not a full reproduction of the paper's training data or protocol. The separately trained RGB experiments use RGB DIV2K and RGB Kodak, and should be compared only with other RGB results.
- CNN/U-Net grayscale regressors do not always preserve the transmitted bits exactly (MSB consistency is below 1); residual U-Net and the generative methods reported 1.0.
- The hybrid and baseline comparisons are single training runs on Kodak. Repeat multiple seeds, report paired uncertainty, and validate on additional image classes/datasets before general claims.
- ViT hyperparameters (patch size, width, depth, learning rate, EMA, warm-starting the ViT hybrid from a trained ViT) were not tuned; try a frozen/warm-started ViT stage 1 for a like-for-like comparison with the U-Net hybrid.
- Hybrid inference currently runs on the whole image rather than using tiled inference; assess memory and latency at larger resolutions.
- Direct CNN/U-Net regressors have small MSB-consistency errors; test hard projection and whether it affects image metrics.
- Repeat the RGB sequential-bitplane versus full-image comparison across seeds and additional test datasets, alongside actual entropy-coded rate measurements.
- The entropy coder is a correctness/reference implementation. Improve context modeling and include container/side-information costs before claiming production codec rates.
