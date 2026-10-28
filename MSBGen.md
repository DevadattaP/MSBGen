01/10/2026

# MSB-Conditioned Generative Image Reconstruction

## Problem & Motivation

Conventional 8-bit images require 8 bits/pixel for representation. However, the Most Significant Bit (MSB) plane contains important coarse structural and intensity information. Through this project I will investigate whether the remaining image information can be **generated at the receiver rather than transmitted**, enabling extremely low-bitrate image communication.

For an 8-bit grayscale image, `I = 128B_7 + 64B_6 + … + B_0`  
where (B\_7) is the MSB plane. The proposed system initially transmits only (B\_7) (1 bit/pixel before entropy coding) and reconstructs the complete image by learning `p(I|B_7)`

Equivalently, the model can learn the missing residual `R=I-128B_7`  
and reconstruct  `I’=128B_7+R’`

The core research question is:  
> **Can a generative model reconstruct a perceptually useful full image from only its MSB plane, thereby enabling extremely low-bitrate image transmission?**

## Proposed Approach

At the transmitter, the image is converted into bit-planes and initially only the MSB (B\_7) is retained and transmitted. At the receiver, (B\_7) is provided as conditioning information to a generative reconstruction model.

The study will progressively investigate:  
`B_7 -> CNN/U-Net baseline`  
`B_7 -> Conditional Diffusion`  
`B_7 -> Conditional Normalizing Flow`  
`B_7 -> Conditional Flow Matching`  
followed, if justified by the results, by a **hybrid architecture** combining these approaches.

The models will be compared using **PSNR, SSIM, LPIPS/perceptual quality, actual bits-per-pixel, reconstruction time, and computational complexity**. If MSB-only reconstruction is insufficient, the approach will be extended to two or three transmitted MSB planes: B\_7+B\_6 or B\_7+B\_6+B\_5

This will allow investigation of the trade-off between transmitted information and reconstruction quality.

## Comparison with Previous Research

| Work | Approach | Main Idea | Difference from Proposed Work |
| :---- | :---- | :---- | :---- |
| [**Han et al. (2025)**](https://doi.org/10.48550/arXiv.2502.21001) – *Towards Lossless Implicit Neural Representation via Bit Plane Decomposition* | Implicit Neural Representation | Represents image bit-planes using an INR and targets lossless representation/reconstruction. | Does not discard the lower bit-planes and generate them using a generative prior; the proposed work intentionally retains only the MSB initially. |
| [**Punnappurath & Brown (2020)**](https://doi.org/10.1109/TPAMI.2021.3125692) – *A Little Bit More: Bitplane-Wise Bit-Depth Recovery* | CNN-based bit-plane prediction | Recursively predicts missing lower-order bit-planes from available higher-order planes for bit-depth recovery. | Closest prior work, but uses deterministic CNN prediction and progressively recovers individual bit-planes. The proposed work investigates stochastic generative models to infer the missing information directly from the MSB. |
| [**Chang et al. (2021)**](https://doi.org/10.1155/2021/9917545) – *Progressive Transmission of Medical Images via a Bank of GANs* | GAN-based progressive transmission | Transmits the MSB and progressively predicts subsequent bit-planes, transmitting residual information for each predicted plane. | The proposed work aims to avoid transmitting the remaining original bit-planes initially and instead reconstruct them using a learned generative image prior. |

While the above three studies specifically exploit **bit-plane/MSB information for image representation, bit-depth recovery, or progressive transmission**, most existing works on diffusion models, flow matching, and normalizing flows focus on **generative modeling or representation of complete images**, rather than reconstructing a full image from an extremely limited representation such as a single MSB plane.

Existing studies demonstrate that MSBs contain useful structural information and that missing bit-planes can be predicted using INR, CNN, or GAN-based methods. However, the proposed work investigates a more aggressive formulation:

`Only MSB transmitted -> Generative reconstruction of full image`

rather than progressively transmitting or explicitly representing the remaining bit-planes.

The main research direction is therefore to study whether **modern generative models—particularly diffusion, normalizing flows, and flow matching—can exploit the strong structural prior contained in the MSB plane to reconstruct the discarded image information at an extremely low transmission rate.**

## Datasets Used in Related Work

| Work | Training Data | Test Data | Image Specification |
| :---- | :---- | :---- | :---- |
| Han et al. (2025) | 1,000 [MIT-FiveK](https://data.csail.mit.edu/graphics/fivek/) | 40 TESTIMAGES, 100 [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), 24 [Kodak](https://r0k.us/graphics/kodak/) | 16-bit/8-bit; all processed at 256×256 |
| Punnappurath & Brown (2020) | 1,000 MIT-FiveK \+ 1,000 [Sintel](https://sintel.is.tue.mpg.de/) | 40 TESTIMAGES, 24 Kodak, 25 [ESPL](https://live.ece.utexas.edu/research/HDRDB/hdr_index.html) v2, [MSCOCO](https://cocodataset.org/), [BSD100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | 16-bit training; 8/16-bit testing; varying resolutions |
| Chang et al. (2021) | 10,000 [NIH Chest X-rays](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) | 10 NIH Chest X-rays | 8-bit grayscale, 256×256 |

## Expected Contribution

The project aims to develop and evaluate an **MSB-conditioned generative image reconstruction framework** and establish its rate–distortion–perception trade-off. The study will determine:

1. How much visual information can be recovered from a single MSB plane.  
2. Whether generative models outperform conventional deterministic bit-plane prediction.  
3. Whether diffusion, normalizing flow, or flow matching provides the most effective reconstruction/complexity trade-off.  
4. Whether transmitting additional MSB planes significantly improves reconstruction quality.  
5. The achievable practical bitrate after entropy coding.

---

08/10/2026

# Initial Report

## Previous Work

The reference method, [A Little Bit More: Bitplane-Wise Bit-Depth Recovery](https://arxiv.org/abs/2005.01091), with the official implementation at [GitHub](https://github.com/abhijithpunnappurath/a-little-bit-more), predicts missing lower bitplanes sequentially: reconstruct one plane, add it to the image, then predict the next. We ported its residual-CNN approach to PyTorch and converted the authors' released D4/D16 RGB checkpoints from Keras to evaluate them on Kodak.

MSBGen studies a related but harder task: reconstructing a complete grayscale image from only its top `k` bits (`k=1,2,3`). It compares direct CNN/U-Net regressors and conditional diffusion/flow models. The paper-style sequential model and the MSBGen full-image models are distinct experiments.

## RGB Pretrained Results

The pretrained RGB benchmark uses released weights on Kodak; it is not a full reproduction of the paper's training data or protocol. (The grayscale experiments I am doing use DIV2K). Mean over 24 RGB Kodak images (author’s training data is my testing data). These are evaluations of the authors' pretrained checkpoints, not models retrained here.

| Input bits \-\> 8 | D4 PSNR | D16 PSNR | D4 SSIM | D16 SSIM |
| :---- | ----: | ----: | ----: | ----: |
| 3 \-\> 8 | 33.13 | 33.48 | 0.9329 | 0.9349 |
| 4 \-\> 8 | 39.34 | 39.53 | 0.9710 | 0.9723 |
| 5 \-\> 8 | 44.46 | 44.65 | 0.9875 | 0.9880 |
| 6 \-\> 8 | 49.21 | 49.29 | 0.9953 | 0.9954 |
| 7 \-\> 8 | 54.93 | 54.99 | 0.9986 | 0.9987 |

All cases preserved the transmitted top bits. More input bits had a much larger effect than increasing depth from D4 to D16.

## Grayscale Kodak Results

Mean PSNR (dB) over 24 images; each model reconstructs the full 8-bit image from the indicated number of top bits. CNN/U-Net/Res U-Net are deterministic. Diffusion and flow columns are single-sample results.

| Method | k=1 | | k=2 | | k=3 | |
| ----- | :---: | :---: | :---: | :---: | :---: | :---: |
| | **PSNR** | **SSIM** | **PSNR** | **SSIM** | **PSNR** | **SSIM** |
| Midpoint fill | 17.01 | 0.5793 | 22.63 | 0.7033 | 28.74 | 0.8237 |
| CNN | 18.04 | 0.6167 | 25.13 | 0.7846 | 31.10 | 0.8915 |
| U-Net | 18.55 | 0.6371 | 25.90 | 0.7983 | 30.87 | 0.8849 |
| Residual U-Net | **18.60** | **0.6373** | **26.14** | **0.8044** | 32.00 | **0.9015** |
| Sequential bitplane CNN (D4) | 18.56 | 0.6328 | 25.65 | 0.7976 | **32.04** | **0.9015** |
| Diffusion | 17.75 | 0.4801 | 25.63 | 0.7885 | 31.58 | 0.8951 |
| Flow matching | 18.11 | 0.4986 | 25.60 | 0.7902 | 31.51 | 0.8944 |

Residual U-Net is strongest or competitive on single-sample distortion; the generative sample means become competitive as more bits are given. The sequential bitplane CNN is close to the best regressors and reaches the highest single-result PSNR at k=3.

## Findings

- For grayscale MSB-only reconstruction, deterministic residual U-Net gives the best single-sample PSNR and perfect transmitted-bit consistency.  
- Diffusion and flow matching are not expected to win single-sample PSNR in a one-to-many problem. Their strongest evidence is lower DISTS, nonzero sample diversity, and improving PSNR of the sample mean.  
- As more MSB planes are transmitted, the generative models become more competitive in both distortion and perceptual metrics.

## Next Steps

- Study the effect of residual depth on grayscale bitplane-wise recovery vs MSB-only recovery.  
- Design conditional generative model using diffusion and flow matching for bitplane-wise recovery.

Example result on MSB-only recovery of grayscale image with different models-  
![image1](outputs/eval/kodak-fm/images/kodim04_compare.png)

Result from bit-plane recovery (literature approach implemented in grayscale)-

| **From B7 -> rest all** | **From B7+B6 -> rest all** | **From B7+B6+B5 -> rest all** |
| :---: | :---: | :---: |
| ![image2](outputs/bitdepth/kodak/kodim04_1to8.png) | ![image3](outputs/bitdepth/kodak/kodim04_2to8.png) | ![image4](outputs/bitdepth/kodak/kodim04_3to8.png) |

---

23/10/2026

# Midterm Report

## Hybrid Reconstruction: Residual U-Net + Flow Matching + Diffusion

### Motivation

The earlier experiments showed complementary strengths. A residual U-Net gives a
stable reconstruction and strong single-image PSNR, but tends to smooth uncertain
fine detail. Flow matching can generate plausible missing texture, although a
single flow sample does not consistently improve pixel-wise distortion. Diffusion
can refine generated detail and improve perceptual metrics, but sampling by itself
is relatively expensive. The hybrid combines these roles: use the U-Net as a
reliable starting point, flow matching to generate a conditional correction, and
diffusion to refine that result. The goal is not to assume that three models must
always be better, but to test whether their complementary behavior improves the
rate-distortion-perception trade-off.

### Architecture

For input image `I`, the transmitter keeps the top `k` bit-planes, producing the
known coarse image `I_top`. The untransmitted information is the residual
`R = I - I_top`. The model processes this residual in three stages:

1. **Residual U-Net:** estimates a deterministic residual `r0` conditioned on the
   transmitted planes. This provides the base reconstruction.
2. **Conditional flow matching:** learns the correction `delta = R - r0`, given
   the transmitted planes and `r0`. At inference it transforms a noise sample into
   a correction and adds it to the base, giving `r1 = r0 + delta`.
3. **Diffusion refinement:** receives the transmitted planes, `r0`, and `r1`. It
   adds noise to `r1` up to a chosen noise level and denoises it with a short
   truncated DDIM trajectory to produce the final residual.

The output is formed by adding the generated residual to `I_top`. Since the model
generates only the missing low-bit residual, it preserves the transmitted MSBs.
The ablations isolate each contribution: `hybrid-unet` uses stage 1 only,
`hybrid-flow` uses stages 1 and 2, and `hybrid` uses all three stages.

### Training and Evaluation

The grayscale hybrid models were trained on DIV2K crops (128 x 128) for 40,000
steps, with batch size 16 and EMA decay 0.999. Stage 1 was initialized from the
matching residual U-Net checkpoint and frozen, so the experiment tests the
generative stages on top of an established reconstruction. The best k=2 checkpoint
was selected at step 38,000; k=1 and k=3 checkpoints were selected at step 40,000.

Evaluation used all 24 grayscale Kodak images. Stochastic methods were evaluated
with eight samples per image; flow used three Euler steps, and the hybrid refiner
used two DDIM steps with `refine_t=0.5` and `eta=0`. LPIPS and DISTS are lower-is-
better perceptual distances. KID is patch KID multiplied by 1,000 (lower is better).
`PSNR of mean` is PSNR computed on the pixel-wise mean of the eight samples; it is
not the average PSNR of the individual samples. The table reports image-set means.

| k | Method | PSNR (dB) | SSIM | LPIPS | DISTS | KID x 10^3 | PSNR of mean (8) | Time / image (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Residual U-Net | **18.60** | **0.6373** | 0.4550 | 0.2931 | 79.75 | — | 0.006 |
| 1 | Sequential bitplane CNN (D4) | 18.56 | 0.6328 | 0.4718 | 0.2872 | 93.87 | — | 0.091 |
| 1 | Diffusion | 17.75 | 0.6101 | 0.4245 | 0.2567 | 54.61 | 18.20 | 0.088 |
| 1 | Flow matching | 18.11 | 0.6151 | **0.4190** | **0.2519** | 54.92 | 18.54 | 0.059 |
| 1 | Hybrid | 18.22 | 0.6242 | 0.4237 | 0.2594 | **46.89** | **18.61** | 0.101 |
| | | | | | | | | |
| 2 | Residual U-Net | **26.14** | **0.8044** | 0.2443 | 0.1554 | 41.76 | — | 0.006 |
| 2 | Sequential bitplane CNN (D4) | 25.65 | 0.7976 | 0.2807 | 0.1657 | 54.33 | — | 0.062 |
| 2 | Diffusion | 25.63 | 0.7885 | 0.2134 | 0.1292 | 26.06 | 26.24 | 0.062 |
| 2 | Flow matching | 25.60 | 0.7902 | 0.2141 | 0.1302 | 25.34 | 26.23 | 0.059 |
| 2 | Hybrid | 25.99 | 0.7936 | **0.2051** | **0.1249** | **17.25** | **26.60** | 0.101 |
| | | | | | | | | |
| 3 | Residual U-Net | 32.00 | **0.9015** | 0.1171 | 0.0852 | 19.04 | — | 0.006 |
| 3 | Sequential bitplane CNN (D4) | **32.04** | **0.9015** | 0.1425 | 0.0943 | 21.15 | — | 0.052 |
| 3 | Diffusion | 31.58 | 0.8951 | 0.0941 | 0.0717 | 8.56 | 32.37 | 0.062 |
| 3 | Flow matching | 31.51 | 0.8944 | 0.0953 | 0.0719 | 9.01 | 32.34 | 0.059 |
| 3 | Hybrid | 31.65 | 0.8930 | **0.0905** | **0.0681** | **5.93** | **32.60** | 0.102 |

For the kodak dataset (24 test images), the hybrid has the best KID among these methods at all three bit depths, and the
best eight-sample mean PSNR. At k=2 and k=3 it also improves LPIPS and DISTS over
standalone diffusion and flow matching. At k=1, flow matching has slightly better
LPIPS and DISTS than the full hybrid, while the hybrid has better KID. The hybrid
does not yet beat the best deterministic single-sample PSNR: residual U-Net leads
at k=1 and k=2, and D4 leads at k=3 by 0.04 dB. At k=3 the hybrid's perceptual
metrics are better than D4's (LPIPS 0.0905 vs. 0.1425, DISTS 0.0681 vs. 0.0943,
and KID 5.93 vs. 21.15). These are results from one training run and a 24-image
test set, so broader testing is needed before drawing general conclusions. Timings
are from the evaluation hardware and should not be treated as portable latency
figures.

### Example: Kodak kodim04

The following comparisons show the reconstructions from
the evaluated methods. The per-image PSNR labels are specific to `kodim04` and are not the 24-image means in the table.

| One transmitted MSB (k=1) | Two transmitted MSBs (k=2) | Three transmitted MSBs (k=3) |
| :---: | :---: | :---: |
| ![kodim04 comparison, k=1](outputs/eval/kodak-final-k1/images/kodim04_hybrid.png) | ![kodim04 comparison, k=2](outputs/eval/kodak-final-k2/images/kodim04_hybrid-k2.png) | ![kodim04 comparison, k=3](outputs/eval/kodak-final-k3/images/kodim04_hybrid-k3.png) |
| bitdepth vs hybrid PSNR | | |
| 19.11 dB vs **19.99 dB** | 27.09 dB vs **27.84 dB** | **32.87 dB** vs 32.61 dB |

### Next Steps: Improve Grayscale, Then Extend to Color

1. **Improve distortion:** the k=3 hybrid is still 0.39 dB below D4 and 0.35 dB
   below residual U-Net on single-sample PSNR. Test joint training of the U-Net
   instead of freezing it, stronger stage-1 models, and adjusted weights for
   pixel-wise and perceptual objectives.
2. **Tune generation:** compare flow-step counts, `refine_t`, and diffusion
   refinement steps to find a better PSNR/perceptual-quality trade-off. Keep
   ablation results so each added stage is justified by measured gains.
3. **Validate robustness and efficiency:** repeat training with multiple seeds,
   report per-image paired uncertainty, evaluate on additional datasets, and
   investigate tiled inference for larger images.
4. **Extend to color:** update the data pipeline and model to use three-channel
   RGB images, initially transmitting the top `k` bits of each channel. Train on
   color DIV2K data and evaluate on color Kodak images. Compare direct RGB
   processing with a luminance/chrominance representation, and verify exact
   preservation of all transmitted channel MSBs. Report color-aware distortion
   and perceptual metrics under the same protocol for every baseline.


29/10/2026

# Experiment using ViT


## Motivation

The U-Net-based models only see local-to-regional context through convolutions and
downsampling. A Vision Transformer (ViT) uses global self-attention over patch tokens, so
every patch can use scene-wide structure from the transmitted planes. Two variants were
tested for k=1, 2, 3:

- **ViT:** a residual ViT regressor (`--preset vit`), the counterpart of the residual U-Net.
  8x8 patches are embedded to 192-d tokens, mixed by 6 transformer blocks (6 heads) with fixed
  2D sin-cos position embeddings, then unpatchified with a linear head and fused with a small
  3x3 conv branch on the raw planes to avoid patch-grid artefacts. It predicts the missing
  low-bit residual, so transmitted bits are preserved exactly (2.9 M parameters).
- **ViT hybrid:** the three-stage hybrid with the residual ViT as stage 1 instead of the
  residual U-Net (`--preset vit-hybrid`): ViT -> flow-matching correction -> truncated
  diffusion refinement (10.9 M parameters, versus 12.3 M for the U-Net hybrid).

## Training and Evaluation

Both variants were trained on DIV2K crops (128 x 128) for 40,000 steps, batch size 16, learning
rate 2e-4. The ViT has no EMA; the ViT hybrid uses EMA 0.999. Unlike the U-Net hybrid, the ViT
hybrid's stage 1 was **trained jointly from scratch** (no `--init-unet`, no `--freeze-unet`),
so `vit-hybrid-unet` below is that run's own stage 1, not the standalone `vit` checkpoint. The
best checkpoints were at step 10k / 22k / 30k for ViT (k=1/2/3) and 36k / 40k / 40k for the
ViT hybrid. Evaluation protocol is unchanged: 24 grayscale Kodak images, eight samples for
stochastic methods, three flow steps, two DDIM refinement steps, `refine_t=0.5`, `eta=0`.

| k | Method | PSNR (dB) | SSIM | LPIPS | DISTS | KID x 10^3 | PSNR of mean (8) | Time / image (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Residual U-Net | 18.60 | **0.6373** | 0.4550 | 0.2931 | 79.75 | — | 0.006 |
| 1 | Sequential bitplane CNN (D4) | 18.56 | 0.6328 | 0.4718 | 0.2872 | 93.87 | — | 0.074 |
| 1 | Hybrid (U-Net) | 18.22 | 0.6242 | 0.4237 | 0.2594 | **46.89** | 18.61 | 0.102 |
| 1 | ViT | 18.35 | 0.6206 | 0.4662 | 0.2932 | 89.81 | — | 0.018 |
| 1 | ViT hybrid | **18.64** | 0.6230 | 0.4293 | 0.2618 | 49.17 | **18.83** | 0.114 |
| 1 | ViT hybrid, stage 1 only | 18.57 | 0.6216 | 0.4492 | 0.2923 | 94.11 | — | 0.018 |
| 1 | ViT hybrid, stages 1+2 | 18.63 | 0.6200 | 0.4246 | 0.2560 | 54.02 | 18.81 | 0.076 |
| | | | | | | | | |
| 2 | Residual U-Net | **26.14** | **0.8044** | 0.2443 | 0.1554 | 41.76 | — | 0.006 |
| 2 | Sequential bitplane CNN (D4) | 25.65 | 0.7976 | 0.2807 | 0.1657 | 54.33 | — | 0.062 |
| 2 | Hybrid (U-Net) | 25.99 | 0.7936 | **0.2051** | **0.1249** | **17.25** | **26.60** | 0.102 |
| 2 | ViT | 24.51 | 0.7777 | 0.2676 | 0.1644 | 48.19 | — | 0.017 |
| 2 | ViT hybrid | 24.71 | 0.7698 | 0.2251 | 0.1314 | 19.46 | 25.30 | 0.114 |
| 2 | ViT hybrid, stage 1 only | 24.39 | 0.7744 | 0.2666 | 0.1671 | 47.82 | — | 0.018 |
| 2 | ViT hybrid, stages 1+2 | 24.65 | 0.7682 | 0.2224 | 0.1297 | 26.47 | 25.20 | 0.076 |
| | | | | | | | | |
| 3 | Residual U-Net | 32.00 | **0.9015** | 0.1171 | 0.0852 | 19.04 | — | 0.006 |
| 3 | Sequential bitplane CNN (D4) | **32.04** | **0.9015** | 0.1425 | 0.0943 | 21.15 | — | 0.052 |
| 3 | Hybrid (U-Net) | 31.65 | 0.8930 | **0.0905** | **0.0681** | **5.93** | **32.60** | 0.102 |
| 3 | ViT | 30.78 | 0.8845 | 0.1487 | 0.0976 | 23.40 | — | 0.018 |
| 3 | ViT hybrid | 31.21 | 0.8896 | 0.0951 | 0.0700 | 7.12 | 32.10 | 0.115 |
| 3 | ViT hybrid, stage 1 only | 30.64 | 0.8828 | 0.1496 | 0.0975 | 22.90 | — | 0.018 |
| 3 | ViT hybrid, stages 1+2 | 31.20 | 0.8896 | 0.0986 | 0.0717 | 9.51 | 32.00 | 0.076 |

## Findings

- The standalone ViT is worse than the residual U-Net on every distortion metric (PSNR
  -0.25 / -1.63 / -1.22 dB at k=1/2/3) and below the plain CNN at k=2 and k=3. Global attention
  did not compensate for the lack of a convolutional locality prior at this size and training
  budget. It preserves the transmitted bits exactly (MSB consistency 1.0000).
- The ViT hybrid has the highest single-sample PSNR of all methods at k=1 (18.64 dB, above the
  residual U-Net's 18.60 and D4's 18.56) and the highest eight-sample-mean PSNR (18.83 dB). Its
  KID (49.17) and DISTS (0.2618) are slightly behind the U-Net hybrid (46.89 / 0.2594).
- At k=2 and k=3 the ViT hybrid is behind the U-Net hybrid on every metric (k=3: 31.21 dB /
  DISTS 0.0700 / KID 7.12 versus 31.65 dB / 0.0681 / 5.93), but it is far better than standalone ViT.
- The flow and diffusion stages help the ViT substantially on perceptual metrics: at k=3, LPIPS
  0.1496 -> 0.0951, DISTS 0.0975 -> 0.0700 and KID 22.90 -> 7.12 compared with the hybrid's own
  stage 1, and PSNR rises by 0.57 dB. Part of that PSNR gain comes from joint training of the
  stage 1 rather than from sampling, so it is not directly comparable to the frozen U-Net hybrid.
- Diffusion refinement improves KID relative to stages 1+2 at all k (54.02 -> 49.17,
  26.47 -> 19.46, 9.51 -> 7.12) and slightly improves PSNR (+0.01 to +0.06 dB). Unlike the U-Net
  hybrid, it did not cost PSNR here; DISTS/LPIPS at k=2 and k=3 change little.
- Inference costs 0.018 s for ViT (3x the U-Net) and 0.114 s for the ViT hybrid (U-Net hybrid:
  0.102 s) on the evaluation GPU.
- Caveats: single training runs, 24 Kodak images, no ViT hyperparameter tuning (patch size,
  width, depth, learning rate), and the hybrid training protocols differ (joint vs frozen stage 1).
  The standalone ViT peaked early in validation (step 10k at k=1) and uses no EMA, so a tuned
  schedule might change its ranking.

![vit-hybrid comparison for k=1](outputs/eval/kodak-vit-hybrid-k1/images/kodim04_compare.png)