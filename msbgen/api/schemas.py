"""Response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Channel(BaseModel):
    name: str = Field(description="Short channel name: 'L', 'R', 'G' or 'B'")
    label: str = Field(description="Human-readable name: 'Gray', 'Red', 'Green' or 'Blue'")


class BitPlane(BaseModel):
    channel: str = Field(description="Channel name (see `channels`)")
    channel_index: int = Field(ge=0, description="Index into `channels`")
    bit: int = Field(ge=0, description="Bit index k (bit_depth-1 = MSB, 0 = LSB)")
    weight: int = Field(description="Place value 2^k")
    ones_ratio: float = Field(description="Fraction of pixels where B_k = 1")
    entropy: float = Field(description="Binary entropy of the plane in bits/pixel (i.i.d. bound)")
    image: str = Field(description="1-bit PNG as a data URL (white = 1, black = 0)")


class CombinedPlane(BaseModel):
    bit: int = Field(ge=0, description="Bit index k (bit_depth-1 = MSB, 0 = LSB)")
    weight: int = Field(description="Place value 2^k")
    image: str = Field(
        description="B_k of R, G, B stacked as one colour PNG data URL; each channel is 0 or 255"
    )


class ImageInfo(BaseModel):
    filename: str | None
    width: int
    height: int
    original_mode: str = Field(description="Pillow mode of the uploaded file")
    original_format: str | None
    color_mode: str = Field(description="Mode that was sliced: 'L' (grayscale) or 'RGB'")
    bit_depth: int = Field(description="Bits (= planes) per channel: 1, 8 or 16")
    channels: list[Channel]
    notes: list[str]


class BitPlaneResponse(ImageInfo):
    preview: str = Field(description="8-bit display rendering of the image, as a PNG data URL")
    planes: list[BitPlane] = Field(description="Ordered by channel, then from MSB to LSB")
    combined: list[CombinedPlane] = Field(
        default_factory=list,
        description="RGB images only: per-bit colour planes, from MSB to LSB",
    )


class PlaneAnalysis(BaseModel):
    channel: str
    channel_index: int = Field(ge=0)
    bit: int = Field(ge=0)
    ones_ratio: float
    entropy: float = Field(description="i.i.d. entropy H(B_k), bits/pixel")
    context_entropy: float = Field(
        description="H(B_k | W, NW, N, NE neighbours in the same plane), bits/pixel"
    )
    spatial_correlation: float | None = Field(
        description="Mean correlation with the right and lower neighbour; null for a constant plane"
    )


class FillScores(BaseModel):
    zero: float | None = Field(description="Missing bits set to 0 (truncation)")
    mid: float | None = Field(description="Missing bits set to the middle of their range")
    random: float | None = Field(description="Missing bits drawn uniformly at random (fixed seed)")


class Reconstruction(BaseModel):
    k: int = Field(ge=1, description="Number of most-significant planes kept")
    bits: list[int] = Field(description="Bit indices kept, MSB first")
    bits_per_pixel: int = Field(description="Raw cost: k × channels")
    coded_bpp: float = Field(description="Sum of the kept planes' context entropies")
    psnr: FillScores = Field(description="dB; null means lossless (infinite)")
    ssim: FillScores = Field(description="null if the image is smaller than the 11×11 window")
    mae: FillScores
    image: str = Field(description="Midpoint-fill reconstruction as an 8-bit PNG data URL")


class Correlation(BaseModel):
    labels: list[str] = Field(description="Plane labels, ordered by channel then MSB → LSB")
    matrix: list[list[float | None]] = Field(description="Pearson correlation; null if undefined")


class AnalysisResponse(ImageInfo):
    planes: list[PlaneAnalysis] = Field(description="Ordered by channel, then from MSB to LSB")
    correlation: Correlation
    reconstructions: list[Reconstruction] = Field(description="k = 1 … bit_depth")


class ModelInfo(BaseModel):
    name: str = Field(description="Run name, used to select the model")
    label: str
    kind: str = Field(description="'regression' (one deterministic estimate), or 'diffusion' / 'flow' "
                                   "(flow matching): a sample of p(R | B7)")
    arch: str = Field(description="'cnn', 'unet' or 'tunet' (time-conditioned U-Net, generative models)")
    output: str = Field(description="'direct' (predicts I) or 'residual' (predicts the missing bits)")
    k: int = Field(description="Number of transmitted MSB planes the model expects")
    bit_depth: int
    channels: int
    parameters: int
    step: int = Field(description="Training step of the checkpoint")
    val_psnr: float | None = Field(description="Validation PSNR at that step (dB); for generative models, of one sample")
    val_gain_db: float | None = Field(description="Validation PSNR gain over midpoint fill (dB)")


class ModelsResponse(BaseModel):
    torch_available: bool
    models_dir: str
    models: list[ModelInfo]


class MethodResult(BaseModel):
    name: str
    label: str
    kind: str = Field(description="'fill' (no-model baseline: zero, mid or random) or 'model'")
    psnr: float | None = Field(description="dB; null means identical")
    ssim: float | None
    mae: float
    msb_consistency: float = Field(description="Fraction of pixels whose top k bits match the transmitted ones")
    lpips: float | None = Field(None, description="LPIPS, lower is better; null if unavailable (see `perceptual`)")
    dists: float | None = Field(None, description="DISTS, lower is better; null if unavailable (see `perceptual`)")
    psnr_gain: float | None = Field(description="PSNR minus midpoint-fill PSNR (dB)")
    time_s: float | None = Field(
        None, description="Reconstruction time in seconds; excludes model loading and thermal-guard pauses"
    )
    image: str = Field(description="Reconstruction as an 8-bit PNG data URL")
    error_map: str = Field(description="|I - Î| × 4 as an 8-bit PNG data URL (brighter = larger error)")


class ModelCandidate(BaseModel):
    name: str = Field(description="Model name, as accepted by /api/reconstruct/model")
    kind: str = Field(description="'checkpoint' or 'bitdepth'")


class ModelRunResponse(BaseModel):
    result: MethodResult
    note: str | None = Field(None, description="E.g. that the model fell back to CPU")


class ReconstructResponse(ImageInfo):
    k: int
    original: str = Field(description="The selected-color image that was reconstructed, as a PNG data URL")
    results: list[MethodResult] = Field(description="Fills first (zero, mid, random), then models")
    perceptual: list[str] = Field(default_factory=list, description="Perceptual metrics computed: 'lpips', 'dists'")
    candidates: list[ModelCandidate] = Field(
        default_factory=list, description="Models compatible with this color mode and k, in run order"
    )


class MetricStat(BaseModel):
    mean: float
    std: float = Field(description="Std over images; for kid, over KID subsets")


class EvaluationMethod(BaseModel):
    name: str
    label: str
    kind: str = Field(description="'fill', 'regression' or 'generative'")
    metrics: dict[str, MetricStat | None] = Field(
        description="psnr, ssim, lpips, dists, mae, msb_consistency, psnr_avg, diversity, time_s, kid (null if not computed)")


class Evaluation(BaseModel):
    name: str = Field(description="Evaluation folder name, e.g. 'kodak-s8'")
    data: str
    images: int
    k: int
    samples: int = Field(description="Draws per image for stochastic methods")
    kid_patch: int | None = Field(description="KID patch size in pixels; null if KID was not computed")
    modified: float = Field(description="summary.json modification time (Unix seconds)")
    methods: list[EvaluationMethod]


class EvaluationsResponse(BaseModel):
    eval_dir: str
    evaluations: list[Evaluation] = Field(description="Newest first")
