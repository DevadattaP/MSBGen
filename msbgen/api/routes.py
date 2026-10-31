"""API endpoints."""

from __future__ import annotations

import base64
import json
import math

import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from msbgen import __version__
from msbgen.api.schemas import (
    AnalysisResponse,
    BitPlane,
    BitPlaneResponse,
    Channel,
    CombinedPlane,
    Correlation,
    Evaluation,
    EvaluationMethod,
    EvaluationsResponse,
    FillScores,
    ImageInfo,
    MetricStat,
    MethodResult,
    ModelInfo,
    ModelsResponse,
    PlaneAnalysis,
    Reconstruction,
    ReconstructResponse,
)
from msbgen.api import model_registry
from msbgen.config import settings
from msbgen.core import (
    FILL_RECONSTRUCTORS,
    ImageDecodeError,
    LoadedImage,
    analyze_image,
    binary_entropy,
    combine_channel_planes,
    convert_color_mode,
    encode_png,
    encode_rgb_bits_png,
    load_image,
    msb_reconstruction,
    reconstruct_and_evaluate,
    round_estimate,
    slice_bitplanes,
    summary_method_kind,
    to_preview,
)

ERROR_MAP_GAIN = 4  # |I - Î| is multiplied by this so small errors are visible
EVAL_METRICS = ("psnr", "ssim", "lpips", "dists", "mae", "msb_consistency", "psnr_avg", "diversity", "time_s", "kid")

router = APIRouter(prefix="/api", tags=["api"])


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _finite(x: float | None) -> float | None:
    """JSON has no NaN / inf: map them to null."""
    return None if x is None or not math.isfinite(x) else float(x)


def _image_info(loaded: LoadedImage, filename: str | None) -> dict:
    return ImageInfo(
        filename=filename,
        width=loaded.width,
        height=loaded.height,
        original_mode=loaded.original_mode,
        original_format=loaded.original_format,
        color_mode=loaded.color_mode,
        bit_depth=loaded.bit_depth,
        channels=[
            Channel(name=n, label=lbl)
            for n, lbl in zip(loaded.channel_names, loaded.channel_labels, strict=True)
        ],
        notes=loaded.notes,
    ).model_dump()


def _build_response(loaded: LoadedImage, filename: str | None) -> BitPlaneResponse:
    planes = slice_bitplanes(loaded.array, loaded.bit_depth)
    items: list[BitPlane] = []
    for c, channel in enumerate(loaded.channel_names):
        for k in reversed(range(loaded.bit_depth)):
            plane = planes[c, k]
            ones = float(plane.mean())
            items.append(
                BitPlane(
                    channel=channel,
                    channel_index=c,
                    bit=k,
                    weight=1 << k,
                    ones_ratio=ones,
                    entropy=binary_entropy(ones),
                    image=_data_url(encode_png(plane, binary=True)),
                )
            )
    combined = []
    if loaded.color_mode == "RGB":
        combined = [
            CombinedPlane(
                bit=k,
                weight=1 << k,
                image=_data_url(encode_rgb_bits_png(combine_channel_planes(planes, k))),
            )
            for k in reversed(range(loaded.bit_depth))
        ]
    return BitPlaneResponse(
        **_image_info(loaded, filename),
        preview=_data_url(encode_png(loaded.preview)),
        planes=items,
        combined=combined,
    )


def _reconstruction_png(loaded: LoadedImage, k: int) -> str:
    n = loaded.bit_depth
    estimate = msb_reconstruction(loaded.array, k, "mid", n)
    rounded = np.minimum(np.floor(estimate + 0.5), (1 << n) - 1)
    rounded = rounded.astype(np.uint16 if n > 8 else np.uint8)
    return _data_url(encode_png(to_preview(rounded, n)))


def _build_analysis(loaded: LoadedImage, filename: str | None) -> AnalysisResponse:
    n = loaded.bit_depth
    names = loaded.channel_names
    result = analyze_image(loaded.array, n)
    multi = len(names) > 1

    planes = [
        PlaneAnalysis(
            channel=names[p.channel_index],
            channel_index=p.channel_index,
            bit=p.bit,
            ones_ratio=p.ones_ratio,
            entropy=p.entropy,
            context_entropy=p.context_entropy,
            spatial_correlation=_finite(p.spatial_correlation),
        )
        for p in result.planes
    ]
    labels = [f"{names[p.channel_index]}·B{p.bit}" if multi else f"B{p.bit}" for p in result.planes]
    matrix = [[_finite(v) for v in row] for row in result.correlation.tolist()]

    def scores(d: dict[str, float | None]) -> FillScores:
        return FillScores(**{fill: _finite(v) for fill, v in d.items()})

    reconstructions = [
        Reconstruction(
            k=r.k,
            bits=list(range(n - 1, n - 1 - r.k, -1)),
            bits_per_pixel=r.bits_per_pixel,
            coded_bpp=r.coded_bpp,
            psnr=scores(r.psnr),
            ssim=scores(r.ssim),
            mae=scores(r.mae),
            image=_reconstruction_png(loaded, r.k),
        )
        for r in result.reconstructions
    ]
    return AnalysisResponse(
        **_image_info(loaded, filename),
        planes=planes,
        correlation=Correlation(labels=labels, matrix=matrix),
        reconstructions=reconstructions,
    )


async def _load_upload(file: UploadFile, color_mode: str | None = None) -> LoadedImage:
    data = await file.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty upload.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File exceeds the {settings.max_upload_mb:g} MB limit.",
        )
    try:
        loaded = await run_in_threadpool(load_image, data, settings.max_pixels)
        return convert_color_mode(loaded, color_mode)
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.post("/bitplanes", response_model=BitPlaneResponse)
async def bitplanes(file: UploadFile, color_mode: str | None = Form(None)) -> BitPlaneResponse:
    """Slice an uploaded image into bit_depth planes per channel after normalising it to
    grayscale or RGB (8 / 24 planes for 8-bit, 16 / 48 for 16-bit)."""
    loaded = await _load_upload(file, color_mode)
    return await run_in_threadpool(_build_response, loaded, file.filename)


@router.post("/analysis", response_model=AnalysisResponse)
async def analysis(file: UploadFile, color_mode: str | None = Form(None)) -> AnalysisResponse:
    """Bit-plane analysis: per-plane entropy (i.i.d. and context), spatial and
    inter-plane correlation, and quality of reconstructions from the top k planes."""
    loaded = await _load_upload(file, color_mode)
    return await run_in_threadpool(_build_analysis, loaded, file.filename)


def _model_info(m) -> ModelInfo:
    c = m.config
    return ModelInfo(
        name=m.name, label=m.label, kind=m.model.kind, arch=c.arch, output=c.output, k=c.k, bit_depth=c.bit_depth,
        channels=c.channels, parameters=m.model.num_parameters(), step=m.step,
        val_psnr=_finite(m.metrics.get("val_psnr")), val_gain_db=_finite(m.metrics.get("val_gain_db")),
    )


@router.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    """Trained reconstruction models found in MSBGEN_MODELS_DIR (default `runs/`)."""
    paths = await run_in_threadpool(model_registry.model_paths)
    found = []
    for path in paths:
        model = None
        try:
            model = await run_in_threadpool(model_registry.load_model, path, "cpu")
            found.append(_model_info(model))
        except Exception:  # noqa: BLE001 - a broken checkpoint must not break the model list
            continue
        finally:
            del model
            model_registry.release_model_memory()
    return ModelsResponse(
        torch_available=model_registry.torch_available(),
        models_dir=str(settings.models_dir),
        models=found,
    )


def _build_reconstruction(loaded: LoadedImage, filename: str | None, k: int, names: list[str] | None) -> ReconstructResponse:
    n = loaded.bit_depth
    if n != 8:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Models work on 8-bit images; this one is {n}-bit.")
    if not 1 <= k < n:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"k must be between 1 and {n - 1}.")
    info = _image_info(loaded, filename)
    image = loaded.array[..., 0] if loaded.color_mode == "L" else loaded.array
    channels = 1 if loaded.color_mode == "L" else 3
    perceptual = {}
    if image.size <= settings.perceptual_max_pixels:
        perceptual = model_registry.perceptual_metrics()
    elif settings.perceptual:
        info["notes"] = [*info["notes"], "LPIPS and DISTS skipped: the image is too large for the perceptual metrics."]

    results: list[MethodResult] = []
    for method in FILL_RECONSTRUCTORS.values():
        estimate, scores = reconstruct_and_evaluate(method, image, k, n)
        error = np.clip(np.abs(image.astype(np.float64) - estimate) * ERROR_MAP_GAIN, 0, 255).astype(np.uint8)
        with model_registry.inference_lock():
            extra = {name: _finite(fn(image, estimate, 255.0)) for name, fn in perceptual.items()}
        results.append(MethodResult(
            name=method.name, label=method.label, kind="fill",
            psnr=_finite(scores["psnr"]), ssim=_finite(scores["ssim"]), mae=scores["mae"],
            msb_consistency=scores["msb_consistency"], **extra,
            psnr_gain=None,
            image=_data_url(encode_png(round_estimate(estimate, n))),
            error_map=_data_url(encode_png(error)),
        ))

    candidates = model_registry.model_candidates(loaded.color_mode, k)

    def run_candidate(kind: str, path, device: str | None = None, include_perceptual: bool = True):
        if kind == "checkpoint":
            with model_registry.inference_lock():
                method = model_registry.load_model(path, device)
        else:
            from msbgen.models.reconstructor import BitplaneReconstructor

            method = BitplaneReconstructor(path, device=device or settings.device)
        if not method.supports(k, n, channels):
            return None
        with model_registry.inference_lock():
            estimate, scores = reconstruct_and_evaluate(method, image, k, n)
        error = np.clip(np.abs(image.astype(np.float64) - estimate) * ERROR_MAP_GAIN, 0, 255).astype(np.uint8)
        extra = {}
        if include_perceptual:
            with model_registry.inference_lock():
                extra = {name: _finite(fn(image, estimate, 255.0)) for name, fn in perceptual.items()}
        return MethodResult(
            name=method.name, label=method.label, kind="model",
            psnr=_finite(scores["psnr"]), ssim=_finite(scores["ssim"]), mae=scores["mae"],
            msb_consistency=scores["msb_consistency"], **extra,
            psnr_gain=None,
            image=_data_url(encode_png(round_estimate(estimate, n))),
            error_map=_data_url(encode_png(error)),
        )

    for kind, path in candidates:
        candidate_name = path.parent.name if kind == "checkpoint" else path.name
        if names and candidate_name not in names:
            continue
        try:
            result = run_candidate(kind, path)
        except Exception as exc:  # noqa: BLE001 - handle each model independently
            if not model_registry.cuda_requested() or not model_registry.is_cuda_oom(exc):
                info["notes"].append(f"Model {candidate_name} failed: {exc}")
                model_registry.release_model_memory()
                continue
            model_registry.release_model_memory()
            try:
                result = run_candidate(kind, path, device="cpu", include_perceptual=False)
                info["notes"].append(
                    f"Model {candidate_name} ran on CPU after CUDA ran out of memory; perceptual scores were skipped."
                )
            except Exception as cpu_exc:  # noqa: BLE001 - report the CPU retry failure and continue
                info["notes"].append(f"Model {candidate_name} failed on GPU and CPU retry: {cpu_exc}")
                result = None
        finally:
            model_registry.release_model_memory()
        if result is not None:
            results.append(result)

    if names:
        found = {result.name for result in results if result.kind == "model"}
        unknown = set(names) - found
        if unknown:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No compatible model named {', '.join(sorted(unknown))}.")

    mid = next(r.psnr for r in results if r.name == "mid")
    for r in results:
        r.psnr_gain = r.psnr - mid if r.psnr is not None and mid is not None else None
    return ReconstructResponse(**info, k=k, original=_data_url(encode_png(image)), results=results,
                               perceptual=list(perceptual))


@router.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct(
    file: UploadFile, k: int = Form(1), models: str | None = Form(None), color_mode: str | None = Form(None)
) -> ReconstructResponse:
    """Reconstruct the image from its top k planes with the fills (zero, mid, random) and every compatible
    trained model (or only `models`, comma-separated), preserving the selected color mode."""
    loaded = await _load_upload(file, color_mode)
    names = [m.strip() for m in models.split(",") if m.strip()] if models else None
    return await run_in_threadpool(_build_reconstruction, loaded, file.filename, k, names)


def _metric_stat(value) -> MetricStat | None:
    if not isinstance(value, dict) or value.get("mean") is None or not math.isfinite(value["mean"]):
        return None
    std = value.get("std")
    return MetricStat(mean=value["mean"], std=std if std is not None and math.isfinite(std) else 0.0)


def _read_evaluation(path) -> Evaluation | None:
    try:
        summary = json.loads(path.read_text())
        meta = summary["meta"]
        methods = [
            EvaluationMethod(name=name, label=e.get("label", name), kind=summary_method_kind(name, e),
                             metrics={m: _metric_stat(e.get(m)) for m in EVAL_METRICS})
            for name, e in summary["methods"].items()
        ]
        return Evaluation(name=path.parent.name, data=str(meta.get("data", "")), images=meta["images"], k=meta["k"],
                          samples=meta.get("samples", 1), kid_patch=meta.get("kid_patch"),
                          modified=path.stat().st_mtime, methods=methods)
    except (OSError, ValueError, KeyError, TypeError):  # a half-written or foreign file must not break the app
        return None


@router.get("/evaluations", response_model=EvaluationsResponse)
async def evaluations() -> EvaluationsResponse:
    """Saved test-set evaluations (`msbgen.cli.evaluate` output) in MSBGEN_EVAL_DIR (default `outputs/eval/`),
    newest first. This is where set-level metrics such as KID live."""
    def read() -> list[Evaluation]:
        found = [_read_evaluation(p) for p in settings.eval_dir.glob("*/summary.json")]
        return sorted((e for e in found if e), key=lambda e: e.modified, reverse=True)

    return EvaluationsResponse(eval_dir=str(settings.eval_dir), evaluations=await run_in_threadpool(read))
