"""API endpoints."""

from __future__ import annotations

import base64
import math

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from msbgen import __version__
from msbgen.api.schemas import (
    AnalysisResponse,
    BitPlane,
    BitPlaneResponse,
    Channel,
    CombinedPlane,
    Correlation,
    FillScores,
    ImageInfo,
    PlaneAnalysis,
    Reconstruction,
)
from msbgen.config import settings
from msbgen.core import (
    ImageDecodeError,
    LoadedImage,
    analyze_image,
    binary_entropy,
    combine_channel_planes,
    encode_png,
    encode_rgb_bits_png,
    load_image,
    msb_reconstruction,
    slice_bitplanes,
    to_preview,
)

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
        return FillScores(zero=_finite(d["zero"]), mid=_finite(d["mid"]))

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


async def _load_upload(file: UploadFile) -> LoadedImage:
    data = await file.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty upload.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File exceeds the {settings.max_upload_mb:g} MB limit.",
        )
    try:
        return await run_in_threadpool(load_image, data, settings.max_pixels)
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.post("/bitplanes", response_model=BitPlaneResponse)
async def bitplanes(file: UploadFile) -> BitPlaneResponse:
    """Slice an uploaded image into bit_depth planes per channel after normalising it to
    grayscale or RGB (8 / 24 planes for 8-bit, 16 / 48 for 16-bit)."""
    loaded = await _load_upload(file)
    return await run_in_threadpool(_build_response, loaded, file.filename)


@router.post("/analysis", response_model=AnalysisResponse)
async def analysis(file: UploadFile) -> AnalysisResponse:
    """Bit-plane analysis: per-plane entropy (i.i.d. and context), spatial and
    inter-plane correlation, and quality of reconstructions from the top k planes."""
    loaded = await _load_upload(file)
    return await run_in_threadpool(_build_analysis, loaded, file.filename)
