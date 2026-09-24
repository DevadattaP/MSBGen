"""API endpoints."""

from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from msbgen import __version__
from msbgen.api.schemas import BitPlane, BitPlaneResponse, Channel, CombinedPlane
from msbgen.config import settings
from msbgen.core import (
    ImageDecodeError,
    LoadedImage,
    binary_entropy,
    combine_channel_planes,
    encode_png,
    encode_rgb_bits_png,
    load_image,
    slice_bitplanes,
)

router = APIRouter(prefix="/api", tags=["api"])


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


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
        preview=_data_url(encode_png(loaded.preview)),
        planes=items,
        combined=combined,
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.post("/bitplanes", response_model=BitPlaneResponse)
async def bitplanes(file: UploadFile) -> BitPlaneResponse:
    """Slice an uploaded image into bit_depth planes per channel after normalising it to
    grayscale or RGB (8 / 24 planes for 8-bit, 16 / 48 for 16-bit)."""
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
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return await run_in_threadpool(_build_response, loaded, file.filename)
