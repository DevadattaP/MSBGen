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


class BitPlaneResponse(BaseModel):
    filename: str | None
    width: int
    height: int
    original_mode: str = Field(description="Pillow mode of the uploaded file")
    original_format: str | None
    color_mode: str = Field(description="Mode that was sliced: 'L' (grayscale) or 'RGB'")
    bit_depth: int = Field(description="Bits (= planes) per channel: 1, 8 or 16")
    channels: list[Channel]
    notes: list[str]
    preview: str = Field(description="8-bit display rendering of the image, as a PNG data URL")
    planes: list[BitPlane] = Field(description="Ordered by channel, then from MSB to LSB")
    combined: list[CombinedPlane] = Field(
        default_factory=list,
        description="RGB images only: per-bit colour planes, from MSB to LSB",
    )
