"""Measure lossless rates for top-k bit-plane streams.

    python -m msbgen.cli.rate data/processed/kodak --gray --ks 1 2 3

The reported payload is an adaptive binary range-coded stream using the same four
causal neighbours as the analysis context entropy. ``framed_bpp`` includes the
four-byte length prefix stored before every plane; image dimensions and k are
assumed to be supplied by the surrounding container and are not counted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from msbgen.core import ImageDecodeError, analyze_image, decode_top_planes, encode_top_planes, load_image, to_grayscale

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".npy"}
FIELDS = ["file", "group", "width", "height", "channels", "k", "pixels", "raw_bpp",
          "context_bpp", "payload_bytes", "payload_bpp", "framed_bpp", "roundtrip"]


def _images(root: Path, recursive: bool, limit: int | None) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    paths = sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit]


def _load(path: Path, gray: bool) -> tuple[np.ndarray, str, int]:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if array.dtype != np.uint8 or array.ndim not in (2, 3):
            raise ValueError(f"expected uint8 grayscale/RGB array in {path}, got {array.dtype} {array.shape}")
        mode = "L" if array.ndim == 2 else "RGB"
        if gray and mode == "RGB":
            array, mode = to_grayscale(array), "L"
        return array, mode, 8
    loaded = load_image(path.read_bytes())
    array, mode = loaded.array, loaded.color_mode
    if gray and mode == "RGB":
        array, mode = to_grayscale(array), "L"
    return array, mode, loaded.bit_depth


def measure(path: Path, root: Path, gray: bool, ks: list[int]) -> list[dict]:
    image, mode, bit_depth = _load(path, gray)
    analysis = analyze_image(image, bit_depth, with_ssim=False)
    channels = 1 if image.ndim == 2 else image.shape[2]
    height, width = image.shape[:2]
    group = f"{mode}-{bit_depth}"
    rows = []
    for k in ks:
        if not 1 <= k < bit_depth:
            raise ValueError(f"k must be in 1..{bit_depth - 1}, got {k}")
        payload = encode_top_planes(image, k, bit_depth)
        decoded = decode_top_planes(payload, image.shape, k, bit_depth)
        pixels = height * width
        context_bpp = analysis.reconstructions[k - 1].coded_bpp
        framing_bytes = 4 * k * channels
        rows.append({
            "file": str(path.relative_to(root)), "group": group, "width": width, "height": height,
            "channels": channels, "k": k, "pixels": pixels, "raw_bpp": float(k * channels),
            "context_bpp": context_bpp, "payload_bytes": len(payload),
            "payload_bpp": (len(payload) - framing_bytes) * 8 / (pixels * channels),
            "framed_bpp": len(payload) * 8 / (pixels * channels),
            "roundtrip": bool(np.array_equal(decoded, ((image.astype(np.uint32) >> (bit_depth - k))
                                                << (bit_depth - k)).astype(image.dtype))),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="folder of images")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3], help="top-plane counts (default: 1 2 3)")
    parser.add_argument("--gray", action="store_true", help="convert RGB images to grayscale")
    parser.add_argument("--recursive", "-r", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, help="output folder (default outputs/rate/<input name>)")
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f"{args.input} is not a folder")
    paths = _images(args.input, args.recursive, args.limit)
    if not paths:
        parser.error(f"no images found in {args.input}")
    out = args.out or Path("outputs") / "rate" / args.input.resolve().name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for index, path in enumerate(paths, 1):
        try:
            rows.extend(measure(path, args.input, args.gray, args.ks))
        except ImageDecodeError as exc:
            errors.append(str(path))
            print(f"skipped {path}: {exc}", file=sys.stderr)
        print(f"\r[{index}/{len(paths)}] {path.relative_to(args.input)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    with (out / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for k in args.ks:
        values = [row for row in rows if row["k"] == k]
        summary[str(k)] = {
            "images": len(values),
            "raw_bpp": float(np.mean([row["raw_bpp"] for row in values])),
            "context_bpp": float(np.mean([row["context_bpp"] for row in values])),
            "payload_bpp": float(np.mean([row["payload_bpp"] for row in values])),
            "framed_bpp": float(np.mean([row["framed_bpp"] for row in values])),
            "roundtrip": all(row["roundtrip"] for row in values),
        }
    (out / "summary.json").write_text(json.dumps({
        "meta": {"data": str(args.input), "gray": args.gray, "images": len(paths), "ks": args.ks,
                 "skipped": errors,
                 "coder": "adaptive binary range coder; four causal neighbours; Laplace(1,1) start"},
        "rates": summary,
    }, indent=2))
    print("k  raw bpp  context bpp  payload bpp  framed bpp  roundtrip")
    for k in args.ks:
        row = summary[str(k)]
        print(f"{k:>1}  {row['raw_bpp']:7.3f}  {row['context_bpp']:11.3f}  {row['payload_bpp']:11.3f}  "
              f"{row['framed_bpp']:10.3f}  {row['roundtrip']}")
    print(f"\nWrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())