"""Download, extract and convert datasets to 8-bit grayscale `.npy` files.

Layout under the data root (default `data/`, git-ignored):

    raw/<download files>                     archives / original images, kept for re-runs
    raw/<extracted_dir>/                     extracted archive contents
    processed/<name>/<image stem>.npy        (H, W) uint8 grayscale (BT.601 luma)
    processed/<name>-rgb/<image stem>.npy    (H, W, 3) uint8 RGB, with `color="rgb"`
    processed/<name>/manifest.json           provenance + list of files and shapes

Every step is skipped when its output already exists, so re-running is cheap and an
interrupted download resumes where it stopped.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from msbgen.core import ImageDecodeError, load_image, to_grayscale
from msbgen.data.sources import Download, Source

CHUNK = 1 << 20
USER_AGENT = "msbgen-prepare-data/0.1"


def _progress(iterable_total: int | None, desc: str, initial: int = 0):
    try:
        from tqdm import tqdm
    except ImportError:  # progress bars are optional
        return None
    return tqdm(total=iterable_total, initial=initial, desc=desc, unit="B", unit_scale=True, unit_divisor=1024, leave=False)


def download(item: Download, dest_dir: Path) -> Path:
    """Download one file with resume support; verifies the size when it is known."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / item.filename
    if dest.exists() and (item.size is None or dest.stat().st_size == item.size):
        return dest

    partial = dest.with_suffix(dest.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(item.url, headers={"User-Agent": USER_AGENT})
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if start and response.status != 206:  # server ignored the range: start over
            start = 0
        total = item.size or (int(response.headers["Content-Length"]) + start if response.headers.get("Content-Length") else None)
        bar = _progress(total, item.filename, start)
        with partial.open("ab" if start else "wb") as f:
            while chunk := response.read(CHUNK):
                f.write(chunk)
                if bar:
                    bar.update(len(chunk))
        if bar:
            bar.close()

    size = partial.stat().st_size
    if item.size is not None and size != item.size:
        raise OSError(f"{item.filename}: expected {item.size:,} bytes, got {size:,}; re-run to resume")
    partial.rename(dest)
    return dest


def extract(archive: Path, dest_dir: Path, expected_dir: str) -> Path:
    target = dest_dir / expected_dir
    if target.is_dir() and any(target.iterdir()):
        return target
    tmp = dest_dir / f".{expected_dir}.extracting"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp)
    inner = tmp / expected_dir
    (inner if inner.is_dir() else tmp).rename(target)
    shutil.rmtree(tmp, ignore_errors=True)
    return target


def convert(images: list[Path], out_dir: Path, source: Source, color: str = "gray") -> dict:
    """Convert images to grayscale (or RGB) .npy and write the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if len(manifest["files"]) == len(images):
            return manifest

    files = []
    for i, path in enumerate(images, 1):
        out = out_dir / f"{path.stem}.npy"
        if out.exists():
            shape = list(np.load(out, mmap_mode="r").shape)
        else:
            try:
                loaded = load_image(path.read_bytes())
            except ImageDecodeError as exc:
                print(f"\n  skipped {path.name}: {exc}", file=sys.stderr)
                continue
            if loaded.bit_depth != 8:
                print(f"\n  skipped {path.name}: {loaded.bit_depth}-bit", file=sys.stderr)
                continue
            if color == "rgb":
                array = loaded.array if loaded.array.shape[-1] == 3 else np.repeat(loaded.array[..., :1], 3, axis=-1)
            else:
                array = to_grayscale(loaded.array)[..., 0]
            np.save(out, np.ascontiguousarray(array))
            shape = list(array.shape)
        files.append({"file": out.name, "source": path.name, "shape": shape})
        print(f"\r  converting {source.name}: {i}/{len(images)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    manifest = {
        "name": source.name,
        "description": source.description,
        "citation": source.citation,
        "urls": [d.url for d in source.downloads],
        "format": "(H, W, 3) uint8 RGB, no resizing" if color == "rgb"
        else "(H, W) uint8 grayscale, ITU-R BT.601 luma (as Pillow 'L'), no resizing",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(files),
        "pixels": int(sum(f["shape"][0] * f["shape"][1] for f in files)),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def processed_name(source: Source, color: str = "gray") -> str:
    return f"{source.name}-rgb" if color == "rgb" else source.name


def prepare(source: Source, root: Path, keep_raw: bool = True, color: str = "gray") -> dict:
    raw = root / "raw"
    print(f"[{source.name}] {source.description}", file=sys.stderr)
    paths = [download(d, raw if source.archive else raw / source.name) for d in source.downloads]
    if source.archive:
        folder = extract(paths[0], raw, source.extracted_dir or paths[0].stem)
        images = sorted(folder.glob(source.image_glob))
    else:
        images = paths
    manifest = convert(images, root / "processed" / processed_name(source, color), source, color)
    if not keep_raw:
        for p in paths:
            p.unlink(missing_ok=True)
        if source.archive:
            shutil.rmtree(raw / (source.extracted_dir or paths[0].stem), ignore_errors=True)
    return manifest
