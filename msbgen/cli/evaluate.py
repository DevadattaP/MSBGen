"""Evaluate reconstructors (constant fills + trained checkpoints) on a test set.

    python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips

Every method receives only the top-k bits of each image (8-bit grayscale) and is
scored against the original with PSNR, SSIM, MAE, MSB consistency and, with
--lpips, LPIPS. Writes to --out (default outputs/eval/<folder name>):

    per_image.csv    one row per image × method
    summary.json     mean and std per method
    images/          with --save-images N: reconstructions of the first N images
                     and a side-by-side comparison strip per image
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from msbgen.core import FILL_RECONSTRUCTORS, load_image, reconstruct_and_evaluate, round_estimate, to_grayscale

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
METRICS = ["psnr", "ssim", "lpips", "mae", "msb_consistency"]


def load_test_images(folder: Path, limit: int | None = None) -> list[tuple[str, np.ndarray]]:
    """(name, (H, W) uint8) pairs from prepared .npy files or ordinary image files."""
    arrays = sorted(folder.glob("*.npy"))
    if arrays:
        return [(p.stem, np.load(p)) for p in arrays[:limit]]
    images = []
    for p in sorted(q for q in folder.iterdir() if q.suffix.lower() in IMAGE_EXTENSIONS)[:limit]:
        loaded = load_image(p.read_bytes())
        if loaded.bit_depth != 8:
            print(f"skipped {p.name}: {loaded.bit_depth}-bit", file=sys.stderr)
            continue
        images.append((p.stem, to_grayscale(loaded.array)[..., 0]))
    return images


def comparison_strip(tiles: list[tuple[str, np.ndarray]], height: int = 320) -> Image.Image:
    """Label each (title, uint8 image) and place them side by side."""
    pad, label_h = 6, 22
    scaled = []
    for title, arr in tiles:
        img = Image.fromarray(arr)
        w = round(img.width * height / img.height)
        scaled.append((title, img.resize((w, height), Image.Resampling.LANCZOS)))
    width = sum(img.width for _, img in scaled) + pad * (len(scaled) + 1)
    canvas = Image.new("L", (width, height + label_h + 2 * pad), 255)
    draw = ImageDraw.Draw(canvas)
    x = pad
    for title, img in scaled:
        canvas.paste(img, (x, pad + label_h))
        draw.text((x, pad + 4), title, fill=0)
        x += img.width + pad
    return canvas


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help="folder of prepared .npy files or images")
    p.add_argument("--models", nargs="*", type=Path, default=[], help="checkpoints (e.g. runs/*/best.pt)")
    p.add_argument("--k", type=int, default=1, help="transmitted planes (default 1)")
    p.add_argument("--fills", nargs="*", default=list(FILL_RECONSTRUCTORS), help="constant fills to include")
    p.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs `lpips`)")
    p.add_argument("--no-ssim", action="store_true")
    p.add_argument("--limit", type=int, help="evaluate only the first N images")
    p.add_argument("--save-images", type=int, default=4, metavar="N", help="save outputs for the first N images (default 4)")
    p.add_argument("--out", type=Path, help="output folder (default outputs/eval/<data folder name>)")
    p.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    p.add_argument("--no-thermal-guard", action="store_true",
                   help="don't pause between images when the GPU/CPU is hot (default limits as in train)")
    args = p.parse_args(argv)

    images = load_test_images(args.data, args.limit)
    if not images:
        p.error(f"no images in {args.data}")
    n = 8
    methods = [FILL_RECONSTRUCTORS[f] for f in args.fills]
    if args.models:
        from msbgen.models import ModelReconstructor

        for path in args.models:
            model = ModelReconstructor(path, device=args.device)
            if not model.supports(args.k, n, 1):
                print(f"skipped {path}: trained for k={model.config.k}", file=sys.stderr)
                continue
            methods.append(model)
    names = [m.name for m in methods]
    if len(set(names)) != len(names):
        p.error(f"method names must be unique, got {names} (run folder names are used for checkpoints)")
    lpips_fn = None
    if args.lpips:
        from msbgen.models.perceptual import LPIPS

        lpips_fn = LPIPS(args.device)

    out = args.out or Path("outputs") / "eval" / args.data.resolve().name
    img_dir = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        img_dir.mkdir(exist_ok=True)

    from msbgen.utils.thermal import ThermalGuard

    guard = ThermalGuard(enabled=not args.no_thermal_guard)
    rows = []
    for i, (name, image) in enumerate(images):
        guard.check()
        tiles = [("original", image)]
        for method in methods:
            estimate, scores = reconstruct_and_evaluate(method, image, args.k, n, with_ssim=not args.no_ssim)
            scores["lpips"] = lpips_fn(image, estimate, 255.0) if lpips_fn else None
            rows.append({"image": name, "method": method.name, **{m: scores[m] for m in METRICS}})
            if i < args.save_images:
                rounded = round_estimate(estimate, n)
                Image.fromarray(rounded).save(img_dir / f"{name}_{method.name}.png")
                tiles.append((f"{method.name}  {scores['psnr']:.2f} dB", rounded))
        if i < args.save_images:
            comparison_strip(tiles).save(img_dir / f"{name}_compare.png")
        print(f"\r[{i + 1}/{len(images)}] {name}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    with (out / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "method", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in methods:
        mrows = [r for r in rows if r["method"] == method.name]
        entry = {"label": method.label, "images": len(mrows)}
        for m in METRICS:
            vals = [r[m] for r in mrows if r[m] is not None and math.isfinite(r[m])]
            entry[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None
        if hasattr(method, "path"):
            entry.update(checkpoint=str(method.path), step=method.step, config=method.config.to_dict())
        summary[method.name] = entry
    meta = {"data": str(args.data), "images": len(images), "k": args.k, "bit_depth": n}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "methods": summary}, indent=2))

    print(f"\n{len(images)} images, k = {args.k}: mean over images\n")
    head = f"{'method':<20} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'MAE':>7} {'MSB ok':>7}"
    print(head + "\n" + "-" * len(head))
    for name_, e in summary.items():
        def cell(m, fmt):
            return format(e[m]["mean"], fmt) if e[m] else "-"
        print(f"{name_:<20} {cell('psnr', '7.2f')} {cell('ssim', '7.4f')} {cell('lpips', '7.4f')} "
              f"{cell('mae', '7.2f')} {cell('msb_consistency', '7.4f')}")
    print(f"\nWrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
