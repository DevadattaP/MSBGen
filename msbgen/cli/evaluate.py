"""Evaluate reconstructors (zero / mid / random fills + trained checkpoints) on a test set.

    python -m msbgen.cli.evaluate data/processed/kodak --models runs/*/best.pt --lpips --dists --kid

Every method receives only the top-k bits of each image (8-bit grayscale) and is
scored against the original with PSNR, SSIM, MAE, MSB consistency and, with
--lpips / --dists, LPIPS / DISTS. `time_s` is the time to produce one reconstruction.

Stochastic methods (random fill, diffusion, flow matching) are scored on their first sample. With
--samples N > 1 they are drawn N times (seeds 0 … N-1) and two more metrics are added:

    psnr_avg     PSNR of the average of the N samples. For a well-calibrated generative
                 model it rises towards the MMSE (regression) PSNR as N grows.
    diversity    mean per-pixel standard deviation across the N samples, in intensity
                 levels: how much the samples differ from each other (0 = identical).

--kid adds a set-level score per method: KID between Inception features of
--kid-patch-sized patches of its outputs (every sample of a stochastic method) and
of the originals. It asks "do the outputs look like real images", regardless of
pixel alignment; a test set of 24 images is far too few for FID, but its ~900
patches are enough for KID.

Writes to --out (default outputs/eval/<folder name>):

    per_image.csv    one row per image × method
    summary.json     mean and std per method (for kid: mean and std over KID subsets)
    images/          with --save-images N: reconstructions of the first N images, a
                     side-by-side comparison strip per image and, with --samples > 1,
                     a strip per stochastic method: samples, their mean and their std
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from msbgen.core import FILL_RECONSTRUCTORS, evaluate, keep_top_bits, load_image, round_estimate, to_grayscale
from msbgen.core.metrics import image_patches, kid, kid_subset_size, psnr

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
METRICS = ["psnr", "ssim", "lpips", "dists", "mae", "msb_consistency", "psnr_avg", "diversity", "time_s"]
STD_GAIN = 4  # the std map in the sample strips is scaled by this, like the web app's error maps


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
    p.add_argument("--bitdepth-models", nargs="*", type=Path, default=[], metavar="DIR",
                   help="folders of sequential bitplane CNNs (bit_NN.pt), e.g. runs/bitdepth-d4")
    p.add_argument("--samples", type=int, default=1, help="draws per image for stochastic methods (default 1)")
    p.add_argument("--sample-steps", type=int,
                   help="generative models: sampling steps (DDIM / ODE; default: the checkpoint's, 3 for new runs)")
    p.add_argument("--eta", type=float, help="diffusion: DDIM eta, 0 = deterministic given z (default: checkpoint's)")
    p.add_argument("--solver", choices=["euler", "heun"], help="flow matching: ODE solver (default: checkpoint's)")
    p.add_argument("--k", type=int, default=1, help="transmitted planes (default 1)")
    p.add_argument("--refine-steps", type=int, help="hybrid: refiner DDIM steps (default: checkpoint's)")
    p.add_argument("--refine-t", type=float, help="hybrid: refiner start noise level (default: checkpoint's)")
    p.add_argument("--hybrid-ablation", action="store_true",
                   help="for hybrid checkpoints also score '<name>-unet' (stage 1) and '<name>-flow' (stages 1+2)")
    p.add_argument("--fills", nargs="*", default=list(FILL_RECONSTRUCTORS), help="fill baselines to include (default: zero mid random)")
    p.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs `lpips`)")
    p.add_argument("--dists", action="store_true", help="also compute DISTS (needs `piq`)")
    p.add_argument("--kid", action="store_true", help="also compute patch KID per method (needs `piq`)")
    p.add_argument("--kid-patch", type=int, default=128, metavar="PX", help="KID patch size (default 128)")
    p.add_argument("--no-ssim", action="store_true")
    p.add_argument("--limit", type=int, help="evaluate only the first N images")
    p.add_argument("--save-images", type=int, default=4, metavar="N", help="save outputs for the first N images (default 4)")
    p.add_argument("--out", type=Path, help="output folder (default outputs/eval/<data folder name>)")
    p.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    p.add_argument("--no-thermal-guard", action="store_true",
                   help="don't pause when the GPU/CPU is hot (default limits as in train)")
    args = p.parse_args(argv)

    images = load_test_images(args.data, args.limit)
    if not images:
        p.error(f"no images in {args.data}")
    if args.samples < 1:
        p.error("--samples must be at least 1")
    if args.kid_patch < 16:
        p.error("--kid-patch must be at least 16")
    n = 8
    from msbgen.utils.thermal import ThermalGuard

    # Checked between images and, for generative models, after every sampling step: one image with
    # several samples is a minute or more of full GPU load.
    guard = ThermalGuard(enabled=not args.no_thermal_guard)
    methods = [FILL_RECONSTRUCTORS[f] for f in args.fills]
    if args.models:
        from msbgen.models import load_reconstructor

        for path in args.models:
            model = load_reconstructor(path, device=args.device, steps=args.sample_steps, eta=args.eta,
                                       solver=args.solver, refine_steps=args.refine_steps,
                                       refine_t=args.refine_t, on_step=guard.check)
            if not model.supports(args.k, n, 1):
                print(f"skipped {path}: trained for k={model.config.k}", file=sys.stderr)
                continue
            methods.append(model)
            if args.hybrid_ablation and model.model.kind == "hybrid":
                for suffix, extra in (("unet", {"steps": 0, "refine_steps": 0}),
                                      ("flow", {"steps": args.sample_steps, "refine_steps": 0})):
                    methods.append(load_reconstructor(path, device=args.device, name=f"{model.name}-{suffix}",
                                                      on_step=guard.check, **extra))
    if args.bitdepth_models:
        from msbgen.models.reconstructor import BitplaneReconstructor

        for folder in args.bitdepth_models:
            bitplane = BitplaneReconstructor(folder, device=args.device)
            if not bitplane.supports(args.k, n, 1):
                print(f"skipped {folder}: missing bit_NN.pt for k={args.k}", file=sys.stderr)
                continue
            methods.append(bitplane)
    names = [m.name for m in methods]
    if len(set(names)) != len(names):
        p.error(f"method names must be unique, got {names} (run folder names are used for checkpoints)")
    lpips_fn = None
    if args.lpips:
        from msbgen.models.perceptual import LPIPS

        lpips_fn = LPIPS(args.device)
    dists_fn = features_fn = None
    if args.dists:
        from msbgen.models.perceptual import DISTS

        dists_fn = DISTS(args.device)
    if args.kid:
        from msbgen.models.perceptual import InceptionFeatures

        features_fn = InceptionFeatures(args.device)
    ref_features: list[np.ndarray] = []
    method_features: dict[str, list[np.ndarray]] = {m.name: [] for m in methods}

    out = args.out or Path("outputs") / "eval" / args.data.resolve().name
    img_dir = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        img_dir.mkdir(exist_ok=True)

    print(guard.describe(), file=sys.stderr)
    rows = []
    for i, (name, image) in enumerate(images):
        guard.check()
        tiles = [("original", image)]
        top = keep_top_bits(image, args.k, n)
        if features_fn:
            ref_features.append(features_fn(image_patches(image, args.kid_patch), 255.0))
        for method in methods:
            draws = args.samples if method.stochastic else 1
            t0 = time.perf_counter()
            estimate = method.sample(top, args.k, n, seed=0)
            seconds = time.perf_counter() - t0
            scores = evaluate(image, estimate, args.k, n, with_ssim=not args.no_ssim)
            scores["lpips"] = lpips_fn(image, estimate, 255.0) if lpips_fn else None
            scores["dists"] = dists_fn(image, estimate, 255.0) if dists_fn else None
            scores["time_s"] = seconds
            scores["psnr_avg"] = scores["diversity"] = None
            samples = [estimate] + [method.sample(top, args.k, n, seed=j) for j in range(1, draws)]
            if draws > 1:
                stack = np.stack(samples)
                scores["psnr_avg"] = psnr(image, stack.mean(0), 255.0)
                scores["diversity"] = float(stack.std(0).mean())
            if features_fn:
                method_features[method.name] += [features_fn(image_patches(x, args.kid_patch), 255.0) for x in samples]
            rows.append({"image": name, "method": method.name, **{m: scores[m] for m in METRICS}})
            if i < args.save_images:
                rounded = round_estimate(estimate, n)
                Image.fromarray(rounded).save(img_dir / f"{name}_{method.name}.png")
                tiles.append((f"{method.name}  {scores['psnr']:.2f} dB", rounded))
                if draws > 1:
                    stack = np.stack(samples)
                    strip = [(f"sample {j}  {psnr(image, x, 255.0):.2f} dB", round_estimate(x, n))
                             for j, x in enumerate(samples)]
                    strip.append((f"mean of {draws}  {scores['psnr_avg']:.2f} dB", round_estimate(stack.mean(0), n)))
                    strip.append((f"std x{STD_GAIN}", np.clip(stack.std(0) * STD_GAIN, 0, 255).astype(np.uint8)))
                    comparison_strip([("original", image), *strip]).save(img_dir / f"{name}_{method.name}_samples.png")
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
        entry["kid"] = None
        if features_fn:
            feats, ref = np.concatenate(method_features[method.name]), np.concatenate(ref_features)
            if kid_subset_size(len(feats), len(ref)) >= 2:
                mean, std = kid(feats, ref)
                entry["kid"] = {"mean": mean, "std": std, "patches": len(feats), "reference_patches": len(ref),
                                "subset_size": kid_subset_size(len(feats), len(ref))}
        if hasattr(method, "path"):
            entry.update(checkpoint=str(method.path), step=method.step, config=method.config.to_dict(),
                         model_kind=method.model.kind)
        if hasattr(method, "sampler"):
            entry.update(method.sampler)
        summary[method.name] = entry
    meta = {"data": str(args.data), "images": len(images), "k": args.k, "bit_depth": n, "samples": args.samples}
    if args.kid:
        meta["kid_patch"] = args.kid_patch
    (out / "summary.json").write_text(json.dumps({"meta": meta, "methods": summary}, indent=2))

    print(f"\n{len(images)} images, k = {args.k}: mean over images\n")
    extra = args.samples > 1
    head = (f"{'method':<20} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'DISTS':>7} {'MAE':>7} {'MSB ok':>7}"
            + (f" {'PSNR avg':>9} {'divers.':>8}" if extra else "")
            + (f" {'KID e3':>7}" if args.kid else "") + f" {'time s':>7}")
    print(head + "\n" + "-" * len(head))
    for name_, e in summary.items():
        def cell(m, fmt, width=7, scale=1.0):
            return format(e[m]["mean"] * scale, fmt) if e[m] else f"{'-':>{width}}"
        print(f"{name_:<20} {cell('psnr', '7.2f')} {cell('ssim', '7.4f')} {cell('lpips', '7.4f')} "
              f"{cell('dists', '7.4f')} {cell('mae', '7.2f')} {cell('msb_consistency', '7.4f')}"
              + (f" {cell('psnr_avg', '9.2f', 9)} {cell('diversity', '8.2f', 8)}" if extra else "")
              + (f" {cell('kid', '7.2f', scale=1e3)}" if args.kid else "")
              + f" {cell('time_s', '7.3f')}")
    if extra:
        print(f"\nPSNR avg: PSNR of the mean of {args.samples} samples; divers.: mean per-pixel std across them (levels).")
    if args.kid:
        print(f"KID e3: patch KID × 1000 vs the originals ({args.kid_patch} px patches, every sample); lower is better.")
    print(f"\nWrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
