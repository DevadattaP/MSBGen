"""Bit-plane analysis over a folder of images (dataset-level statistics).

    python -m msbgen.cli.analyze data/kodak --gray --plots

Every image is decoded the same way as in the web app (grayscale or RGB, native bit
depth), optionally converted to grayscale, and analysed with `analyze_image`. Images
are grouped by (colour mode, bit depth), e.g. "L-8" or "RGB-8", because statistics
of different shapes can't be averaged together.

Writes to --out (default outputs/analysis/<folder name>):

    planes.csv          one row per image × channel × bit-plane
    reconstruction.csv  one row per image × k (top-k planes kept)
    summary.json        per group: means over images, mean correlation matrix
    *.png               with --plots: entropy per plane, quality vs k, correlation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from msbgen.core import ImageDecodeError, analyze_image, load_image, to_grayscale

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
PLANE_FIELDS = ["file", "group", "channel", "bit", "ones_ratio", "entropy", "context_entropy", "spatial_correlation"]
REC_FIELDS = [
    "file", "group", "k", "bits_per_pixel", "coded_bpp",
    "psnr_zero", "psnr_mid", "ssim_zero", "ssim_mid", "mae_zero", "mae_mid",
]


def find_images(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def analyse_file(path: Path, root: Path, gray: bool, with_ssim: bool) -> dict:
    """Analyse one file; returns plain rows (picklable) or an error message."""
    name = str(path.relative_to(root))
    try:
        loaded = load_image(path.read_bytes())
    except ImageDecodeError as exc:
        return {"file": name, "error": str(exc)}

    array, mode = loaded.array, loaded.color_mode
    if gray and mode == "RGB":
        array, mode = to_grayscale(array), "L"
    channels = ["L"] if mode == "L" else ["R", "G", "B"]
    group = f"{mode}-{loaded.bit_depth}"
    result = analyze_image(array, loaded.bit_depth, with_ssim=with_ssim)

    planes = [
        {
            "file": name, "group": group, "channel": channels[p.channel_index], "bit": p.bit,
            "ones_ratio": p.ones_ratio, "entropy": p.entropy, "context_entropy": p.context_entropy,
            "spatial_correlation": p.spatial_correlation,
        }
        for p in result.planes
    ]
    recs = [
        {
            "file": name, "group": group, "k": r.k, "bits_per_pixel": r.bits_per_pixel, "coded_bpp": r.coded_bpp,
            **{f"{metric}_{fill}": getattr(r, metric)[fill] for metric in ("psnr", "ssim", "mae") for fill in ("zero", "mid")},
        }
        for r in result.reconstructions
    ]
    return {
        "file": name, "group": group, "bit_depth": loaded.bit_depth, "channels": channels,
        "width": loaded.width, "height": loaded.height,
        "planes": planes, "reconstructions": recs, "correlation": result.correlation,
    }


def _mean(values: list[float | None]) -> float | None:
    """Mean over finite values (PSNR of a lossless reconstruction is inf and is skipped)."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def summarise(results: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)

    summary = {}
    for group, items in sorted(groups.items()):
        first = items[0]
        n = first["bit_depth"]
        planes = []
        for channel in first["channels"]:
            for bit in range(n - 1, -1, -1):
                rows = [p for r in items for p in r["planes"] if p["channel"] == channel and p["bit"] == bit]
                planes.append({
                    "channel": channel, "bit": bit,
                    **{f: _mean([row[f] for row in rows]) for f in PLANE_FIELDS[4:]},
                })
        recs = []
        for k in range(1, n + 1):
            rows = [x for r in items for x in r["reconstructions"] if x["k"] == k]
            recs.append({
                "k": k, "bits_per_pixel": rows[0]["bits_per_pixel"],
                **{f: _mean([row[f] for row in rows]) for f in REC_FIELDS[4:]},
                "lossless_images": sum(1 for row in rows if row["psnr_mid"] == math.inf),
            })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells: plane constant in every image
            corr = np.nanmean(np.stack([r["correlation"] for r in items]), axis=0)
        labels = [f"{c}·B{b}" if len(first["channels"]) > 1 else f"B{b}" for c in first["channels"] for b in range(n - 1, -1, -1)]
        summary[group] = {
            "images": len(items),
            "bit_depth": n,
            "channels": first["channels"],
            "mean_pixels": float(np.mean([r["width"] * r["height"] for r in items])),
            "planes": planes,
            "reconstructions": recs,
            "correlation": {
                "labels": labels,
                "matrix": [[None if math.isnan(v) else round(float(v), 6) for v in row] for row in corr],
            },
        }
    return summary


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict) -> None:
    for group, s in summary.items():
        print(f"\n== {group}: {s['images']} image(s), mean {s['mean_pixels'] / 1e6:.2f} MP ==")
        print(f"{'plane':>8} {'H iid':>7} {'H|ctx':>7} {'rho':>6}")
        for p in s["planes"]:
            label = f"{p['channel']}·B{p['bit']}" if len(s["channels"]) > 1 else f"B{p['bit']}"
            rho = "-" if p["spatial_correlation"] is None else f"{p['spatial_correlation']:.3f}"
            print(f"{label:>8} {p['entropy']:7.3f} {p['context_entropy']:7.3f} {rho:>6}")
        print(f"\n{'k':>3} {'bpp':>4} {'coded':>7} {'PSNR mid':>9} {'PSNR zero':>10} {'SSIM mid':>9} {'SSIM zero':>10}")
        for r in s["reconstructions"]:
            def f(v, d):
                return "inf" if v is None else f"{v:.{d}f}"
            print(
                f"{r['k']:>3} {r['bits_per_pixel']:>4} {r['coded_bpp']:7.3f} {f(r['psnr_mid'], 2):>9} "
                f"{f(r['psnr_zero'], 2):>10} {f(r['ssim_mid'], 4):>9} {f(r['ssim_zero'], 4):>10}"
            )


def make_plots(summary: dict, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"L": "0.25", "R": "#e0384b", "G": "#1f9d55", "B": "#2f6fe4"}
    written = []
    for group, s in summary.items():
        n = s["bit_depth"]
        bits = list(range(n - 1, -1, -1))

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for ch in s["channels"]:
            rows = [p for p in s["planes"] if p["channel"] == ch]
            prefix = f"{ch} " if len(s["channels"]) > 1 else ""
            ax.plot(range(n), [p["context_entropy"] for p in rows], "-o", color=colours[ch], ms=4, label=f"{prefix}H | ctx")
            ax.plot(range(n), [p["entropy"] for p in rows], "--", color=colours[ch], alpha=0.7, label=f"{prefix}H i.i.d.")
        ax.set(xticks=range(n), xticklabels=[f"B{b}" for b in bits], ylim=(0, 1.05),
               xlabel="bit-plane (MSB → LSB)", ylabel="bits / pixel", title=f"Entropy per plane · {group} (n={s['images']})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        written.append(_save(fig, out / f"{group}_entropy.png"))

        ks = [r["k"] for r in s["reconstructions"]]
        for metric, label in (("psnr", "PSNR (dB)"), ("ssim", "SSIM")):
            if all(r[f"{metric}_mid"] is None for r in s["reconstructions"]):
                continue
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            for fill, style in (("mid", "-o"), ("zero", "--s")):
                ax.plot(ks, [r[f"{metric}_{fill}"] for r in s["reconstructions"]], style, ms=4,
                        label=f"{'midpoint' if fill == 'mid' else 'zero'} fill")
            ax.set(xticks=ks, xlabel="k (most-significant planes kept)", ylabel=label,
                   title=f"Top-k reconstruction · {group} (n={s['images']})")
            ax.grid(alpha=0.3)
            ax.legend()
            written.append(_save(fig, out / f"{group}_{metric}.png"))

        matrix = np.array([[np.nan if v is None else v for v in row] for row in s["correlation"]["matrix"]])
        size = len(matrix)
        fig, ax = plt.subplots(figsize=(1.6 + size * 0.28, 1.2 + size * 0.28))
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        labels = s["correlation"]["labels"]
        ax.set_xticks(range(size), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(size), labels, fontsize=7)
        ax.set_title(f"Mean plane correlation · {group}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        written.append(_save(fig, out / f"{group}_correlation.png"))
    return written


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="folder of images")
    parser.add_argument("--out", type=Path, help="output folder (default: outputs/analysis/<input name>)")
    parser.add_argument("--gray", action="store_true", help="convert RGB images to grayscale (BT.601 luma) first")
    parser.add_argument("--recursive", "-r", action="store_true", help="search sub-folders too")
    parser.add_argument("--limit", type=int, help="analyse at most this many images")
    parser.add_argument("--no-ssim", action="store_true", help="skip SSIM (faster)")
    parser.add_argument("--plots", action="store_true", help="save PNG plots (needs matplotlib)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="parallel processes")
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        parser.error(f"{args.input} is not a folder")
    paths = find_images(args.input, args.recursive)[: args.limit]
    if not paths:
        parser.error(f"no images found in {args.input}")
    out = args.out or Path("outputs") / "analysis" / args.input.resolve().name
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(p, args.input, args.gray, not args.no_ssim) for p in paths]
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyse_file, *job) for job in jobs]
        for i, future in enumerate(futures, 1):
            r = future.result()
            (errors if "error" in r else results).append(r)
            print(f"\r[{i}/{len(paths)}] {r['file'][:60]:<60}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    for e in errors:
        print(f"skipped {e['file']}: {e['error']}", file=sys.stderr)
    if not results:
        print("no images could be analysed", file=sys.stderr)
        return 1

    write_csv(out / "planes.csv", PLANE_FIELDS, [p for r in results for p in r["planes"]])
    write_csv(out / "reconstruction.csv", REC_FIELDS, [x for r in results for x in r["reconstructions"]])
    summary = summarise(results)
    meta = {"input": str(args.input), "gray": args.gray, "images": len(results), "skipped": [e["file"] for e in errors]}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "groups": summary}, indent=2))

    print_summary(summary)
    written = make_plots(summary, out) if args.plots else []
    print(f"\nWrote {out}/ (planes.csv, reconstruction.csv, summary.json{', ' + str(len(written)) + ' plots' if written else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
