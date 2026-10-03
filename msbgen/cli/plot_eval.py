"""Plot evaluation results (the output folders of `msbgen.cli.evaluate`).

    python -m msbgen.cli.plot_eval                              # every outputs/eval/*/
    python -m msbgen.cli.plot_eval outputs/eval/kodak-s8 --pair diffusion unet-res

Writes to --out (default outputs/plots), per evaluation folder <name>:

    eval_<name>_metrics.png      one panel per metric (PSNR, SSIM, LPIPS, DISTS, KID),
                                 methods in the same order in every panel, coloured by
                                 kind: fill baseline, regression model, generative model
    eval_<name>_perception.png   perception vs distortion: PSNR against DISTS and LPIPS
    eval_<name>_paired.png       per image, method A against method B on each metric,
                                 with the y = x line; A is the generative model and B the
                                 regression model with the best PSNR, unless --pair is given

Error bars on per-image metrics are 95 % intervals of the mean over images. They are
unpaired, so they overstate how uncertain a difference between two methods is: the
paired plot shows that directly. KID error bars are the std over KID subsets (only in
summaries written by the current evaluate; older ones show none).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from msbgen.core import summary_method_kind as method_kind

# Reference data-viz palette: categorical slots 1 and 2 for the two model kinds, neutral
# gray for the no-model fills; ink and grid tokens for everything that is not data.
KIND_COLOURS = {"fill": "#8a8a86", "regression": "#2a78d6", "generative": "#eb6834"}
KIND_LABELS = {"fill": "fill baseline", "regression": "regression model", "generative": "generative model"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# key → (title, higher is better, scale)
PANELS = {
    "psnr": ("PSNR (dB)", True, 1.0),
    "ssim": ("SSIM", True, 1.0),
    "lpips": ("LPIPS", False, 1.0),
    "dists": ("DISTS", False, 1.0),
    "kid": ("KID × 10³", False, 1e3),
}



def read_eval(folder: Path) -> tuple[dict, dict[str, dict[str, dict[str, float]]]]:
    """(summary.json, {method: {image: {metric: value}}} from per_image.csv)."""
    summary = json.loads((folder / "summary.json").read_text())
    per_image: dict[str, dict[str, dict[str, float]]] = {}
    path = folder / "per_image.csv"
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f):
                values = {k: float(v) for k, v in row.items() if k not in ("image", "method") and v not in ("", None)}
                per_image.setdefault(row["method"], {})[row["image"]] = values
    return summary, per_image


def default_pair(methods: dict[str, dict]) -> tuple[str, str] | None:
    """(generative, best-PSNR regression model) if there is exactly one generative model."""
    gen = [n for n, e in methods.items() if method_kind(n, e) == "generative"]
    reg = [n for n, e in methods.items() if method_kind(n, e) == "regression" and e.get("psnr")]
    if len(gen) != 1 or not reg:
        return None
    return gen[0], max(reg, key=lambda n: methods[n]["psnr"]["mean"])


def _style(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelcolor=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


LABEL_OFFSETS = [(7, 4, "left"), (7, -11, "left"), (-7, 4, "right"), (-7, -11, "right"),
                 (7, 15, "left"), (7, -22, "left"), (-7, 15, "right"), (-7, -22, "right")]


def _place_labels(fig, ax, points: list[tuple[str, float, float]], offsets=LABEL_OFFSETS,
                  taken: list | None = None) -> list:
    """Name every point, trying `offsets` in order so labels don't overlap each other, the
    points, or the boxes in `taken` (from an earlier call). Returns all boxes now taken."""
    renderer = fig.canvas.get_renderer()
    taken = list(taken or [])
    markers = [ax.transData.transform((x, y)) for _, x, y in points]
    taken += [(px - 6, py - 6, px + 6, py + 6) for px, py in markers]  # marker boxes, display px
    for n, x, y in points:
        for dx, dy, ha in offsets:
            text = ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8, color=INK)
            bb = text.get_window_extent(renderer)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            if not any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
                break
            text.remove()
        else:  # nothing fits: keep the first position
            dx, dy, ha = offsets[0]
            text = ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8,
                               color=INK)
            bb = text.get_window_extent(renderer)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
        taken.append(box)
    return taken


def plot_metrics(plt, name: str, summary: dict, path: Path) -> bool:
    methods = summary["methods"]
    panels = [k for k in PANELS if any(e.get(k) for e in methods.values())]
    if not panels:
        return False
    names = list(methods)
    kinds = [method_kind(n, methods[n]) for n in names]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels) + 1.4, 0.42 * len(names) + 1.6),
                             sharey=True, squeeze=False)
    y = np.arange(len(names))[::-1]
    for ax, key in zip(axes[0], panels):
        title, higher, scale = PANELS[key]
        stats = [methods[n].get(key) for n in names]
        means = np.array([s["mean"] * scale if s else np.nan for s in stats])
        if key == "kid":
            err = [s["std"] * scale if s and "subset_size" in s else 0.0 for s in stats]
        else:
            count = [methods[n].get("images", 1) for n in names]
            err = [1.96 * s["std"] * scale / math.sqrt(c) if s and c > 1 else 0.0 for s, c in zip(stats, count)]
        ok = ~np.isnan(means)
        ax.barh(y[ok], means[ok], height=0.62, color=[KIND_COLOURS[k] for k, o in zip(kinds, ok) if o],
                edgecolor="white", linewidth=2)
        if any(err):
            ax.errorbar(means[ok], y[ok], xerr=np.array(err)[ok], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
        if key == "psnr":  # mean of N samples, for stochastic methods
            for yi, n in zip(y, names):
                avg = methods[n].get("psnr_avg")
                if avg:
                    ax.plot(avg["mean"], yi, "o", ms=8, mfc="white", mec=INK, mew=1.5, zorder=3)
        if ok.any():  # the best value, just past its whisker (or PSNR-avg marker)
            best = int(np.nanargmax(means) if higher else np.nanargmin(means))
            end = means[best] + err[best]
            avg = methods[names[best]].get("psnr_avg") if key == "psnr" else None
            end = max(end, avg["mean"]) if avg else end
            ax.annotate(f"{means[best]:.3g}", (end, y[best]), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8, fontweight="bold", color=INK)
        lo = np.nanmin(means[ok]) if ok.any() else 0
        hi = np.nanmax(means[ok] + np.array(err)[ok]) if ok.any() else 1
        ax.set_xlim(min(0.0, lo * 1.1), hi * 1.18)
        note = ""
        if key == "kid" and any(s and "subset_size" not in s for s in stats):
            note = " (no spread: rerun evaluate)"
        ax.set_title(f"{title}\n{'higher' if higher else 'lower'} is better{note}", fontsize=9, color=INK)
        _style(ax)
    axes[0, 0].set_yticks(y, names)
    axes[0, 0].tick_params(axis="y", labelcolor=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLOURS[k]) for k in KIND_COLOURS if k in kinds]
    labels = [KIND_LABELS[k] for k in KIND_COLOURS if k in kinds]
    if any(methods[n].get("psnr_avg") for n in names):
        handles.append(plt.Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=INK, mew=1.5))
        labels.append(f"PSNR of the mean of {summary['meta'].get('samples', 'N')} samples")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK_2)
    meta = summary["meta"]
    fig.suptitle(f"{name}: {meta['images']} images, k = {meta['k']} (bars: mean; whiskers: 95 % interval)",
                 fontsize=10, color=INK, y=1.07)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_perception(plt, name: str, summary: dict, path: Path) -> bool:
    methods = summary["methods"]
    ys = [k for k in ("dists", "lpips") if any(e.get(k) for e in methods.values())]
    if not ys or not any(e.get("psnr") for e in methods.values()):
        return False
    fig, axes = plt.subplots(1, len(ys), figsize=(5.2 * len(ys), 4.4), squeeze=False)
    labelled = []
    for ax, key in zip(axes[0], ys):
        points = [(n, e["psnr"]["mean"], e[key]["mean"]) for n, e in methods.items() if e.get("psnr") and e.get(key)]
        for n, x, y in points:
            ax.plot(x, y, "o", ms=9, color=KIND_COLOURS[method_kind(n, methods[n])], mec="white", mew=2)
        ax.set_title("better: towards the bottom right", fontsize=8, color=INK_2, loc="right")
        ax.set_xlabel("PSNR (dB) — distortion, higher is better", color=INK_2, fontsize=9)
        ax.set_ylabel(f"{PANELS[key][0]} — perception, lower is better", color=INK_2, fontsize=9)
        ax.margins(0.15)
        _style(ax)
        labelled.append((ax, points))
    kinds = {method_kind(n, e) for n, e in methods.items()}
    handles = [plt.Line2D([], [], ls="", marker="o", ms=9, color=KIND_COLOURS[k]) for k in KIND_COLOURS if k in kinds]
    fig.legend(handles, [KIND_LABELS[k] for k in KIND_COLOURS if k in kinds], loc="upper center",
               ncol=len(handles), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0), labelcolor=INK_2)
    fig.suptitle(f"{name}: perception vs distortion (mean over images; one sample per stochastic method)",
                 fontsize=10, color=INK, y=1.06)
    fig.tight_layout()
    fig.canvas.draw()  # final axes positions and limits, so label boxes are measured where they end up
    for ax, points in labelled:
        _place_labels(fig, ax, points)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_paired(plt, name: str, per_image: dict, pair: tuple[str, str], path: Path) -> bool:
    a, b = pair
    if a not in per_image or b not in per_image:
        return False
    images = sorted(set(per_image[a]) & set(per_image[b]))
    keys = [k for k in ("psnr", "ssim", "lpips", "dists")
            if all(k in per_image[m][i] for m in pair for i in images)]
    if not images or not keys:
        return False
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.9), squeeze=False)
    for ax, key in zip(axes[0], keys):
        title, higher, _ = PANELS[key]
        xs = np.array([per_image[b][i][key] for i in images])
        ys = np.array([per_image[a][i][key] for i in images])
        lo, hi = min(xs.min(), ys.min()), max(xs.max(), ys.max())
        pad = (hi - lo) * 0.06 or 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_2, lw=1, ls="--")
        ax.plot(xs, ys, "o", ms=7, color=KIND_COLOURS["generative"], mec="white", mew=1.5)
        wins = int(np.sum(ys > xs if higher else ys < xs))
        ax.set_title(f"{title}: {a} better on {wins}/{len(images)}\n({'higher' if higher else 'lower'} is better)",
                     fontsize=9, color=INK)
        ax.set_xlabel(b, color=INK_2, fontsize=9)
        ax.set_ylabel(a, color=INK_2, fontsize=9)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        _style(ax)
    fig.suptitle(f"{name}: {a} vs {b}, one point per image (dashed: equal)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evals", nargs="*", type=Path, help="evaluation folders (default: every outputs/eval/*/ with a summary.json)")
    p.add_argument("--out", type=Path, default=Path("outputs/plots"), help="output folder (default outputs/plots)")
    p.add_argument("--pair", nargs=2, metavar=("A", "B"),
                   help="methods for the paired plot (default: the generative model vs the best-PSNR regression model)")
    args = p.parse_args(argv)

    evals = args.evals or sorted(d for d in Path("outputs/eval").glob("*") if (d / "summary.json").exists())
    evals = [e for e in evals if (e / "summary.json").exists()]
    if not evals:
        p.error("no evaluation folders with a summary.json found")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for folder in evals:
        name = folder.resolve().name
        summary, per_image = read_eval(folder)
        if plot_metrics(plt, name, summary, args.out / f"eval_{name}_metrics.png"):
            written.append(f"eval_{name}_metrics.png")
        if plot_perception(plt, name, summary, args.out / f"eval_{name}_perception.png"):
            written.append(f"eval_{name}_perception.png")
        pair = tuple(args.pair) if args.pair else default_pair(summary["methods"])
        if pair and plot_paired(plt, name, per_image, pair, args.out / f"eval_{name}_paired.png"):
            written.append(f"eval_{name}_paired.png")
        elif args.pair:
            print(f"{name}: no paired plot, {pair[0]} or {pair[1]} is not in per_image.csv", file=sys.stderr)
    for w in written:
        print(args.out / w)
    print(f"Wrote {len(written)} files to {args.out}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
