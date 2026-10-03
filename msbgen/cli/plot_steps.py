"""Plot generative models against their number of sampling steps.

    python -m msbgen.cli.plot_steps                              # every outputs/eval/*/
    python -m msbgen.cli.plot_steps outputs/eval/kodak-gen-s* outputs/eval/kodak-fm

Collects every generative method from the evaluation folders (typically one folder per
step count, written by `evaluate --sample-steps N`) and writes to --out (default
outputs/plots), with <name> = <data folder>-k<k>:

    steps_<name>_curves.png     PSNR, DISTS, KID and sample diversity against the number
                                of steps, one line per model (and sampler setting), with
                                the best-PSNR regression model as a dashed reference
    steps_<name>_tradeoff.png   perception vs distortion: PSNR against DISTS, each model's
                                step counts joined in order, with the regression models
                                and the midpoint fill as reference points

All folders must be on the same data and k. When the same model and step count is in
several folders, the most recently written summary wins. Bands are 95 % intervals of the
mean over images (unpaired; per_image.csv has the paired view) and, for KID, ± the std
over KID subsets.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from msbgen.cli.plot_eval import GRID, INK, INK_2, KIND_COLOURS, KIND_LABELS, _place_labels, _style, read_eval
from msbgen.core import summary_method_kind as method_kind

# Categorical slots 2…8 of the reference palette, in order; slot 1 (blue) is the regression reference.
SERIES_COLOURS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

# key → (title, subtitle, scale)
CURVES = {
    "psnr": ("PSNR (dB)", "higher is better", 1.0),
    "dists": ("DISTS", "lower is better", 1.0),
    "kid": ("KID × 10³", "lower is better", 1e3),
    "diversity": ("Diversity (levels)", "per-pixel std across samples", 1.0),
}
REFERENCE_FILLS = ("mid",)  # fills shown in the trade-off plot; zero / random are far off the scale
# Label offsets (points) for step numbers: above the line first, or below it first.
ABOVE = [(0, 7, "center"), (6, 5, "left"), (-6, 5, "right"), (0, 15, "center")]
BELOW = [(0, -14, "center"), (6, -12, "left"), (-6, -12, "right"), (0, -22, "center")]


def series_label(name: str, entry: dict) -> str:
    """The method name, plus the sampler settings that differ from the usual defaults."""
    if entry.get("solver") not in (None, "euler"):
        return f"{name} ({entry['solver']})"
    if entry.get("eta"):
        return f"{name} (η = {entry['eta']:g})"
    return name


def collect(folders: list[Path]) -> tuple[dict, dict[str, dict[int, dict]], dict[str, dict]]:
    """(meta, {series: {steps: entry}}, {reference method: entry}) over all folders."""
    meta = None
    series: dict[str, dict[int, dict]] = {}
    refs: dict[str, dict] = {}
    for folder in sorted(folders, key=lambda f: (f / "summary.json").stat().st_mtime):  # newest wins
        summary, _ = read_eval(folder)
        m = summary["meta"]
        if meta is None:
            meta = m
        elif (Path(m["data"]).name, m["k"]) != (Path(meta["data"]).name, meta["k"]):
            raise ValueError(f"{folder}: data {m['data']}, k = {m['k']} differs from "
                             f"{meta['data']}, k = {meta['k']}")
        for name, entry in summary["methods"].items():
            entry = {**entry, "images": m["images"]}
            if method_kind(name, entry) == "generative":
                if entry.get("sample_steps"):
                    series.setdefault(series_label(name, entry), {})[int(entry["sample_steps"])] = entry
            else:
                refs[name] = entry
    return meta, series, refs


def _stat(entry: dict, key: str, scale: float) -> tuple[float, float] | None:
    """(mean, half-width of the band) or None."""
    s = entry.get(key)
    if not s:
        return None
    if key == "kid":
        return s["mean"] * scale, s.get("std", 0.0) * scale if "subset_size" in s else 0.0
    n = entry.get("images", 1)
    return s["mean"] * scale, 1.96 * s["std"] * scale / math.sqrt(n) if n > 1 else 0.0


def _best_regression(refs: dict[str, dict]) -> str | None:
    reg = [n for n, e in refs.items() if method_kind(n, e) == "regression" and e.get("psnr")]
    return max(reg, key=lambda n: refs[n]["psnr"]["mean"]) if reg else None


def _styles(series: dict) -> dict[str, tuple[str, str]]:
    """Colour and marker per series, by sorted name, so a series keeps its look across plots."""
    names = sorted(series)
    if len(names) > len(SERIES_COLOURS):
        raise ValueError(f"{len(names)} generative series; at most {len(SERIES_COLOURS)} fit in one plot")
    return {n: (SERIES_COLOURS[i], SERIES_MARKERS[i]) for i, n in enumerate(names)}


def plot_curves(plt, name: str, meta: dict, series: dict, refs: dict, path: Path) -> bool:
    keys = [k for k in CURVES if any(_stat(e, k, 1.0) for s in series.values() for e in s.values())]
    if not keys:
        return False
    styles = _styles(series)
    ref = _best_regression(refs)
    all_steps = sorted({s for points in series.values() for s in points})
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.5), squeeze=False)
    for ax, key in zip(axes[0], keys):
        title, subtitle, scale = CURVES[key]
        for label in sorted(series):
            colour, marker = styles[label]
            pts = [(s, _stat(e, key, scale)) for s, e in sorted(series[label].items())]
            pts = [(s, st) for s, st in pts if st]
            if not pts:
                continue
            xs = [s for s, _ in pts]
            ys = [m for _, (m, _) in pts]
            err = [h for _, (_, h) in pts]
            ax.fill_between(xs, [y - h for y, h in zip(ys, err)], [y + h for y, h in zip(ys, err)],
                            color=colour, alpha=0.15, lw=0)
            ax.plot(xs, ys, color=colour, lw=2, marker=marker, ms=6, mec="white", mew=1.5, zorder=3)
        ref_stat = _stat(refs[ref], key, scale) if ref else None
        if ref_stat:
            ax.axhline(ref_stat[0], color=KIND_COLOURS["regression"], lw=1.2, ls="--", zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(all_steps, [str(s) for s in all_steps])
        ax.minorticks_off()
        ax.set_xlabel("sampling steps", color=INK_2, fontsize=9)
        ax.set_title(f"{title}\n{subtitle}", fontsize=9, color=INK)
        _style(ax)
    handles = [plt.Line2D([], [], color=c, lw=2, marker=mk, ms=6, mec="white", mew=1.5) for c, mk in
               (styles[n] for n in sorted(series))]
    labels = sorted(series)
    if ref:
        handles.append(plt.Line2D([], [], color=KIND_COLOURS["regression"], lw=1.2, ls="--"))
        labels.append(f"{ref} (best regression model)")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK_2)
    fig.suptitle(f"{name}: generative models by number of sampling steps "
                 f"({meta['images']} images; bands: 95 % interval, KID ± std)", fontsize=10, color=INK, y=1.08)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_tradeoff(plt, name: str, series: dict, refs: dict, path: Path) -> bool:
    lines = {label: [(s, e["psnr"]["mean"], e["dists"]["mean"]) for s, e in sorted(points.items())
                     if e.get("psnr") and e.get("dists")] for label, points in series.items()}
    lines = {label: pts for label, pts in lines.items() if pts}
    if not lines:
        return False
    styles = _styles(series)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    step_labels = {}
    for label in sorted(lines):
        colour, marker = styles[label]
        pts = lines[label]
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=colour, lw=2, marker=marker, ms=7,
                mec="white", mew=1.5, zorder=3)
        step_labels[label] = [(str(s), x, y) for s, x, y in pts]
    ref_points = []
    kinds = set()
    for n, e in refs.items():
        kind = method_kind(n, e)
        if (kind == "regression" or n in REFERENCE_FILLS) and e.get("psnr") and e.get("dists"):
            ax.plot(e["psnr"]["mean"], e["dists"]["mean"], "o", ms=9, color=KIND_COLOURS[kind], mec="white",
                    mew=2, zorder=3)
            ref_points.append((n, e["psnr"]["mean"], e["dists"]["mean"]))
            kinds.add(kind)
    ax.set_title("better: towards the bottom right", fontsize=8, color=INK_2, loc="right")
    ax.set_xlabel("PSNR (dB) — distortion, higher is better", color=INK_2, fontsize=9)
    ax.set_ylabel("DISTS — perception, lower is better", color=INK_2, fontsize=9)
    ax.margins(0.12)
    _style(ax)
    handles = [plt.Line2D([], [], color=c, lw=2, marker=mk, ms=7, mec="white", mew=1.5) for c, mk in
               (styles[n] for n in sorted(lines))]
    labels = sorted(lines)
    for kind in ("regression", "fill"):
        if kind in kinds:
            handles.append(plt.Line2D([], [], ls="", marker="o", ms=9, color=KIND_COLOURS[kind]))
            labels.append(KIND_LABELS[kind])
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK_2)
    fig.suptitle(f"{name}: perception vs distortion; numbers are sampling steps", fontsize=10, color=INK, y=1.06)
    fig.tight_layout()
    fig.canvas.draw()  # final limits before measuring label boxes
    # Step numbers: alternate series take the side above / below their line, so a number
    # never sits between two lines where it could belong to either.
    taken = _place_labels(fig, ax, ref_points)
    for i, label in enumerate(sorted(lines)):
        offsets = ABOVE if i % 2 == 0 else BELOW
        taken = _place_labels(fig, ax, step_labels[label], offsets, taken)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evals", nargs="*", type=Path, help="evaluation folders (default: every outputs/eval/*/ with a summary.json)")
    p.add_argument("--out", type=Path, default=Path("outputs/plots"), help="output folder (default outputs/plots)")
    p.add_argument("--name", help="name in the output files (default: <data folder>-k<k>)")
    args = p.parse_args(argv)

    evals = args.evals or sorted(d for d in Path("outputs/eval").glob("*") if (d / "summary.json").exists())
    evals = [e for e in evals if (e / "summary.json").exists()]
    if not evals:
        p.error("no evaluation folders with a summary.json found")
    try:
        meta, series, refs = collect(evals)
        _styles(series)
    except ValueError as exc:
        p.error(str(exc))
    if not series:
        p.error("no generative models with a sample_steps entry in these folders")
    name = args.name or f"{Path(meta['data']).name}-k{meta['k']}"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    if plot_curves(plt, name, meta, series, refs, args.out / f"steps_{name}_curves.png"):
        written.append(args.out / f"steps_{name}_curves.png")
    if plot_tradeoff(plt, name, series, refs, args.out / f"steps_{name}_tradeoff.png"):
        written.append(args.out / f"steps_{name}_tradeoff.png")
    for w in written:
        print(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
