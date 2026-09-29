"""Plot training histories (loss, PSNR, SSIM, learning rate) of one or more runs.

    python -m msbgen.cli.plot_runs                       # every runs/*/log.jsonl
    python -m msbgen.cli.plot_runs runs/unet-res runs/unet --x epoch

Writes to --out (default outputs/plots):

    <run>_history.csv     training and validation records merged into one row per step
    <run>_overview.png    per run: train vs val loss, train vs val PSNR, val SSIM, LR
    loss.png              all runs: training loss (faint) and validation loss (markers)
    val_psnr.png          all runs: validation PSNR, with the midpoint- and random-fill baselines
    val_gain.png          all runs: PSNR gain over midpoint fill
    val_ssim.png          all runs: validation SSIM, with the midpoint- and random-fill baselines

Runs logged before the random fill was added have no random-fill values; their line is
simply left out.
    lr.png                all runs: learning-rate schedule
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

TRAIN_FIELDS = ["loss", "train_psnr", "lr", "it_per_s", "gpu_temp", "cpu_temp", "paused_s"]
VAL_FIELDS = ["val_loss", "val_psnr", "val_psnr_mid", "val_psnr_random", "val_gain_db", "val_ssim", "val_ssim_mid",
              "val_ssim_random", "val_msb_consistency", "best_val_psnr"]
BASELINES = {"mid": ("--", "midpoint fill"), "random": (":", "random fill")}  # fill → (line style, label)


def read_history(run_dir: Path) -> tuple[list[dict], list[dict]]:
    """(train records, validation records), each sorted by step, one record per step.

    If a step appears twice (a crash followed by --resume in an older version), the
    later record wins.
    """
    train: dict[int, dict] = {}
    val: dict[int, dict] = {}
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        (val if "val_psnr" in rec else train)[rec["step"]] = rec
    return [train[s] for s in sorted(train)], [val[s] for s in sorted(val)]


def write_history_csv(path: Path, train: list[dict], val: list[dict]) -> None:
    rows: dict[int, dict] = {}
    for rec in train + val:
        rows.setdefault(rec["step"], {"step": rec["step"]}).update(rec)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", *TRAIN_FIELDS, *VAL_FIELDS], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[s] for s in sorted(rows))


def smooth(values: list[float], window: int) -> np.ndarray:
    """Trailing moving average (window 1 = unchanged); keeps the curve's length."""
    v = np.asarray(values, dtype=float)
    if window <= 1 or len(v) < 2:
        return v
    kernel = np.ones(min(window, len(v)))
    return np.convolve(v, kernel, "full")[: len(v)] / np.convolve(np.ones_like(v), kernel, "full")[: len(v)]


def series(records: list[dict], key: str, x: str) -> tuple[list[float], list[float]]:
    pts = [(r.get(x, r["step"]), r[key]) for r in records if r.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", type=Path, help="run folders (default: every runs/*/ with a log.jsonl)")
    p.add_argument("--out", type=Path, default=Path("outputs/plots"), help="output folder (default outputs/plots)")
    p.add_argument("--x", choices=["step", "epoch"], default="step", help="x axis (default step)")
    p.add_argument("--smooth", type=int, default=5, help="moving-average window for training curves (default 5)")
    args = p.parse_args(argv)

    runs = args.runs or sorted(d for d in Path("runs").glob("*") if (d / "log.jsonl").exists())
    runs = [r for r in runs if (r / "log.jsonl").exists()]
    if not runs:
        p.error("no runs with a log.jsonl found")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    histories = {r.name: read_history(r) for r in runs}
    xlabel = "training step" if args.x == "step" else "epoch (passes over the training pixels)"
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    written = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(args.out / name, dpi=150)
        plt.close(fig)
        written.append(name)

    # ---- per run ----
    for name, (train, val) in histories.items():
        write_history_csv(args.out / f"{name}_history.csv", train, val)
        written.append(f"{name}_history.csv")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        ax = axes[0, 0]
        x, y = series(train, "loss", args.x)
        ax.plot(x, y, color="C0", alpha=0.25, lw=0.8)
        ax.plot(x, smooth(y, args.smooth), color="C0", label="train (smoothed)")
        ax.plot(*series(val, "val_loss", args.x), "o-", color="C1", ms=3, label="validation")
        ax.set(yscale="log", title="Loss", xlabel=xlabel)
        ax.legend()
        ax = axes[0, 1]
        x, y = series(train, "train_psnr", args.x)
        if y:
            ax.plot(x, smooth(y, args.smooth), color="C0", label="train (smoothed)")
        ax.plot(*series(val, "val_psnr", args.x), "o-", color="C1", ms=3, label="validation")
        for fill, (ls, label) in BASELINES.items():
            x, y = series(val, f"val_psnr_{fill}", args.x)
            if y:
                ax.plot(x, y, ls, color="0.5", label=f"{label} (val)")
        ax.set(title="PSNR (dB)", xlabel=xlabel)
        ax.legend()
        ax = axes[1, 0]
        ax.plot(*series(val, "val_ssim", args.x), "o-", color="C1", ms=3, label="validation")
        for fill, (ls, label) in BASELINES.items():
            x, y = series(val, f"val_ssim_{fill}", args.x)
            if y:
                ax.plot(x, y, ls, color="0.5", label=f"{label} (val)")
        ax.set(title="SSIM", xlabel=xlabel)
        ax.legend()
        ax = axes[1, 1]
        ax.plot(*series(train, "lr", args.x), color="C2")
        ax.set(title="Learning rate", xlabel=xlabel, yscale="log")
        for a in axes.flat:
            a.grid(alpha=0.3)
        fig.suptitle(name)
        save(fig, f"{name}_overview.png")

    # ---- comparisons ----
    def compare(filename: str, title: str, ylabel: str, key: str, baseline: str | None = None,
                train_key: str | None = None, logy: bool = False) -> None:
        """`baseline`: metric prefix such as "val_psnr"; the fills in BASELINES are drawn as
        horizontal lines (mean over the first run that logged them)."""
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for i, (name, (train, val)) in enumerate(histories.items()):
            c = colours[i % len(colours)]
            if train_key:
                x, y = series(train, train_key, args.x)
                if y:
                    ax.plot(x, smooth(y, args.smooth), color=c, alpha=0.35, lw=1)
            ax.plot(*series(val, key, args.x), "o-", color=c, ms=3, label=name)
        for fill, (ls, label) in BASELINES.items() if baseline else ():
            y = next((y for _, val in histories.values() if (y := series(val, f"{baseline}_{fill}", args.x)[1])), None)
            if y:
                ax.axhline(float(np.mean(y)), ls=ls, color="0.5", label=label)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
        save(fig, filename)

    compare("loss.png", "Loss (faint: training, smoothed; markers: validation)", "loss", "val_loss",
            train_key="loss", logy=True)
    compare("val_psnr.png", "Validation PSNR", "PSNR (dB)", "val_psnr", baseline="val_psnr")
    compare("val_gain.png", "Validation PSNR gain over midpoint fill", "gain (dB)", "val_gain_db")
    compare("val_ssim.png", "Validation SSIM", "SSIM", "val_ssim", baseline="val_ssim")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (name, (train, _)) in enumerate(histories.items()):
        ax.plot(*series(train, "lr", args.x), color=colours[i % len(colours)], label=name)
    ax.set(title="Learning rate", xlabel=xlabel, yscale="log")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "lr.png")

    for name, (train, val) in histories.items():
        last = val[-1] if val else {}
        gain = f"{last['val_gain_db']:+.2f} dB" if "val_gain_db" in last else "-"
        print(f"{name:<16} {len(train):>5} train records, {len(val):>3} validations, "
              f"last step {max([r['step'] for r in train + val], default=0)}, last val gain {gain}")
    print(f"Wrote {len(written)} files to {args.out}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
