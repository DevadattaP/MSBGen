"""Train, evaluate, and plot the paper's bitplane-wise PyTorch baseline.

Examples:
    python -m msbgen.cli.bitdepth train --input-bits 1 --train data/processed/div2k-train
    python -m msbgen.cli.bitdepth evaluate --models runs/bitdepth --test data/processed/kodak
    python -m msbgen.cli.bitdepth plot outputs/bitdepth/evaluation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from msbgen.core.metrics import mae, psnr, ssim
from msbgen.models.bitdepth_recovery import (
    BitplaneDataset,
    BitplaneResidualCNN,
    BitplaneTrainConfig,
    recover_bitplanes,
    train_bitplane,
)
from msbgen.utils.thermal import ThermalLimits


def _arrays(folder: Path, limit: int | None = None) -> list[tuple[str, np.ndarray]]:
    files = sorted(folder.glob("*.npy"))[:limit]
    if not files:
        raise FileNotFoundError(f"no .npy images found in {folder}")
    return [(path.stem, np.load(path)) for path in files]


def _model_path(folder: Path, input_bits: int) -> Path:
    return folder / f"bit_{input_bits:02d}.pt"


def train_command(args: argparse.Namespace) -> int:
    train_images = [image for _, image in _arrays(args.train, args.limit)]
    val_images = [image for _, image in _arrays(args.val, args.val_limit)] if args.val else None
    channels = 1 if train_images[0].ndim == 2 else train_images[0].shape[-1]
    model = BitplaneResidualCNN(channels=channels, depth=args.depth)
    dataset = BitplaneDataset(train_images, args.input_bits, args.bit_depth, args.patch_size, args.augment)
    val_data = (BitplaneDataset(val_images, args.input_bits, args.bit_depth, args.patch_size)
                if val_images else None)
    config = BitplaneTrainConfig(args.epochs, args.batch_size, args.lr, args.lr_drop_epoch,
                                 args.lr_drop_factor, args.workers, args.device,
                                 not args.no_thermal_guard,
                                 ThermalLimits(args.gpu_max_temp, args.gpu_resume_temp,
                                               args.cpu_max_temp, args.cpu_resume_temp))
    history = train_bitplane(model, dataset, config, val_data)
    args.out.mkdir(parents=True, exist_ok=True)
    path = _model_path(args.out, args.input_bits)
    torch.save({"config": model.config, "state_dict": model.state_dict(), "history": history,
                "input_bits": args.input_bits, "bit_depth": args.bit_depth}, path)
    (args.out / f"bit_{args.input_bits:02d}_history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {path}")
    return 0


def _load_model(path: Path, device: str) -> BitplaneResidualCNN:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = BitplaneResidualCNN(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def evaluate_command(args: argparse.Namespace) -> int:
    images = _arrays(args.test, args.limit)
    device = torch.device(args.device)
    models = {bits: _load_model(_model_path(args.models, bits), args.device)
              for bits in range(min(args.start_bits), args.end_bits)}
    rows = []
    for name, image in images:
        source = torch.from_numpy(np.ascontiguousarray(image)).float()
        if image.ndim == 2:
            source = source[None, None]
        else:
            source = source.permute(2, 0, 1)[None]
        source = source / ((1 << args.bit_depth) - 1)
        for start_bits in args.start_bits:
            estimate = recover_bitplanes(source, [models[b] for b in range(start_bits, args.end_bits)],
                                         start_bits, args.bit_depth, device)[0].cpu().numpy()
            estimate = estimate[0] if image.ndim == 2 else np.moveaxis(estimate, 0, -1)
            estimate *= (1 << args.bit_depth) - 1
            reference = image.astype(np.float32)
            row = {"image": name, "start_bits": start_bits,
                   "psnr": psnr(reference, estimate, (1 << args.bit_depth) - 1),
                   "ssim": ssim(reference, estimate, (1 << args.bit_depth) - 1),
                   "mae": mae(reference, estimate)}
            rows.append(row)
            if args.save_images:
                out = args.out / f"{name}_{start_bits}to{args.end_bits}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.clip(np.rint(estimate), 0, 255).astype(np.uint8)).save(out)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "start_bits", "psnr", "ssim", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out / 'evaluation.csv'}")
    return 0


def plot_command(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(args.csv.open()))
    starts = sorted({int(row["start_bits"]) for row in rows})
    means = [np.mean([float(row["psnr"]) for row in rows if int(row["start_bits"]) == start]) for start in starts]
    ssims = [np.mean([float(row["ssim"]) for row in rows if int(row["start_bits"]) == start]) for start in starts]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(starts, means, marker="o")
    axes[0].set(xlabel="input bits", ylabel="PSNR (dB)", title="Bitplane recovery")
    axes[1].plot(starts, ssims, marker="o")
    axes[1].set(xlabel="input bits", ylabel="SSIM")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--input-bits", type=int, required=True)
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--val", type=Path)
    train.add_argument("--out", type=Path, default=Path("runs/bitdepth"))
    train.add_argument("--bit-depth", type=int, default=8)
    train.add_argument("--depth", type=int, default=4)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--patch-size", type=int, default=48)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--lr-drop-epoch", type=int, default=16)
    train.add_argument("--lr-drop-factor", type=float, default=5.0)
    train.add_argument("--workers", type=int, default=4,
                       help="data-loader workers (default 4; use the allocated CPU count)")
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--augment", action="store_true")
    train.add_argument("--limit", type=int)
    train.add_argument("--val-limit", type=int)
    train.add_argument("--no-thermal-guard", action="store_true",
                       help="disable GPU/CPU thermal throttling")
    train.add_argument("--gpu-max-temp", type=float, default=80.0)
    train.add_argument("--gpu-resume-temp", type=float, default=72.0)
    train.add_argument("--cpu-max-temp", type=float, default=90.0)
    train.add_argument("--cpu-resume-temp", type=float, default=80.0)
    train.set_defaults(function=train_command)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--models", type=Path, required=True)
    evaluate.add_argument("--test", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, default=Path("outputs/bitdepth"))
    evaluate.add_argument("--start-bits", nargs="+", type=int, default=[1, 2, 3],
                          help="input bit depths to evaluate (default: 1 2 3)")
    evaluate.add_argument("--end-bits", type=int, default=8)
    evaluate.add_argument("--bit-depth", type=int, default=8)
    evaluate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--save-images", action="store_true")
    evaluate.set_defaults(function=evaluate_command)

    plot = commands.add_parser("plot")
    plot.add_argument("csv", type=Path)
    plot.add_argument("--out", type=Path, default=Path("outputs/bitdepth/recovery.png"))
    plot.set_defaults(function=plot_command)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())