"""Download, convert, and evaluate the authors' pretrained Keras models on Kodak.

The released checkpoints contain RGB D4/D16 models for bit positions 4--16.
For an 8-bit Kodak evaluation, the available pretrained models support 3->8
and 4->8 recovery; 1->8 and 2->8 checkpoints were not released.

Example:
    python -m msbgen.cli.evaluate_pretrained --data data/raw/kodak --depths 4 16 \
        --start-bits 3 4 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from msbgen.core.reconstruction import evaluate
from msbgen.models.bitdepth_recovery import BitplaneResidualCNN, recover_bitplanes

REPO_RAW = "https://raw.githubusercontent.com/abhijithpunnappurath/a-little-bit-more/master"
AVAILABLE_INPUT_BITS = range(3, 16)


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 conversion requires h5py; install it with `pip install h5py`") from exc
    return h5py


def _natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _keras_weights(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, np.ndarray]]]:
    h5py = _require_h5py()
    kernels: list[tuple[str, np.ndarray]] = []
    biases: list[tuple[str, np.ndarray]] = []
    batch_norms: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as file:
        def visit(name: str, item) -> None:
            if not isinstance(item, h5py.Dataset):
                return
            array = np.asarray(item, dtype=np.float32)
            if name.endswith("/kernel:0"):
                kernels.append((name, array))
            elif name.endswith("/bias:0"):
                biases.append((name, array))
            else:
                match = re.search(r"/(gamma|beta|moving_mean|moving_variance):0$", name)
                if match:
                    layer = next((part for part in name.split("/") if re.fullmatch(r"bn_?\d+", part)), None)
                    if layer is not None:
                        batch_norms.setdefault(layer, {})[match.group(1)] = array
        file.visititems(visit)
    kernels.sort(key=lambda item: _natural_key(item[0]))
    biases.sort(key=lambda item: _natural_key(item[0]))
    bn = [batch_norms[name] for name in sorted(batch_norms, key=_natural_key)]
    if not kernels or not bn:
        raise ValueError(f"no Keras convolution/BatchNorm weights found in {path}")
    return [array for _, array in kernels], [array for _, array in biases], bn


def convert_hdf5(source: Path, destination: Path, depth: int) -> Path:
    """Convert one released Keras checkpoint into the local PyTorch checkpoint format."""
    kernels, biases, batch_norms = _keras_weights(source)
    model = BitplaneResidualCNN(channels=3, depth=depth, filters=64)
    convolutions = [model.input]
    for block in model.blocks:
        convolutions.extend((block.conv1, block.conv2))
    convolutions.append(model.output)
    if len(kernels) != len(convolutions) or len(batch_norms) != 2 * depth + 1:
        raise ValueError(f"unexpected layer count in {source}: {len(kernels)} conv, {len(batch_norms)} BN")
    for layer, kernel in zip(convolutions, kernels, strict=True):
        expected = tuple(layer.weight.shape)
        converted = kernel.transpose(3, 2, 0, 1)
        if converted.shape != expected:
            raise ValueError(f"kernel shape mismatch: Keras {kernel.shape}, PyTorch {expected}")
        layer.weight.data.copy_(torch.from_numpy(converted))
    bias_layers = [layer for layer in convolutions if layer.bias is not None]
    if len(biases) != len(bias_layers):
        raise ValueError(f"unexpected bias count in {source}: {len(biases)} vs {len(bias_layers)}")
    for layer, bias in zip(bias_layers, biases, strict=True):
        if tuple(bias.shape) != tuple(layer.bias.shape):
            raise ValueError(f"bias shape mismatch in {source}")
        layer.bias.data.copy_(torch.from_numpy(bias))
    batch_layers = []
    for block in model.blocks:
        batch_layers.extend((block.bn1, block.bn2))
    batch_layers.append(model.output_bn)
    for layer, values in zip(batch_layers, batch_norms, strict=True):
        for key, target in (("gamma", layer.weight), ("beta", layer.bias),
                            ("moving_mean", layer.running_mean), ("moving_variance", layer.running_var)):
            if key not in values or tuple(values[key].shape) != tuple(target.shape):
                raise ValueError(f"BatchNorm {key} mismatch in {source}")
            target.data.copy_(torch.from_numpy(values[key]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                "source": str(source), "input_channels": 3}, destination)
    return destination


def download_checkpoint(depth: int, input_bits: int, cache: Path) -> Path:
    remote_bit = input_bits + 1
    if input_bits not in AVAILABLE_INPUT_BITS:
        raise ValueError(f"released checkpoints support input bits {AVAILABLE_INPUT_BITS.start}..{AVAILABLE_INPUT_BITS.stop - 1}")
    destination = cache / f"D{depth}" / f"bitpos_{remote_bit:02d}" / "model_030.hdf5"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{REPO_RAW}/pretrained_models/D{depth}/bitpos_{remote_bit:02d}/model_030.hdf5"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, destination)
    return destination


def _load_models(depth: int, starts: list[int], bit_depth: int, cache: Path, device: str) -> dict[int, BitplaneResidualCNN]:
    models = {}
    for input_bits in range(min(starts), bit_depth):
        source = download_checkpoint(depth, input_bits, cache)
        converted = cache / f"D{depth}" / f"bit_{input_bits:02d}.pt"
        if not converted.exists() or converted.stat().st_mtime < source.stat().st_mtime:
            convert_hdf5(source, converted, depth)
        payload = torch.load(converted, map_location=device, weights_only=True)
        model = BitplaneResidualCNN(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        models[input_bits] = model.to(device).eval()
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/raw/kodak"))
    parser.add_argument("--cache", type=Path, default=Path("pretrained_models/a-little-bit-more"))
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/kodak-pretrained"))
    parser.add_argument("--depths", nargs="+", type=int, default=[4, 16], choices=[4, 16])
    parser.add_argument("--start-bits", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--bit-depth", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args(argv)
    if args.bit_depth != 8:
        parser.error("the released Kodak checkpoints are evaluated here at 8-bit output")
    if any(bits < 3 or bits >= args.bit_depth for bits in args.start_bits):
        parser.error("released pretrained models support Kodak input depths 3 through 7")
    files = sorted(args.data.glob("*.png"))[:args.limit]
    if not files:
        parser.error(f"no PNG Kodak images found in {args.data}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for depth in args.depths:
        models = _load_models(depth, args.start_bits, args.bit_depth, args.cache, args.device)
        for path in files:
            reference = np.array(Image.open(path).convert("RGB"), dtype=np.uint8, copy=True)
            source = torch.from_numpy(reference).permute(2, 0, 1)[None].float() / 255.0
            for start in args.start_bits:
                estimate = recover_bitplanes(source, [models[b] for b in range(start, args.bit_depth)],
                                             start, args.bit_depth, args.device)[0]
                output = np.moveaxis(estimate.cpu().numpy(), 0, -1) * 255.0
                scores = evaluate(reference, output, start, args.bit_depth)
                rows.append({"image": path.stem, "model": f"D{depth}", "start_bits": start, **scores})
                if args.save_images:
                    Image.fromarray(np.clip(np.rint(output), 0, 255).astype(np.uint8)).save(
                        args.out / f"{path.stem}_D{depth}_{start}to8.png")
        print(f"evaluated D{depth}", flush=True)
    with (args.out / "per_image.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "model", "start_bits", "psnr", "ssim", "mae", "msb_consistency"])
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for depth in args.depths:
        for start in args.start_bits:
            subset = [row for row in rows if row["model"] == f"D{depth}" and row["start_bits"] == start]
            summary.append({"model": f"D{depth}", "start_bits": start,
                            **{metric: float(np.mean([row[metric] for row in subset])) for metric in ("psnr", "ssim", "mae", "msb_consistency")}})
    (args.out / "summary.json").write_text(json.dumps({"data": str(args.data), "results": summary}, indent=2))
    _plot(summary, args.out / "metrics.png")
    print(f"wrote {args.out}")
    return 0


def _plot(summary: list[dict], destination: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model in sorted({row["model"] for row in summary}):
        rows = [row for row in summary if row["model"] == model]
        axes[0].plot([row["start_bits"] for row in rows], [row["psnr"] for row in rows], marker="o", label=model)
        axes[1].plot([row["start_bits"] for row in rows], [row["ssim"] for row in rows], marker="o", label=model)
    axes[0].set(xlabel="input bits", ylabel="PSNR (dB)")
    axes[1].set(xlabel="input bits", ylabel="SSIM")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())