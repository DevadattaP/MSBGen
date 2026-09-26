"""Train a deterministic reconstruction baseline (B7 → Î).

    python -m msbgen.cli.train --name unet-res --arch unet --output residual

Presets for the PLAN.md baselines (override any option after them):

    --preset cnn        baseline 2: small CNN, direct output
    --preset unet       baseline 3: U-Net, direct output
    --preset unet-res   baseline 4: U-Net, residual output (predicts R = I - 128·B7)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PRESETS = {
    "cnn": {"arch": "cnn", "output": "direct", "arch_args": {"width": 64, "depth": 8}},
    "unet": {"arch": "unet", "output": "direct", "arch_args": {"width": 24, "levels": 4}},
    "unet-res": {"arch": "unet", "output": "residual", "arch_args": {"width": 24, "levels": 4}},
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, help="baseline preset (sets arch, output, size)")
    p.add_argument("--name", help="run name (default: the preset name); output goes to <out-dir>/<name>/")
    p.add_argument("--arch", choices=["cnn", "unet"])
    p.add_argument("--output", choices=["direct", "residual"])
    p.add_argument("--width", type=int, help="channels (CNN) / first-level channels (U-Net)")
    p.add_argument("--depth", type=int, help="CNN layers")
    p.add_argument("--levels", type=int, help="U-Net downsampling levels")
    p.add_argument("--k", type=int, default=1, help="number of transmitted MSB planes (default 1)")
    p.add_argument("--train", nargs="+", default=["data/processed/div2k-train"], help="training folders of .npy")
    p.add_argument("--val", nargs="+", default=["data/processed/div2k-valid"], help="validation folders of .npy")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss", choices=["mse", "l1"], default="mse")
    p.add_argument("--val-every", type=int, default=1_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-limit", type=int, help="use only the first N validation images")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cuda, cpu, cuda:1, …")
    p.add_argument("--resume", type=Path, help="continue from a last.pt checkpoint")
    t = p.add_argument_group("thermal guard", "pause training (after saving last.pt) when the GPU or CPU gets "
                             "too hot, resume once both have cooled")
    t.add_argument("--gpu-max-temp", type=float, default=80.0, help="pause at or above this GPU °C (default 80)")
    t.add_argument("--gpu-resume-temp", type=float, default=72.0, help="resume at or below this GPU °C (default 72)")
    t.add_argument("--cpu-max-temp", type=float, default=90.0, help="pause at or above this CPU °C (default 90)")
    t.add_argument("--cpu-resume-temp", type=float, default=80.0, help="resume at or below this CPU °C (default 80)")
    t.add_argument("--no-thermal-guard", action="store_true", help="disable the guard (desktops / servers)")
    args = p.parse_args(argv)

    from msbgen.models.msb_net import ModelConfig
    from msbgen.training import TrainConfig, train
    from msbgen.utils.thermal import ThermalLimits

    base = dict(PRESETS.get(args.preset, {"arch": "unet", "output": "residual", "arch_args": {"width": 24, "levels": 4}}))
    arch = args.arch or base["arch"]
    arch_args = dict(base["arch_args"]) if arch == base["arch"] else {}
    for key in ("width", "depth", "levels"):
        if getattr(args, key) is not None:
            arch_args[key] = getattr(args, key)
    name = args.name or args.preset
    if not name:
        p.error("give --name or --preset")

    try:
        model = ModelConfig(arch=arch, output=args.output or base["output"], k=args.k, arch_args=arch_args)
        thermal = ThermalLimits(args.gpu_max_temp, args.gpu_resume_temp, args.cpu_max_temp, args.cpu_resume_temp)
    except ValueError as exc:
        p.error(str(exc))
    cfg = TrainConfig(
        name=name, model=model, train_dirs=args.train, val_dirs=args.val, out_dir=args.out_dir,
        steps=args.steps, batch_size=args.batch_size, crop=args.crop, lr=args.lr, loss=args.loss,
        val_every=args.val_every, log_every=args.log_every, val_limit=args.val_limit, workers=args.workers,
        seed=args.seed, device=args.device, thermal_guard=not args.no_thermal_guard, thermal=thermal,
    )
    try:
        best = train(cfg, resume=args.resume)
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
