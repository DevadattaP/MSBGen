"""Train a reconstruction model from the top-k planes (B7 → Î).

    python -m msbgen.cli.train --name unet-res --arch unet --output residual

Presets (override any option after them):

    --preset cnn        baseline 2: small CNN, direct output
    --preset unet       baseline 3: U-Net, direct output
    --preset unet-res   baseline 4: U-Net, residual output (predicts R = I - 128·B7)
    --preset diffusion  Milestone 3: conditional diffusion on R (time-conditioned U-Net,
                        v-prediction, EMA 0.999, 40k steps, validation every 2k steps)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PRESETS = {
    "cnn": {"arch": "cnn", "output": "direct", "arch_args": {"width": 64, "depth": 8}},
    "unet": {"arch": "unet", "output": "direct", "arch_args": {"width": 24, "levels": 4}},
    "unet-res": {"arch": "unet", "output": "residual", "arch_args": {"width": 24, "levels": 4}},
    "diffusion": {"kind": "diffusion", "arch_args": {"width": 32, "levels": 4, "blocks": 1},
                  "train": {"steps": 40_000, "ema": 0.999, "val_every": 2_000}},
}
TRAIN_DEFAULTS = {"steps": 20_000, "ema": 0.0, "val_every": 1_000}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, help="preset (sets model kind, arch, output, size)")
    p.add_argument("--name", help="run name (default: the preset name); output goes to <out-dir>/<name>/")
    p.add_argument("--kind", choices=["regression", "diffusion"], help="model kind (default: from the preset)")
    p.add_argument("--arch", choices=["cnn", "unet"], help="regression backbone")
    p.add_argument("--output", choices=["direct", "residual"], help="regression output mode")
    p.add_argument("--width", type=int, help="channels (CNN) / first-level channels (U-Nets)")
    p.add_argument("--depth", type=int, help="CNN layers")
    p.add_argument("--levels", type=int, help="U-Net downsampling levels")
    p.add_argument("--blocks", type=int, help="diffusion: ResBlocks per resolution")
    p.add_argument("--k", type=int, default=1, help="number of transmitted MSB planes (default 1)")
    p.add_argument("--train", nargs="+", default=["data/processed/div2k-train"], help="training folders of .npy")
    p.add_argument("--val", nargs="+", default=["data/processed/div2k-valid"], help="validation folders of .npy")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--steps", type=int, help="training steps (default 20k; diffusion preset 40k)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss", choices=["mse", "l1"], default="mse")
    p.add_argument("--val-every", type=int, help="validate every N steps (default 1k; diffusion preset 2k)")
    p.add_argument("--ema", type=float, help="EMA decay of the weights, 0 = off (default 0; diffusion preset 0.999)")
    p.add_argument("--val-sample-steps", type=int, default=25, help="diffusion: DDIM steps for validation samples")
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

    from msbgen.models.diffusion import DiffusionConfig
    from msbgen.models.msb_net import ModelConfig
    from msbgen.training import TrainConfig, train
    from msbgen.utils.thermal import ThermalLimits

    base = dict(PRESETS.get(args.preset, {"arch": "unet", "output": "residual", "arch_args": {"width": 24, "levels": 4}}))
    kind = args.kind or base.get("kind", "regression")
    if kind != base.get("kind", "regression"):  # e.g. --preset unet --kind diffusion: preset sizes don't apply
        base = PRESETS["diffusion"] if kind == "diffusion" else {"arch": "unet", "output": "residual", "arch_args": {}}
    if kind == "diffusion":
        arch_args = dict(base["arch_args"])
        options = ("width", "levels", "blocks")
    else:
        arch = args.arch or base["arch"]
        arch_args = dict(base["arch_args"]) if arch == base["arch"] else {}
        options = ("width", "depth", "levels")
    for key in options:
        if getattr(args, key) is not None:
            arch_args[key] = getattr(args, key)
    name = args.name or args.preset
    if not name:
        p.error("give --name or --preset")
    train_opts = {**TRAIN_DEFAULTS, **base.get("train", {})}
    for key in train_opts:
        if getattr(args, key) is not None:
            train_opts[key] = getattr(args, key)

    try:
        if kind == "diffusion":
            model = DiffusionConfig(k=args.k, arch_args=arch_args)
        else:
            model = ModelConfig(arch=arch, output=args.output or base["output"], k=args.k, arch_args=arch_args)
        thermal = ThermalLimits(args.gpu_max_temp, args.gpu_resume_temp, args.cpu_max_temp, args.cpu_resume_temp)
    except ValueError as exc:
        p.error(str(exc))
    cfg = TrainConfig(
        name=name, model=model, train_dirs=args.train, val_dirs=args.val, out_dir=args.out_dir,
        batch_size=args.batch_size, crop=args.crop, lr=args.lr, loss=args.loss,
        log_every=args.log_every, val_limit=args.val_limit, workers=args.workers,
        val_sample_steps=args.val_sample_steps, seed=args.seed, device=args.device,
        thermal_guard=not args.no_thermal_guard, thermal=thermal, **train_opts,
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
