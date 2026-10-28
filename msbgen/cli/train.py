"""Train a reconstruction model from the top-k planes (B7 → Î).

    python -m msbgen.cli.train --name unet-res --arch unet --output residual

Presets (override any option after them):

    --preset cnn        baseline 2: small CNN, direct output
    --preset unet       baseline 3: U-Net, direct output
    --preset unet-res   baseline 4: U-Net, residual output (predicts R = I - 128·B7)
    --preset diffusion  Milestone 3: conditional diffusion on R (time-conditioned U-Net,
                        v-prediction, EMA 0.999, 40k steps, validation every 2k steps)
    --preset flow       Milestone 4: conditional flow matching on R (same network and
                        training as the diffusion preset, straight-path velocity loss)
    --preset normalizing-flow
                        Milestone 5: conditional Glow likelihood on fixed-size residual patches
    --preset hybrid     Milestone 6: residual U-Net -> flow matching -> diffusion refinement; use
                        --init-unet runs/unet-res/best.pt [--freeze-unet] to start from a trained U-Net
    --preset vit        residual ViT regressor (patch tokens + global attention + conv fusion)
    --preset vit-hybrid hybrid with a residual ViT as stage 1 (ViT -> flow matching -> diffusion refinement)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIT_ARGS = {"patch": 8, "dim": 192, "depth": 6, "heads": 6}
PRESETS = {
    "cnn": {"arch": "cnn", "output": "direct", "arch_args": {"width": 64, "depth": 8}},
    "unet": {"arch": "unet", "output": "direct", "arch_args": {"width": 24, "levels": 4}},
    "unet-res": {"arch": "unet", "output": "residual", "arch_args": {"width": 24, "levels": 4}},
    "diffusion": {"kind": "diffusion", "arch_args": {"width": 32, "levels": 4, "blocks": 1},
                  "train": {"steps": 40_000, "ema": 0.999, "val_every": 2_000}},
    "flow": {"kind": "flow", "arch_args": {"width": 32, "levels": 4, "blocks": 1},
             "train": {"steps": 40_000, "ema": 0.999, "val_every": 2_000}},
    "normalizing-flow": {"kind": "normalizing_flow", "train": {"steps": 40_000, "val_every": 2_000}},
    "hybrid": {"kind": "hybrid", "arch_args": {"width": 32, "levels": 4, "blocks": 1},
               "train": {"steps": 40_000, "ema": 0.999, "val_every": 2_000}},
    "vit": {"arch": "vit", "output": "residual", "arch_args": dict(VIT_ARGS), "train": {"steps": 40_000}},
    "vit-hybrid": {"kind": "hybrid", "unet_arch": "vit", "arch_args": {"width": 32, "levels": 4, "blocks": 1},
                   "train": {"steps": 40_000, "ema": 0.999, "val_every": 2_000}},
}
GENERATIVE = ("diffusion", "flow", "normalizing_flow", "hybrid")
TRAIN_DEFAULTS = {"steps": 20_000, "ema": 0.0, "val_every": 1_000}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, help="preset (sets model kind, arch, output, size)")
    p.add_argument("--name", help="run name (default: the preset name); output goes to <out-dir>/<name>/")
    p.add_argument("--kind", choices=["regression", *GENERATIVE], help="model kind (default: from the preset)")
    p.add_argument("--arch", choices=["cnn", "unet", "vit"], help="regression backbone")
    p.add_argument("--output", choices=["direct", "residual"], help="regression output mode")
    p.add_argument("--width", type=int, help="channels (CNN) / first-level channels (U-Nets)")
    p.add_argument("--depth", type=int, help="CNN layers / ViT transformer blocks")
    p.add_argument("--patch", type=int, help="ViT patch size (default 8)")
    p.add_argument("--dim", type=int, help="ViT token width (default 192)")
    p.add_argument("--heads", type=int, help="ViT attention heads (default 6)")
    p.add_argument("--unet-arch", choices=["unet", "vit"], help="hybrid: stage-1 backbone (default unet)")
    p.add_argument("--levels", type=int, help="U-Net downsampling levels")
    p.add_argument("--blocks", type=int, help="generative models: ResBlocks per resolution")
    p.add_argument("--crop-size", type=int, help="normalizing flow: fixed patch size (default: --crop)")
    p.add_argument("--scales", type=int, help="normalizing flow: squeeze scales (default 4)")
    p.add_argument("--blocks-per-scale", type=int, help="normalizing flow: Glow blocks per scale (default 4)")
    p.add_argument("--condition-width", type=int, help="normalizing flow: B7 feature width (default 32)")
    p.add_argument("--hidden-width", type=int, help="normalizing flow: coupling subnet width (default 64)")
    p.add_argument("--k", type=int, default=1, help="number of transmitted MSB planes (default 1)")
    p.add_argument("--train", nargs="+", default=["data/processed/div2k-train"], help="training folders of .npy")
    p.add_argument("--val", nargs="+", default=["data/processed/div2k-valid"], help="validation folders of .npy")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--steps", type=int, help="training steps (default 20k; generative presets 40k)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss", choices=["mse", "l1"], default="mse")
    p.add_argument("--val-every", type=int, help="validate every N steps (default 1k; generative presets 2k)")
    p.add_argument("--ema", type=float, help="EMA decay of the weights, 0 = off (default 0; generative presets 0.999)")
    p.add_argument("--sample-steps", type=int,
                   help="generative models: default sampling steps, stored in the checkpoint (default 3)")
    p.add_argument("--val-sample-steps", type=int,
                   help="generative models: sampling steps for validation samples (default: --sample-steps)")
    p.add_argument("--solver", choices=["euler", "heun"],
                   help="flow matching: default ODE solver, used in validation and stored in the checkpoint (euler)")
    p.add_argument("--log-every", type=int, default=100)
    h = p.add_argument_group("hybrid", "residual U-Net + flow matching + diffusion refinement")
    h.add_argument("--init-unet", type=Path, help="residual U-Net regression checkpoint that initialises stage 1")
    h.add_argument("--freeze-unet", action="store_true", help="keep stage 1 fixed (train only flow + refiner)")
    h.add_argument("--refine-steps", type=int, help="default DDIM steps of the refiner, 0 = off (default 2)")
    h.add_argument("--refine-t", type=float, help="noise level before refinement, in (0, 1] (default 0.5)")
    h.add_argument("--unet-weight", type=float, help="weight of the stage-1 MSE when the U-Net trains (default 1)")
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
    from msbgen.models.flow import FlowConfig
    from msbgen.models.hybrid import HybridConfig
    from msbgen.models.msb_net import ModelConfig
    from msbgen.models.normalizing_flow import NormalizingFlowConfig
    from msbgen.training import TrainConfig, train
    from msbgen.utils.thermal import ThermalLimits

    base = dict(PRESETS.get(args.preset, {"arch": "unet", "output": "residual", "arch_args": {"width": 24, "levels": 4}}))
    kind = args.kind or base.get("kind", "regression")
    if kind != base.get("kind", "regression"):  # e.g. --preset unet --kind diffusion: preset sizes don't apply
        preset = "normalizing-flow" if kind == "normalizing_flow" else kind
        base = PRESETS[preset] if preset in PRESETS else {"arch": "unet", "output": "residual", "arch_args": {}}
    if args.solver and kind != "flow":
        p.error("--solver only applies to flow matching")
    if kind == "normalizing_flow" and (args.sample_steps or args.val_sample_steps):
        p.error("--sample-steps / --val-sample-steps do not apply to normalizing flows")
    if (args.sample_steps or args.val_sample_steps) and kind not in GENERATIVE:
        p.error("--sample-steps / --val-sample-steps only apply to the generative models")
    if kind in ("diffusion", "flow", "hybrid"):
        arch_args = dict(base["arch_args"])
        options = ("width", "levels", "blocks")
    elif kind == "normalizing_flow":
        arch_args = {}
        options = ()
    else:
        arch = args.arch or base["arch"]
        arch_args = dict(base["arch_args"]) if arch == base["arch"] else (dict(VIT_ARGS) if arch == "vit" else {})
        options = ("patch", "dim", "depth", "heads") if arch == "vit" else ("width", "depth", "levels")
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
        sampler = {"sample_steps": args.sample_steps} if args.sample_steps is not None else {}
        if kind == "diffusion":
            model = DiffusionConfig(k=args.k, arch_args=arch_args, **sampler)
        elif kind == "flow":
            model = FlowConfig(k=args.k, arch_args=arch_args, solver=args.solver or "euler", **sampler)
        elif kind == "hybrid":
            unet_arch = args.unet_arch or base.get("unet_arch", "unet")
            unet_args = dict(VIT_ARGS) if unet_arch == "vit" else {"width": 24, "levels": 4}
            if unet_arch == "vit":
                unet_args.update({key: getattr(args, key) for key in ("patch", "dim", "depth", "heads")
                                  if getattr(args, key) is not None})
            if args.init_unet:
                from msbgen.models.checkpoint import read_checkpoint

                init_cfg = read_checkpoint(args.init_unet)["model_config"]
                unet_arch, unet_args = init_cfg["arch"], init_cfg["arch_args"]
            extra = {key: getattr(args, key) for key in ("refine_steps", "refine_t", "unet_weight")
                     if getattr(args, key) is not None}
            model = HybridConfig(k=args.k, arch_args=arch_args, unet_arch=unet_arch, unet_args=unet_args,
                                 **extra, **sampler)
        elif kind == "normalizing_flow":
            model = NormalizingFlowConfig(
                k=args.k, crop_size=args.crop_size if args.crop_size is not None else args.crop,
                scales=args.scales if args.scales is not None else 4,
                blocks_per_scale=args.blocks_per_scale if args.blocks_per_scale is not None else 4,
                condition_width=args.condition_width if args.condition_width is not None else 32,
                hidden_width=args.hidden_width if args.hidden_width is not None else 64,
            )
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
        init_unet=str(args.init_unet) if args.init_unet else None, freeze_unet=args.freeze_unet,
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
