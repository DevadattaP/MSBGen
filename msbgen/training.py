"""Training loop for the reconstruction models, from the top-k planes:

* regression (`ModelConfig`): CNN / U-Net baselines, MSE (or L1) to the original image;
  `best.pt` is the checkpoint with the highest validation PSNR.
* generative: diffusion (`DiffusionConfig`), flow matching (`FlowConfig`), or a
    conditional normalizing flow (`NormalizingFlowConfig`). Generative `best.pt`
    checkpoints minimize validation loss; stochastic validation uses a fixed random
    stream, and the normalizing-flow loss is dequantized NLL. Validation PSNR / SSIM
    are of one sample per crop, which is expected to score *below* a regression model
    on PSNR (see msbgen/models/README.md).

With `ema` > 0 an exponential moving average of the weights is kept; validation and the
saved inference weights use it (standard for generative models).

A run lives in `<out_dir>/<name>/`:

    config.json   full training + model configuration
    log.jsonl     training history: one JSON record per log / validation event (see below)
    last.pt       latest checkpoint (also written on Ctrl-C), used by --resume
    best.pt       best checkpoint (see above)

`log.jsonl` records (plot them with `python -m msbgen.cli.plot_runs`):

    {"step", "epoch", "loss", "train_psnr", "lr", "it_per_s",
     "gpu_temp", "cpu_temp", "paused_s"}                                       every `log_every` steps
    {"step", "epoch", "val_loss", "val_psnr", "val_psnr_mid", "val_psnr_random",
     "val_gain_db", "val_ssim", "val_ssim_mid", "val_ssim_random",
     "val_msb_consistency", "best_val_psnr" or "best_val_loss"}             every `val_every` steps and at step 0

`loss` is the mean training loss over the last `log_every` steps and `train_psnr` the PSNR
of that mean MSE (MSE loss only). `epoch` = pixels seen / pixels in the training set, so
it is fractional. `it_per_s` excludes thermal pauses; `paused_s` is the run's total time
spent cooling down. On --resume, records after the checkpoint's step are dropped first,
so the history never contains a step twice.

Thermal guard (on by default): every few seconds the GPU and CPU temperatures are read;
when either reaches its limit, `last.pt` is saved and training pauses until both have
cooled (see `msbgen.utils.thermal`). This keeps a laptop from overheating and shutting
down mid-run.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from msbgen.core.metrics import ssim
from msbgen.data.crops import CenterCrops, RandomCrops
from msbgen.models.checkpoint import GENERATIVE_KINDS, ModelConfigs, Models, build_model, read_checkpoint, save_checkpoint
from msbgen.models.msb_net import ModelConfig
from msbgen.utils.thermal import ThermalGuard, ThermalLimits


@dataclass
class TrainConfig:
    name: str
    model: ModelConfigs = field(default_factory=ModelConfig)
    train_dirs: list[str] = field(default_factory=lambda: ["data/processed/div2k-train"])
    val_dirs: list[str] = field(default_factory=lambda: ["data/processed/div2k-valid"])
    out_dir: str = "runs"
    steps: int = 20_000
    batch_size: int = 16
    crop: int = 128
    lr: float = 2e-4
    min_lr: float = 1e-6
    warmup: int = 500
    loss: str = "mse"  # regression only: "mse" or "l1"
    grad_clip: float = 1.0
    val_every: int = 1_000
    val_crop: int = 256
    val_limit: int | None = None
    log_every: int = 100
    ema: float = 0.0  # EMA decay of the weights (0 = off); e.g. 0.999 for the generative models
    val_sample_steps: int | None = None  # generative models: validation sampling steps (None: the model's)
    init_unet: str | None = None  # hybrid: residual U-Net checkpoint that initialises stage 1
    freeze_unet: bool = False  # hybrid: keep stage 1 fixed (needs init_unet to be useful)
    workers: int = 4
    seed: int = 0
    device: str = "auto"
    thermal_guard: bool = True
    thermal: ThermalLimits = field(default_factory=ThermalLimits)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model"] = self.model.to_dict()
        d["model_kind"] = self.model.kind
        d["thermal"] = asdict(self.thermal)
        return d


def _lr_lambda(cfg: TrainConfig):
    def f(step: int) -> float:
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        floor = cfg.min_lr / cfg.lr
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


def keep_top_bits_t(x: torch.Tensor, k: int, bit_depth: int) -> torch.Tensor:
    drop = bit_depth - k
    return ((x.to(torch.int32) >> drop) << drop).float()


def _psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred / target in [0, 1]; returns PSNR (dB) per batch item, capped at 100 for exact matches."""
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).clamp_max(100.0)


@torch.no_grad()
def validate(model: Models, loader: DataLoader, device: torch.device, loss_fn=F.mse_loss,
             sample_steps: int | None = None, on_step: Callable[[], object] | None = None) -> dict[str, float]:
    """Validation loss, and mean per-image PSNR / SSIM of the model and of the midpoint-fill
    and random-fill baselines on the same crops, plus MSB consistency.

    For the generative models, the loss uses fixed t and noise and the estimate is one
    sample (`sample_steps` steps, default the model's own, with its default sampler) from fixed noise. The random fill uses a fixed seed too, so
    every validation of every run sees the same noise.

    `on_step` is called after every batch and every sampling step: generative validation
    is minutes of sustained load, so training passes its thermal-guard check here."""
    cfg = model.config
    n, k = cfg.bit_depth, cfg.k
    peak = model.max_value
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)  # random fill
    model_gen = torch.Generator(device=device).manual_seed(1)  # generative loss / samples
    losses, consistent = [], []
    psnrs: dict[str, list] = {"model": [], "mid": [], "random": []}
    ssims: dict[str, list] = {"model": [], "mid": [], "random": []}
    for batch in loader:
        x = batch.to(device)
        target = x.float() / peak
        top = keep_top_bits_t(x, k, n)
        if on_step is not None:
            on_step()
        if model.kind in GENERATIVE_KINDS:
            losses.append(model.loss(x, generator=model_gen, reduction="none"))
            pred = model.sample(top, steps=sample_steps, generator=model_gen, on_step=on_step)
        else:
            pred = model(top)
            losses.append(loss_fn(pred, target, reduction="none").flatten(1).mean(1))
        noise = torch.randint(0, int(model.residual_max) + 1, top.shape, generator=gen, device=device)
        estimates = {"model": pred, "mid": (top + model.residual_max / 2) / peak, "random": (top + noise) / peak}
        ref = x.cpu().numpy()
        for name, est in estimates.items():
            psnrs[name].append(_psnr_per_image(est, target))
            ssims[name] += [ssim(r[0], e[0], peak) for r, e in zip(ref, (est * peak).cpu().numpy())]
        rounded = torch.floor(pred * peak + 0.5).clamp(0, peak)
        consistent.append((keep_top_bits_t(rounded, k, n) == top).float().flatten(1).mean(1))
    model.train(was_training)
    psnr = {name: torch.cat(v).mean().item() for name, v in psnrs.items()}

    def mean_ssim(values: list) -> float | None:
        values = [v for v in values if v is not None]  # None for crops smaller than the SSIM window
        return float(np.mean(values)) if values else None

    return {"val_loss": torch.cat(losses).mean().item(), "val_psnr": psnr["model"], "val_psnr_mid": psnr["mid"],
            "val_psnr_random": psnr["random"], "val_gain_db": psnr["model"] - psnr["mid"],
            "val_ssim": mean_ssim(ssims["model"]), "val_ssim_mid": mean_ssim(ssims["mid"]),
            "val_ssim_random": mean_ssim(ssims["random"]),
            "val_msb_consistency": torch.cat(consistent).mean().item()}


@torch.no_grad()
def _update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for e, p in zip(ema.parameters(), model.parameters(), strict=True):
        e.lerp_(p, 1 - decay)
    for e, b in zip(ema.buffers(), model.buffers(), strict=True):
        e.copy_(b)


def _log(run_dir: Path, record: dict) -> None:
    with (run_dir / "log.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def _truncate_log(run_dir: Path, step: int) -> None:
    """Drop history after `step` (e.g. logged after the checkpoint we resume from), and any torn records
    (a hard power-off can leave the tail of the file as NUL bytes or a half-written line)."""
    path = run_dir / "log.jsonl"
    if path.exists():
        kept = []
        for line in path.read_text(errors="replace").splitlines():
            try:
                if json.loads(line)["step"] <= step:
                    kept.append(line)
            except (ValueError, KeyError, TypeError):
                continue
        path.write_text("".join(line + "\n" for line in kept))


def train(cfg: TrainConfig, resume: Path | None = None) -> Path:
    """Train, validate and checkpoint. Returns the path of the best checkpoint."""
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    model = build_model(cfg.model).to(device)
    if cfg.init_unet:
        model.load_unet(Path(cfg.init_unet))
    if cfg.freeze_unet:
        model.unet.requires_grad_(False)
    ema = copy.deepcopy(model).requires_grad_(False) if cfg.ema > 0 else None
    eval_model = ema if ema is not None else model  # validated, and saved as the inference weights
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(cfg))
    generative = model.kind in GENERATIVE_KINDS
    # best.pt criterion: highest validation PSNR (regression) or lowest validation loss (generative)
    best_key, sign = ("val_loss", -1.0) if generative else ("val_psnr", 1.0)
    start, best = 0, -math.inf  # best is stored as sign * metric, so larger is always better
    if resume:
        ckpt = read_checkpoint(resume, map_location=device)
        model.load_state_dict(ckpt.get("train_state_dict", ckpt["state_dict"]))
        if ema is not None:
            ema.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start = ckpt["step"]
        best = sign * ckpt["metrics"].get(f"best_{best_key}", -sign * math.inf)
        _truncate_log(run_dir, start)
        print(f"resumed from {resume} at step {start}", file=sys.stderr)

    # Crop i of the whole run is fully determined by (seed, i), so resuming continues the same stream.
    train_set = RandomCrops([Path(d) for d in cfg.train_dirs], cfg.crop, samples=cfg.steps * cfg.batch_size, seed=cfg.seed)
    loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=range(start * cfg.batch_size, len(train_set)),
                        num_workers=cfg.workers, pin_memory=device.type == "cuda", persistent_workers=cfg.workers > 0,
                        drop_last=True)
    val_loader = DataLoader(CenterCrops([Path(d) for d in cfg.val_dirs], cfg.val_crop, cfg.val_limit),
                            batch_size=4, num_workers=min(cfg.workers, 2))
    print(f"{cfg.name}: {model.kind} {cfg.model.arch}/{cfg.model.output}, {model.num_parameters() / 1e6:.2f}M params, "
          f"{len(train_set.files)} train images, {len(val_loader.dataset)} val crops, device {device}", file=sys.stderr)
    gpu_index = device.index or 0 if device.type == "cuda" else 0
    guard = ThermalGuard(cfg.thermal, enabled=cfg.thermal_guard, gpu_index=gpu_index)
    print(guard.describe(), file=sys.stderr)

    loss_fn = F.mse_loss if cfg.loss == "mse" else F.l1_loss
    n, k = cfg.model.bit_depth, cfg.model.k
    step, running, t0, window_paused = start, 0.0, time.time(), 0.0
    train_cfg = cfg.to_dict()
    pixels_per_step = cfg.batch_size * cfg.crop * cfg.crop

    def epoch() -> float:
        return round(step * pixels_per_step / train_set.total_pixels, 4)

    def checkpoint(name: str, metrics: dict) -> None:
        save_checkpoint(run_dir / name, eval_model, step, metrics, train_cfg, optimizer, scheduler,
                        train_model=model if ema is not None else None)

    def best_metrics() -> dict:
        return {f"best_{best_key}": sign * best}

    def cool_down(force: bool = False) -> float:
        """Pause if the GPU/CPU is too hot, saving last.pt first. Returns seconds paused."""
        return guard.check(before_pause=lambda: checkpoint("last.pt", best_metrics()), force=force)

    def run_validation() -> dict:
        nonlocal best
        cool_down(force=True)  # validation is a burst of load: start it cool
        metrics = validate(eval_model, val_loader, device, loss_fn, cfg.val_sample_steps, on_step=cool_down)
        score = sign * metrics[best_key]
        improved = score > best
        best = max(best, score)
        metrics.update(best_metrics())
        ssim_text = f", SSIM {metrics['val_ssim']:.4f}" if metrics["val_ssim"] is not None else ""
        print(f"  val PSNR {metrics['val_psnr']:.2f} dB{' (sample)' if generative else ''} "
              f"(midpoint {metrics['val_psnr_mid']:.2f}, gain {metrics['val_gain_db']:+.2f} dB{ssim_text}, "
              f"loss {metrics['val_loss']:.5f}, MSB consistency {metrics['val_msb_consistency']:.4f})"
              f"{'  ★ best' if improved and step else ''}", file=sys.stderr)
        _log(run_dir, {"step": step, "epoch": epoch(), **metrics})
        return {**metrics, "improved": improved}

    if start == 0:
        run_validation()  # the untrained model: the starting point of every curve
        best = -math.inf  # step 0 never counts as "best"
        t0 = time.time()

    model.train()
    try:
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            if generative:
                loss = model.loss(x)
            else:
                loss = loss_fn(model(keep_top_bits_t(x, k, n)), x.float() / model.max_value)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if ema is not None:
                _update_ema(ema, model, min(cfg.ema, (1 + step) / (10 + step)))  # warm-up: short memory early on
            running += loss.item()
            window_paused += cool_down()

            if step % cfg.log_every == 0:
                elapsed = max(time.time() - t0 - window_paused, 1e-6)  # compute time only
                mean_loss = running / cfg.log_every
                rec = {"step": step, "epoch": epoch(), "loss": mean_loss,
                       "train_psnr": -10 * math.log10(max(mean_loss, 1e-10)) if cfg.loss == "mse" and not generative else None,
                       "lr": scheduler.get_last_lr()[0], "it_per_s": cfg.log_every / elapsed, **guard.status()}
                eta = (cfg.steps - step) / rec["it_per_s"] / 60
                temps = "".join(f"  {name} {rec[key]:.0f}°C" for name, key in (("GPU", "gpu_temp"), ("CPU", "cpu_temp"))
                                if rec[key] is not None)
                print(f"step {step:>6}/{cfg.steps}  loss {rec['loss']:.5f}  lr {rec['lr']:.2e}  "
                      f"{rec['it_per_s']:.1f} it/s  eta {eta:.0f} min{temps}", file=sys.stderr)
                _log(run_dir, rec)
                running, t0, window_paused = 0.0, time.time(), 0.0

            if step % cfg.val_every == 0 or step == cfg.steps:
                metrics = run_validation()
                improved = metrics.pop("improved")
                checkpoint("last.pt", metrics)
                if improved:
                    checkpoint("best.pt", metrics)
                t0, window_paused = time.time(), 0.0
    except KeyboardInterrupt:
        print(f"\ninterrupted at step {step}; saving last.pt (resume with --resume {run_dir / 'last.pt'})", file=sys.stderr)
        checkpoint("last.pt", best_metrics())
        raise
    if guard.pauses:
        print(f"thermal guard paused training {guard.pauses} time(s), {guard.paused_s / 60:.1f} min in total",
              file=sys.stderr)
    return run_dir / "best.pt"
