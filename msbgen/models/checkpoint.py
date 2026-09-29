"""Saving / loading model checkpoints.

A checkpoint is a plain dict saved with `torch.save`, loadable with
`torch.load(..., weights_only=True)` (no pickled classes):

    {"format": "msbgen-checkpoint-v1", "model_kind": "regression" | "diffusion",
     "model_config": {...}, "state_dict": {...}, "step": int, "metrics": {...},
     "train_config": {...}, "optimizer": {...}, "scheduler": {...},   (optional, for --resume)
     "train_state_dict": {...}}                                        (optional, see below)

With EMA (exponential moving average of the weights, used for the generative models),
`state_dict` holds the EMA weights, which are the ones to use for inference, and
`train_state_dict` the raw weights that training resumes from. Checkpoints written
before `model_kind` existed are regression models.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from msbgen.models.diffusion import DiffusionConfig, MSBDiffusion
from msbgen.models.msb_net import ModelConfig, MSBNet

FORMAT = "msbgen-checkpoint-v1"
KINDS = {"regression": (ModelConfig, MSBNet), "diffusion": (DiffusionConfig, MSBDiffusion)}


def build_model(config: ModelConfig | DiffusionConfig) -> MSBNet | MSBDiffusion:
    """The (untrained) model for a config; the config's `kind` picks the class."""
    return KINDS[config.kind][1](config)


def config_from_dict(kind: str, d: dict) -> ModelConfig | DiffusionConfig:
    if kind not in KINDS:
        raise ValueError(f"unknown model kind {kind!r}")
    return KINDS[kind][0](**d)


def save_checkpoint(path: Path, model: nn.Module, step: int, metrics: dict | None = None,
                    train_config: dict | None = None, optimizer: torch.optim.Optimizer | None = None,
                    scheduler=None, train_model: nn.Module | None = None) -> None:
    """Write `model` (the weights to infer with). `train_model`: raw weights when `model` is an EMA copy."""
    payload = {
        "format": FORMAT,
        "model_kind": model.kind,
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "step": step,
        "metrics": metrics or {},
        "train_config": train_config or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if train_model is not None:
        payload["train_state_dict"] = train_model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint behind


def read_checkpoint(path: Path, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not an MSBGen checkpoint")
    payload.setdefault("model_kind", "regression")
    return payload


def load_model(path: Path, device: str | torch.device = "cpu") -> tuple[MSBNet | MSBDiffusion, dict]:
    """Build the model from its stored config and load the weights. Returns (model, checkpoint)."""
    payload = read_checkpoint(path)
    model = build_model(config_from_dict(payload["model_kind"], payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload
