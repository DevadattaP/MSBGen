"""Saving / loading model checkpoints.

A checkpoint is a plain dict saved with `torch.save`, loadable with
`torch.load(..., weights_only=True)` (no pickled classes):

    {"format": "msbgen-checkpoint-v1", "model_config": {...}, "state_dict": {...},
     "step": int, "metrics": {...}, "train_config": {...}, "optimizer": {...} (optional)}
"""

from __future__ import annotations

from pathlib import Path

import torch

from msbgen.models.msb_net import ModelConfig, MSBNet

FORMAT = "msbgen-checkpoint-v1"


def save_checkpoint(path: Path, model: MSBNet, step: int, metrics: dict | None = None,
                    train_config: dict | None = None, optimizer: torch.optim.Optimizer | None = None,
                    scheduler=None) -> None:
    payload = {
        "format": FORMAT,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint behind


def read_checkpoint(path: Path, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not an MSBGen checkpoint")
    return payload


def load_model(path: Path, device: str | torch.device = "cpu") -> tuple[MSBNet, dict]:
    """Build the model from its stored config and load the weights. Returns (model, checkpoint)."""
    payload = read_checkpoint(path)
    model = MSBNet(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload
