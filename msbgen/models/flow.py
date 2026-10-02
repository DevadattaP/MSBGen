"""Conditional flow matching on the residual: a generative model of p(R | B7) (PLAN Milestone 4).

Same residual, conditioning and network as the diffusion model (see
`msbgen.models.generative`); only the way noise is turned into a residual differs.

Probability path (rectified flow / conditional OT path, continuous t ∈ [0, 1]):

    x_t = (1 − t) r + t ε,   ε ~ N(0, I)      t = 0: the residual,  t = 1: pure noise

The straight line from r to ε has the constant velocity dx_t/dt = ε − r. The network
sees [x_t, B7 planes] and t and is trained to predict it (MSE). Averaged over all the
residuals consistent with B7, that regression gives the velocity field that transports
the noise distribution onto p(R | B7).

Sampling integrates the ODE dx/dt = v_θ(x, t, B7) from t = 1 (x = ε) to t = 0 with a
fixed grid of `steps` steps:

    euler   one network evaluation per step
    heun    Euler step, then a trapezoidal correction: two evaluations per step

The sample is deterministic given ε, like DDIM with eta = 0. Because the paths are
straight, few steps can already be enough; that speed / quality trade-off against
diffusion is one of the comparisons this model exists for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, ClassVar

import torch

from msbgen.models.generative import Predictor, ResidualGenerator, rand, randn

SOLVERS = ("euler", "heun")


@dataclass
class FlowConfig:
    kind: ClassVar[str] = "flow"
    arch: ClassVar[str] = "tunet"  # time-conditioned U-Net (for listings; there is only one)
    output: ClassVar[str] = "residual"  # always generates R, never the whole image

    k: int = 1
    bit_depth: int = 8
    channels: int = 1
    arch_args: dict = field(default_factory=lambda: {"width": 32, "levels": 4, "blocks": 1})
    sample_steps: int = 50  # default number of ODE steps at inference
    solver: str = "euler"  # default ODE solver at inference: "euler" or "heun"

    def __post_init__(self):
        if not 1 <= self.k < self.bit_depth:
            raise ValueError(f"k must be in 1..{self.bit_depth - 1}, got {self.k}")
        if self.sample_steps < 1:
            raise ValueError("sample_steps must be at least 1")
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {', '.join(SOLVERS)}, got {self.solver!r}")

    def to_dict(self) -> dict:
        return asdict(self)


class MSBFlow(ResidualGenerator):
    """`predict(x_t, t, cond)` is the velocity dx_t/dt = ε − r."""

    kind = "flow"

    def loss(self, x: torch.Tensor, generator: torch.Generator | None = None, reduction: str = "mean") -> torch.Tensor:
        """Velocity MSE for a batch of original images x (B, C, H, W), integer levels.

        With a seeded `generator`, t and ε are fixed, so the loss is repeatable (validation).
        """
        top = self.keep_top(x)
        r = self.to_residual(x, top)
        t = rand(x.shape[0], generator, x.device)
        eps = randn(r.shape, generator, x.device)
        tb = t[:, None, None, None]
        v_pred = self.predict((1 - tb) * r + tb * eps, t, self.condition(top))
        per_item = (v_pred - (eps - r)).square().flatten(1).mean(1)
        return per_item.mean() if reduction == "mean" else per_item

    @torch.no_grad()
    def sample(self, top: torch.Tensor, steps: int | None = None, solver: str | None = None,
               generator: torch.Generator | None = None, predict: Predictor | None = None,
               on_step: Callable[[], object] | None = None) -> torch.Tensor:
        """Draw Î / (2^n - 1) given the top-k image (B, C, H, W) by integrating the ODE.

        `predict` replaces `self.predict`, e.g. to run the network tile by tile.
        `on_step` is called after every network evaluation (`steps`, or 2·`steps` − 1
        for Heun); callers pass a thermal-guard check here (see msbgen.utils.thermal).
        """
        steps = steps or self.config.sample_steps
        solver = solver or self.config.solver
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {', '.join(SOLVERS)}, got {solver!r}")
        predict = predict or self.predict
        top = top.float()
        cond = self.condition(top)

        def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            v = predict(x, t.expand(x.shape[0]), cond)
            if on_step is not None:
                on_step()
            return v

        x = randn(top.shape, generator, top.device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=top.device)
        for i in range(steps):
            t, s = ts[i], ts[i + 1]
            v = velocity(x, t)
            x_next = x + (s - t) * v
            if solver == "heun" and i < steps - 1:  # the last step lands on t = 0: plain Euler there
                x_next = x + (s - t) * (v + velocity(x_next, s)) / 2
            x = x_next
        return self.from_residual(x, top)
