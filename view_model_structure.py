from pathlib import Path
import torch
from torchinfo import summary

from msbgen.models import MSBHybrid, load_model
from msbgen.models.diffusion import cosine_schedule
from msbgen.models.msb_net import pad_to_multiple

device = "cuda" if torch.cuda.is_available() else "cpu"

model, ckpt = load_model(
    Path("runs/hybrid/best.pt"),
    device=device,
)
if not isinstance(model, MSBHybrid):
    raise TypeError(f"Expected an MSBHybrid checkpoint, got {type(model).__name__}")

model.eval()

print(f"Model: {type(model).__name__}")
print(f"Training step: {ckpt['step']}")
print(f"Configuration: {ckpt['model_config']}")

# Print the high-level component structure
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    trainable = sum(
        p.numel() for p in module.parameters()
        if p.requires_grad
    )
    print(
        f"{name:15} "
        f"{type(module).__name__:15} "
        f"params={params:,} "
        f"trainable={trainable:,}"
    )

# Total parameter counts
total = sum(p.numel() for p in model.parameters())
trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"\nTotal parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")


summary(
    model.unet,
    input_size=(1, 1, 128, 128),
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable",
    ],
    depth=4,
)

import inspect

print("flow:", inspect.signature(model.flow.forward))
print("refiner:", inspect.signature(model.refiner.forward))

# Flow: 3 input channels
x_flow = torch.randn(1, 3, 128, 128, device=device)
t_flow = torch.rand(1, device=device)

print("\n========== FLOW ==========")
summary(
    model.flow,
    input_data=(x_flow, t_flow),
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable",
    ],
    depth=4,
)

# Refiner: 4 input channels
x_refiner = torch.randn(1, 4, 128, 128, device=device)
t_refiner = torch.rand(1, device=device)

print("\n========== REFINER ==========")
summary(
    model.refiner,
    input_data=(x_refiner, t_refiner),
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable",
    ],
    depth=4,
)


import onnx

print("Sampler signature:", inspect.signature(model.sample))


class HybridSampler(torch.nn.Module):
    def __init__(self, hybrid):
        super().__init__()
        self.hybrid = hybrid
        if hybrid.config.eta != 0:
            raise ValueError("This ONNX wrapper currently supports eta=0 only")

    def forward(self, planes, flow_noise, refine_noise):
        hybrid = self.hybrid
        config = hybrid.config
        batch, _, height, width = planes.shape
        grouped_planes = planes.reshape(
            batch, config.channels, config.k, height, width
        )
        bit_weights = torch.tensor(
            [1 << (config.bit_depth - 1 - i) for i in range(config.k)],
            device=planes.device,
            dtype=planes.dtype,
        ).reshape(1, 1, config.k, 1, 1)
        top = (grouped_planes * bit_weights).sum(dim=2)
        condition = planes - 0.5

        unet_input = pad_to_multiple(condition, hybrid.unet.backbone.multiple)
        estimate = hybrid.unet.backbone(unet_input)[..., :height, :width]
        residual = torch.sigmoid(estimate) * 2 - 1

        flow_condition = torch.cat([condition, residual], dim=1)
        coarse_noise = flow_noise
        flow_times = torch.linspace(
            1.0, 0.0, config.sample_steps + 1, device=planes.device
        )
        for index in range(config.sample_steps):
            time = flow_times[index].expand(batch)
            velocity = hybrid._run(
                hybrid.flow, coarse_noise, time, flow_condition
            )
            coarse_noise = coarse_noise + (
                flow_times[index + 1] - flow_times[index]
            ) * velocity
        coarse = (residual + coarse_noise).clamp(-1, 1)

        full_condition = torch.cat([condition, residual, coarse], dim=1)
        refine_times = torch.linspace(
            config.refine_t,
            0.0,
            config.refine_steps + 1,
            device=planes.device,
        )
        alphas, sigmas = cosine_schedule(refine_times)
        refined = alphas[0] * coarse + sigmas[0] * refine_noise
        for index in range(config.refine_steps):
            alpha_t, sigma_t = alphas[index], sigmas[index]
            alpha_next, sigma_next = alphas[index + 1], sigmas[index + 1]
            time = refine_times[index].expand(batch)
            velocity = hybrid._run(
                hybrid.refiner, refined, time, full_condition
            )
            clean = (alpha_t * refined - sigma_t * velocity).clamp(-1, 1)
            if index == config.refine_steps - 1:
                refined = clean
            else:
                noise = (refined - alpha_t * clean) / sigma_t
                refined = alpha_next * clean + sigma_next * noise

        return (
            top + (refined.clamp(-1, 1) + 1) / 2 * hybrid.residual_max
        ) / hybrid.max_value


sampler = HybridSampler(model).eval()

# The wrapper accepts binary top-k planes and explicit noise for reproducible sampling.
config = model.config
dummy_planes = torch.randint(
    0, 2, (1, config.channels * config.k, 128, 128), device=device
).float()
dummy_flow_noise = torch.randn(
    1, config.channels, 128, 128, device=device
)
dummy_refine_noise = torch.randn_like(dummy_flow_noise)

output_path = Path("runs/onnx/msbhybrid.onnx")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Export the sampler with its stochastic inputs explicit.
with torch.no_grad():
    torch.onnx.export(
        sampler,
        (dummy_planes, dummy_flow_noise, dummy_refine_noise),
        str(output_path),
        input_names=["top_planes", "flow_noise", "refine_noise"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

# Verify the exported ONNX file
onnx_model = onnx.load(str(output_path))
onnx.checker.check_model(onnx_model)

print(f"Exported successfully: {output_path}")