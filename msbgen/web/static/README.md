# Web app and API

A FastAPI backend (`msbgen/api/`) and a plain HTML / CSS / JS front-end (this folder, no build step).
[`msbgen/main.py`](../../main.py) serves `index.html` at `/`, these files under `/static`, and the API under `/api`.

```bash
source ~/venv/bin/activate
uvicorn msbgen.main:app --reload        # from the project root
```

Open <http://127.0.0.1:8000>. Interactive API docs are at <http://127.0.0.1:8000/docs>.
To reach it from other machines: `uvicorn msbgen.main:app --host 0.0.0.0 --port 8000`.

## Files

| File | Role |
| --- | --- |
| `index.html` | Page structure: the five numbered sections and the image dialog |
| `css/styles.css` | Styles, light and dark themes |
| `js/app.js` | All behaviour: upload, request pipeline, rendering, charts |
| [`../../api/routes.py`](../../api/routes.py) | Endpoints |
| [`../../api/model_registry.py`](../../api/model_registry.py) | Which models run, thermal guard, GPU memory release |
| [`../../api/schemas.py`](../../api/schemas.py) | Response models |
| [`../../config.py`](../../config.py) | Settings read from `MSBGEN_*` environment variables |

## What the page does

1. **Input image.** Drop, browse or paste an image. Choose **RGB** or **Grayscale** before pressing **Slice & analyse**; every later step uses that mode.
2. **Bit-planes**, then 3. **Analysis**, then 4. **Model reconstruction**. They run one after another, never in parallel. After the analysis, pick **k = 1, 2 or 3** (the number of transmitted most-significant planes).
3. **Model reconstruction** shows the original and the zero, midpoint and random baselines straight away. Each model gets a blank card; click it to run that model only. A failure is shown on that card (click again to retry) and does not affect other cards or earlier sections. The table below the cards compares every result, including its reconstruction time. Click an image to see all of its metrics, including time.

Results are kept per colour mode and per `k`, so switching modes or `k` only changes what is shown. Everything is cleared only when the input image is changed or cleared. Model runs are queued, so only one model runs on the server at a time.

The saved test-set evaluation table is no longer part of the page. `GET /api/evaluations` still exists.

## Configuration

Set these before starting `uvicorn`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MSBGEN_MAX_UPLOAD_MB` | `25` | Maximum upload size |
| `MSBGEN_MAX_PIXELS` | `16777216` (4096²) | Maximum image area |
| `MSBGEN_MODELS_DIR` | `runs` | Auto-discovery root: `<root>/*/best.pt` for grayscale, `<root>/rgb/*/best.pt` for RGB |
| `MSBGEN_BITDEPTH_MODELS_DIR` | `runs` | Auto-discovery root for `bitdepth-*` folders; RGB folders are under `<root>/rgb/` |
| `MSBGEN_MODEL_REGISTRY` | unset | Optional JSON file selecting ordered model paths by colour mode and `k` (see below) |
| `MSBGEN_DEVICE` | auto | Device for model inference (`cuda`, `cpu`, …); auto uses CUDA when available |
| `MSBGEN_GPU_MAX_C` | `80` | Inference pauses at this GPU temperature and resumes 8 °C lower |
| `MSBGEN_CPU_MAX_C` | `90` | Inference pauses at this CPU temperature and resumes 10 °C lower |
| `MSBGEN_EVAL_DIR` | `outputs/eval` | Where `/api/evaluations` looks for saved evaluations (`<name>/summary.json`) |
| `MSBGEN_PERCEPTUAL` | `1` | `0` turns off LPIPS / DISTS per reconstruction |
| `MSBGEN_PERCEPTUAL_MAX_PIXELS` | `4194304` (2048²) | Larger images skip LPIPS / DISTS (GPU memory) |

Restart the server after changing any of them. A shell prefix works:
`MSBGEN_DEVICE=cpu uvicorn msbgen.main:app --reload`.

## Which models run

Without a registry, models are discovered on every request, so a newly finished training run appears without a restart:

- Grayscale: `<MSBGEN_MODELS_DIR>/*/best.pt`
- RGB: `<MSBGEN_MODELS_DIR>/rgb/*/best.pt`
- Sequential bit-plane CNNs: `bitdepth-*` folders containing `bit_NN.pt`, in the same two places

Only models trained for the selected colour mode, `k` and 8-bit depth are offered.

To pick exact models and their order, set `MSBGEN_MODEL_REGISTRY` to a JSON file. Paths are relative to that file; a `.pt` path is a standard checkpoint and a directory is a sequential bit-depth model. Only listed paths are used, in the listed order:

```json
{
  "grayscale": {
    "1": ["runs/unet-res/best.pt", "runs/bitdepth-d4"],
    "2": ["runs/unet-res-k2/best.pt"],
    "3": []
  },
  "rgb": {
    "1": ["runs/rgb/unet-res/best.pt"],
    "2": [],
    "3": []
  }
}
```

A card is named after its run folder (for example `vit-hybrid`) until it has run; the finished card also shows the model's label (for example "Hybrid (ViT + flow)").

## Memory, devices and temperature

- **One model at a time.** Each model is loaded, run, and released (including `torch.cuda.empty_cache()`) before the next, so GPU memory does not accumulate.
- **CPU fallback.** If a model runs out of CUDA memory it is retried on the CPU, with a note. LPIPS / DISTS are skipped for that result. Large models can be slow on the CPU.
- **Thermal guard.** The temperatures are checked before each model starts, after every sampling step, every image tile of a regression model, every bit-plane network of a sequential model, and after the first stage of a hybrid. If the GPU or CPU reaches its limit the request waits until both have cooled; the wait is not counted in the reported time. A sensor that cannot be read is not guarded. A single very large network pass cannot be interrupted, so lower the limits (`MSBGEN_GPU_MAX_C`, `MSBGEN_CPU_MAX_C`) if your laptop runs close to its limit.
- **Timing.** `time_s` covers `reconstruct` only: not model loading, scoring or thermal pauses. It is measured on the device that actually ran (after a CPU fallback, the CPU).

## API

All endpoints take multipart form data for uploads.

### `POST /api/bitplanes`

Fields: `file`, and optional `color_mode` (`L` or `RGB`). The web UI always sends it; without it the image keeps the mode it was decoded as. Converting RGB to grayscale uses BT.601 luma; grayscale to RGB repeats the channel.

```bash
curl -F "file=@photo.jpg" -F "color_mode=RGB" http://127.0.0.1:8000/api/bitplanes
```

Response (abridged):

```json
{
  "filename": "photo.jpg",
  "width": 800, "height": 600,
  "original_mode": "RGBA", "original_format": "PNG",
  "color_mode": "RGB", "bit_depth": 8,
  "channels": [
    { "name": "R", "label": "Red" }, { "name": "G", "label": "Green" },
    { "name": "B", "label": "Blue" }
  ],
  "notes": ["Dropped alpha channel."],
  "preview": "data:image/png;base64,...",
  "planes": [
    { "channel": "R", "channel_index": 0, "bit": 7, "weight": 128,
      "ones_ratio": 0.498, "entropy": 0.99999,
      "image": "data:image/png;base64,..." }
  ]
}
```

`planes` are ordered by channel, then from MSB ($B_{n-1}$, where n = `bit_depth`) down to LSB ($B_0$). Each plane image is a 1-bit PNG (white = 1, black = 0).

For RGB images the response also has `combined`, a list of `{ "bit", "weight", "image" }` from MSB to LSB. Each `image` is an 8-colour PNG of bit $k$ across R, G and B. For grayscale images the list is empty.

Errors: `400` for an empty or undecodable file, a bad `color_mode`, or an image over the pixel limit, and `413` for a file over the size limit.

### `POST /api/analysis`

Same upload as `/api/bitplanes`, including `color_mode`. Returns the same image fields (`filename`, `width`, `color_mode`, `bit_depth`, `channels`, `notes`, …) plus:

```json
{
  "planes": [
    { "channel": "R", "channel_index": 0, "bit": 7, "ones_ratio": 0.635,
      "entropy": 0.947, "context_entropy": 0.098, "spatial_correlation": 0.935 }
  ],
  "correlation": { "labels": ["R·B7", "R·B6", "..."], "matrix": [[1.0, 0.26, "..."]] },
  "reconstructions": [
    { "k": 1, "bits": [7], "bits_per_pixel": 3, "coded_bpp": 0.288,
      "psnr": { "zero": 11.85, "mid": 16.27, "random": 13.50 },
      "ssim": { "zero": 0.468, "mid": 0.505, "random": 0.121 },
      "mae": { "zero": 53.15, "mid": 33.14, "random": 44.00 }, "image": "data:image/png;base64,..." }
  ]
}
```

`reconstructions` runs from k = 1 to `bit_depth`, and each `image` is the midpoint-fill reconstruction as an 8-bit PNG. JSON has no infinity, so the lossless PSNR (k = `bit_depth`) is `null`. SSIM is `null` for images smaller than 11×11, and correlations involving a constant plane are `null`.

### `GET /api/models`

The trained checkpoints that can be found (not filtered by colour mode or `k`):
`{"torch_available", "models_dir", "models": [{"name", "label", "kind", "arch", "output", "k", "bit_depth", "channels", "parameters", "step", "val_psnr", "val_gain_db"}]}`.
`kind` is `regression`, `diffusion`, `flow`, `normalizing_flow` or `hybrid`. For generative models, `val_psnr` is of one sample. Each checkpoint is loaded on the CPU to read this, so the call can take a while with many models. With a registry it lists the checkpoints named there; otherwise every discovered one.

### `POST /api/reconstruct`

Reconstruct the image from its top `k` planes with the three baselines, and optionally with the trained models.

| Field | Default | Meaning |
| --- | --- | --- |
| `file` | required | The image; must be 8-bit |
| `k` | `1` | Number of most-significant planes kept |
| `color_mode` | decoded mode | `L` or `RGB`; baselines and models receive that many channels |
| `run_models` | `true` | `false` returns only the baselines and the list of models that could be run. The web UI uses this |
| `models` | all | Comma-separated model names; with `run_models=true`, only these run |

Response:

```json
{
  "k": 1,
  "original": "data:image/png;base64,...",
  "results": [
    { "name": "mid", "label": "Midpoint fill", "kind": "fill",
      "psnr": 17.29, "ssim": 0.635, "mae": 30.66, "msb_consistency": 1.0,
      "lpips": 0.383, "dists": 0.265, "time_s": 0.004,
      "psnr_gain": 0.0, "image": "data:...", "error_map": "data:..." },
    { "name": "unet-res", "label": "U-Net (residual)", "kind": "model", "...": "..." }
  ],
  "perceptual": ["lpips", "dists"],
  "candidates": [{ "name": "unet-res", "kind": "checkpoint" }]
}
```

- `results`: fills first (`zero`, `mid`, `random`), then any models that ran. `psnr_gain` is relative to the midpoint fill, `error_map` is $|I - \hat I| \times 4$, and `time_s` is the reconstruction time (see above).
- `candidates`: the models compatible with this colour mode and `k`, in run order. `kind` is `checkpoint` or `bitdepth`.
- `lpips` and `dists` are `null` (and missing from `perceptual`) when `lpips` / `piq` or their weights are unavailable, when `MSBGEN_PERCEPTUAL=0`, or for images larger than `MSBGEN_PERCEPTUAL_MAX_PIXELS` (with a note). The networks load on first use, which downloads their weights the first time.
- With `run_models=true`, a model that fails is described in `notes` and the others still run.
- Diffusion and flow models return their seed-0 sample with the checkpoint's sampler.

Errors: `400` for non-8-bit input or an invalid `k`, and `404` for an unknown model name.

### `POST /api/reconstruct/model`

Run one model. Fields: `file`, `model` (a `candidates[].name`), `k` (default `1`), `color_mode`.

```json
{ "result": { "name": "unet-res", "label": "U-Net (residual)", "kind": "model", "time_s": 0.41, "...": "..." },
  "note": null }
```

`result` has the same shape as an item of `results`. `note` says when the model fell back to the CPU. Errors: `404` for a name that is not offered for this mode and `k`, `400` for an unsupported combination, and `500` with the error text if the model fails (also after the CPU retry).

### `GET /api/evaluations`

Saved test-set evaluations (`msbgen.cli.evaluate` output) found in `MSBGEN_EVAL_DIR`, newest first:
`{"eval_dir", "evaluations": [{"name", "data", "images", "k", "samples", "kid_patch", "modified", "methods": [{"name", "label", "kind", "metrics": {"psnr": {"mean", "std"}, "...": null}}]}]}`.
`kind` is `fill`, `regression` or `generative`; `metrics` has `psnr`, `ssim`, `lpips`, `dists`, `mae`, `msb_consistency`, `psnr_avg`, `diversity`, `time_s` and `kid` (std over images, or over KID subsets for `kid`; `null` if not computed). Set-level metrics such as KID exist only here, not per uploaded image.

### `GET /api/health`

Returns `{"status": "ok", "version": "..."}`.

## Tests

```bash
python -m pytest tests/test_api.py
```

The API tests use temporary model folders and a patched thermal guard, so they need no GPU or sensors.
