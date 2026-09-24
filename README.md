# MSBGen

## Generative Image Reconstruction from Most-Significant Bitplanes

MSBGen studies how much of an 8-bit image can be recovered from only its
most-significant bit-plane(s). The long-term goal is an ultra-low-bitrate
perceptual image codec: transmit just the MSB plane $B_7$ (≈1 bit/pixel before
entropy coding), then reconstruct the full image with a conditional generative
model (diffusion, flow matching, normalizing flows).

$$
I = \sum_{k=0}^{7} 2^k B_k = 128\,B_7 + R, \qquad R \in [0, 127]
$$

$$
\hat I = 128\,B_7 + \hat R, \qquad \hat R \sim p_\theta(R \mid B_7)
$$

A single MSB plane matches many possible images, so recovering the image is a
one-to-many problem. That is why a **generative** model makes sense here rather
than a single regression. See [PLAN.md](PLAN.md) for the full research plan,
research questions and milestones.

## Current status

| Milestone | Status |
| --- | --- |
| Bit-plane explorer web app (slice & visualise planes) | ✅ done |
| Bit-plane analysis (entropy, correlations, k-MSB reconstruction) | 🔜 next |
| Deterministic baselines (CNN / U-Net, residual prediction) | ⏳ |
| Conditional diffusion / flow matching / normalizing flow | ⏳ |
| Multi-MSB experiments, entropy coding, final codec | ⏳ |

### Bit-plane explorer

A FastAPI web app where you upload an image and see every bit-plane:

Every image is sliced as either **grayscale** or **RGB**, and keeps its **bit
depth**. Each channel gives one plane per bit:

| Input | Sliced as | Planes |
| --- | --- | ---: |
| Grayscale, 8-bit (gray + alpha: alpha dropped) | Grayscale (L) | 8 |
| RGB, RGBA, palette, CMYK, YCbCr, LAB, HSV, 8-bit | RGB | 24 |
| Grayscale, 16-bit (gray + alpha: alpha dropped) | Grayscale (L) | 16 (B15 … B0) |
| RGB / RGBA, 16-bit | RGB | 48 |
| Bilevel (1-bit) | Grayscale (L) | 1 |

- For each plane: its place value $2^k$, the fraction of pixels set to 1, and its
  binary entropy (an i.i.d. upper bound in bits/pixel)
- For RGB images, an extra **Combined RGB** row stacks bit $k$ of R, G and B
  into one colour image for each $k$ (each channel 255 where its bit is set, else
  0). The result has 8 possible colours: yellow means R and G are set, white
  means all three, black means none. The MSB tile is exactly the 1-bit-per-channel
  image $\lfloor I/128 \rfloor \cdot 255$, i.e. what $B_7$ alone "sees" in colour.
- Click a plane to enlarge it or download it as a 1-bit PNG
- Accepts drag-and-drop or pasting from the clipboard. Works in light and dark
  themes and on phone-sized screens.

Input handling:

- **Alpha** (transparency) is dropped. The colour values stay exactly as stored;
  they are not blended onto a background.
- **Palette** (indexed) images become RGB. They have one channel, but its values
  are indices into a colour table, not gray levels.
- **CMYK, YCbCr, LAB and HSV** are converted to RGB with Pillow.
- EXIF rotation is applied. The UI reports every conversion when it happens.
- Pillow can't hold 16-bit colour, so 16-bit RGB/RGBA PNG and TIFF files are
  decoded with OpenCV. The true bit depth is read from the file header first, so
  nothing is silently truncated to 8 bits.
- The preview image is an 8-bit rendering used **for display only**. The planes
  always come from the full-bit-depth data.

## Project structure

```shell
MSBGen/
├── msbgen/
│   ├── main.py            # FastAPI app factory; serves UI + API
│   ├── config.py          # Settings (env vars: MSBGEN_*)
│   ├── core/              # Framework-independent logic, reusable for experiments
│   │   ├── bitplanes.py   # slice_bitplanes, reconstruct_from_planes, binary_entropy
│   │   └── image_io.py    # decode uploads → grayscale/RGB uint8/uint16 arrays, PNG encoding
│   ├── api/
│   │   ├── routes.py      # /api/health, /api/bitplanes
│   │   └── schemas.py     # Pydantic response models
│   └── web/static/        # Front-end (plain HTML / CSS / JS, no build step)
│       ├── index.html
│       ├── css/styles.css
│       └── js/app.js
├── tests/                 # pytest suite (core + API)
├── PLAN.md                # Research plan
├── requirements.txt
└── README.md
```

`msbgen.core` has no web dependencies, so the same bit-plane code will be
reused by the dataset builders and training scripts.

## Getting started

Requires Python 3.10+.

```bash
# 1. Activate your environment (or create one: python -m venv ~/venv)
source ~/venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (from the project root)
uvicorn msbgen.main:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API docs are at
<http://127.0.0.1:8000/docs>.

To make it reachable from other machines on your network:
`uvicorn msbgen.main:app --host 0.0.0.0 --port 8000`.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MSBGEN_MAX_UPLOAD_MB` | `25` | Maximum upload size |
| `MSBGEN_MAX_PIXELS` | `16777216` (4096²) | Maximum image area |

### Tests

```bash
python -m pytest
```

## API

### `POST /api/bitplanes`

Multipart form upload with a single field `file`.

```bash
curl -F "file=@photo.jpg" http://127.0.0.1:8000/api/bitplanes
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

`planes` are ordered by channel, then from MSB ($B_{n-1}$, where n = `bit_depth`)
down to LSB ($B_0$). Each plane image is a 1-bit PNG (white = 1, black = 0).

For RGB images the response also has `combined`, a list of
`{ "bit", "weight", "image" }` from MSB to LSB. Each `image` is an 8-colour PNG
of bit $k$ across R, G and B. For grayscale images the list is empty.

Errors: `400` for an empty or undecodable file or one over the pixel limit, and
`413` for a file over the size limit.

### `GET /api/health`

Returns `{"status": "ok", "version": "..."}`.

## Using the core library

```python
import numpy as np
from PIL import Image
from msbgen.core import slice_bitplanes, reconstruct_from_planes

img = np.asarray(Image.open("photo.png").convert("L"))   # (H, W) uint8
planes = slice_bitplanes(img)                            # (1, 8, H, W), values {0,1}

msb_only = reconstruct_from_planes(planes, bits=[7])     # 128·B7
top3     = reconstruct_from_planes(planes, bits=[7, 6, 5])
assert (reconstruct_from_planes(planes)[..., 0] == img).all()  # lossless
```

The bit depth comes from the dtype (`bool` → 1, `uint8` → 8, `uint16` → 16), and
any number of channels is accepted. For example, a 16-bit RGB array gives
`(3, 16, H, W)`. Use `msbgen.core.load_image(bytes)` to decode a file into
grayscale or RGB the same way the web app does.
