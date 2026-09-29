# `msbgen.data`

Getting datasets onto disk in a reproducible form, and feeding them to training.

| Module | What it does | Needs torch |
| --- | --- | --- |
| [`sources.py`](sources.py) | Registry of downloadable datasets (URLs, sizes, citations) | no |
| [`prepare.py`](prepare.py) | Download (resumable) → extract → convert to grayscale `.npy` + manifest | no |
| [`crops.py`](crops.py) | PyTorch datasets: `RandomCrops` (training) and `CenterCrops` (validation) | yes |

The command-line entry point is
[`python -m msbgen.cli.prepare_data`](../cli/README.md#prepare_data-download-and-prepare-datasets).

## Datasets

| Name | Content | Use | Download |
| --- | --- | --- | ---: |
| `div2k-train` | 800 diverse 2K natural photos (DIV2K) | training | 3.5 GB |
| `div2k-valid` | 100 photos from the same distribution | validation during training | 0.45 GB |
| `kodak` | 24 photos, 768×512 (Kodak PhotoCD) | final test set | 15 MB |

`div2k` is a group name for both DIV2K splits. **Why these:**

- DIV2K is large and varied enough that random 128×128 crops give effectively
  unlimited training patches, which suits a small GPU.
- Kodak is the standard test set in image compression, so the numbers can be
  compared with the literature.

PLAN.md also mentions CelebA-HQ/FFHQ (faces) and CLIC/COCO for later
generalisation experiments. Adding one means adding a `Source` entry (see below).

## On-disk layout

```
data/                                  (git-ignored)
├── raw/
│   ├── DIV2K_train_HR.zip             downloads, kept so re-runs don't download again
│   ├── DIV2K_train_HR/                extracted originals
│   └── kodak/kodim01.png …
└── processed/
    ├── div2k-train/
    │   ├── 0001.npy …                 (H, W) uint8 grayscale
    │   └── manifest.json
    ├── div2k-valid/ …
    └── kodak/ …
```

**Processed format.** Each image is one `.npy` holding an `(H, W)` `uint8` array.
It is 8-bit grayscale made with ITU-R BT.601 luma, identical to Pillow's `L` and to
`msbgen.core.to_grayscale`, with **no resizing**. Decoding uses
`msbgen.core.load_image`, the same path as the web app and the analysis CLI, so
every tool sees identical pixels.

`.npy` is used instead of PNG because training reads random crops from 800 large
images. `np.load(..., mmap_mode="r")` reads only the cropped pixels from disk, with
no PNG decoding and no need to fit the dataset (~2.3 GB) in RAM.

**`manifest.json`** records provenance: source URLs, citation, conversion,
creation time, image count, total pixels, and each file with its original name and
shape.

## Preparation steps (`prepare.py`)

Every step is **idempotent**: it is skipped when its output already exists.

1. **Download** with an HTTP `Range` resume. An interrupted download continues from
   the `.part` file. The file size is checked against the registry when known, and
   a mismatch raises an error; re-running resumes.
2. **Extract** the zip into a temporary folder, then rename it into place, so a
   half-extracted folder is never mistaken for a finished one.
3. **Convert** each image to grayscale `.npy`. Files that are already converted
   are skipped, and undecodable or non-8-bit files are reported and skipped. The
   manifest is written last.

`prepare(source, root, keep_raw=True)` runs all three. `--delete-raw` on the CLI
removes the downloads afterwards, saving about 4 GB for DIV2K.

## Training datasets (`crops.py`)

```python
from msbgen.data.crops import RandomCrops, CenterCrops

train = RandomCrops(["data/processed/div2k-train"], crop=128, samples=320_000, seed=0)
val = CenterCrops(["data/processed/div2k-valid"], crop=256)
train[0]    # torch.uint8 tensor (1, 128, 128)
```

- **`RandomCrops`** picks an image with probability proportional to its area, so
  each pixel is equally likely to be sampled. It then takes a random crop and a
  random flip/90° rotation (dihedral augmentation; bit-planes have no preferred
  orientation). Sample `i` is fully determined by `(seed, i)`. The training loop
  uses global sample indices, so a resumed run continues the same stream of crops.
- **`CenterCrops`** takes one fixed centre crop per image, for fast and repeatable
  validation.
- Both return raw `uint8` crops. The training loop zeroes the lower bits and the
  model splits the planes, so the same data works for any k.

## Adding a dataset

Add a `Source` to `SOURCES` in [`sources.py`](sources.py):

```python
Source(
    name="clic-valid",
    description="CLIC 2020 professional validation set",
    downloads=(Download("https://…/valid.zip", "clic_valid.zip", size_in_bytes),),
    archive=True, extracted_dir="valid",
    citation="…",
)
```

Use `archive=False` with one `Download` per image for image lists like Kodak. Then
run `python -m msbgen.cli.prepare_data clic-valid`.
