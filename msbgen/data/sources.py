"""Where each dataset comes from.

A *source* is one downloadable split. `prepare_data` downloads it into
`<root>/raw/`, extracts it if needed, and converts every image to 8-bit grayscale
in `<root>/processed/<name>/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K"
KODAK_URL = "https://r0k.us/graphics/kodak/kodak"


@dataclass(frozen=True)
class Download:
    url: str
    filename: str
    size: int | None = None  # bytes; checked after download when known


@dataclass(frozen=True)
class Source:
    name: str
    description: str
    downloads: tuple[Download, ...]
    archive: bool  # True: one zip to extract; False: the downloads are the images
    image_glob: str = "*.png"  # inside the extracted folder (archives only)
    extracted_dir: str | None = None  # folder name the archive extracts to
    citation: str = ""
    groups: tuple[str, ...] = field(default=())  # aliases this source belongs to


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            name="div2k-train",
            description="DIV2K training set: 800 high-resolution (2K) natural images",
            downloads=(Download(f"{DIV2K_URL}/DIV2K_train_HR.zip", "DIV2K_train_HR.zip", 3_530_603_713),),
            archive=True,
            extracted_dir="DIV2K_train_HR",
            citation="Agustsson & Timofte, NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study, CVPRW 2017",
            groups=("div2k",),
        ),
        Source(
            name="div2k-valid",
            description="DIV2K validation set: 100 high-resolution (2K) natural images",
            downloads=(Download(f"{DIV2K_URL}/DIV2K_valid_HR.zip", "DIV2K_valid_HR.zip", 448_993_893),),
            archive=True,
            extracted_dir="DIV2K_valid_HR",
            citation="Agustsson & Timofte, NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study, CVPRW 2017",
            groups=("div2k",),
        ),
        Source(
            name="kodak",
            description="Kodak PhotoCD test set: 24 images, 768×512",
            downloads=tuple(Download(f"{KODAK_URL}/kodim{i:02d}.png", f"kodim{i:02d}.png") for i in range(1, 25)),
            archive=False,
            citation="Kodak Lossless True Color Image Suite, http://r0k.us/graphics/kodak/",
        ),
    ]
}


def resolve(names: list[str]) -> list[Source]:
    """Expand names and group aliases (e.g. "div2k" → train + valid), keeping order."""
    out: list[Source] = []
    for name in names:
        matches = [SOURCES[name]] if name in SOURCES else [s for s in SOURCES.values() if name in s.groups]
        if not matches:
            known = sorted(set(SOURCES) | {g for s in SOURCES.values() for g in s.groups})
            raise KeyError(f"unknown dataset {name!r}; choose from {', '.join(known)}")
        out.extend(m for m in matches if m not in out)
    return out
