import csv
import json

import numpy as np
from PIL import Image

from msbgen.cli.analyze import main


def _make_dataset(root):
    y, x = np.mgrid[0:32, 0:40]
    Image.fromarray(((x * 6) % 256).astype(np.uint8), "L").save(root / "gray.png")
    rgb = np.stack([(x * 6) % 256, (y * 8) % 256, (x + y) % 256], axis=-1).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(root / "a.png")
    Image.fromarray(rgb[::-1], "RGB").save(root / "b.png")
    (root / "broken.png").write_bytes(b"not an image")
    (root / "notes.txt").write_text("ignored")


def test_cli_groups_by_mode(tmp_path, capsys):
    _make_dataset(tmp_path)
    out = tmp_path / "out"
    assert main([str(tmp_path), "--out", str(out), "--workers", "1"]) == 0

    summary = json.loads((out / "summary.json").read_text())
    assert summary["meta"]["skipped"] == ["broken.png"]
    groups = summary["groups"]
    assert set(groups) == {"L-8", "RGB-8"}
    assert groups["RGB-8"]["images"] == 2
    assert len(groups["RGB-8"]["planes"]) == 24
    assert len(groups["RGB-8"]["correlation"]["matrix"]) == 24
    recs = groups["L-8"]["reconstructions"]
    assert [r["k"] for r in recs] == list(range(1, 9))
    assert recs[-1]["psnr_mid"] is None and recs[-1]["lossless_images"] == 1

    with (out / "reconstruction.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 8
    with (out / "planes.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 8 + 2 * 24
    assert "RGB-8" in capsys.readouterr().out


def test_cli_gray_and_plots(tmp_path):
    _make_dataset(tmp_path)
    out = tmp_path / "out"
    assert main([str(tmp_path), "--out", str(out), "--gray", "--no-ssim", "--plots", "--workers", "1"]) == 0
    groups = json.loads((out / "summary.json").read_text())["groups"]
    assert set(groups) == {"L-8"} and groups["L-8"]["images"] == 3
    assert groups["L-8"]["reconstructions"][0]["ssim_mid"] is None
    assert (out / "L-8_entropy.png").exists() and (out / "L-8_correlation.png").exists()
    assert not (out / "L-8_ssim.png").exists()  # no SSIM computed → no SSIM plot
