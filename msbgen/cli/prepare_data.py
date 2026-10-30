"""Download and prepare datasets as 8-bit grayscale arrays.

    python -m msbgen.cli.prepare_data div2k kodak

Downloads into <root>/raw/ (resumable), extracts archives and writes
<root>/processed/<name>/*.npy plus a manifest.json. Safe to re-run: finished steps
are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from msbgen.data.sources import SOURCES, resolve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="*", default=["div2k", "kodak"],
                        help="dataset names or groups (default: div2k kodak); see --list")
    parser.add_argument("--root", type=Path, default=Path("data"), help="data folder (default: data)")
    parser.add_argument("--color", choices=["gray", "rgb"], default="gray",
                        help="gray: (H, W) luma into processed/<name>; rgb: (H, W, 3) into processed/<name>-rgb")
    parser.add_argument("--delete-raw", action="store_true", help="delete downloads after conversion")
    parser.add_argument("--list", action="store_true", help="list available datasets and exit")
    args = parser.parse_args(argv)

    if args.list:
        for s in SOURCES.values():
            size = sum(d.size or 0 for d in s.downloads)
            groups = f" (group: {', '.join(s.groups)})" if s.groups else ""
            print(f"{s.name:<12} {s.description}{groups}" + (f", {size / 1e9:.1f} GB" if size else ""))
        return 0

    try:
        sources = resolve(args.datasets)
    except KeyError as exc:
        parser.error(str(exc.args[0]))

    from msbgen.data.prepare import prepare, processed_name

    for source in sources:
        try:
            manifest = prepare(source, args.root, keep_raw=not args.delete_raw, color=args.color)
        except OSError as exc:
            print(f"[{source.name}] failed: {exc}", file=sys.stderr)
            return 1
        print(f"[{source.name}] {manifest['count']} images, {manifest['pixels'] / 1e6:.0f} MP → "
              f"{args.root / 'processed' / processed_name(source, args.color)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
