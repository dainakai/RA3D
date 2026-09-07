"""Build an explicit RA3D frame CSV from paired numeric image stems."""

import argparse
from pathlib import Path
import pandas as pd


def images(root, directory):
    paths = [p for p in (root / directory).iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}]
    if not paths or any(not p.stem.isdecimal() for p in paths):
        raise ValueError("Image filenames must have numeric stems; otherwise write frames.csv explicitly")
    result = {int(p.stem): p for p in paths}
    if len(result) != len(paths):
        raise ValueError("Duplicate image frame stems")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--filename-offset", type=int, default=1)
    parser.add_argument("--output", default="frames.csv")
    args = parser.parse_args()
    primary, secondary = images(args.root, args.primary), images(args.root, args.secondary)
    if primary.keys() != secondary.keys():
        raise ValueError("Primary and secondary frame sets differ")
    frame = pd.DataFrame(
        [
            {
                "frame": index - args.filename_offset,
                "primary": str(primary[index].relative_to(args.root)),
                "secondary": str(secondary[index].relative_to(args.root)),
            }
            for index in sorted(primary)
        ]
    )
    frame.to_csv(args.root / args.output, index=False)


if __name__ == "__main__":
    main()
