"""Explicit adapters to pinned public HoloD3 and HoloD3-Linker implementations."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .io import read_table, write_table, write_json, sha256
from .config import Units
from .tracks import read_tracks

HOLOD3_COMMIT = "bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50"
LINKER_COMMIT = "fbe935a5d7fcd6b0c04ad0b2b2b9786c23f08cdf"


def detect(acquisition, output, *, holod3_root, device="cpu", weights=None, limit=0, dry_run=False):
    root = Path(holod3_root).resolve()
    if not (root / "holod3/cli.py").is_file():
        raise ValueError(
            "holod3_root must point to an installed HoloD3 source checkout; see docs/upstream.md"
        )
    command = [
        sys.executable,
        "-m",
        "holod3.cli",
        "infer",
        "--acquisition",
        str(Path(acquisition).resolve()),
        "--run-dir",
        str(Path(output).resolve()),
        "--preset",
        "portable-torch",
        "--device",
        device,
        "--yolo-device",
        device.replace("cuda:", "") if device.startswith("cuda:") else device,
        "--limit",
        str(limit),
        "--no-visualization",
    ]
    for role, path in (weights or {}).items():
        if role not in {"yolo", "depth-primary", "depth-fallback", "diameter"}:
            raise ValueError(f"Unknown HoloD3 model role: {role}")
        command.extend(["--" + role + "-weights", str(Path(path).resolve())])
    if dry_run:
        command.append("--dry-run")
    env = os.environ.copy()
    env["HOLOD3_ROOT"] = str(root)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(command, cwd=root, env=env, check=True)


def link(particles, output, *, pixel_pitch_um=10.0, slice_spacing_um=100.0, extra_args=()):
    Units(pixel_pitch_um=pixel_pitch_um, slice_spacing_um=slice_spacing_um)
    try:
        from holod3_linker import link_holod3_output
    except ImportError as exc:
        raise RuntimeError('Install the tracking extra: pip install -e ".[tracking]"') from exc
    particles = Path(particles).resolve()
    # The upstream adapter accepts CSV or a pipeline directory. Preserve Parquet
    # values when adapting a selectively downloaded detection table to that API.
    with tempfile.TemporaryDirectory(prefix="ra3d-link-input-") as temporary:
        source = particles
        if particles.suffix == ".parquet":
            source = Path(temporary) / "particles_3d.csv"
            write_table(read_table(particles), source)
        result = link_holod3_output(
            source, output_dir=output, pixel_pitch_um=pixel_pitch_um, linker_args=extra_args
        )
    table = read_tracks(result.trajectory_dir)
    # Link costs use physical Z, but trajectory exports retain the input slice.
    # Therefore slice spacing must not rescale the exported coordinates.
    write_table(table, Path(output) / "trajectory3d.parquet")
    write_json(
        {
            "source": str(particles),
            "sha256": sha256(particles) if particles.is_file() else None,
            "pixel_pitch_um": pixel_pitch_um,
            "slice_spacing_um": slice_spacing_um,
            "linker_commit": LINKER_COMMIT,
        },
        Path(output) / "ra3d_link.json",
    )
    return result
