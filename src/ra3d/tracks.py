"""Validate trajectory tables and adapt them to the paper's numerical kernels."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import Units
from .io import read_table, write_table, require_columns, require_integer, require_unique
from ._core.candidates import Track, finite_difference


def read_tracks(source: str | Path) -> pd.DataFrame:
    source = Path(source)
    if source.is_dir():
        parts = []
        for path in sorted(source.glob("path_*.csv")):
            frame = read_table(path)
            frame["track_id"] = int(path.stem.split("_")[-1])
            parts.append(frame)
        if not parts:
            raise ValueError(f"No path_*.csv trajectories in {source}")
        frame = pd.concat(parts, ignore_index=True)
    else:
        frame = read_table(source)
    if {"xc", "yc", "slice"}.issubset(frame) and "x" not in frame:
        frame = frame.rename(columns={"xc": "x", "yc": "y", "slice": "z"})
    require_columns(frame, ["track_id", "frame", "x", "y", "z", "diameter_px"], name="tracks", finite=True)
    require_integer(frame, ["track_id", "frame"], name="tracks")
    require_unique(frame, ["track_id", "frame"], name="tracks")
    if frame.empty:
        raise ValueError("Trajectory table is empty")
    if (frame["diameter_px"] <= 0).any() or (frame["track_id"] < 0).any():
        raise ValueError("Track IDs must be nonnegative and particle diameters positive")
    return frame.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)


def trajectory_objects(table: pd.DataFrame) -> tuple[dict[int, Track], pd.DataFrame]:
    tracks = {}
    parts = []
    for track_id, original in table.groupby("track_id", sort=True):
        part = original.copy()
        frames = part["frame"].to_numpy(np.int32)
        for axis in "xyz":
            velocity = "v" + axis
            if velocity not in part or not np.isfinite(part[velocity].to_numpy(float)).all():
                part[velocity] = finite_difference(frames, part[axis].to_numpy(float))
        tracks[int(track_id)] = Track(
            int(track_id),
            frames,
            *(part[col].to_numpy(float) for col in ["x", "y", "z", "diameter_px", "vx", "vy", "vz"]),
            part["postOutProb"].to_numpy(float) if "postOutProb" in part else None,
        )
        parts.append(part)
    return tracks, pd.concat(parts, ignore_index=True)


def materialize_tracks(table: pd.DataFrame, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for track_id, frame in table.groupby("track_id", sort=True):
        write_table(frame.drop(columns="track_id"), directory / f"path_{int(track_id):08d}.csv")
    return directory


def from_physical(
    table: pd.DataFrame, units: Units, *, z_origin_um: float, xy_origin_px: float = 1.0
) -> pd.DataFrame:
    """Convert physical measurements; z_origin_um is the physical position of slice 1."""
    require_columns(table, ["track_id", "frame", "x_um", "y_um", "z_um", "diameter_um"], finite=True)
    result = table[["track_id", "frame"]].copy()
    result["x"] = table.x_um / units.pixel_pitch_um + xy_origin_px
    result["y"] = table.y_um / units.pixel_pitch_um + xy_origin_px
    result["z"] = (table.z_um - z_origin_um) / units.slice_spacing_um + 1.0
    result["diameter_px"] = table.diameter_um / units.pixel_pitch_um
    return result
