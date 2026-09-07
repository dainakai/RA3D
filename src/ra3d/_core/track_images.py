"""Numerical kernels derived from generate_collision_gifs.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
import math
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackPoint:
    x: float
    y: float
    z: float
    diameter_px: float


@dataclass(frozen=True)
class DiameterStats:
    diameter_um: float | None
    count: int
    start_frame: int | None
    end_frame: int | None


@dataclass(frozen=True)
class EventSpec:
    scene: str
    event_id: int
    label: str
    frame: int
    vanished_track_id: int
    survivor_track_id: int
    vanished_diameter_um: float | None
    survivor_diameter_um: float | None
    collision_score: float | None
    mode: str


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def point_at(track: pd.DataFrame, frame: int, *, nearest: bool = False) -> TrackPoint | None:
    frames = track["frame"].to_numpy(dtype=np.int32)
    idxs = np.flatnonzero(frames == frame)
    if len(idxs):
        row = track.iloc[int(idxs[0])]
        return TrackPoint(float(row.x), float(row.y), float(row.z), float(row.diameter_px))
    if not nearest or frame < int(frames.min()) or frame > int(frames.max()):
        return None
    idx = int(np.searchsorted(frames, frame))
    if idx <= 0:
        row = track.iloc[0]
        return TrackPoint(float(row.x), float(row.y), float(row.z), float(row.diameter_px))
    if idx >= len(frames):
        row = track.iloc[-1]
        return TrackPoint(float(row.x), float(row.y), float(row.z), float(row.diameter_px))
    f0 = float(frames[idx - 1])
    f1 = float(frames[idx])
    if f1 == f0:
        row = track.iloc[idx]
        return TrackPoint(float(row.x), float(row.y), float(row.z), float(row.diameter_px))
    a = (float(frame) - f0) / (f1 - f0)
    r0 = track.iloc[idx - 1]
    r1 = track.iloc[idx]
    return TrackPoint(
        float(r0.x + a * (r1.x - r0.x)),
        float(r0.y + a * (r1.y - r0.y)),
        float(r0.z + a * (r1.z - r0.z)),
        float(r0.diameter_px + a * (r1.diameter_px - r0.diameter_px)),
    )


def load_track(track_dir: Path, track_id: int) -> pd.DataFrame:
    path = track_dir / f"path_{track_id:08d}.csv"
    df = pd.read_csv(path).sort_values("frame").reset_index(drop=True)
    required = {"frame", "x", "y", "z", "diameter_px"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def choose_center(spec: EventSpec, vanished: pd.DataFrame, survivor: pd.DataFrame, frame: int) -> TrackPoint:
    if spec.mode == "survivor_remains" or frame >= spec.frame:
        pt = point_at(survivor, frame, nearest=True)
        if pt is not None:
            return pt
    vanished_pt = point_at(vanished, min(frame, spec.frame), nearest=True)
    survivor_pt = point_at(survivor, min(frame, spec.frame), nearest=True)
    if vanished_pt is None and survivor_pt is None:
        raise ValueError(f"no center point for {spec.scene}:{spec.event_id} frame={frame}")
    if vanished_pt is None:
        return survivor_pt  # type: ignore[return-value]
    if survivor_pt is None:
        return vanished_pt
    vanished_um = spec.vanished_diameter_um or vanished_pt.diameter_px
    survivor_um = spec.survivor_diameter_um or survivor_pt.diameter_px
    return vanished_pt if vanished_um >= survivor_um else survivor_pt


def crop_with_mean_padding(
    image: np.ndarray, cx: float, cy: float, size: int, *, padding_mean: float | None = None
) -> tuple[np.ndarray, float, float]:
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size
    out = np.full(
        (size, size), float(np.mean(image)) if padding_mean is None else padding_mean, dtype=np.float32
    )
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(image.shape[1], x1)
    src_y1 = min(image.shape[0], y1)
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = image[
            src_y0:src_y1, src_x0:src_x1
        ]
    return out, float(x0), float(y0)


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(crop, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(crop)), float(np.max(crop))
    if hi <= lo:
        return np.zeros_like(crop, dtype=np.uint8)
    return np.rint(np.clip((crop - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def determine_mode(track_dir: Path, frame: int, survivor_track_id: int, pre_frames: int) -> str:
    survivor = load_track(track_dir, survivor_track_id)
    has_survivor_before_window = int(survivor["frame"].min()) <= frame - pre_frames
    return "survivor_remains" if has_survivor_before_window else "merge_to_child"


def collect_diameter(
    track: pd.DataFrame,
    *,
    pixel_pitch_um: float,
    min_frame: int | None = None,
    max_frame: int | None = None,
    take_last: int | None = None,
    take_first: int | None = None,
) -> DiameterStats:
    df = track.copy()
    if min_frame is not None:
        df = df[df["frame"].astype(int) >= min_frame]
    if max_frame is not None:
        df = df[df["frame"].astype(int) <= max_frame]
    df = df.sort_values("frame")
    if take_last is not None and take_last > 0:
        df = df.tail(take_last)
    if take_first is not None and take_first > 0:
        df = df.head(take_first)
    diam = pd.to_numeric(df["diameter_px"], errors="coerce").to_numpy(np.float64) * float(pixel_pitch_um)
    frames = pd.to_numeric(df["frame"], errors="coerce").to_numpy(np.float64)
    good = np.isfinite(diam) & np.isfinite(frames) & (diam > 0.0)
    if not good.any():
        return DiameterStats(None, 0, None, None)
    good_frames = frames[good].astype(np.int64)
    good_diam = diam[good]
    return DiameterStats(
        diameter_um=float(np.median(good_diam)),
        count=int(good_diam.size),
        start_frame=int(good_frames.min()),
        end_frame=int(good_frames.max()),
    )
