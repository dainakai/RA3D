"""Numerical kernels derived from kinematic_pre_event.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
import math
from pathlib import Path
from dataclasses import dataclass


DX_UM = 10.0


DZ_UM = 100.0


Z_WEIGHT = 0.15


@dataclass
class Track:
    frame: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray


@dataclass
class Point:
    x_um: float
    y_um: float
    z_um: float
    vx_um_frame: float
    vy_um_frame: float
    vz_um_frame: float


def finite_difference(frames: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if len(values) < 2:
        return out
    for i in range(len(values)):
        if i == 0:
            left, right = 0, 1
        elif i == len(values) - 1:
            left, right = len(values) - 2, len(values) - 1
        else:
            left, right = i - 1, i + 1
        dt = float(frames[right] - frames[left])
        out[i] = (float(values[right]) - float(values[left])) / dt if dt > 0 else 0.0
    return out


def load_track(track_dir: Path, track_id: int, *, dx_um: float = DX_UM, dz_um: float = DZ_UM) -> Track:
    path = track_dir / f"path_{track_id:08d}.csv"
    df = pd.read_csv(path).sort_values("frame").reset_index(drop=True)
    if {"x", "y", "z"}.issubset(df.columns):
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=np.float64)
        z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        x = pd.to_numeric(df["xc"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(df["yc"], errors="coerce").to_numpy(dtype=np.float64)
        z = pd.to_numeric(df["slice"], errors="coerce").to_numpy(dtype=np.float64)
    frames = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=np.int32)
    vx = (
        pd.to_numeric(df["vx"], errors="coerce").to_numpy(dtype=np.float64)
        if "vx" in df
        else np.full(len(df), np.nan)
    )
    vy = (
        pd.to_numeric(df["vy"], errors="coerce").to_numpy(dtype=np.float64)
        if "vy" in df
        else np.full(len(df), np.nan)
    )
    vz = (
        pd.to_numeric(df["vz"], errors="coerce").to_numpy(dtype=np.float64)
        if "vz" in df
        else np.full(len(df), np.nan)
    )
    if not np.all(np.isfinite(vx)):
        vx = finite_difference(frames, x)
    if not np.all(np.isfinite(vy)):
        vy = finite_difference(frames, y)
    if not np.all(np.isfinite(vz)):
        vz = finite_difference(frames, z)
    good = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
    )
    return Track(frame=frames[good], x=x[good], y=y[good], z=z[good], vx=vx[good], vy=vy[good], vz=vz[good])


def point_exact(track: Track, frame: int, *, dx_um: float = DX_UM, dz_um: float = DZ_UM) -> Point | None:
    idx = int(np.searchsorted(track.frame, frame))
    if idx >= len(track.frame) or int(track.frame[idx]) != int(frame):
        return None
    return Point(
        x_um=float(track.x[idx] * dx_um),
        y_um=float(track.y[idx] * dx_um),
        z_um=float(track.z[idx] * dz_um),
        vx_um_frame=float(track.vx[idx] * dx_um),
        vy_um_frame=float(track.vy[idx] * dx_um),
        vz_um_frame=float(track.vz[idx] * dz_um),
    )


def eff_distance(delta: np.ndarray) -> float:
    return float(math.sqrt(delta[0] ** 2 + delta[1] ** 2 + (Z_WEIGHT * delta[2]) ** 2))


def slope(frames: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2:
        return math.nan
    x = frames.astype(np.float64)
    y = values.astype(np.float64)
    x = x - float(x.mean())
    denom = float(np.dot(x, x))
    if denom <= 0:
        return math.nan
    return float(np.dot(x, y - float(y.mean())) / denom)


def positive_fraction(values: np.ndarray, threshold: float = 0.0) -> float:
    if len(values) < 2:
        return math.nan
    diffs = np.diff(values)
    return float(np.mean(diffs > threshold))


def analyze_row(
    row: pd.Series,
    *,
    track_dir: Path,
    track_cache: dict[tuple[str, int], Track],
    pre_frames: int = 10,
    dx_um: float = DX_UM,
    dz_um: float = DZ_UM,
) -> dict[str, Any]:
    vanished_id = int(row["vanished_track_id"])
    survivor_id = int(row["survivor_track_id"])
    candidate_frame = int(row["candidate_frame"])
    key_prefix = str(track_dir)
    vanished_key = (key_prefix, vanished_id)
    survivor_key = (key_prefix, survivor_id)
    if vanished_key not in track_cache:
        track_cache[vanished_key] = load_track(track_dir, vanished_id, dx_um=dx_um, dz_um=dz_um)
    if survivor_key not in track_cache:
        track_cache[survivor_key] = load_track(track_dir, survivor_id, dx_um=dx_um, dz_um=dz_um)
    vanished = track_cache[vanished_key]
    survivor = track_cache[survivor_key]

    frames: list[int] = []
    xy_values: list[float] = []
    z_values: list[float] = []
    eff_values: list[float] = []
    closing_eff_values: list[float] = []
    closing_xy_values: list[float] = []

    for frame in range(candidate_frame - pre_frames, candidate_frame):
        pv = point_exact(vanished, frame, dx_um=dx_um, dz_um=dz_um)
        ps = point_exact(survivor, frame, dx_um=dx_um, dz_um=dz_um)
        if pv is None or ps is None:
            continue
        delta = np.asarray([pv.x_um - ps.x_um, pv.y_um - ps.y_um, pv.z_um - ps.z_um], dtype=np.float64)
        rel_vel = np.asarray(
            [
                pv.vx_um_frame - ps.vx_um_frame,
                pv.vy_um_frame - ps.vy_um_frame,
                pv.vz_um_frame - ps.vz_um_frame,
            ],
            dtype=np.float64,
        )
        xy = float(math.hypot(delta[0], delta[1]))
        z_abs = float(abs(delta[2]))
        eff = eff_distance(delta)
        if eff > 1.0e-9:
            d_eff = np.asarray([delta[0], delta[1], Z_WEIGHT * delta[2]], dtype=np.float64)
            v_eff = np.asarray([rel_vel[0], rel_vel[1], Z_WEIGHT * rel_vel[2]], dtype=np.float64)
            closing_eff = float(-np.dot(d_eff, v_eff) / eff)
        else:
            closing_eff = 0.0
        if xy > 1.0e-9:
            closing_xy = float(-np.dot(delta[:2], rel_vel[:2]) / xy)
        else:
            closing_xy = 0.0
        frames.append(frame)
        xy_values.append(xy)
        z_values.append(z_abs)
        eff_values.append(eff)
        closing_eff_values.append(closing_eff)
        closing_xy_values.append(closing_xy)

    out: dict[str, Any] = {
        "pre_kin_both_count": len(frames),
        "pre_kin_first_offset": math.nan,
        "pre_kin_last_offset": math.nan,
    }
    if not frames:
        return out

    f = np.asarray(frames, dtype=np.int32)
    xy_arr = np.asarray(xy_values, dtype=np.float64)
    z_arr = np.asarray(z_values, dtype=np.float64)
    eff_arr = np.asarray(eff_values, dtype=np.float64)
    closing_eff = np.asarray(closing_eff_values, dtype=np.float64)
    closing_xy = np.asarray(closing_xy_values, dtype=np.float64)
    tail = min(4, len(f))

    event_xy = float(row.get("xy_dist_um", math.nan))
    event_z = float(row.get("z_dist_um", math.nan))
    event_eff = float(row.get("effective_distance_um", math.nan))
    if not math.isfinite(event_eff) and math.isfinite(event_xy) and math.isfinite(event_z):
        event_eff = float(math.sqrt(event_xy**2 + (Z_WEIGHT * event_z) ** 2))

    out.update(
        {
            "pre_kin_first_offset": int(candidate_frame - int(f[0])),
            "pre_kin_last_offset": int(candidate_frame - int(f[-1])),
            "pre_kin_span_frames": int(f[-1] - f[0]),
            "pre_eff_first_um": float(eff_arr[0]),
            "pre_eff_last_um": float(eff_arr[-1]),
            "pre_eff_min_um": float(eff_arr.min()),
            "pre_eff_max_um": float(eff_arr.max()),
            "pre_eff_last_minus_first_um": float(eff_arr[-1] - eff_arr[0]),
            "pre_eff_slope_um_frame": slope(f, eff_arr),
            "pre_eff_tail_slope_um_frame": slope(f[-tail:], eff_arr[-tail:]),
            "pre_eff_increase_fraction": positive_fraction(eff_arr),
            "pre_eff_tail_increase_fraction": positive_fraction(eff_arr[-tail:]),
            "pre_eff_event_minus_last_um": float(event_eff - eff_arr[-1])
            if math.isfinite(event_eff)
            else math.nan,
            "pre_eff_event_minus_min_um": float(event_eff - eff_arr.min())
            if math.isfinite(event_eff)
            else math.nan,
            "pre_xy_first_um": float(xy_arr[0]),
            "pre_xy_last_um": float(xy_arr[-1]),
            "pre_xy_min_um": float(xy_arr.min()),
            "pre_xy_max_um": float(xy_arr.max()),
            "pre_xy_last_minus_first_um": float(xy_arr[-1] - xy_arr[0]),
            "pre_xy_slope_um_frame": slope(f, xy_arr),
            "pre_xy_tail_slope_um_frame": slope(f[-tail:], xy_arr[-tail:]),
            "pre_xy_increase_fraction": positive_fraction(xy_arr),
            "pre_xy_tail_increase_fraction": positive_fraction(xy_arr[-tail:]),
            "pre_xy_event_minus_last_um": float(event_xy - xy_arr[-1])
            if math.isfinite(event_xy)
            else math.nan,
            "pre_xy_event_minus_min_um": float(event_xy - xy_arr.min())
            if math.isfinite(event_xy)
            else math.nan,
            "pre_z_first_um": float(z_arr[0]),
            "pre_z_last_um": float(z_arr[-1]),
            "pre_z_last_minus_first_um": float(z_arr[-1] - z_arr[0]),
            "pre_z_slope_um_frame": slope(f, z_arr),
            "pre_closing_eff_last_um_frame": float(closing_eff[-1]),
            "pre_closing_xy_last_um_frame": float(closing_xy[-1]),
            "pre_closing_eff_median_um_frame": float(np.median(closing_eff)),
            "pre_closing_xy_median_um_frame": float(np.median(closing_xy)),
        }
    )
    return out
