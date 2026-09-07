"""Numerical kernels derived from detect_vanish_survive_collisions.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial import cKDTree


TRACK_RE = re.compile(r"^path_([0-9]+)\.csv$")


@dataclass
class Track:
    track_id: int
    frame: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    diameter_px: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    post_out_prob: np.ndarray | None = None

    @property
    def first_frame(self) -> int:
        return int(self.frame[0])

    @property
    def last_frame(self) -> int:
        return int(self.frame[-1])

    @property
    def length(self) -> int:
        return int(len(self.frame))

    def point_at_or_near(self, target_frame: int) -> int:
        idx = int(np.searchsorted(self.frame, target_frame))
        if idx <= 0:
            return 0
        if idx >= self.length:
            return self.length - 1
        if abs(int(self.frame[idx]) - target_frame) < abs(int(self.frame[idx - 1]) - target_frame):
            return idx
        return idx - 1


@dataclass
class FrameIndex:
    frame: int
    rows: pd.DataFrame
    tree: cKDTree


def track_id_from_path(path: Path) -> int:
    match = TRACK_RE.match(path.name)
    if match is None:
        raise ValueError(f"not a path_*.csv file: {path}")
    return int(match.group(1))


def iter_track_files(directory: Path) -> list[Path]:
    files = sorted((p for p in directory.iterdir() if TRACK_RE.match(p.name)), key=track_id_from_path)
    if not files:
        raise FileNotFoundError(f"no path_*.csv files found in {directory}")
    return files


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


def load_tracks(directory: Path, which: str) -> tuple[dict[int, Track], pd.DataFrame]:
    tracks: dict[int, Track] = {}
    tables: list[pd.DataFrame] = []
    for path in iter_track_files(directory):
        tid = track_id_from_path(path)
        df = pd.read_csv(path).sort_values("frame")
        if df.empty:
            continue
        if which == "smoothed":
            required = {"frame", "x", "y", "z", "diameter_px"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{path} missing columns: {sorted(missing)}")
            use = df[["frame", "x", "y", "z", "diameter_px"]].copy()
            for vcol in ("vx", "vy", "vz"):
                use[vcol] = pd.to_numeric(df[vcol], errors="coerce") if vcol in df.columns else np.nan
            post = (
                pd.to_numeric(df["postOutProb"], errors="coerce").to_numpy(dtype=np.float64)
                if "postOutProb" in df.columns
                else None
            )
        else:
            required = {"frame", "xc", "yc", "slice", "diameter_px"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{path} missing columns: {sorted(missing)}")
            use = df[["frame", "xc", "yc", "slice", "diameter_px"]].rename(
                columns={"xc": "x", "yc": "y", "slice": "z"}
            )
            use["vx"] = np.nan
            use["vy"] = np.nan
            use["vz"] = np.nan
            post = None

        use = use.dropna(subset=["frame", "x", "y", "z", "diameter_px"]).copy()
        if use.empty:
            continue
        use["frame"] = use["frame"].astype(np.int32)
        frames = use["frame"].to_numpy(dtype=np.int32)
        for col, vcol in (("x", "vx"), ("y", "vy"), ("z", "vz")):
            vals = pd.to_numeric(use[col], errors="coerce").to_numpy(dtype=np.float64)
            vel = pd.to_numeric(use[vcol], errors="coerce").to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(vel)):
                vel = finite_difference(frames, vals)
            use[vcol] = vel

        use = use.dropna(subset=["x", "y", "z", "diameter_px", "vx", "vy", "vz"]).copy()
        if use.empty:
            continue
        use["track_id"] = np.int32(tid)
        tr = Track(
            track_id=tid,
            frame=use["frame"].to_numpy(dtype=np.int32),
            x=use["x"].to_numpy(dtype=np.float64),
            y=use["y"].to_numpy(dtype=np.float64),
            z=use["z"].to_numpy(dtype=np.float64),
            diameter_px=use["diameter_px"].to_numpy(dtype=np.float64),
            vx=use["vx"].to_numpy(dtype=np.float64),
            vy=use["vy"].to_numpy(dtype=np.float64),
            vz=use["vz"].to_numpy(dtype=np.float64),
            post_out_prob=post,
        )
        tracks[tid] = tr
        tables.append(use)
    if not tables:
        raise ValueError(f"no loadable tracks in {directory}")
    table = pd.concat(tables, ignore_index=True)
    return tracks, table


def build_frame_indices(table: pd.DataFrame, args: argparse.Namespace) -> dict[int, FrameIndex]:
    indices: dict[int, FrameIndex] = {}
    for frame, rows in table.groupby("frame", sort=True):
        rows = rows.reset_index(drop=True)
        coords = np.column_stack(
            (
                rows["x"].to_numpy(dtype=np.float64) * args.dx_um / args.xy_gate_um,
                rows["y"].to_numpy(dtype=np.float64) * args.dx_um / args.xy_gate_um,
                rows["z"].to_numpy(dtype=np.float64) * args.dz_um / args.z_gate_um,
            )
        )
        indices[int(frame)] = FrameIndex(frame=int(frame), rows=rows, tree=cKDTree(coords))
    return indices


def vector_um(x: float, y: float, z: float, args: argparse.Namespace) -> np.ndarray:
    return np.asarray([x * args.dx_um, y * args.dx_um, z * args.dz_um], dtype=np.float64)


def velocity_um_frame(vx: float, vy: float, vz: float, args: argparse.Namespace) -> np.ndarray:
    return np.asarray([vx * args.dx_um, vy * args.dx_um, vz * args.dz_um], dtype=np.float64)


def effective_distance_um(delta_um: np.ndarray, z_weight: float) -> float:
    return float(math.sqrt(delta_um[0] ** 2 + delta_um[1] ** 2 + (z_weight * delta_um[2]) ** 2))


def boundary_penalty(track: Track, max_frame: int, args: argparse.Namespace) -> float:
    x = float(track.x[-1])
    y = float(track.y[-1])
    z = float(track.z[-1])
    edge_xy = min(x, y, args.datlen - x, args.datlen - y)
    edge_z = min(z, args.datlen - z)
    margin = max(args.boundary_margin_px, 1.0)
    penalty = 0.0
    if edge_xy < margin:
        penalty = max(penalty, (margin - edge_xy) / margin)
    if edge_z < 0.5 * margin:
        penalty = max(penalty, (0.5 * margin - edge_z) / (0.5 * margin))
    if max_frame - track.last_frame <= args.max_final_frame_gap:
        penalty = max(penalty, 1.0)
    return float(min(max(penalty, 0.0), 1.0))


def median_window(values: np.ndarray, frames: np.ndarray, lo: int, hi: int) -> float:
    left = int(np.searchsorted(frames, lo, side="left"))
    right = int(np.searchsorted(frames, hi, side="right"))
    if right <= left:
        return math.nan
    arr = values[left:right]
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else math.nan


def survivor_change_score(track: Track, event_frame: int) -> tuple[float, dict[str, float]]:
    pre_lo = event_frame - 8
    pre_hi = event_frame - 2
    post_lo = event_frame + 2
    post_hi = event_frame + 8
    d_pre = median_window(track.diameter_px, track.frame, pre_lo, pre_hi)
    d_post = median_window(track.diameter_px, track.frame, post_lo, post_hi)
    vy_pre = median_window(track.vy, track.frame, pre_lo, pre_hi)
    vy_post = median_window(track.vy, track.frame, post_lo, post_hi)
    vx_pre = median_window(track.vx, track.frame, pre_lo, pre_hi)
    vx_post = median_window(track.vx, track.frame, post_lo, post_hi)
    vz_pre = median_window(track.vz, track.frame, pre_lo, pre_hi)
    vz_post = median_window(track.vz, track.frame, post_lo, post_hi)
    delta_diameter = d_post - d_pre if math.isfinite(d_pre) and math.isfinite(d_post) else math.nan
    if all(math.isfinite(v) for v in (vx_pre, vx_post, vy_pre, vy_post, vz_pre, vz_post)):
        delta_speed = math.sqrt((vx_post - vx_pre) ** 2 + (vy_post - vy_pre) ** 2 + (vz_post - vz_pre) ** 2)
    else:
        delta_speed = math.nan

    score = 0.0
    if math.isfinite(delta_diameter) and delta_diameter > 0:
        score += min(delta_diameter / 2.0, 1.0)
    if math.isfinite(delta_speed):
        score += min(delta_speed / 2.0, 1.0) * 0.5
    return float(min(score, 1.5)), {
        "delta_diameter_px": float(delta_diameter) if math.isfinite(delta_diameter) else math.nan,
        "delta_speed_px_frame": float(delta_speed) if math.isfinite(delta_speed) else math.nan,
    }


def vertical_chase_features(
    p_v: np.ndarray, vel_v: np.ndarray, p_s: np.ndarray, vel_s: np.ndarray, args: argparse.Namespace
) -> dict[str, float | str]:
    y_gap = float(p_s[1] - p_v[1])
    vy_v = float(vel_v[1])
    vy_s = float(vel_s[1])
    label = "none"
    closing_y = 0.0
    if y_gap > 0 and vy_v > vy_s:
        label = "vanished_chases_survivor"
        closing_y = vy_v - vy_s
    elif y_gap < 0 and vy_s > vy_v:
        label = "survivor_chases_vanished"
        closing_y = vy_s - vy_v
    time_to_y_catch = abs(y_gap) / closing_y if closing_y > 1.0e-9 else math.inf
    if not math.isfinite(time_to_y_catch):
        penalty = 1.0
    else:
        penalty = min(time_to_y_catch / max(args.vertical_catch_window, 1.0), 1.0)
    return {
        "vertical_chase_label": label,
        "vertical_y_gap_um": y_gap,
        "vertical_closing_um_frame": closing_y,
        "vertical_time_to_catch_frame": time_to_y_catch,
        "vertical_penalty": penalty,
    }


def generate_candidate_pool(
    tracks: dict[int, Track],
    table: pd.DataFrame,
    frame_indices: dict[int, FrameIndex],
    args: argparse.Namespace,
    max_slack_um: float,
) -> list[dict[str, object]]:
    max_frame = int(table["frame"].max())
    candidates: list[dict[str, object]] = []
    survivor_last = {tid: tr.last_frame for tid, tr in tracks.items()}
    survivor_first = {tid: tr.first_frame for tid, tr in tracks.items()}
    boundary = {tid: boundary_penalty(tr, max_frame=max_frame, args=args) for tid, tr in tracks.items()}

    for vanished in tracks.values():
        if vanished.length < args.min_vanished_len:
            continue
        end_frame = vanished.last_frame
        if end_frame >= max_frame - args.max_final_frame_gap:
            continue
        v_last = vanished.length - 1
        p_last = vector_um(vanished.x[v_last], vanished.y[v_last], vanished.z[v_last], args)
        vel_v = velocity_um_frame(vanished.vx[v_last], vanished.vy[v_last], vanished.vz[v_last], args)
        diam_v_um = float(vanished.diameter_px[v_last] * args.dx_um)
        if not np.all(np.isfinite(p_last)) or not np.all(np.isfinite(vel_v)) or not math.isfinite(diam_v_um):
            continue

        for frame in range(end_frame - args.pre_window, end_frame + args.post_window + 1):
            frame_index = frame_indices.get(frame)
            if frame_index is None:
                continue
            dt = frame - end_frame
            p_v = p_last + vel_v * dt
            query = np.asarray(
                [
                    p_v[0] / args.xy_gate_um,
                    p_v[1] / args.xy_gate_um,
                    p_v[2] / args.z_gate_um,
                ],
                dtype=np.float64,
            )
            hit_indices = frame_index.tree.query_ball_point(query, r=1.0)
            if not hit_indices:
                continue
            rows = frame_index.rows.iloc[hit_indices]
            for row in rows.itertuples(index=False):
                survivor_id = int(row.track_id)
                if survivor_id == vanished.track_id:
                    continue
                survivor = tracks.get(survivor_id)
                if survivor is None or survivor.length < args.min_survivor_len:
                    continue
                if survivor_first[survivor_id] > end_frame:
                    continue
                post_survival = survivor_last[survivor_id] - end_frame
                if post_survival < args.min_post_frames:
                    continue

                p_s = vector_um(float(row.x), float(row.y), float(row.z), args)
                vel_s = velocity_um_frame(float(row.vx), float(row.vy), float(row.vz), args)
                delta = p_v - p_s
                xy_dist_um = float(math.hypot(delta[0], delta[1]))
                z_dist_um = float(abs(delta[2]))
                true_distance_um = float(np.linalg.norm(delta))
                eff_distance_um = effective_distance_um(delta, args.z_score_weight)
                diam_s_um = float(row.diameter_px * args.dx_um)
                radius_sum_um = 0.5 * (diam_v_um + diam_s_um)
                eff_margin_um = eff_distance_um - radius_sum_um
                if eff_margin_um > max_slack_um:
                    continue

                if true_distance_um > 1.0e-9:
                    closing_um_frame = float(-np.dot(delta, vel_v - vel_s) / true_distance_um)
                else:
                    closing_um_frame = 0.0
                chase = vertical_chase_features(p_v, vel_v, p_s, vel_s, args)
                change_score, change = survivor_change_score(survivor, end_frame)
                wi = max(diam_v_um, 0.0) ** 3
                wj = max(diam_s_um, 0.0) ** 3
                if wi + wj <= 0:
                    wi = wj = 1.0
                centroid = (p_v * wi + p_s * wj) / (wi + wj)
                candidates.append(
                    {
                        "vanished_track_id": vanished.track_id,
                        "survivor_track_id": survivor_id,
                        "candidate_frame": int(frame),
                        "vanished_end_frame": int(end_frame),
                        "time_offset_frame": int(frame - end_frame),
                        "x_um": float(centroid[0]),
                        "y_um": float(centroid[1]),
                        "z_um": float(centroid[2]),
                        "vanished_x_um": float(p_v[0]),
                        "vanished_y_um": float(p_v[1]),
                        "vanished_z_um": float(p_v[2]),
                        "survivor_x_um": float(p_s[0]),
                        "survivor_y_um": float(p_s[1]),
                        "survivor_z_um": float(p_s[2]),
                        "xy_dist_um": xy_dist_um,
                        "z_dist_um": z_dist_um,
                        "closest_distance_um": true_distance_um,
                        "effective_distance_um": eff_distance_um,
                        "radius_sum_um": radius_sum_um,
                        "distance_margin_um": true_distance_um - radius_sum_um,
                        "effective_margin_um": eff_margin_um,
                        "vanished_diameter_um": diam_v_um,
                        "survivor_diameter_um": diam_s_um,
                        "closing_speed_um_s": closing_um_frame * args.fps,
                        "closing_speed_um_frame": closing_um_frame,
                        "post_survival_frames": int(post_survival),
                        "pre_overlap_frames": int(
                            end_frame - max(vanished.first_frame, survivor.first_frame)
                        ),
                        "boundary_exit_score": float(boundary[vanished.track_id]),
                        "survivor_change_score": float(change_score),
                        "survivor_delta_diameter_px": change["delta_diameter_px"],
                        "survivor_delta_speed_px_frame": change["delta_speed_px_frame"],
                        **chase,
                    }
                )
    return candidates
