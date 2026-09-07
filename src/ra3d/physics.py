"""Physical consistency and Hall-corrected gravitational collision diagnostics."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from ._core.physics import (
    HallCorrectedLaminarKernel,
    pair_prediction,
    predictive_standard_deviation,
    median_velocity_px_per_frame,
    summary_metrics,
)
from ._core.feature_primitives import summarize_diameter
from .config import Units
from .io import read_table, write_table, write_json, require_columns, require_integer
from .tracks import read_tracks


def consistency(candidates, tracks, units=Units(), *, minimum_points=3):
    """Use observed pre/post windows; raw candidate time anchors the measurements."""
    if minimum_points < 2:
        raise ValueError("At least two points are required for an interval velocity")
    require_integer(candidates, ["vanished_track_id", "survivor_track_id", "candidate_frame"])
    groups = {int(k): v for k, v in tracks.groupby("track_id")}
    rows = []
    for row in candidates.to_dict("records"):
        vanished, survivor = groups[int(row["vanished_track_id"])], groups[int(row["survivor_track_id"])]
        frame = int(row.get("candidate_frame_raw", row["candidate_frame"]))
        diameter = summarize_diameter(
            vanished,
            survivor,
            frame,
            pixel_pitch_um=units.pixel_pitch_um,
            pre_gap_frames=2,
            pre_window_frames=10,
            post_window_frames=20,
        )
        dv, ds, post = [
            diameter[k] for k in ["pre_diameter_vanished_um", "pre_diameter_survivor_um", "post_diameter_um"]
        ]
        parent_valid = np.isfinite([dv, ds]).all() and min(dv, ds) > 0
        expected = np.cbrt(dv**3 + ds**3) if parent_valid else np.nan
        windows = [(vanished, "pre"), (survivor, "pre"), (survivor, "post")]
        velocities, counts = [], []
        for track, when in windows:
            start, end = diameter[f"{when}_window_start_frame"], diameter[f"{when}_window_end_frame"]
            value, count = (
                median_velocity_px_per_frame(track, int(start), int(end))
                if np.isfinite([start, end]).all()
                else (np.nan, 0)
            )
            velocities.append(value * units.pixel_pitch_um * units.fps * 1e-6)
            counts.append(count)
        available = bool(parent_valid and min(counts) >= minimum_points and np.isfinite(velocities).all())
        predicted_velocity = (
            (dv**3 * velocities[0] + ds**3 * velocities[1]) / (dv**3 + ds**3) if available else np.nan
        )
        rows.append(
            {
                **row,
                **diameter,
                "expected_post_diameter_um": expected,
                "measured_post_diameter_um": post,
                "diameter_error_um": post - expected,
                "expected_post_vy_m_s": predicted_velocity,
                "measured_post_vy_m_s": velocities[2] if available else np.nan,
                "pre_vy_vanished_n": counts[0],
                "pre_vy_survivor_n": counts[1],
                "post_vy_n": counts[2],
                "velocity_available": available,
            }
        )
    return pd.DataFrame(rows)


def collision_rate(
    particles,
    *,
    recorded_frames,
    volume_cm3,
    fps=4000.0,
    max_final_frame_gap=12,
    time_blocks=11,
    bootstrap_replicates=4000,
    seed=260724,
    kernel=None,
):
    """Predict unordered diameter-pair counts from measured frame-wise density.

    The last max_final_frame_gap+1 frames are outside vanishing-endpoint support.
    Bootstrap blocks estimate density variation; Poisson variance is added separately.
    """
    require_integer(particles, ["frame"])
    require_columns(particles, ["diameter_um"], finite=True)
    effective = int(recorded_frames) - 1 - int(max_final_frame_gap)
    if min(volume_cm3, fps) <= 0 or not np.isfinite([volume_cm3, fps]).all():
        raise ValueError("Volume and frame rate must be finite and positive")
    if time_blocks < 2 or effective < time_blocks or bootstrap_replicates < 2:
        raise ValueError("Use at least two nonempty time blocks and two bootstrap replicates")
    if (particles.frame < 0).any() or (particles.frame >= recorded_frames).any():
        raise ValueError("Particle frames are outside the recording interval")
    if (particles.diameter_um <= 0).any():
        raise ValueError("Particle diameters must be positive")
    selected = particles.loc[particles.frame < effective]
    fine_edges, pair_edges = np.arange(25.0, 305.0, 5.0), np.arange(25.0, 325.0, 25.0)
    centers = (fine_edges[:-1] + fine_edges[1:]) / 2
    block_edges = np.linspace(0, effective, time_blocks + 1, dtype=np.int64)
    indices = np.searchsorted(block_edges[1:], selected.frame, side="right")
    histogram, _, _ = np.histogram2d(
        indices, selected.diameter_um, bins=[np.arange(time_blocks + 1) - 0.5, fine_edges]
    )
    density = histogram.sum(axis=0) / (effective * volume_cm3)
    block_density = histogram / (np.diff(block_edges)[:, None] * volume_cm3)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, time_blocks, size=(bootstrap_replicates, time_blocks))
    bootstrap_density = block_density[sampled].mean(axis=1)
    kernel = kernel or HallCorrectedLaminarKernel()
    a, b = np.meshgrid(centers, centers, indexing="ij")
    matrix = kernel.kernel_m3_s(a, b)
    exposure = effective / fps * volume_cm3 * 1e-6
    expected = pair_prediction(density * 1e6, matrix, centers, pair_edges, exposure_volume_m3_s=exposure)[0]
    bootstrap = pair_prediction(
        bootstrap_density * 1e6, matrix, centers, pair_edges, exposure_volume_m3_s=exposure
    )
    density_sd, poisson_sd, total_sd = predictive_standard_deviation(expected, bootstrap)
    rows = [
        {
            "small_diameter_low_um": pair_edges[i],
            "large_diameter_low_um": pair_edges[j],
            "predicted_count": expected[i, j],
            "density_sd": density_sd[i, j],
            "poisson_sd": poisson_sd[i, j],
            "total_sd": total_sd[i, j],
        }
        for i in range(len(pair_edges) - 1)
        for j in range(i, len(pair_edges) - 1)
    ]
    total = float(np.nansum(expected))
    total_density_sd = float(np.std(np.nansum(bootstrap, axis=(1, 2)), ddof=1))
    audit = {
        "effective_frames": effective,
        "exposure_s": effective / fps,
        "exposure_volume_m3_s": exposure,
        "predicted_count": total,
        "predicted_rate_m3_s": total / exposure,
        "total_sd": float(np.sqrt(total + total_density_sd**2)),
        "density_sd": total_density_sd,
        "time_blocks": time_blocks,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "coalescence_efficiency": 1.0,
        "density_bin_width_um": 5.0,
        "pair_bin_width_um": 25.0,
    }
    return pd.DataFrame(rows), audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    con = sub.add_parser("consistency")
    con.add_argument("--candidates", required=True)
    con.add_argument("--tracks", required=True)
    con.add_argument("--output", required=True)
    con.add_argument("--pixel-pitch-um", type=float, default=10.0)
    con.add_argument("--fps", type=float, default=4000.0)
    rate = sub.add_parser("rate")
    rate.add_argument("--particles", required=True)
    rate.add_argument("--recorded-frames", required=True, type=int)
    rate.add_argument("--volume-cm3", required=True, type=float)
    rate.add_argument("--fps", default=4000.0, type=float)
    rate.add_argument("--output", required=True)
    rate.add_argument("--diameter-column", default="final_diameter_um")
    args = parser.parse_args(argv)
    if args.command == "consistency":
        result = consistency(
            read_table(args.candidates),
            read_tracks(args.tracks),
            Units(pixel_pitch_um=args.pixel_pitch_um, fps=args.fps),
        )
        audit = {
            "diameter": summary_metrics(result, "expected_post_diameter_um", "measured_post_diameter_um"),
            "vertical_velocity": summary_metrics(result, "expected_post_vy_m_s", "measured_post_vy_m_s"),
        }
    else:
        particles = read_table(args.particles)
        particles = particles[["frame", args.diameter_column]].rename(
            columns={args.diameter_column: "diameter_um"}
        )
        result, audit = collision_rate(
            particles, recorded_frames=args.recorded_frames, volume_cm3=args.volume_cm3, fps=args.fps
        )
    write_table(result, args.output)
    write_json(audit, str(args.output) + ".json")


if __name__ == "__main__":
    main()
