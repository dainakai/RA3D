"""Candidate generation and full trajectory/image feature extraction."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Units, KEY_COLUMNS, STAGE1_FEATURES, STATIC_FEATURES, RAW_IMAGE_FEATURES
from .io import require_columns, require_integer, require_unique
from .tracks import trajectory_objects
from ._core.candidates import build_frame_indices, generate_candidate_pool
from ._core.features import add_derived_features
from ._core.feature_primitives import summarize_diameter, make_dark_mask
from ._core.kinematics import analyze_row
from ._core.track_images import TrackPoint, finite_float, crop_with_mean_padding, normalize_crop
from ._core.components import point_hit, component_area
from .acquisition import Reconstructor


def validate_candidates(table: pd.DataFrame):
    require_integer(table, KEY_COLUMNS, name="candidates")
    require_unique(table, KEY_COLUMNS, name="candidates")
    require_columns(table, ["x_um", "y_um", "z_um"], name="candidates", finite=True)
    if table.vanished_track_id.eq(table.survivor_track_id).any():
        raise ValueError("A candidate must contain two different trajectories")


def generate_candidates(tracks: pd.DataFrame, units: Units = Units()) -> pd.DataFrame:
    objects, table = trajectory_objects(tracks)
    parameters = units.candidate_parameters()
    indices = build_frame_indices(table, parameters)
    rows = generate_candidate_pool(objects, table, indices, parameters, max_slack_um=2000.0)
    if not rows:
        return pd.DataFrame(
            columns=list(dict.fromkeys(KEY_COLUMNS + ["x_um", "y_um", "z_um"] + STAGE1_FEATURES))
        )
    result = add_derived_features(pd.DataFrame(rows))
    validate_candidates(result)
    return result


def static_features(
    candidates: pd.DataFrame, tracks: pd.DataFrame, track_dir: Path, units: Units = Units(), progress=None
) -> pd.DataFrame:
    """Add 53 trajectory features; image columns remain unavailable."""
    validate_candidates(candidates)
    track_frames = {int(k): v.reset_index(drop=True) for k, v in tracks.groupby("track_id", sort=False)}
    kinematic_cache = {}
    rows = []
    for idx, row in candidates.reset_index(drop=True).iterrows():
        vanished, survivor = int(row.vanished_track_id), int(row.survivor_track_id)
        if vanished not in track_frames or survivor not in track_frames:
            raise ValueError(f"Candidate refers to missing trajectory {vanished} or {survivor}")
        result = summarize_diameter(
            track_frames[vanished],
            track_frames[survivor],
            int(row.candidate_frame),
            pixel_pitch_um=units.pixel_pitch_um,
            pre_gap_frames=2,
            pre_window_frames=10,
            post_window_frames=20,
        )
        result.update(
            analyze_row(
                row,
                track_dir=track_dir,
                track_cache=kinematic_cache,
                pre_frames=10,
                dx_um=units.pixel_pitch_um,
                dz_um=units.slice_spacing_um,
            )
        )
        rows.append(result)
        if progress and (idx + 1) % 1000 == 0:
            progress({"stage": "static_features", "rows": idx + 1, "total": len(candidates)})
    result = candidates.reset_index(drop=True).copy()
    if rows:
        extra = pd.DataFrame(rows)
        for col in extra:
            result[col] = extra[col].to_numpy()
    result = add_derived_features(result)
    require_columns(result, STATIC_FEATURES)
    return result


def image_features(
    candidates: pd.DataFrame,
    tracks: pd.DataFrame,
    reconstructor: Reconstructor,
    *,
    progress=None,
    checkpoint=None,
    checkpoint_every=50,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute all 15 image features, sharing frame FFTs and depth reconstructions.

    Missing observations outside a trajectory are recorded as skipped frames.
    Missing image files and reconstruction errors fail instead of being converted to false measurements.
    """
    validate_candidates(candidates)
    work = candidates.reset_index(drop=True)
    track_arrays = {
        int(k): v[["frame", "x", "y", "z", "diameter_px"]].to_numpy(float)
        for k, v in tracks.groupby("track_id", sort=False)
    }

    @lru_cache(maxsize=262144)
    def point(track_id, frame, nearest=False):
        array = track_arrays[track_id]
        frames = array[:, 0]
        idx = int(np.searchsorted(frames, frame))
        if idx < len(frames) and frames[idx] == frame:
            return TrackPoint(*map(float, array[idx, 1:]))
        if not nearest or idx == 0 or idx == len(frames):
            return None
        a = (float(frame) - float(frames[idx - 1])) / (float(frames[idx]) - float(frames[idx - 1]))
        return TrackPoint(*map(float, array[idx - 1, 1:] + a * (array[idx, 1:] - array[idx - 1, 1:])))

    counts = np.zeros((len(work), 5), dtype=np.int16)
    areas = [[] for _ in range(len(work))]
    last_frame = -1
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path:
        relevant = [*KEY_COLUMNS, "vanished_diameter_um", "survivor_diameter_um"]
        relevant = [c for c in relevant if c in work]
        acq = reconstructor.acquisition
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(work[relevant], index=False).to_numpy().tobytes()
            + pd.util.hash_pandas_object(tracks, index=False).to_numpy().tobytes()
            + pd.util.hash_pandas_object(acq.frames, index=True).to_numpy().tobytes()
            + json.dumps(
                {
                    "units": vars(acq.units),
                    "wavelength": acq.wavelength_um,
                    "phase": acq.phase_distance_um,
                    "start": acq.reconstruction_start_um,
                    "padding": acq.padding_px,
                    "iterations": acq.phase_iterations,
                    "coefficients": acq.coefficients.tolist() if acq.coefficients is not None else None,
                    "mode": acq.mode,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if checkpoint_path.exists():
            with np.load(checkpoint_path, allow_pickle=False) as saved:
                if str(saved["fingerprint"]) != fingerprint:
                    raise ValueError("Image checkpoint inputs differ; choose a new checkpoint path")
                counts = saved["counts"]
                areas = [row[row > 0].tolist() for row in saved["areas"]]
                last_frame = int(saved["frame"])
            if progress:
                progress({"stage": "image_resume", "completed_through_frame": last_frame})

    def save_checkpoint(frame):
        if checkpoint:
            checkpoint(frame, counts, areas)
        if checkpoint_path:
            packed = np.zeros((len(work), 9), dtype=np.int32)
            for i, values in enumerate(areas):
                packed[i, : len(values)] = values
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, frame=frame, counts=counts, areas=packed, fingerprint=fingerprint)
            temporary.replace(checkpoint_path)

    requests = defaultdict(list)
    for index, row in enumerate(work.itertuples(index=False)):
        vanished, survivor = int(row.vanished_track_id), int(row.survivor_track_id)
        frame = int(row.candidate_frame)
        remains = int(track_arrays[survivor][0, 0]) <= frame - 10
        vd = finite_float(getattr(row, "vanished_diameter_um", None))
        sd = finite_float(getattr(row, "survivor_diameter_um", None))
        for before in range(frame - 10, frame - 1):
            vp, sp = point(vanished, before, True), point(survivor, before, True)
            if remains and sp is not None:
                center = sp
            elif vp is None:
                center = sp
            elif sp is None:
                center = vp
            else:
                center = vp if (vd or vp.diameter_px) >= (sd or sp.diameter_px) else sp
            if center is None:
                if last_frame < 0:
                    counts[index, 1] += 1
                continue
            if before not in reconstructor.acquisition.frames.index:
                # Windows before acquisition starts are expected; holes inside it are errors.
                if (
                    before < reconstructor.acquisition.frames.index.min()
                    or before > reconstructor.acquisition.frames.index.max()
                ):
                    if last_frame < 0:
                        counts[index, 1] += 1
                    continue
                raise ValueError(f"Acquisition has an interior frame gap at {before}")
            if before <= last_frame:
                continue
            points = (point(vanished, before), point(survivor, before), vp, sp)
            requests[before].append((index, center, points))
        if progress and (index + 1) % 10000 == 0:
            progress({"stage": "image_requests", "rows": index + 1, "total": len(work)})
    for completed, frame in enumerate(sorted(requests), start=1):
        by_depth = defaultdict(list)
        for request in requests.pop(frame):
            by_depth[request[1].z].append(request)
        for z, group in by_depth.items():
            full = reconstructor.intensity(frame, z)
            padding_mean = float(full.mean())
            crop_cache = {}
            for index, center, points in group:
                crop_key = (round(center.x), round(center.y))
                if crop_key not in crop_cache:
                    crop, x0, y0 = crop_with_mean_padding(
                        full, center.x, center.y, 128, padding_mean=padding_mean
                    )
                    _, labels = make_dark_mask(normalize_crop(crop), 30)
                    crop_cache[crop_key] = (labels, x0, y0)
                labels, x0, y0 = crop_cache[crop_key]
                hits = [point_hit(labels, p, x0, y0, observed=p is not None) for p in points]
                observed_same = hits[0] is not None and hits[1] is not None and hits[0].label == hits[1].label
                interpolated_same = (
                    hits[2] is not None and hits[3] is not None and hits[2].label == hits[3].label
                )
                one_observed = (points[0] is None) ^ (points[1] is None)
                counts[index, 0] += 1
                counts[index, 2] += int(interpolated_same)
                counts[index, 3] += int(observed_same)
                counts[index, 4] += int(one_observed and interpolated_same)
                if interpolated_same:
                    areas[index].append(component_area(labels, hits[2].label))
        if progress and (completed == 1 or completed % checkpoint_every == 0):
            progress(
                {
                    "stage": "image_features",
                    "frame": frame,
                    "completed_frames": completed,
                    "remaining_frames": len(requests),
                }
            )
        if completed % checkpoint_every == 0:
            save_checkpoint(frame)
    save_checkpoint(frame if "completed" in locals() else last_frame)
    result = work.drop(columns=RAW_IMAGE_FEATURES, errors="ignore").copy()
    result["rendered_pre_frames"] = counts[:, 0]
    result["skipped_pre_frames"] = counts[:, 1]
    result["pre_same_interp_count"] = counts[:, 2]
    result["pre_same_interp_fraction"] = np.divide(
        counts[:, 2], counts[:, 0], out=np.zeros(len(work)), where=counts[:, 0] > 0
    )
    result["pre_same_observed_count"] = counts[:, 3]
    result["pre_alternating_same_count"] = counts[:, 4]
    result["same_component_area_median_px"] = [float(np.median(v)) if v else 0.0 for v in areas]
    result["same_component_score"] = result.pre_same_interp_fraction
    for column, threshold in [("fragment_pre_full_v1", 9), ("fragment_pre8_v1", 8), ("fragment_pre7_v1", 7)]:
        result[column] = counts[:, 2] >= threshold
    result["image_features_computed"] = True
    if not np.equal(counts[:, :2].sum(axis=1), 9).all():
        raise RuntimeError("Image window accounting failed")
    return add_derived_features(result)
