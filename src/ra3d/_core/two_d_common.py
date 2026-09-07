"""Frozen depth-free feature and identity-labeling contracts."""

from __future__ import annotations
from typing import Any, Iterable
import math
import numpy as np
import pandas as pd
from .features import feature_frame as feature_frame

KEY_COLUMNS = ["vanished_track_id", "survivor_track_id", "candidate_frame"]


STAGE1_FEATURES = (
    "time_offset_frame",
    "xy_dist_um",
    "radius_sum_um",
    "vanished_diameter_um",
    "survivor_diameter_um",
    "post_survival_frames",
    "pre_overlap_frames",
    "survivor_delta_diameter_px",
    "vertical_y_gap_um",
    "vertical_closing_um_frame",
    "vertical_time_to_catch_frame",
    "vertical_penalty",
    "xy_gate_ratio_400",
)

STATIC_ONLY_FEATURES = (
    "pre_diameter_vanished_um",
    "pre_diameter_survivor_um",
    "pre_diameter_large_um",
    "pre_diameter_small_um",
    "post_diameter_um",
    "pre_diameter_vanished_n",
    "pre_diameter_survivor_n",
    "post_diameter_n",
    "post_over_large",
    "post_over_expected",
    "post_minus_large",
    "post_minus_expected",
    "volume_resid_frac",
    "signed_volume_resid_frac",
    "small_large_ratio",
    "expected_growth_um",
    "growth_fraction_of_expected",
    "pre_n_min",
    "pre_kin_both_count",
    "pre_kin_span_frames",
    "pre_xy_first_um",
    "pre_xy_last_um",
    "pre_xy_min_um",
    "pre_xy_max_um",
    "pre_xy_last_minus_first_um",
    "pre_xy_slope_um_frame",
    "pre_xy_tail_slope_um_frame",
    "pre_xy_increase_fraction",
    "pre_xy_tail_increase_fraction",
    "pre_xy_event_minus_last_um",
    "pre_xy_event_minus_min_um",
    "pre_closing_xy_last_um_frame",
    "pre_closing_xy_median_um_frame",
    "diameter_under_0p725",
    "diameter_under_0p90",
    "moving_apart_xy_rule",
)

TWO_D_FEATURES = STAGE1_FEATURES + STATIC_ONLY_FEATURES

EXCLUDED_DEPTH_FEATURES = (
    "z_dist_um",
    "closest_distance_um",
    "effective_distance_um",
    "distance_margin_um",
    "effective_margin_um",
    "closing_speed_um_frame",
    "boundary_exit_score",
    "survivor_change_score",
    "survivor_delta_speed_px_frame",
    "z_gate_ratio_6000",
    "xy_z_gate_value_400_6000",
    "pre_eff_first_um",
    "pre_eff_last_um",
    "pre_eff_min_um",
    "pre_eff_max_um",
    "pre_eff_last_minus_first_um",
    "pre_eff_slope_um_frame",
    "pre_eff_tail_slope_um_frame",
    "pre_eff_increase_fraction",
    "pre_eff_tail_increase_fraction",
    "pre_eff_event_minus_last_um",
    "pre_eff_event_minus_min_um",
    "pre_z_first_um",
    "pre_z_last_um",
    "pre_z_last_minus_first_um",
    "pre_z_slope_um_frame",
    "pre_closing_eff_last_um_frame",
    "pre_closing_eff_median_um_frame",
    "rendered_pre_frames",
    "skipped_pre_frames",
    "pre_same_interp_count",
    "pre_same_interp_fraction",
    "pre_same_observed_count",
    "pre_alternating_same_count",
    "same_component_area_median_px",
    "same_component_score",
    "fragment_pre_full_v1",
    "fragment_pre8_v1",
    "fragment_pre7_v1",
    "same_ge7",
    "same_ge8",
    "same_ge9",
    "same7_and_diameter_under_0p90",
)


def percentile_score(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy(float)


def add_2d_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    default = pd.Series(np.nan, index=output.index)
    xy = pd.to_numeric(output["xy_dist_um"], errors="coerce")
    output["xy_gate_ratio_400"] = xy / 400.0
    pre_count = pd.to_numeric(output.get("pre_kin_both_count", default), errors="coerce").fillna(0)
    pre_slope = pd.to_numeric(output.get("pre_xy_slope_um_frame", default), errors="coerce")
    pre_delta = pd.to_numeric(output.get("pre_xy_last_minus_first_um", default), errors="coerce")
    pre_fraction = pd.to_numeric(output.get("pre_xy_increase_fraction", default), errors="coerce")
    if "post_over_expected" in output:
        post_expected = pd.to_numeric(output["post_over_expected"], errors="coerce")
        post_count = pd.to_numeric(output.get("post_diameter_n", default), errors="coerce").fillna(0)
        ratio = pd.to_numeric(output.get("small_large_ratio", default), errors="coerce")
        output["diameter_under_0p725"] = (
            ((post_expected < 0.725) & (post_count >= 8) & (ratio >= 0.4)).fillna(False).astype(float)
        )
        output["diameter_under_0p90"] = (
            ((post_expected < 0.9) & (post_count >= 3) & (ratio >= 0.4)).fillna(False).astype(float)
        )
    output["moving_apart_xy_rule"] = (
        ((pre_count >= 5) & (pre_slope >= 5.0) & (pre_delta >= 20.0) & (pre_fraction >= 0.75))
        .fillna(False)
        .astype(float)
    )
    return output


def xy_spatial_adjacency(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    frame_column: str = "candidate_frame",
    frame_tolerance: int = 5,
    xy_tolerance_um: float = 350.0,
) -> dict[int, list[int]]:
    cframe = pd.to_numeric(candidates[frame_column], errors="raise").to_numpy(np.int64)
    cx = pd.to_numeric(candidates["x_um"], errors="coerce").to_numpy(float)
    cy = pd.to_numeric(candidates["y_um"], errors="coerce").to_numpy(float)
    tframe = pd.to_numeric(truth["frame"], errors="raise").to_numpy(np.int64)
    tx = pd.to_numeric(truth["x_um"], errors="coerce").to_numpy(float)
    ty = pd.to_numeric(truth["y_um"], errors="coerce").to_numpy(float)
    truth_by_frame = {
        int(frame): np.asarray(indices, dtype=np.int64)
        for (frame, indices) in truth.groupby("frame", sort=False).indices.items()
    }
    adjacency: dict[int, list[int]] = {}
    for frame, candidate_indices_raw in candidates.groupby(frame_column, sort=False).indices.items():
        candidate_indices = np.asarray(candidate_indices_raw, dtype=np.int64)
        possible = [
            truth_by_frame[value]
            for value in range(int(frame) - int(frame_tolerance), int(frame) + int(frame_tolerance) + 1)
            if value in truth_by_frame
        ]
        if not possible:
            continue
        truth_indices = np.concatenate(possible)
        dxy = np.hypot(
            cx[candidate_indices, None] - tx[truth_indices][None, :],
            cy[candidate_indices, None] - ty[truth_indices][None, :],
        )
        valid = np.isfinite(dxy) & (dxy <= float(xy_tolerance_um))
        for local_index, columns in enumerate(valid):
            eligible = np.flatnonzero(columns)
            if not len(eligible):
                continue
            candidate_index = int(candidate_indices[local_index])
            selected = truth_indices[eligible]
            cost = ((cframe[candidate_index] - tframe[selected]) / frame_tolerance) ** 2 + (
                dxy[local_index, eligible] / xy_tolerance_um
            ) ** 2
            order = np.lexsort((selected, cost))
            adjacency[candidate_index] = [int(value) for value in selected[order]]
    return adjacency


def incremental_maximum_matching(
    candidate_order: Iterable[int], adjacency: dict[int, list[int]]
) -> tuple[dict[int, int], dict[int, int]]:
    truth_to_prediction: dict[int, int] = {}
    prediction_to_truth: dict[int, int] = {}

    def augment(root: int) -> bool:
        queue = [root]
        cursor = 0
        seen_predictions = {root}
        seen_truth: set[int] = set()
        parent: dict[int, tuple[int, int]] = {}
        endpoint: tuple[int, int] | None = None
        while cursor < len(queue) and endpoint is None:
            prediction = queue[cursor]
            cursor += 1
            for truth_index in adjacency.get(prediction, []):
                if truth_index in seen_truth:
                    continue
                seen_truth.add(truth_index)
                previous = truth_to_prediction.get(truth_index)
                if previous is None:
                    endpoint = (prediction, truth_index)
                    break
                if previous not in seen_predictions:
                    seen_predictions.add(previous)
                    parent[previous] = (prediction, truth_index)
                    queue.append(previous)
        if endpoint is None:
            return False
        (prediction, truth_index) = endpoint
        while True:
            previous_truth = prediction_to_truth.get(prediction)
            truth_to_prediction[truth_index] = prediction
            prediction_to_truth[prediction] = truth_index
            if prediction == root:
                break
            (parent_prediction, parent_truth) = parent[prediction]
            if previous_truth != parent_truth:
                raise RuntimeError("inconsistent augmenting path")
            prediction = parent_prediction
            truth_index = parent_truth
        return True

    for candidate_index in candidate_order:
        augment(int(candidate_index))
    return (prediction_to_truth, truth_to_prediction)


def role_consistent(vanished_source: float, survivor_source: float, truth_row: pd.Series) -> bool:
    if (
        not np.isfinite(vanished_source)
        or not np.isfinite(survivor_source)
        or vanished_source < 0
        or (survivor_source < 0)
    ):
        return False
    vanished = int(vanished_source)
    survivor = int(survivor_source)
    parent1 = int(truth_row["parent1_id"])
    parent2 = int(truth_row["parent2_id"])
    child = int(truth_row["child_id"])
    return (
        vanished == parent1
        and survivor in {parent2, child}
        or (vanished == parent2 and survivor in {parent1, child})
    )


def attach_track_sources(candidates: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    output = candidates.drop(
        columns=["vanished_source_particle_id", "survivor_source_particle_id"], errors="ignore"
    ).copy()
    if "identity_candidate_frame" not in output:
        output["identity_candidate_frame"] = output["candidate_frame"]
    if "vanished_end_frame" not in output:
        if "time_offset_frame" not in output:
            raise RuntimeError("vanished_end_frame and time_offset_frame are absent")
        output["vanished_end_frame"] = output["identity_candidate_frame"].to_numpy(np.int64) - output[
            "time_offset_frame"
        ].to_numpy(np.int64)
    source = mapping[["frame", "observed_track_id", "source_particle_id"]].copy()
    if source.duplicated(["frame", "observed_track_id"]).any():
        raise RuntimeError("mapping keys are not unique")
    vanished = source.rename(
        columns={
            "frame": "vanished_end_frame",
            "observed_track_id": "vanished_track_id",
            "source_particle_id": "vanished_source_particle_id",
        }
    )
    survivor = source.rename(
        columns={
            "frame": "identity_candidate_frame",
            "observed_track_id": "survivor_track_id",
            "source_particle_id": "survivor_source_particle_id",
        }
    )
    output = output.merge(
        vanished, on=["vanished_end_frame", "vanished_track_id"], how="left", validate="many_to_one"
    ).merge(
        survivor, on=["identity_candidate_frame", "survivor_track_id"], how="left", validate="many_to_one"
    )
    return output


def attach_training_labels(
    candidates: pd.DataFrame, truth: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    adjacency = xy_spatial_adjacency(candidates, truth)
    (prediction_to_truth, truth_to_prediction) = incremental_maximum_matching(
        range(len(candidates)), adjacency
    )
    sources = attach_track_sources(candidates, mapping)
    truth_reset = truth.reset_index(drop=True)
    assigned = np.full(len(sources), -1, dtype=np.int32)
    state = np.full(len(sources), "negative", dtype=object)
    for candidate_index, truth_choices in adjacency.items():
        vanished = float(sources.loc[candidate_index, "vanished_source_particle_id"])
        survivor = float(sources.loc[candidate_index, "survivor_source_particle_id"])
        truth_index = int(prediction_to_truth.get(candidate_index, truth_choices[0]))
        for possible in truth_choices:
            if role_consistent(vanished, survivor, truth_reset.iloc[int(possible)]):
                truth_index = int(possible)
                break
        assigned[candidate_index] = truth_index
        known = np.isfinite(vanished) and np.isfinite(survivor) and (vanished >= 0) and (survivor >= 0)
        if role_consistent(vanished, survivor, truth_reset.iloc[truth_index]):
            state[candidate_index] = "identity_consistent"
        elif known:
            state[candidate_index] = "identity_inconsistent"
        else:
            state[candidate_index] = "identity_unknown"
    sources["spatial_positive"] = (assigned >= 0).astype(np.int8)
    sources["assigned_truth_index"] = assigned
    sources["identity_state"] = state
    event_ids = np.full(len(sources), -1, dtype=np.int32)
    positive_indices = np.flatnonzero(assigned >= 0)
    if len(positive_indices):
        event_ids[positive_indices] = truth_reset.loc[
            assigned[positive_indices], "collision_event_id"
        ].to_numpy(np.int32)
    sources["matched_truth_event_id"] = event_ids
    positive_mask = sources["spatial_positive"].eq(1)
    strict_mask = sources["identity_state"].eq("identity_consistent")
    selected_positive = np.zeros(len(sources), dtype=bool)
    if positive_mask.any():
        event_has_strict = (
            sources.loc[positive_mask]
            .groupby("matched_truth_event_id")["identity_state"]
            .transform(lambda values: values.eq("identity_consistent").any())
            .to_numpy(bool)
        )
        local_positive = np.flatnonzero(positive_mask.to_numpy())
        selected_positive[local_positive] = strict_mask.to_numpy()[local_positive] | ~event_has_strict
    sources["identity_training_selected"] = selected_positive
    sources["identity_event_factor"] = np.float32(0.0)
    if selected_positive.any():
        counts = (
            sources.loc[selected_positive]
            .groupby("matched_truth_event_id")["matched_truth_event_id"]
            .transform("size")
            .to_numpy(float)
        )
        sources.loc[selected_positive, "identity_event_factor"] = (1.0 / counts).astype(np.float32)
    sources["spatial_event_factor"] = np.float32(0.0)
    if positive_mask.any():
        counts = (
            sources.loc[positive_mask]
            .groupby("matched_truth_event_id")["matched_truth_event_id"]
            .transform("size")
            .to_numpy(float)
        )
        sources.loc[positive_mask, "spatial_event_factor"] = (1.0 / counts).astype(np.float32)
    sources["training_source"] = "synth"
    state_counts = sources["identity_state"].value_counts().to_dict()
    strict_events = set(assigned[np.flatnonzero(strict_mask.to_numpy())].astype(int).tolist())
    audit = {
        "candidate_rows": int(len(sources)),
        "spatial_positive_rows": int(positive_mask.sum()),
        "identity_training_positive_rows": int(selected_positive.sum()),
        "xy_exact_covered_events": int(len(truth_to_prediction)),
        "identity_consistent_events": int(len(strict_events)),
        "identity_state_rows": {str(key): int(value) for (key, value) in state_counts.items()},
    }
    return (sources, audit)


def logit_blend(left: np.ndarray, right: np.ndarray, right_weight: float) -> np.ndarray:
    a = np.clip(np.asarray(left, dtype=float), 1e-09, 1.0 - 1e-09)
    b = np.clip(np.asarray(right, dtype=float), 1e-09, 1.0 - 1e-09)
    a_logit = np.log(a / (1.0 - a))
    b_logit = np.log(b / (1.0 - b))
    value = (1.0 - right_weight) * a_logit + right_weight * b_logit
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def selected_gate_mask(frame: pd.DataFrame) -> np.ndarray:
    offset = pd.to_numeric(frame["time_offset_frame"], errors="raise").to_numpy(np.int64)
    post = pd.to_numeric(frame["post_survival_frames"], errors="raise").to_numpy(np.int64)
    xy = pd.to_numeric(frame["xy_dist_um"], errors="coerce").to_numpy(float)
    radius = pd.to_numeric(frame["radius_sum_um"], errors="coerce").to_numpy(float)
    return (
        (post >= 1)
        & (offset >= -12)
        & (offset <= 12)
        & np.isfinite(xy)
        & np.isfinite(radius)
        & (xy <= 1200.0)
        & (xy - radius <= 1200.0)
    )


def attach_outside_rank(frame: pd.DataFrame, score: np.ndarray, base_mask: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["stage1_score"] = np.asarray(score, dtype=float)
    output["selected_gate_member"] = np.asarray(base_mask, dtype=np.int8)
    output["outside_rank"] = np.int32(-1)
    outside = np.flatnonzero(~np.asarray(base_mask, dtype=bool))
    if len(outside):
        local_order = np.lexsort(
            (
                output["survivor_track_id"].to_numpy(np.int64)[outside],
                output["vanished_track_id"].to_numpy(np.int64)[outside],
                output["candidate_frame"].to_numpy(np.int64)[outside],
                -np.asarray(score, dtype=float)[outside],
            )
        )
        ranked = outside[local_order]
        output.loc[ranked, "outside_rank"] = np.arange(len(ranked), dtype=np.int32)
    return output


def shortlist_mask(frame: pd.DataFrame, outside_fraction: float = 0.4) -> np.ndarray:
    base = frame["selected_gate_member"].eq(1).to_numpy()
    outside_count = int((~base).sum())
    keep = int(math.ceil(float(outside_fraction) * outside_count))
    rank = frame["outside_rank"].to_numpy(np.int64)
    return base | (rank >= 0) & (rank < keep)


def rank_within_vanished(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    work = frame[["candidate_frame", "vanished_track_id", "survivor_track_id"]].copy()
    work["_score"] = np.asarray(score, dtype=float)
    work["_row"] = np.arange(len(work), dtype=np.int64)
    ordered = work.sort_values(
        ["vanished_track_id", "_score", "candidate_frame", "survivor_track_id", "_row"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    ordered["_rank"] = ordered.groupby("vanished_track_id", sort=False).cumcount()
    return ordered.set_index("_row")["_rank"].reindex(np.arange(len(work))).to_numpy(np.int32)


def identity_adjacency(
    predictions: pd.DataFrame, truth: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[dict[int, list[int]], pd.DataFrame]:
    spatial = xy_spatial_adjacency(predictions, truth)
    if not spatial:
        return ({}, predictions.iloc[0:0].copy())
    spatial_indices = np.fromiter(spatial.keys(), dtype=np.int64, count=len(spatial))
    source_input = predictions.iloc[spatial_indices].copy()
    source_input["_candidate_index"] = spatial_indices
    sourced = attach_track_sources(source_input, mapping).set_index("_candidate_index", drop=False)
    truth_reset = truth.reset_index(drop=True)
    adjacency: dict[int, list[int]] = {}
    for candidate_index, truth_choices in spatial.items():
        vanished = float(sourced.loc[candidate_index, "vanished_source_particle_id"])
        survivor = float(sourced.loc[candidate_index, "survivor_source_particle_id"])
        mapping_known = (
            np.isfinite(vanished) and np.isfinite(survivor) and (vanished >= 0) and (survivor >= 0)
        )
        consistent = [
            int(truth_index)
            for truth_index in truth_choices
            if role_consistent(vanished, survivor, truth_reset.iloc[int(truth_index)]) or not mapping_known
        ]
        if consistent:
            adjacency[int(candidate_index)] = consistent
    return (adjacency, sourced)
