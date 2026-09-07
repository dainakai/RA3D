"""Numerical kernels derived from common.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


def normalized_match_cost(
    candidate_frame: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    candidate_z: np.ndarray,
    truth_frame: np.ndarray,
    truth_x: np.ndarray,
    truth_y: np.ndarray,
    truth_z: np.ndarray,
) -> np.ndarray:
    return (
        ((candidate_frame - truth_frame) / 5.0) ** 2
        + (np.hypot(candidate_x - truth_x, candidate_y - truth_y) / 350.0) ** 2
        + (np.abs(candidate_z - truth_z) / 15000.0) ** 2
    )


def candidate_truth_adjacency(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    frame_column: str = "candidate_frame",
    frame_tolerance: int = 5,
    xy_tolerance_um: float = 350.0,
    z_tolerance_um: float = 15000.0,
) -> dict[int, list[int]]:
    """Build sparse current-metric adjacency ordered by normalized cost."""

    required_candidates = {frame_column, "x_um", "y_um", "z_um"}
    required_truth = {
        "collision_event_id",
        "frame",
        "x_um",
        "y_um",
        "z_um",
    }
    missing_candidates = sorted(required_candidates - set(candidates.columns))
    missing_truth = sorted(required_truth - set(truth.columns))
    if missing_candidates or missing_truth:
        raise RuntimeError(
            f"matching columns missing: candidates={missing_candidates}, truth={missing_truth}"
        )
    cframe = pd.to_numeric(candidates[frame_column], errors="raise").to_numpy(np.int64)
    cx = pd.to_numeric(candidates["x_um"], errors="coerce").to_numpy(float)
    cy = pd.to_numeric(candidates["y_um"], errors="coerce").to_numpy(float)
    cz = pd.to_numeric(candidates["z_um"], errors="coerce").to_numpy(float)
    tframe = pd.to_numeric(truth["frame"], errors="raise").to_numpy(np.int64)
    tx = pd.to_numeric(truth["x_um"], errors="coerce").to_numpy(float)
    ty = pd.to_numeric(truth["y_um"], errors="coerce").to_numpy(float)
    tz = pd.to_numeric(truth["z_um"], errors="coerce").to_numpy(float)
    truth_by_frame: dict[int, np.ndarray] = {}
    for frame, indices in truth.groupby("frame", sort=False).indices.items():
        truth_by_frame[int(frame)] = np.asarray(indices, dtype=np.int64)

    adjacency: dict[int, list[int]] = {}
    candidate_groups: dict[int, np.ndarray] = {}
    for frame, indices in candidates.groupby(frame_column, sort=False).indices.items():
        candidate_groups[int(frame)] = np.asarray(indices, dtype=np.int64)
    for frame, candidate_indices in candidate_groups.items():
        possible = [
            truth_by_frame[value]
            for value in range(
                frame - int(frame_tolerance),
                frame + int(frame_tolerance) + 1,
            )
            if value in truth_by_frame
        ]
        if not possible:
            continue
        truth_indices = np.concatenate(possible)
        local_cx = cx[candidate_indices, None]
        local_cy = cy[candidate_indices, None]
        local_cz = cz[candidate_indices, None]
        dxy = np.hypot(
            local_cx - tx[truth_indices][None, :],
            local_cy - ty[truth_indices][None, :],
        )
        dz = np.abs(local_cz - tz[truth_indices][None, :])
        valid = (
            np.isfinite(dxy)
            & np.isfinite(dz)
            & (dxy <= float(xy_tolerance_um))
            & (dz <= float(z_tolerance_um))
        )
        for local_index, columns in enumerate(valid):
            eligible = np.flatnonzero(columns)
            if not len(eligible):
                continue
            candidate_index = int(candidate_indices[local_index])
            selected_truth = truth_indices[eligible]
            costs = normalized_match_cost(
                np.full(len(eligible), cframe[candidate_index], dtype=float),
                np.full(len(eligible), cx[candidate_index], dtype=float),
                np.full(len(eligible), cy[candidate_index], dtype=float),
                np.full(len(eligible), cz[candidate_index], dtype=float),
                tframe[selected_truth].astype(float),
                tx[selected_truth],
                ty[selected_truth],
                tz[selected_truth],
            )
            order = np.lexsort((selected_truth, costs))
            adjacency[candidate_index] = [int(value) for value in selected_truth[order]]
    return adjacency


def incremental_maximum_matching(
    candidate_order: Iterable[int],
    adjacency: dict[int, list[int]],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return a maximum prediction-to-truth matching for an insertion order."""

    truth_to_prediction: dict[int, int] = {}
    prediction_to_truth: dict[int, int] = {}

    def augment(root_prediction: int) -> bool:
        queue = [root_prediction]
        queue_cursor = 0
        seen_predictions = {root_prediction}
        seen_truth: set[int] = set()
        parent: dict[int, tuple[int, int]] = {}
        endpoint: tuple[int, int] | None = None
        while queue_cursor < len(queue) and endpoint is None:
            prediction = queue[queue_cursor]
            queue_cursor += 1
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
        prediction, truth_index = endpoint
        while True:
            previous_truth = prediction_to_truth.get(prediction)
            truth_to_prediction[truth_index] = prediction
            prediction_to_truth[prediction] = truth_index
            if prediction == root_prediction:
                break
            parent_prediction, parent_truth = parent[prediction]
            if previous_truth != parent_truth:
                raise RuntimeError("incremental matching path is inconsistent")
            prediction = parent_prediction
            truth_index = parent_truth
        return True

    for prediction in candidate_order:
        augment(int(prediction))
    return prediction_to_truth, truth_to_prediction
