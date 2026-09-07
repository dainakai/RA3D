"""Numerical kernels derived from run_identity_label_ranking.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from .matching import candidate_truth_adjacency, incremental_maximum_matching


def role_consistent(
    vanished_source: float,
    survivor_source: float,
    truth_row: pd.Series,
) -> bool:
    if (
        not np.isfinite(vanished_source)
        or not np.isfinite(survivor_source)
        or vanished_source < 0
        or survivor_source < 0
    ):
        return False
    vanished = int(vanished_source)
    survivor = int(survivor_source)
    parent1 = int(truth_row["parent1_id"])
    parent2 = int(truth_row["parent2_id"])
    child = int(truth_row["child_id"])
    return (vanished == parent1 and survivor in {parent2, child}) or (
        vanished == parent2 and survivor in {parent1, child}
    )


def attach_identity_labels(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    adjacency = candidate_truth_adjacency(candidates, truth)
    prediction_to_truth, truth_to_prediction = incremental_maximum_matching(range(len(candidates)), adjacency)
    positive_indices = np.fromiter(adjacency.keys(), dtype=np.int64, count=len(adjacency))
    positive = candidates.loc[
        positive_indices,
        [
            "candidate_frame",
            "vanished_end_frame",
            "vanished_track_id",
            "survivor_track_id",
        ],
    ].copy()
    positive["candidate_index"] = positive_indices

    mapping_columns = [
        "frame",
        "observed_track_id",
        "source_particle_id",
    ]
    mapping = mapping[mapping_columns].copy()
    if mapping.duplicated(["frame", "observed_track_id"]).any():
        raise RuntimeError("track-to-truth mapping keys are not unique")
    vanished_mapping = mapping.rename(
        columns={
            "frame": "vanished_end_frame",
            "observed_track_id": "vanished_track_id",
            "source_particle_id": "vanished_source_particle_id",
        }
    )
    survivor_mapping = mapping.rename(
        columns={
            "frame": "candidate_frame",
            "observed_track_id": "survivor_track_id",
            "source_particle_id": "survivor_source_particle_id",
        }
    )
    positive = positive.merge(
        vanished_mapping,
        on=["vanished_end_frame", "vanished_track_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        survivor_mapping,
        on=["candidate_frame", "survivor_track_id"],
        how="left",
        validate="many_to_one",
    )
    positive = positive.set_index("candidate_index").reindex(positive_indices)
    vanished_source = pd.to_numeric(positive["vanished_source_particle_id"], errors="coerce").to_numpy(float)
    survivor_source = pd.to_numeric(positive["survivor_source_particle_id"], errors="coerce").to_numpy(float)

    truth_reset = truth.reset_index(drop=True)
    assigned = np.full(len(candidates), -1, dtype=np.int32)
    state = np.full(len(candidates), "negative", dtype=object)
    vanished_map_values = np.full(len(candidates), np.nan, dtype=float)
    survivor_map_values = np.full(len(candidates), np.nan, dtype=float)
    for position, candidate_index in enumerate(positive_indices):
        candidate_index = int(candidate_index)
        vanished = float(vanished_source[position])
        survivor = float(survivor_source[position])
        anchored_truth = prediction_to_truth.get(candidate_index)
        if anchored_truth is not None:
            truth_index = int(anchored_truth)
        else:
            truth_index = int(adjacency[candidate_index][0])
            for possible_truth in adjacency[candidate_index]:
                if role_consistent(
                    vanished,
                    survivor,
                    truth_reset.iloc[int(possible_truth)],
                ):
                    truth_index = int(possible_truth)
                    break
        assigned[candidate_index] = truth_index
        vanished_map_values[candidate_index] = vanished
        survivor_map_values[candidate_index] = survivor
        known = np.isfinite(vanished) and np.isfinite(survivor) and vanished >= 0 and survivor >= 0
        if role_consistent(
            vanished,
            survivor,
            truth_reset.iloc[truth_index],
        ):
            state[candidate_index] = "identity_consistent"
        elif known:
            state[candidate_index] = "identity_inconsistent"
        else:
            state[candidate_index] = "identity_unknown"

    output = candidates.copy()
    output["spatial_positive"] = (assigned >= 0).astype(np.int8)
    output["assigned_truth_index"] = assigned
    assigned_event = np.full(len(output), -1, dtype=np.int32)
    assigned_component = np.full(len(output), "", dtype=object)
    if len(positive_indices):
        truth_indices = assigned[positive_indices]
        assigned_event[positive_indices] = truth_reset.loc[truth_indices, "collision_event_id"].to_numpy(
            np.int32
        )
        assigned_component[positive_indices] = (
            truth_reset.loc[truth_indices, "sampling_component"].astype(str).to_numpy()
        )
    output["matched_truth_event_id"] = assigned_event
    output["truth_sampling_component"] = assigned_component
    output["identity_state"] = state
    output["vanished_source_particle_id"] = vanished_map_values
    output["survivor_source_particle_id"] = survivor_map_values

    positive_mask = output["spatial_positive"].eq(1)
    spatial_counts = (
        output.loc[positive_mask]
        .groupby("matched_truth_event_id")["matched_truth_event_id"]
        .transform("size")
        .to_numpy(float)
    )
    output["spatial_event_factor"] = np.float32(0.0)
    output.loc[positive_mask, "spatial_event_factor"] = (1.0 / spatial_counts).astype(np.float32)

    strict_mask = output["identity_state"].eq("identity_consistent")
    event_has_strict = (
        output.loc[positive_mask]
        .groupby("matched_truth_event_id")["identity_state"]
        .transform(lambda values: values.eq("identity_consistent").any())
        .to_numpy(bool)
    )
    selected_positive = np.zeros(len(output), dtype=bool)
    local_positive_indices = np.flatnonzero(positive_mask.to_numpy())
    selected_positive[local_positive_indices] = (
        strict_mask.to_numpy()[local_positive_indices] | ~event_has_strict
    )
    output["identity_training_selected"] = selected_positive
    selected_mask = output["identity_training_selected"]
    identity_counts = (
        output.loc[selected_mask]
        .groupby("matched_truth_event_id")["matched_truth_event_id"]
        .transform("size")
        .to_numpy(float)
    )
    output["identity_event_factor"] = np.float32(0.0)
    output.loc[selected_mask, "identity_event_factor"] = (1.0 / identity_counts).astype(np.float32)

    assigned_truth_events = set(assigned[positive_indices].astype(int).tolist())
    strict_truth_events = set(
        assigned[np.flatnonzero(output["identity_state"].eq("identity_consistent").to_numpy())]
        .astype(int)
        .tolist()
    )
    component_rows = []
    for component, indices in truth_reset.groupby("sampling_component", sort=True).indices.items():
        component_truth = {int(value) for value in indices}
        component_rows.append(
            {
                "sampling_component": str(component),
                "truth_events": int(len(component_truth)),
                "exact_covered_events": int(len(component_truth & set(truth_to_prediction))),
                "attributed_events": int(len(component_truth & assigned_truth_events)),
                "identity_consistent_events": int(len(component_truth & strict_truth_events)),
            }
        )
    component_table = pd.DataFrame(
        component_rows,
        columns=[
            "sampling_component",
            "truth_events",
            "exact_covered_events",
            "attributed_events",
            "identity_consistent_events",
        ],
    )
    for numerator in (
        "exact_covered_events",
        "attributed_events",
        "identity_consistent_events",
    ):
        component_table[numerator.replace("_events", "_recall")] = (
            component_table[numerator] / component_table["truth_events"]
        )

    state_counts = output["identity_state"].value_counts().to_dict()
    audit = {
        "candidate_rows": int(len(output)),
        "spatial_positive_rows": int(positive_mask.sum()),
        "spatial_negative_rows": int((~positive_mask).sum()),
        "exact_covered_events": int(len(truth_to_prediction)),
        "attributed_events": int(len(assigned_truth_events)),
        "identity_consistent_events": int(len(strict_truth_events)),
        "identity_fallback_events": int(len(assigned_truth_events - strict_truth_events)),
        "identity_training_positive_rows": int(selected_positive.sum()),
        "identity_state_rows": {str(key): int(value) for key, value in state_counts.items()},
    }
    return output, audit, component_table
