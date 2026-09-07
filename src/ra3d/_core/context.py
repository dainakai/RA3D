"""Numerical kernels derived from run_group_context.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from . import common as campaign


BASE_FEATURES = (
    "score",
    "score_percentile",
    "structural_score",
    "image_enriched",
    "time_offset_frame",
)


GROUP_SPECS = {
    "vanished": ["vanished_track_id"],
    "survivor": ["survivor_track_id"],
    "pair": ["vanished_track_id", "survivor_track_id"],
}


CONTEXT_FEATURES = tuple(
    list(BASE_FEATURES)
    + [
        f"{prefix}_{suffix}"
        for prefix in GROUP_SPECS
        for suffix in (
            "max",
            "second",
            "top3_mean",
            "mean",
            "std",
            "log_count",
            "rank",
            "below_max",
            "frame_span",
        )
    ]
    + [
        "vanished_log_unique_survivors",
        "survivor_log_unique_vanished",
        "vanished_image_fraction",
        "survivor_image_fraction",
    ]
)


def add_one_group_context(
    frame: pd.DataFrame,
    group_columns: list[str],
    prefix: str,
) -> None:
    group = frame.groupby(group_columns, sort=False)
    frame[f"{prefix}_max"] = group["score"].transform("max")
    frame[f"{prefix}_mean"] = group["score"].transform("mean")
    frame[f"{prefix}_std"] = group["score"].transform("std").fillna(0.0)
    frame[f"{prefix}_log_count"] = np.log1p(group["score"].transform("size").to_numpy(float))
    frame[f"{prefix}_frame_span"] = (
        group["candidate_frame"].transform("max") - group["candidate_frame"].transform("min")
    ).astype(float)
    ordered = frame.sort_values(
        group_columns + ["score", "candidate_frame", "_source_index"],
        ascending=[True] * len(group_columns) + [False, True, True],
        kind="mergesort",
    ).copy()
    ordered["_local_rank"] = ordered.groupby(group_columns, sort=False).cumcount()
    rank_by_source = ordered.set_index("_source_index")["_local_rank"]
    frame[f"{prefix}_rank"] = frame["_source_index"].map(rank_by_source).astype(np.int32)
    second_rows = ordered[ordered["_local_rank"].eq(1)]
    top3 = ordered[ordered["_local_rank"].lt(3)].groupby(group_columns, sort=False)["score"].mean()
    if len(group_columns) == 1:
        second = second_rows.set_index(group_columns[0])["score"]
        keys: Any = frame[group_columns[0]]
        frame[f"{prefix}_second"] = keys.map(second)
        frame[f"{prefix}_top3_mean"] = keys.map(top3)
    else:
        second = second_rows.set_index(group_columns)["score"]
        keys = pd.MultiIndex.from_frame(frame[group_columns])
        frame[f"{prefix}_second"] = second.reindex(keys).to_numpy(float)
        frame[f"{prefix}_top3_mean"] = top3.reindex(keys).to_numpy(float)
    frame[f"{prefix}_second"] = frame[f"{prefix}_second"].fillna(frame[f"{prefix}_max"])
    frame[f"{prefix}_top3_mean"] = frame[f"{prefix}_top3_mean"].fillna(frame[f"{prefix}_max"])
    frame[f"{prefix}_below_max"] = frame[f"{prefix}_max"] - frame["score"]


def context_frame(
    compact: pd.DataFrame,
    score: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "synth_seed",
        *campaign.KEY_COLUMNS,
        *campaign.GEOMETRY_COLUMNS,
        "time_offset_frame",
        "structural_score",
        "image_enriched",
        "spatial_positive",
        "matched_truth_event_id",
    ]
    frame = compact[[column for column in columns if column in compact]].copy()
    frame["_source_index"] = np.arange(len(frame), dtype=np.int64)
    frame["score"] = np.asarray(score, dtype=np.float64)
    frame["score_percentile"] = campaign.percentile_score(score)
    for prefix, group_columns in GROUP_SPECS.items():
        add_one_group_context(frame, group_columns, prefix)
    vanished = frame.groupby("vanished_track_id", sort=False)
    survivor = frame.groupby("survivor_track_id", sort=False)
    frame["vanished_log_unique_survivors"] = np.log1p(
        vanished["survivor_track_id"].transform("nunique").to_numpy(float)
    )
    frame["survivor_log_unique_vanished"] = np.log1p(
        survivor["vanished_track_id"].transform("nunique").to_numpy(float)
    )
    frame["vanished_image_fraction"] = vanished["image_enriched"].transform("mean")
    frame["survivor_image_fraction"] = survivor["image_enriched"].transform("mean")
    return frame


def blend_scores(base: np.ndarray, context: np.ndarray, weight: float) -> np.ndarray:
    base = np.clip(np.asarray(base, dtype=float), 1.0e-9, 1.0 - 1.0e-9)
    context = np.clip(np.asarray(context, dtype=float), 1.0e-9, 1.0 - 1.0e-9)
    base_logit = np.log(base / (1.0 - base))
    context_logit = np.log(context / (1.0 - context))
    value = (1.0 - weight) * base_logit + weight * context_logit
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))
