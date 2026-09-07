"""Numerical kernels derived from run_structural_policy.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from . import common as campaign
from .suppression import coordinate_nms


def adjusted_score(
    score: np.ndarray,
    rank: np.ndarray,
    penalty: float,
) -> np.ndarray:
    probability = np.clip(np.asarray(score, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    log_odds = np.log(probability / (1.0 - probability))
    return 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                log_odds - float(penalty) * np.asarray(rank, dtype=float),
                -40.0,
                40.0,
            )
        )
    )


def temporal_mode_predictions(
    frame: pd.DataFrame,
    *,
    seed: int,
    method: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    score_column = str(config.get("score_column", "base_percentile"))
    mode_gap = int(config["pair_mode_gap_frames"])
    top_k = int(config["maximum_modes_per_vanished_track"])
    columns = campaign.KEY_COLUMNS + campaign.GEOMETRY_COLUMNS
    work = frame[columns].copy()
    work["_source_index"] = np.arange(len(work), dtype=np.int64)
    work["_policy_score"] = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    ordered = work.sort_values(
        [
            "vanished_track_id",
            "survivor_track_id",
            "candidate_frame",
            "_policy_score",
            "_source_index",
        ],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).copy()
    frame_difference = ordered.groupby(["vanished_track_id", "survivor_track_id"], sort=False)[
        "candidate_frame"
    ].diff()
    ordered["_new_temporal_mode"] = (frame_difference.isna() | frame_difference.gt(mode_gap)).astype(np.int32)
    ordered["_temporal_mode"] = (
        ordered.groupby(["vanished_track_id", "survivor_track_id"], sort=False)["_new_temporal_mode"]
        .cumsum()
        .astype(np.int32)
    )
    representatives = (
        ordered.sort_values(
            [
                "_policy_score",
                "candidate_frame",
                "vanished_track_id",
                "survivor_track_id",
                "_source_index",
            ],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        .drop_duplicates(
            ["vanished_track_id", "survivor_track_id", "_temporal_mode"],
            keep="first",
        )
        .copy()
    )
    representatives = representatives.sort_values(
        ["_policy_score", "candidate_frame", "_source_index"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    representatives["_lineage_rank"] = representatives.groupby("vanished_track_id", sort=False).cumcount()
    capped = representatives[representatives["_lineage_rank"] < top_k].copy()
    if bool(config.get("coordinate_suppression", True)):
        selected = coordinate_nms(
            capped,
            frame_tolerance=int(config.get("coordinate_frame_tolerance", 2)),
            xy_tolerance_um=float(config.get("coordinate_xy_tolerance_um", 250.0)),
            z_tolerance_um=float(config.get("coordinate_z_tolerance_um", 7500.0)),
        )
    else:
        selected = capped
    selected[method] = adjusted_score(
        selected["_policy_score"].to_numpy(float),
        selected["_lineage_rank"].to_numpy(float),
        float(config.get("lineage_log_odds_penalty", 1.0)),
    )
    selected["synth_seed"] = int(seed)
    selected["method"] = method
    audit = {
        "seed": int(seed),
        "input_candidates": int(len(work)),
        "temporal_pair_modes": int(len(representatives)),
        "post_vanished_cap": int(len(capped)),
        "post_coordinate_suppression": int(len(selected)),
        "pair_mode_gap_frames": mode_gap,
        "maximum_modes_per_vanished_track": top_k,
    }
    return selected, audit
