"""Preserve one candidate per vanishing track within a fixed static budget."""

from __future__ import annotations
import numpy as np
import pandas as pd


def budget_select(
    frame: pd.DataFrame, scores: np.ndarray, budget: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = frame.copy()
    work["stage1_score"] = np.asarray(scores, dtype=float)
    work["_row"] = np.arange(len(work), dtype=np.int64)
    ordered = work.sort_values(
        ["stage1_score", "candidate_frame", "vanished_track_id", "survivor_track_id", "_row"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    guaranteed = ordered.drop_duplicates("vanished_track_id", keep="first")
    guaranteed_rows = set(guaranteed["_row"].astype(int).tolist())
    if len(guaranteed_rows) > int(budget):
        raise RuntimeError("static budget is smaller than vanished-track count")
    fill = ordered.loc[~ordered["_row"].isin(guaranteed_rows)].head(int(budget) - len(guaranteed_rows))
    selected_rows = guaranteed_rows | set(fill["_row"].astype(int).tolist())
    selected = work.loc[work["_row"].isin(selected_rows)].copy()
    selected = selected.sort_values("_row").drop(columns=["_row"])
    if len(selected) != min(int(budget), len(frame)):
        raise RuntimeError("budget selection returned an unexpected row count")
    return (
        selected.reset_index(drop=True),
        {
            "budget_rows": int(budget),
            "guaranteed_vanished_best_rows": int(len(guaranteed_rows)),
            "global_fill_rows": int(len(fill)),
            "output_rows": int(len(selected)),
        },
    )
