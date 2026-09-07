"""Numerical kernels derived from run_asymmetric_full.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_ranks(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Attach deterministic global, pair, and vanished-track score ranks.

    ``scripts/retraincurrent_recall90`` is inserted at the front of
    ``sys.path`` for the shared experiment helpers.  That directory also has a
    historical ``run_image_shortlist`` module whose rank helper is tied to one
    global score column.  Keep the score-column-aware implementation local so
    that module-name shadowing cannot silently change this experiment.
    """
    output = frame.reset_index(drop=True).copy()
    output["_local_row"] = np.arange(len(output), dtype=np.int64)
    ordered = output.sort_values(
        [
            score_column,
            "candidate_frame",
            "vanished_track_id",
            "survivor_track_id",
            "_local_row",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).copy()
    ordered["global_rank"] = np.arange(len(ordered), dtype=np.int64)
    ordered["pair_rank"] = ordered.groupby(["vanished_track_id", "survivor_track_id"], sort=False).cumcount()
    ordered["vanished_rank"] = ordered.groupby("vanished_track_id", sort=False).cumcount()
    ranks = ordered.set_index("_local_row")[["global_rank", "pair_rank", "vanished_rank"]]
    output = output.join(ranks, on="_local_row")
    for column in ("global_rank", "pair_rank", "vanished_rank"):
        output[column] = output[column].astype(np.int32)
    return output
