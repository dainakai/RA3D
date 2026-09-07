"""Numerical kernels derived from run_full_feature_cascade.py."""

from __future__ import annotations

import pandas as pd
import math
from collections import defaultdict


def coordinate_nms(
    frame: pd.DataFrame,
    *,
    frame_tolerance: int,
    xy_tolerance_um: float,
    z_tolerance_um: float,
) -> pd.DataFrame:
    ordered = frame.sort_values("_policy_score", ascending=False, kind="mergesort")
    kept_indices = []
    kept_by_frame: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for row in ordered.itertuples(index=True):
        frame_value = int(row.candidate_frame)
        x = float(row.x_um)
        y = float(row.y_um)
        z = float(row.z_um)
        conflict = False
        for nearby_frame in range(
            frame_value - frame_tolerance,
            frame_value + frame_tolerance + 1,
        ):
            for other_x, other_y, other_z in kept_by_frame.get(nearby_frame, ()):
                if math.hypot(x - other_x, y - other_y) > xy_tolerance_um:
                    continue
                if abs(z - other_z) <= z_tolerance_um:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        kept_indices.append(int(row.Index))
        kept_by_frame[frame_value].append((x, y, z))
    return frame.loc[kept_indices].copy()
