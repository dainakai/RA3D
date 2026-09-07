"""Numerical kernels derived from run_permissive_static.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_score(score: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(score, dtype=float)).rank(method="average", pct=True).to_numpy(float)


def logit_blend(
    stage1_score: np.ndarray,
    static_score: np.ndarray,
    static_weight: float,
) -> np.ndarray:
    stage1 = np.clip(np.asarray(stage1_score, dtype=float), 1.0e-9, 1.0 - 1.0e-9)
    static = np.clip(np.asarray(static_score, dtype=float), 1.0e-9, 1.0 - 1.0e-9)
    stage1_logit = np.log(stage1 / (1.0 - stage1))
    static_logit = np.log(static / (1.0 - static))
    blended = (1.0 - float(static_weight)) * stage1_logit + float(static_weight) * static_logit
    return 1.0 / (1.0 + np.exp(-np.clip(blended, -40.0, 40.0)))
