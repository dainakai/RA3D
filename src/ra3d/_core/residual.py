"""Numerical kernels derived from run_coordinate_correction.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from . import common as campaign
from sklearn.ensemble import HistGradientBoostingRegressor


TARGETS = ("frame_delta", "x_delta", "y_delta", "z_delta")


def correct_predictions(
    prediction: pd.DataFrame,
    full_frame: pd.DataFrame,
    models: dict[str, HistGradientBoostingRegressor],
    *,
    source_method: str,
    method: str,
) -> pd.DataFrame:
    source_indices = prediction["_source_index"].to_numpy(np.int64)
    selected_features = full_frame.iloc[source_indices]
    x = campaign.feature_frame(selected_features, list(campaign.ENHANCED_FEATURES))
    correction = {name: models[name].predict(x) for name in TARGETS}
    output = prediction.copy()
    output["candidate_frame"] = np.rint(
        output["candidate_frame"].to_numpy(float) + np.clip(correction["frame_delta"], -5.0, 5.0)
    ).astype(np.int64)
    output["x_um"] = output["x_um"].to_numpy(float) + np.clip(correction["x_delta"], -350.0, 350.0)
    output["y_um"] = output["y_um"].to_numpy(float) + np.clip(correction["y_delta"], -350.0, 350.0)
    output["z_um"] = output["z_um"].to_numpy(float) + np.clip(correction["z_delta"], -15000.0, 15000.0)
    output[method] = output[source_method].to_numpy(float)
    output["method"] = method
    return output
