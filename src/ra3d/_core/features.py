"""Numerical kernels derived from enhanced_model_features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


OLD_FEATURES = [
    "time_offset_frame",
    "xy_dist_um",
    "z_dist_um",
    "closest_distance_um",
    "effective_distance_um",
    "radius_sum_um",
    "distance_margin_um",
    "effective_margin_um",
    "vanished_diameter_um",
    "survivor_diameter_um",
    "closing_speed_um_frame",
    "post_survival_frames",
    "pre_overlap_frames",
    "boundary_exit_score",
    "survivor_change_score",
    "survivor_delta_diameter_px",
    "survivor_delta_speed_px_frame",
    "vertical_y_gap_um",
    "vertical_closing_um_frame",
    "vertical_time_to_catch_frame",
    "vertical_penalty",
]


NEW_FEATURES = [
    "xy_gate_ratio_400",
    "z_gate_ratio_6000",
    "xy_z_gate_value_400_6000",
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
    "pre_z_first_um",
    "pre_z_last_um",
    "pre_z_last_minus_first_um",
    "pre_z_slope_um_frame",
    "pre_closing_eff_last_um_frame",
    "pre_closing_xy_last_um_frame",
    "pre_closing_eff_median_um_frame",
    "pre_closing_xy_median_um_frame",
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
    "diameter_under_0p725",
    "diameter_under_0p90",
    "same7_and_diameter_under_0p90",
    "moving_apart_xy_rule",
]


ENHANCED_FEATURES = OLD_FEATURES + NEW_FEATURES


def finite_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    default = pd.Series(np.nan, index=out.index)
    xy = finite_series(out.get("xy_dist_um", default))
    z = finite_series(out.get("z_dist_um", default))
    out["xy_gate_ratio_400"] = xy / 400.0
    out["z_gate_ratio_6000"] = z / 6000.0
    out["xy_z_gate_value_400_6000"] = (xy / 400.0) ** 2 + (z / 6000.0) ** 2

    pre_same = finite_series(out.get("pre_same_interp_count", default)).fillna(0)
    post_over_expected = finite_series(out.get("post_over_expected", default))
    post_diameter_n = finite_series(out.get("post_diameter_n", default)).fillna(0)
    small_large_ratio = finite_series(out.get("small_large_ratio", default))
    pre_kin_count = finite_series(out.get("pre_kin_both_count", default)).fillna(0)
    pre_xy_slope = finite_series(out.get("pre_xy_slope_um_frame", default))
    pre_xy_delta = finite_series(out.get("pre_xy_last_minus_first_um", default))
    pre_xy_frac = finite_series(out.get("pre_xy_increase_fraction", default))

    out["same_ge7"] = (pre_same >= 7).astype(float)
    out["same_ge8"] = (pre_same >= 8).astype(float)
    out["same_ge9"] = (pre_same >= 9).astype(float)
    out["diameter_under_0p725"] = (
        ((post_over_expected < 0.725) & (post_diameter_n >= 8) & (small_large_ratio >= 0.4))
        .fillna(False)
        .astype(float)
    )
    out["diameter_under_0p90"] = (
        ((post_over_expected < 0.90) & (post_diameter_n >= 3) & (small_large_ratio >= 0.4))
        .fillna(False)
        .astype(float)
    )
    out["same7_and_diameter_under_0p90"] = ((pre_same >= 7) & out["diameter_under_0p90"].astype(bool)).astype(
        float
    )
    out["moving_apart_xy_rule"] = (
        ((pre_kin_count >= 5) & (pre_xy_slope >= 5.0) & (pre_xy_delta >= 20.0) & (pre_xy_frac >= 0.75))
        .fillna(False)
        .astype(float)
    )
    for col in ["fragment_pre_full_v1", "fragment_pre8_v1", "fragment_pre7_v1"]:
        if col in out.columns:
            out[col] = out[col].astype(float)
    return out


def feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        if col not in out.columns:
            out[col] = np.nan
    x = out[features].copy()
    for col in features:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x
