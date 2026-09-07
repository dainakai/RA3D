"""Numerical kernels derived from build_enhanced_candidate_features.py."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Any
from dataclasses import dataclass, field
from scipy import ndimage as ndi
from .track_images import collect_diameter


def make_dark_mask(norm: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    dark = norm <= int(threshold)
    dark = ndi.binary_closing(dark, structure=np.ones((3, 3), dtype=bool), iterations=1)
    dark = ndi.binary_opening(dark, structure=np.ones((2, 2), dtype=bool), iterations=1)
    labels, nlab = ndi.label(dark)
    if nlab == 0:
        return dark, labels
    counts = np.bincount(labels.ravel())
    remove = np.flatnonzero(counts < 6)
    remove = remove[remove != 0]
    if remove.size:
        dark[np.isin(labels, remove)] = False
        labels, _ = ndi.label(dark)
    return dark, labels


def summarize_diameter(
    vanished: pd.DataFrame,
    survivor: pd.DataFrame,
    candidate_frame: int,
    *,
    pixel_pitch_um: float,
    pre_gap_frames: int,
    pre_window_frames: int,
    post_window_frames: int,
) -> dict[str, Any]:
    pre_v = collect_diameter(
        vanished,
        pixel_pitch_um=pixel_pitch_um,
        max_frame=int(candidate_frame) - int(pre_gap_frames),
        take_last=int(pre_window_frames),
    )
    pre_s = collect_diameter(
        survivor,
        pixel_pitch_um=pixel_pitch_um,
        max_frame=int(candidate_frame) - int(pre_gap_frames),
        take_last=int(pre_window_frames),
    )
    post_s = collect_diameter(
        survivor,
        pixel_pitch_um=pixel_pitch_um,
        min_frame=int(candidate_frame) + 1,
        take_first=int(post_window_frames),
    )
    pre_values = [v for v in (pre_v.diameter_um, pre_s.diameter_um) if v is not None and math.isfinite(v)]
    pre_large = max(pre_values) if pre_values else math.nan
    pre_small = min(pre_values) if pre_values else math.nan
    d_v = pre_v.diameter_um if pre_v.diameter_um is not None else math.nan
    d_s = pre_s.diameter_um if pre_s.diameter_um is not None else math.nan
    d_post = post_s.diameter_um if post_s.diameter_um is not None else math.nan
    expected = math.nan
    if math.isfinite(d_v) and math.isfinite(d_s) and d_v > 0 and d_s > 0:
        expected = float((d_v**3 + d_s**3) ** (1.0 / 3.0))
    volume_resid_frac = math.nan
    signed_volume_resid_frac = math.nan
    if math.isfinite(expected) and expected > 0 and math.isfinite(d_post):
        expected_vol = expected**3
        signed_volume_resid_frac = (expected_vol - d_post**3) / expected_vol
        volume_resid_frac = abs(signed_volume_resid_frac)
    return {
        "pre_diameter_vanished_um": d_v,
        "pre_diameter_survivor_um": d_s,
        "pre_diameter_large_um": pre_large,
        "pre_diameter_small_um": pre_small,
        "post_diameter_um": d_post,
        "pre_diameter_vanished_n": int(pre_v.count),
        "pre_diameter_survivor_n": int(pre_s.count),
        "post_diameter_n": int(post_s.count),
        "pixel_pitch_um": float(pixel_pitch_um),
        "pre_window_start_frame": min(x for x in [pre_v.start_frame, pre_s.start_frame] if x is not None)
        if pre_v.start_frame is not None or pre_s.start_frame is not None
        else math.nan,
        "pre_window_end_frame": max(x for x in [pre_v.end_frame, pre_s.end_frame] if x is not None)
        if pre_v.end_frame is not None or pre_s.end_frame is not None
        else math.nan,
        "post_window_start_frame": post_s.start_frame if post_s.start_frame is not None else math.nan,
        "post_window_end_frame": post_s.end_frame if post_s.end_frame is not None else math.nan,
        "post_over_large": d_post / pre_large
        if math.isfinite(d_post) and math.isfinite(pre_large) and pre_large > 0
        else math.nan,
        "post_over_expected": d_post / expected
        if math.isfinite(d_post) and math.isfinite(expected) and expected > 0
        else math.nan,
        "post_minus_large": d_post - pre_large
        if math.isfinite(d_post) and math.isfinite(pre_large)
        else math.nan,
        "post_minus_expected": d_post - expected
        if math.isfinite(d_post) and math.isfinite(expected)
        else math.nan,
        "volume_resid_frac": volume_resid_frac,
        "signed_volume_resid_frac": signed_volume_resid_frac,
        "small_large_ratio": pre_small / pre_large
        if math.isfinite(pre_small) and math.isfinite(pre_large) and pre_large > 0
        else math.nan,
        "expected_growth_um": expected - pre_large
        if math.isfinite(expected) and math.isfinite(pre_large)
        else math.nan,
        "growth_fraction_of_expected": (d_post - pre_large) / (expected - pre_large)
        if math.isfinite(d_post)
        and math.isfinite(expected)
        and math.isfinite(pre_large)
        and abs(expected - pre_large) > 1.0e-9
        else math.nan,
        "pre_n_min": min(int(pre_v.count), int(pre_s.count)),
    }


@dataclass
class SameAccumulator:
    candidate_key: str
    row_index: int
    vanished_track_id: int
    survivor_track_id: int
    candidate_frame: int
    x_um: float
    y_um: float
    z_um: float
    xy_dist_um: float
    z_dist_um: float
    base_score: float
    mode: str
    rendered_pre_frames: int = 0
    skipped_pre_frames: int = 0
    pre_same_interp_count: int = 0
    pre_same_observed_count: int = 0
    pre_alternating_same_count: int = 0
    component_areas: list[int] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        frac = self.pre_same_interp_count / self.rendered_pre_frames if self.rendered_pre_frames else 0.0
        area = float(np.median(self.component_areas)) if self.component_areas else 0.0
        return {
            "candidate_key": self.candidate_key,
            "row_index": self.row_index,
            "vanished_track_id": self.vanished_track_id,
            "survivor_track_id": self.survivor_track_id,
            "candidate_frame": self.candidate_frame,
            "x_um": self.x_um,
            "y_um": self.y_um,
            "z_um": self.z_um,
            "xy_dist_um": self.xy_dist_um,
            "z_dist_um": self.z_dist_um,
            "base_score_for_features": self.base_score,
            "same_component_mode": self.mode,
            "rendered_pre_frames": self.rendered_pre_frames,
            "skipped_pre_frames": self.skipped_pre_frames,
            "pre_same_interp_count": self.pre_same_interp_count,
            "pre_same_interp_fraction": frac,
            "pre_same_observed_count": self.pre_same_observed_count,
            "pre_alternating_same_count": self.pre_alternating_same_count,
            "same_component_area_median_px": area,
            "same_component_score": frac,
            "fragment_pre_full_v1": bool(self.pre_same_interp_count >= 9),
            "fragment_pre8_v1": bool(self.pre_same_interp_count >= 8),
            "fragment_pre7_v1": bool(self.pre_same_interp_count >= 7),
        }
