"""Frozen scientific recipe and explicit acquisition units."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json

from ._core.features import ENHANCED_FEATURES, OLD_FEATURES

KEY_COLUMNS = ["vanished_track_id", "survivor_track_id", "candidate_frame"]
GEOMETRY_COLUMNS = ["x_um", "y_um", "z_um"]
STAGE1_FEATURES = OLD_FEATURES + ["xy_gate_ratio_400", "z_gate_ratio_6000", "xy_z_gate_value_400_6000"]
RAW_IMAGE_FEATURES = [
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
]
IMAGE_FEATURES = RAW_IMAGE_FEATURES + ["same_ge7", "same_ge8", "same_ge9", "same7_and_diameter_under_0p90"]
STATIC_FEATURES = [name for name in ENHANCED_FEATURES if name not in IMAGE_FEATURES]
DEVELOPMENT_SEEDS = (260740, 260741, 260742, 260743, 260744)
TERMINAL_SEED = 260745
HGB_PARAMS = dict(
    loss="log_loss",
    learning_rate=0.045,
    max_iter=350,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.02,
)


def paper_recipe() -> dict:
    return json.loads(files("ra3d").joinpath("config/paper.json").read_text())


@dataclass(frozen=True)
class Units:
    """Legacy trajectory indices: x/y pixels, z slices, velocity per frame."""

    pixel_pitch_um: float = 10.0
    slice_spacing_um: float = 100.0
    fps: float = 4000.0
    image_size_px: int = 1024

    def __post_init__(self):
        import math

        if not all(math.isfinite(float(v)) and float(v) > 0 for v in vars(self).values()):
            raise ValueError("All acquisition units and image dimensions must be finite and positive")

    def candidate_parameters(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            dx_um=self.pixel_pitch_um,
            dz_um=self.slice_spacing_um,
            fps=self.fps,
            datlen=self.image_size_px,
            min_vanished_len=2,
            min_survivor_len=2,
            min_post_frames=1,
            pre_window=12,
            post_window=12,
            xy_gate_um=1500.0,
            z_gate_um=30000.0,
            z_score_weight=0.15,
            time_tolerance=12.0,
            vertical_catch_window=16.0,
            boundary_margin_px=5.0,
            max_final_frame_gap=12,
        )
