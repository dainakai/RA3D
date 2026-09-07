from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ra3d.config import (
    STAGE1_FEATURES,
    STATIC_FEATURES,
    ENHANCED_FEATURES,
    Units,
)
from ra3d.io import read_table
from ra3d.tracks import materialize_tracks
from ra3d.features import static_features, image_features
from ra3d.smoothing import savgol_reflected, smooth_tracks, SmoothingOptions
from ra3d.evaluation import event_curve
from ra3d._core.matching import incremental_maximum_matching

FIXTURES = Path(__file__).parent / "fixtures"


def test_real_trajectory_feature_reference(tmp_path):
    candidates = read_table(FIXTURES / "paper_candidates.parquet")
    tracks = read_table(FIXTURES / "paper_tracks.parquet")
    computed = static_features(candidates, tracks, materialize_tracks(tracks, tmp_path / "tracks"))
    assert (len(STAGE1_FEATURES), len(STATIC_FEATURES), len(ENHANCED_FEATURES)) == (24, 77, 92)
    for column in STATIC_FEATURES:
        np.testing.assert_allclose(
            computed[column], candidates[column], rtol=1e-9, atol=1e-8, equal_nan=True, err_msg=column
        )


def test_maximum_matching_can_reassign_a_previous_prediction():
    forward, reverse = incremental_maximum_matching([0, 1], {0: [0, 1], 1: [0]})
    assert forward == {0: 1, 1: 0}
    assert len(reverse) == 2


def test_scene_isolation_and_projected_xy_protocol():
    truth = pd.DataFrame({"frame": [10], "x_um": [0.0], "y_um": [0.0], "z_um": [100000.0]})
    pred = pd.DataFrame(
        {"candidate_frame": [10], "x_um": [0.0], "y_um": [0.0], "z_um": [0.0], "score": [0.9]}
    )
    _, xy = event_curve({1: pred}, {1: truth})
    _, xyz = event_curve({1: pred}, {1: truth}, projected_xy=False)
    assert xy["event_ap"] == 1 and xyz["event_ap"] == 0
    empty = pred.iloc[:0]
    _, pooled = event_curve({1: empty, 2: pred}, {1: truth, 2: truth.iloc[:0]})
    assert pooled["event_ap"] == 0


def test_reflected_savgol_reproduces_linear_signal_and_derivative():
    value = np.arange(13) * 2.5 + 7
    np.testing.assert_allclose(savgol_reflected(value), value, atol=1e-12)
    np.testing.assert_allclose(savgol_reflected(value, derivative=1), 2.5, atol=1e-12)
    assert np.array_equal(savgol_reflected([3.0], derivative=1), [0.0])


def test_zgmm_downweights_extreme_depth_outliers():
    rng = np.random.default_rng(1)
    frames = np.tile(np.arange(50), 4)
    z = 100 + 0.2 * frames + rng.normal(0, 0.03, 200)
    z[22] += 250
    table = pd.DataFrame(
        {
            "track_id": np.repeat(np.arange(4), 50),
            "frame": frames,
            "x": frames + 0.2,
            "y": frames * 2.0,
            "z": z,
            "diameter_px": 3.0,
        }
    )
    result, audit = smooth_tracks(table, SmoothingOptions(selection="fixed", degree=1, lambda_other=1e-6))
    assert result.loc[22, "postOutProb"] > 0.99
    assert abs(result.loc[22, "z"] - 104.4) < 0.1
    assert audit["fit"]["converged"]


def test_image_checkpoint_resume_is_identical(tmp_path):
    from types import SimpleNamespace

    candidates = read_table(FIXTURES / "paper_candidates.parquet").iloc[:2].copy()
    tracks = read_table(FIXTURES / "paper_tracks.parquet")

    class Engine:
        acquisition = SimpleNamespace(
            frames=pd.DataFrame({"primary": ["demo"] * 6000}, index=np.arange(6000)),
            units=Units(),
            wavelength_um=0.6328,
            phase_distance_um=1.0,
            reconstruction_start_um=1.0,
            padding_px=1024,
            phase_iterations=1,
            coefficients=None,
            mode="single_gabor",
        )

        def intensity(self, frame, z):
            y, x = np.indices((1024, 1024))
            return ((x + y + frame) % 100 / 100).astype(np.float32)

    uninterrupted = image_features(candidates, tracks, Engine())
    checkpoint = tmp_path / "state.npz"

    def interrupt(value):
        if value["stage"] == "image_features" and value["completed_frames"] == 2:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated"):
        image_features(
            candidates, tracks, Engine(), checkpoint_path=checkpoint, checkpoint_every=1, progress=interrupt
        )
    assert checkpoint.is_file()
    resumed = image_features(candidates, tracks, Engine(), checkpoint_path=checkpoint, checkpoint_every=1)
    pd.testing.assert_frame_equal(uninterrupted, resumed)
    assert np.all(resumed.rendered_pre_frames + resumed.skipped_pre_frames == 9)
    changed = candidates.copy()
    changed.loc[changed.index[0], "candidate_frame"] += 1
    with pytest.raises(ValueError, match="inputs differ"):
        image_features(changed, tracks, Engine(), checkpoint_path=checkpoint)
