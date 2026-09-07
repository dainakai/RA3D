from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from ra3d.io import read_table, write_table
from ra3d.upstream import link
from ra3d.physics import collision_rate
from ra3d._core.physics import HallCorrectedLaminarKernel, pair_prediction
from ra3d.simulation.render import HologramSimulator, RenderOptions
from ra3d._core.two_d_budget import budget_select
from ra3d.workflow import label_candidates


def test_link_adapts_parquet_and_preserves_nonstandard_slice_units(tmp_path, monkeypatch):
    particles = tmp_path / "particles.parquet"
    write_table(pd.DataFrame({"frame": [0], "file": ["000001.png"], "slice": [8.5]}), particles)

    def fake_link(source, *, output_dir, pixel_pitch_um, linker_args):
        assert Path(source).suffix == ".csv"
        assert read_table(source).iloc[0].file == "000001.png"
        trajectory = Path(output_dir) / "trajectory3d"
        write_table(
            pd.DataFrame(
                {
                    "frame": [0, 1],
                    "xc": [10.0, 11.0],
                    "yc": [20.0, 21.0],
                    "slice": [8.5, 8.6],
                    "diameter_px": [3.0, 3.0],
                }
            ),
            trajectory / "path_00000001.csv",
        )
        return SimpleNamespace(trajectory_dir=trajectory)

    monkeypatch.setitem(sys.modules, "holod3_linker", SimpleNamespace(link_holod3_output=fake_link))
    link(particles, tmp_path / "linked", pixel_pitch_um=5.0, slice_spacing_um=50.0)
    result = read_table(tmp_path / "linked/trajectory3d.parquet")
    np.testing.assert_allclose(result.z, [8.5, 8.6])


def test_kernel_symmetry_units_and_unordered_pair_counting():
    kernel = HallCorrectedLaminarKernel()
    diameters = np.array([50.0, 100.0])
    matrix = kernel.kernel_m3_s(diameters[:, None], diameters[None, :])
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), 0.0)
    assert 0 < matrix[0, 1] < 1e-7
    result = pair_prediction(
        np.array([2.0, 3.0]), matrix, diameters, np.array([25.0, 125.0]), exposure_volume_m3_s=4.0
    )
    assert result[0, 0, 0] == pytest.approx(4.0 * 2.0 * 3.0 * matrix[0, 1])


def test_rate_excludes_detector_unsupported_frames_and_scales_exposure():
    rows = pd.DataFrame({"frame": np.repeat(np.arange(100), 2), "diameter_um": np.tile([52.5, 102.5], 100)})
    _, first = collision_rate(rows, recorded_frames=100, volume_cm3=1, bootstrap_replicates=10)
    extra = pd.concat([rows, pd.DataFrame({"frame": [99] * 100, "diameter_um": [52.5] * 100})])
    _, second = collision_rate(extra, recorded_frames=100, volume_cm3=1, bootstrap_replicates=10)
    assert first["effective_frames"] == 87
    assert second["predicted_count"] == first["predicted_count"]


def test_empty_optical_scene_preserves_background_intensity():
    engine = HologramSimulator(RenderOptions(image_size_px=16, propagation_size_px=24))
    primary, secondary = engine.render(pd.DataFrame(columns=["x_px", "y_px", "z_um", "diameter_um"]))
    np.testing.assert_allclose(primary, 77.5, atol=1e-4)
    np.testing.assert_allclose(secondary, 77.5, atol=1e-4)


def test_2d_budget_preserves_each_vanishing_track():
    candidates = pd.DataFrame(
        {"vanished_track_id": [1, 1, 2], "survivor_track_id": [3, 4, 5], "candidate_frame": [10, 11, 12]}
    )
    chosen, _ = budget_select(candidates, np.array([0.9, 0.8, 0.1]), 2)
    assert chosen.vanished_track_id.tolist() == [1, 2]
    with pytest.raises(RuntimeError, match="smaller"):
        budget_select(candidates, np.array([0.9, 0.8, 0.1]), 1)


def test_exhaustive_negative_scene_can_be_labeled_without_events():
    candidates = pd.DataFrame(
        {
            "vanished_track_id": [1],
            "survivor_track_id": [2],
            "candidate_frame": [5],
            "vanished_end_frame": [5],
            "x_um": [100.0],
            "y_um": [200.0],
            "z_um": [300.0],
        }
    )
    truth = pd.DataFrame(
        columns=[
            "collision_event_id",
            "frame",
            "x_um",
            "y_um",
            "z_um",
            "parent1_id",
            "parent2_id",
            "child_id",
        ]
    )
    mapping = pd.DataFrame(columns=["frame", "observed_track_id", "source_particle_id"])
    labeled, audit, _ = label_candidates(candidates, truth, mapping)
    assert labeled.identity_state.tolist() == ["negative"]
    assert audit["spatial_positive_rows"] == 0
