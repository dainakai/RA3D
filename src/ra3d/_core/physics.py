"""Paper physical diagnostics; SCALE-SDM-derived table interpolation retains its BSD notice."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from importlib.resources import files


class HallCorrectedLaminarKernel:
    """Davis--Jonas--Hall efficiency with the source-code interpolation rules."""

    def __init__(self, root: Path | None = None):
        root = Path(root) if root is not None else Path(str(files("ra3d").joinpath("config/physics")))
        self.table_path = root / "collision_efficiency_table_long.csv"
        self.metadata_path = root / "extraction_metadata.csv"
        table = pd.read_csv(self.table_path)
        metadata = pd.read_csv(self.metadata_path).set_index("quantity")["value"]
        self.large_nodes_um = np.sort(table["r_large_um"].unique())
        self.ratio_nodes = np.sort(table["radius_ratio"].unique())
        pivot = table.pivot(index="r_large_um", columns="radius_ratio", values="collision_efficiency")
        self.efficiency = pivot.loc[self.large_nodes_um, self.ratio_nodes].to_numpy()
        self.surface_pressure_hpa = float(metadata["surface_pressure_hpa"])
        self.surface_temperature_k = float(metadata["derived_surface_temperature_k"])
        self.surface_density_kg_m3 = float(metadata["derived_surface_density_kg_m3"])

    def collision_efficiency(self, radius_1_um: np.ndarray, radius_2_um: np.ndarray) -> np.ndarray:
        radius_1_um, radius_2_um = np.broadcast_arrays(radius_1_um, radius_2_um)
        large = np.maximum(radius_1_um, radius_2_um)
        ratio = np.minimum(radius_1_um, radius_2_um) / large
        ratio_hi = np.searchsorted(self.ratio_nodes, ratio, side="left")
        ratio_hi = np.clip(ratio_hi, 1, len(self.ratio_nodes) - 1)
        ratio_lo = ratio_hi - 1
        q = (ratio - self.ratio_nodes[ratio_lo]) / (self.ratio_nodes[ratio_hi] - self.ratio_nodes[ratio_lo])
        large_hi = np.searchsorted(self.large_nodes_um, large, side="left")
        below = large <= self.large_nodes_um[0]
        above = large > self.large_nodes_um[-1]
        inside = ~(below | above)
        output = np.empty_like(large, dtype=float)
        output[below] = (1.0 - q[below]) * self.efficiency[0, ratio_lo[below]] + q[below] * self.efficiency[
            0, ratio_hi[below]
        ]
        if np.any(inside):
            hi = large_hi[inside]
            lo = hi - 1
            p = (large[inside] - self.large_nodes_um[lo]) / (
                self.large_nodes_um[hi] - self.large_nodes_um[lo]
            )
            qi = q[inside]
            rlo = ratio_lo[inside]
            rhi = ratio_hi[inside]
            output[inside] = (
                (1.0 - p) * (1.0 - qi) * self.efficiency[lo, rlo]
                + p * (1.0 - qi) * self.efficiency[hi, rlo]
                + (1.0 - p) * qi * self.efficiency[lo, rhi]
                + p * qi * self.efficiency[hi, rhi]
            )
        if np.any(above):
            interpolated = (1.0 - q[above]) * self.efficiency[-1, ratio_lo[above]] + q[
                above
            ] * self.efficiency[-1, ratio_hi[above]]
            output[above] = np.minimum(interpolated, 1.0)
        return output

    def terminal_velocity_m_s(self, radius_um: np.ndarray) -> np.ndarray:
        radius_um = np.asarray(radius_um, dtype=float)
        grav_cm_s2 = 980.665
        rho_water_g_cm3 = 1.0
        lb4l_cm = 6.62e-06
        pb4l_hpa = 1013.25
        tb4l_k = 293.15
        visb4l = 0.0001818
        rho_air_g_cm3 = self.surface_density_kg_m3 * 0.001
        diameter_cm = 2.0 * radius_um * 0.0001
        temperature_c = self.surface_temperature_k - 273.15
        if temperature_c >= 0:
            viscosity = (1.718 + 0.0049 * temperature_c) * 0.0001
        else:
            viscosity = (1.718 + 0.0049 * temperature_c - 1.2e-05 * temperature_c**2) * 0.0001
        mean_free_path_cm = (
            lb4l_cm
            * (viscosity / visb4l)
            * (pb4l_hpa / self.surface_pressure_hpa)
            * math.sqrt(self.surface_temperature_k / tb4l_k)
        )
        gxdrow = grav_cm_s2 * (rho_water_g_cm3 - rho_air_g_cm3)
        slip = 1.0 + 2.51 * mean_free_path_cm / diameter_cm
        output = np.empty_like(radius_um)
        small = diameter_cm <= 0.0019
        output[small] = gxdrow / (18.0 * viscosity) * slip[small] * diameter_cm[small] ** 2 * 0.01
        if np.any(~small):
            coefficients = np.asarray(
                [-3.18657, 0.992696, -0.00153193, -0.000987059, -0.000578878, 8.55176e-05, -3.27815e-06]
            )
            diameter_large = diameter_cm[~small]
            nda = rho_air_g_cm3 * (4.0 * gxdrow) / (3.0 * viscosity**2) * diameter_large**3
            x1 = np.log(nda)
            x2 = x1**2
            x3 = x1 * x2
            y = (
                coefficients[0]
                + coefficients[1] * x1
                + coefficients[2] * x2
                + coefficients[3] * x3
                + coefficients[4] * x2**2
                + coefficients[5] * x3 * x2
                + coefficients[6] * x3**2
            )
            reynolds = slip[~small] * np.exp(y)
            output[~small] = viscosity * reynolds / (rho_air_g_cm3 * diameter_large) * 0.01
        return output

    def kernel_m3_s(self, diameter_1_um: np.ndarray, diameter_2_um: np.ndarray) -> np.ndarray:
        diameter_1_um, diameter_2_um = np.broadcast_arrays(diameter_1_um, diameter_2_um)
        radius_1_um = diameter_1_um / 2.0
        radius_2_um = diameter_2_um / 2.0
        efficiency = self.collision_efficiency(radius_1_um, radius_2_um)
        velocity_1 = self.terminal_velocity_m_s(radius_1_um)
        velocity_2 = self.terminal_velocity_m_s(radius_2_um)
        return (
            efficiency
            * math.pi
            * ((radius_1_um + radius_2_um) * 1e-06) ** 2
            * np.abs(velocity_1 - velocity_2)
        )


def pair_prediction(
    density_m3: np.ndarray,
    kernel_matrix: np.ndarray,
    fine_centers: np.ndarray,
    pair_edges: np.ndarray,
    *,
    exposure_volume_m3_s: float,
) -> np.ndarray:
    density_m3 = np.asarray(density_m3, dtype=float)
    if density_m3.ndim == 1:
        density_m3 = density_m3[None, :]
    size = len(pair_edges) - 1
    output = np.full((len(density_m3), size, size), np.nan)
    for small_bin in range(size):
        small = (fine_centers >= pair_edges[small_bin]) & (fine_centers < pair_edges[small_bin + 1])
        for large_bin in range(small_bin, size):
            large = (fine_centers >= pair_edges[large_bin]) & (fine_centers < pair_edges[large_bin + 1])
            diagonal_factor = 0.5 if small_bin == large_bin else 1.0
            output[:, small_bin, large_bin] = (
                exposure_volume_m3_s
                * diagonal_factor
                * np.einsum(
                    "bi,ij,bj->b",
                    density_m3[:, small],
                    kernel_matrix[np.ix_(small, large)],
                    density_m3[:, large],
                    optimize=True,
                )
            )
    return output


def predictive_standard_deviation(
    expected: np.ndarray, bootstrap_expected: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Lower-triangle cells intentionally have no samples: only unordered pairs
    # are represented. Avoid emitting a degrees-of-freedom warning for them.
    sample = np.asarray(bootstrap_expected).reshape(len(bootstrap_expected), -1)
    valid = np.isfinite(sample).sum(axis=0) >= 2
    flat_sd = np.full(sample.shape[1], np.nan)
    flat_sd[valid] = np.nanstd(sample[:, valid], axis=0, ddof=1)
    density_sd = flat_sd.reshape(np.shape(expected))
    poisson_sd = np.sqrt(np.maximum(expected, 0.0))
    total_sd = np.sqrt(np.maximum(expected, 0.0) + density_sd**2)
    return (density_sd, poisson_sd, total_sd)


def median_velocity_px_per_frame(frame: pd.DataFrame, start: int, end: int) -> tuple[float, int]:
    selected = frame.loc[
        frame["frame"].between(int(start), int(end), inclusive="both"), ["frame", "y"]
    ].copy()
    selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
    selected = (
        selected.loc[np.isfinite(selected["y"])]
        .sort_values("frame", kind="mergesort")
        .drop_duplicates("frame", keep="last")
    )
    if len(selected) < 2:
        return (math.nan, 0)
    frame_values = selected["frame"].to_numpy(np.float64)
    y_values = selected["y"].to_numpy(np.float64)
    frame_delta = np.diff(frame_values)
    y_delta = np.diff(y_values)
    valid = np.isfinite(frame_delta) & np.isfinite(y_delta) & (frame_delta > 0)
    interval_velocity = y_delta[valid] / frame_delta[valid]
    if not len(interval_velocity):
        return (math.nan, int(len(selected)))
    return (float(np.median(interval_velocity)), int(len(selected)))


def summary_metrics(frame: pd.DataFrame, expected: str, measured: str) -> dict[str, Any]:
    x = pd.to_numeric(frame[expected], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(frame[measured], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    error = y - x
    if not len(x):
        return {"rows": 0}
    pearson = stats.pearsonr(x, y) if len(x) >= 2 else None
    spearman = stats.spearmanr(x, y) if len(x) >= 2 else None
    return {
        "rows": int(len(x)),
        "median_signed_error": float(np.median(error)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "root_mean_squared_error": float(np.sqrt(np.mean(error**2))),
        "pearson_r": float(pearson.statistic) if pearson is not None else None,
        "pearson_p": float(pearson.pvalue) if pearson is not None else None,
        "spearman_rho": float(spearman.statistic) if spearman is not None else None,
        "spearman_p": float(spearman.pvalue) if spearman is not None else None,
    }
