"""Shared physical parameters and formulas for collision-benchmark generation."""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SynthParams:
    seed: int = 260734
    frames: int = 300
    fps: float = 4000.0
    datlen: int = 1024
    dx_um: float = 10.0
    wavelength_um: float = 0.6328
    dz_um: float = 100.0
    slices: int = 1024
    baseline_gray: float = 77.5
    target_particles: int = 200
    min_diameter_um: float = 25.0
    max_diameter_um: float = 300.0
    camera1_id: str = "C001"
    camera2_id: str = "C002"
    scene: str = "H001S0011"
    calibration_bundle: str = "C000H001S0010"
    c002_front_distance_um: float = 74900.0
    c001_front_distance_um: float = 96350.0

    @property
    def width_um(self) -> float:
        return self.datlen * self.dx_um

    @property
    def height_um(self) -> float:
        return self.datlen * self.dx_um

    @property
    def depth_um(self) -> float:
        return self.slices * self.dz_um

    @property
    def volume_cm3(self) -> float:
        return self.width_um * self.height_um * self.depth_um * 1e-12

    @property
    def target_number_density_cm3(self) -> float:
        return self.target_particles / self.volume_cm3


def terminal_velocity_beard_1976_um_s(
    diameter_um: float, visco_g_cm_s: float = 0.0001818, pressure_mb: float = 1013.25, kelvin: float = 293.15
) -> float:
    """Return the Beard 1976 terminal velocity in micrometres per second."""
    dp = diameter_um
    eta = visco_g_cm_s * 0.1
    gravity = 9.817
    d0 = dp * 1e-06
    l0 = 6.62e-06 * 0.01
    p0 = 1013.25
    eta0 = 0.0001818 * 0.1
    t0 = 293.15
    rho_w = 1000.0
    rho_f = 1.225
    delta_rho = rho_w - rho_f
    if 0.5 <= dp < 19.0:
        c_l = delta_rho * gravity / (18.0 * eta)
        l_val = l0 * (eta / eta0) * (p0 / pressure_mb) * (kelvin / t0) ** 0.5
        c_ac = 1.0 + 2.51 * l_val / d0
        velocity_m_s = c_l * c_ac * d0**2
    elif 19.0 <= dp < 1070.0:
        l_val = l0 * (eta / eta0) * (p0 / pressure_mb) * (kelvin / t0) ** 0.5
        c_ac = 1.0 + 2.51 * l_val / d0
        c2 = 4.0 * rho_f * delta_rho * gravity / (3.0 * eta**2)
        x_val = math.log(c2 * d0**3)
        y_val = (
            -3.18657
            + 0.992696 * x_val
            - 0.00153193 * x_val**2
            - 0.000987059 * x_val**3
            - 0.000578878 * x_val**4
            + 8.55176e-05 * x_val**5
            - 3.27815e-06 * x_val**6
        )
        n_re = c_ac * math.exp(y_val)
        velocity_m_s = eta * n_re / (rho_f * d0)
    else:
        raise ValueError(f"diameter outside Beard 1976 range: {diameter_um}")
    return velocity_m_s * 1000000.0


def describe_array(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(np.max(values)),
    }
