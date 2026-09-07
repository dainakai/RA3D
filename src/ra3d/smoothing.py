"""Python implementation of polynomial Z-GMM and reflected Savitzky–Golay smoothing.

The model and deterministic PRESS selection follow zsmooth_new.jl. Batched sufficient
statistics replace the per-trajectory Julia solves without changing the objective.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math
import time

import numpy as np
from scipy.signal import savgol_coeffs

from .io import require_columns, require_integer, require_unique


@dataclass(frozen=True)
class SmoothingOptions:
    selection: str = "auto"
    degree: int = 2
    alpha: float = 0.0
    lambda0: float = 1e-6
    lambda_other: float = 1e-4
    em_max_iter: int = 100
    em_tol: float = 5e-5
    pi_init: float = 0.05
    outlier_variance_multiplier: float = 50.0
    tune_min_points: int = 8
    tune_max_trajectories: int = 512
    tune_relative_tolerance: float = 0.01
    sg_window: int = 7
    sg_order: int = 2

    def validate(self):
        if self.selection not in {"auto", "fixed"}:
            raise ValueError("Smoothing selection must be auto or fixed")
        if self.degree not in (1, 2) or min(self.lambda0, self.lambda_other) <= 0 or self.alpha < 0:
            raise ValueError("Use degree 1 or 2, positive ridge strengths, and nonnegative alpha")
        if not 0 < self.pi_init < 1 or self.outlier_variance_multiplier <= 1:
            raise ValueError("Invalid mixture initialization")
        if min(self.em_max_iter, self.tune_max_trajectories, self.tune_min_points) < 1 or self.em_tol <= 0:
            raise ValueError("Iteration, sample, and convergence settings must be positive")
        if self.sg_window < 3 or self.sg_window % 2 != 1 or self.sg_order < 1:
            raise ValueError("Savitzky–Golay requires an odd window >=3 and order >=1")


def savgol_reflected(values, window=7, order=2, derivative=0):
    """Use odd reflection at both endpoints, including the derivative filter."""
    y = np.asarray(values, dtype=np.float64)
    w = min(window, len(y))
    w -= int(w % 2 == 0)
    p = min(order, w - 2)
    if w < 3 or p < derivative:
        return y.copy() if derivative == 0 else np.zeros(len(y))
    half = w // 2
    padded = np.concatenate((2 * y[0] - y[1 : half + 1][::-1], y, 2 * y[-1] - y[-half - 1 : -1][::-1]))
    return np.convolve(padded, savgol_coeffs(w, p, deriv=derivative, delta=1.0, use="conv"), mode="valid")


class PolynomialTracks:
    def __init__(self, tracks, options):
        self.table = tracks.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)
        self.ids, self.group = np.unique(self.table.track_id.to_numpy(), return_inverse=True)
        self.n = len(self.ids)
        self.lengths = np.bincount(self.group, minlength=self.n)
        self.starts = np.r_[0, np.cumsum(self.lengths)[:-1]]
        frames = self.table.frame.to_numpy(float)
        spans = frames[self.starts + self.lengths - 1] - frames[self.starts]
        self.scale = float(max(1.0, np.max(spans)))
        self.tau = (frames - frames[self.starts][self.group]) / self.scale
        self.phi = np.stack([self.tau**k for k in range(options.degree + 1)], axis=1)
        self.z = self.table.z.to_numpy(float)
        self.precision = np.diag(
            [options.lambda0]
            + [options.lambda_other + options.alpha * k * k for k in range(1, options.degree + 1)]
        )
        self.degree = options.degree

    def systems(self, weight):
        p = self.degree + 1
        matrices = np.broadcast_to(self.precision, (self.n, p, p)).copy()
        rhs = np.empty((self.n, p), dtype=np.float64)
        for j in range(p):
            rhs[:, j] = np.bincount(self.group, weights=weight * self.phi[:, j] * self.z, minlength=self.n)
            for k in range(j, p):
                value = np.bincount(
                    self.group, weights=weight * self.phi[:, j] * self.phi[:, k], minlength=self.n
                )
                matrices[:, j, k] += value
                if k != j:
                    matrices[:, k, j] += value
        return matrices, rhs

    def solve(self, weight, prior):
        matrices, rhs = self.systems(weight)
        rhs += self.precision @ prior
        factor = np.linalg.cholesky(matrices)
        intermediate = np.linalg.solve(factor, rhs[..., None])
        return np.linalg.solve(factor.transpose(0, 2, 1), intermediate)[..., 0]

    def prediction(self, beta):
        return np.sum(self.phi * beta[self.group], axis=1)


def _posterior(residual_sq, pi_out, var0, var1):
    lp0 = np.log(1 - pi_out + 1e-12) - 0.5 * np.log(2 * np.pi * var0) - 0.5 * residual_sq / var0
    lp1 = np.log(pi_out + 1e-12) - 0.5 * np.log(2 * np.pi * var1) - 0.5 * residual_sq / var1
    return np.clip(np.exp(lp1 - np.logaddexp(lp0, lp1)), 0.0, 1.0)


def fit_z_gmm(tracks, options=SmoothingOptions(), progress=None):
    options.validate()
    p = PolynomialTracks(tracks, options)
    beta = p.solve(np.ones(len(p.z)), np.zeros(options.degree + 1))
    prior = beta.mean(axis=0)
    beta = p.solve(np.ones(len(p.z)), prior)
    var0 = max(1e-9, float(np.mean((p.z - p.prediction(beta)) ** 2)))
    var1 = max(var0 * 1.000001, var0 * options.outlier_variance_multiplier)
    pi_out = options.pi_init
    converged = False
    for iteration in range(1, options.em_max_iter + 1):
        old = np.array([pi_out, var0, var1, np.linalg.norm(beta, axis=1).mean()])
        posterior = _posterior((p.z - p.prediction(beta)) ** 2, pi_out, var0, var1)
        pi_out = float(np.clip(posterior.mean(), 1e-9, 1 - 1e-9))
        weight = (1 - posterior) / var0 + posterior / var1
        beta = p.solve(weight, prior)
        residual_sq = (p.z - p.prediction(beta)) ** 2
        var0 = max(1e-9, float(np.sum((1 - posterior) * residual_sq) / max(np.sum(1 - posterior), 1e-9)))
        var1 = max(var0 * 1.000001, float(np.sum(posterior * residual_sq) / max(np.sum(posterior), 1e-9)))
        new = np.array([pi_out, var0, var1, np.linalg.norm(beta, axis=1).mean()])
        difference = float(np.max(np.abs(new - old) / (np.abs(old) + 1e-8)))
        if progress and (iteration == 1 or iteration % 10 == 0):
            progress({"stage": "z_gmm", "iteration": iteration, "relative_change": difference})
        if difference <= options.em_tol:
            converged = True
            break
    posterior = _posterior((p.z - p.prediction(beta)) ** 2, pi_out, var0, var1)
    state = {
        "pi_out": pi_out,
        "sigma2_normal": var0,
        "sigma2_outlier": var1,
        "time_scale": p.scale,
        "prior_mean": prior.tolist(),
        "iterations": iteration,
        "final_diff": difference,
        "converged": converged,
    }
    return p, beta, posterior, state


def press_objective(p, beta, posterior, state):
    weight = (1 - posterior) / state["sigma2_normal"] + posterior / state["sigma2_outlier"]
    matrices, _ = p.systems(weight)
    inverse = np.linalg.inv(matrices)
    leverage = weight * np.einsum("ij,ijk,ik->i", p.phi, inverse[p.group], p.phi)
    errors = np.abs((p.z - p.prediction(beta)) / np.maximum(1 - leverage, 1e-6))
    score_weight = np.clip(1 - posterior, 1e-6, 1.0)
    order = np.argsort(errors, kind="stable")
    index = min(
        int(np.searchsorted(np.cumsum(score_weight[order]), 0.99 * score_weight.sum())), len(order) - 1
    )
    p99 = float(errors[order[index]])
    return float(np.sqrt(np.sum(np.minimum(errors, p99) ** 2 * score_weight) / score_weight.sum()))


def select_options(tracks, options=SmoothingOptions(), progress=None):
    options.validate()
    if options.selection == "fixed":
        return options, {"selection": "fixed", "selected": asdict(options), "candidates": []}
    lengths = tracks.groupby("track_id").size().rename("length").reset_index()
    eligible = lengths.loc[lengths.length >= options.tune_min_points].sort_values(["length", "track_id"])
    if eligible.empty:
        raise ValueError(
            f"Automatic smoothing needs trajectories with at least {options.tune_min_points} points; use fixed settings for shorter inputs"
        )
    if len(eligible) > options.tune_max_trajectories:
        idx = np.unique(np.rint(np.linspace(1, len(eligible), options.tune_max_trajectories)).astype(int) - 1)
        eligible = eligible.iloc[idx]
    sample = tracks.loc[tracks.track_id.isin(eligible.track_id)]
    results = []
    for degree in [1, 2]:
        for lam in [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]:
            for alpha in [0.0, 1e-4, 1e-2] if degree == 2 else [0.0]:
                candidate = replace(options, degree=degree, alpha=alpha, lambda_other=lam)
                start = time.monotonic()
                p, beta, posterior, state = fit_z_gmm(sample, candidate)
                entry = {
                    "degree": degree,
                    "alpha": alpha,
                    "lambda_other": lam,
                    "objective": press_objective(p, beta, posterior, state),
                    "em_converged": state["converged"],
                    "iterations": state["iterations"],
                    "seconds": time.monotonic() - start,
                }
                results.append(entry)
                if progress:
                    progress(
                        {"stage": "smoothing_selection", "candidate": len(results), "total": 24, **entry}
                    )
    converged = [r for r in results if r["em_converged"] and math.isfinite(r["objective"])]
    if not converged:
        raise ValueError(
            "No smoothing selection candidate converged; increase em_max_iter or choose fixed settings"
        )
    best = min(r["objective"] for r in converged)
    near = [r for r in converged if r["objective"] <= best * (1 + options.tune_relative_tolerance)]
    selected = min(near, key=lambda r: (r["degree"], r["objective"], -r["lambda_other"], -r["alpha"]))
    chosen = replace(
        options, degree=selected["degree"], alpha=selected["alpha"], lambda_other=selected["lambda_other"]
    )
    return chosen, {
        "selection": "auto",
        "sampled_trajectories": len(eligible),
        "sampled_points": len(sample),
        "selected": selected,
        "candidates": results,
        "minimum_objective": best,
    }


def smooth_tracks(tracks, options=SmoothingOptions(), progress=None):
    table = tracks.rename(columns={"xc": "x", "yc": "y", "slice": "z"})
    require_columns(table, ["track_id", "frame", "x", "y", "z", "diameter_px"], finite=True)
    require_integer(table, ["track_id", "frame"])
    require_unique(table, ["track_id", "frame"])
    if table.empty:
        raise ValueError("No trajectories to smooth")
    selected, tuning = select_options(table, options, progress)
    p, beta, posterior, state = fit_z_gmm(table, selected, progress)
    result = p.table.copy()
    result["z"] = p.prediction(beta)
    result["vz"] = (
        np.sum(
            np.stack([k * p.tau ** (k - 1) * beta[p.group, k] for k in range(1, selected.degree + 1)]), axis=0
        )
        / p.scale
    )
    result["postOutProb"] = posterior
    for axis in "xy":
        smooth = []
        velocity = []
        values = p.table[axis].to_numpy(float)
        for start, length in zip(p.starts, p.lengths):
            original = values[start : start + length]
            smooth.append(savgol_reflected(original, selected.sg_window, selected.sg_order))
            velocity.append(savgol_reflected(original, selected.sg_window, selected.sg_order, 1))
        result[axis] = np.concatenate(smooth)
        result["v" + axis] = np.concatenate(velocity)
    return result, {"options": asdict(selected), "tuning": tuning, "fit": state}
