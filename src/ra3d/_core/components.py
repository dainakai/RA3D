"""Numerical kernels derived from same_component_features.py."""

from __future__ import annotations

import math
import numpy as np
from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentHit:
    label: int
    distance_px: float
    x: float
    y: float
    observed: bool


def point_hit(
    labels: np.ndarray,
    pt: Any,
    x0: float,
    y0: float,
    *,
    observed: bool,
    search_radius_px: int = 5,
) -> ComponentHit | None:
    if pt is None:
        return None
    x = float(pt.x) - float(x0)
    y = float(pt.y) - float(y0)
    h, w = labels.shape
    if not (-search_radius_px <= x < w + search_radius_px and -search_radius_px <= y < h + search_radius_px):
        return None
    xi = int(round(x))
    yi = int(round(y))
    if 0 <= xi < w and 0 <= yi < h and labels[yi, xi] > 0:
        return ComponentHit(int(labels[yi, xi]), 0.0, x, y, observed)
    best_label = 0
    best_dist = math.inf
    xlo = max(0, xi - search_radius_px)
    xhi = min(w - 1, xi + search_radius_px)
    ylo = max(0, yi - search_radius_px)
    yhi = min(h - 1, yi + search_radius_px)
    for yy in range(ylo, yhi + 1):
        for xx in range(xlo, xhi + 1):
            lab = int(labels[yy, xx])
            if lab <= 0:
                continue
            dist = math.hypot(float(xx) - x, float(yy) - y)
            if dist < best_dist:
                best_dist = dist
                best_label = lab
    if best_label <= 0:
        return None
    return ComponentHit(best_label, float(best_dist), x, y, observed)


def component_area(labels: np.ndarray, label: int) -> int:
    if label <= 0:
        return 0
    return int(np.sum(labels == int(label)))
