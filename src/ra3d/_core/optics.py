"""Numerical kernels derived from depth_model_estimate.py."""

from __future__ import annotations

import numpy as np
import math
import torch
from pathlib import Path
from PIL import Image

_DISTORTION_MAP_CACHE = {}


def transfer_sqrt_arr(datlen: int, wavlen: float, dx: float, device: torch.device) -> torch.Tensor:
    axis = torch.arange(datlen, device=device, dtype=torch.float32) - (datlen / 2)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    scale = float(wavlen) / float(datlen) / float(dx)
    return 1.0 - (xx * scale) ** 2 - (yy * scale) ** 2


def transfer(z_um: float, datlen: int, wavlen: float, sqrt_part: torch.Tensor) -> torch.Tensor:
    phase = (2.0 * math.pi * float(z_um) / float(wavlen)) * torch.sqrt(sqrt_part)
    return torch.exp(1j * phase).to(torch.complex64)


def load_gray2float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def distortion_maps(n: int, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeff_key = np.ascontiguousarray(coeffs, dtype=np.float64).tobytes()
    cache_key = (n, coeff_key)
    cached = _DISTORTION_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    i, j = np.indices((n, n), dtype=np.float64)
    i += 1.0
    j += 1.0
    ref_x = np.rint(
        coeffs[0] + coeffs[1] * j + coeffs[2] * i + coeffs[3] * j * j + coeffs[4] * i * j + coeffs[5] * i * i
    ).astype(np.int64)
    ref_y = np.rint(
        coeffs[6]
        + coeffs[7] * j
        + coeffs[8] * i
        + coeffs[9] * j * j
        + coeffs[10] * i * j
        + coeffs[11] * i * i
    ).astype(np.int64)
    valid = (ref_x >= 1) & (ref_x <= n) & (ref_y >= 1) & (ref_y <= n)
    maps = (ref_y - 1, ref_x - 1, valid)
    _DISTORTION_MAP_CACHE[cache_key] = maps
    return maps


def quadratic_distortion_correction(img: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    n = img.shape[0]
    ref_y, ref_x, valid = distortion_maps(n, coeffs)
    out = np.full((n, n), float(img.mean()), dtype=np.float32)
    out[valid] = img[ref_y[valid], ref_x[valid]]
    return out


def fft2(x: torch.Tensor, backend: str) -> torch.Tensor:
    if backend != "torch":
        raise ValueError(f"Unsupported fft backend: {backend}")
    return torch.fft.fft2(x)


def ifft2(x: torch.Tensor, backend: str) -> torch.Tensor:
    if backend != "torch":
        raise ValueError(f"Unsupported fft backend: {backend}")
    return torch.fft.ifft2(x)


def phase_retrieval_holo(
    holo1: torch.Tensor,
    holo2: torch.Tensor,
    d_pr: torch.Tensor,
    d_pr_inv: torch.Tensor,
    priter: int,
    fft_backend: str,
) -> torch.Tensor:
    sqrt_i1 = torch.sqrt(holo1)
    sqrt_i2 = torch.sqrt(holo2)
    light1 = sqrt_i1.to(torch.complex64)
    d_pr_unshifted = torch.fft.ifftshift(d_pr)
    d_pr_inv_unshifted = torch.fft.ifftshift(d_pr_inv)
    for _ in range(priter):
        light2 = ifft2(fft2(light1, fft_backend) * d_pr_unshifted, fft_backend)
        light2 = sqrt_i2 * torch.exp(1j * torch.angle(light2))
        light1 = ifft2(fft2(light2, fft_backend) * d_pr_inv_unshifted, fft_backend)
        light1 = sqrt_i1 * torch.exp(1j * torch.angle(light1))
    return light1
