"""Numerical kernels derived from proc.py."""

from __future__ import annotations

import torch


def mean_pad_complex(field: torch.Tensor, outlen: int) -> torch.Tensor:
    inlen = int(field.shape[0])
    out = torch.empty((outlen, outlen), device=field.device, dtype=torch.complex64)
    out.fill_(field.mean())
    offset = (outlen - inlen) // 2
    out[offset : offset + inlen, offset : offset + inlen] = field
    return out
