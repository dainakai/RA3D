"""Explicit, portable image-pair manifest for collision features and visual review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ._core.optics import (
    load_gray2float,
    quadratic_distortion_correction,
    transfer_sqrt_arr,
    transfer,
    phase_retrieval_holo,
)
from ._core.padding import mean_pad_complex
from .config import Units
from .io import read_table, require_columns, require_integer, require_unique


@dataclass
class Acquisition:
    root: Path
    frames: pd.DataFrame
    units: Units
    wavelength_um: float = 0.6328
    phase_distance_um: float = 56300.0
    reconstruction_start_um: float = 61650.0
    padding_px: int = 2048
    phase_iterations: int = 12
    coefficients: np.ndarray | None = None
    mode: str = "dual_phase_retrieval"

    @classmethod
    def load(cls, path: str | Path) -> Acquisition:
        path = Path(path).resolve()
        config = yaml.safe_load(path.read_text())
        if config.get("schema_version") != 1:
            raise ValueError("Acquisition schema_version must be 1; see docs/custom-data.md")
        units = Units(**config["units"])
        frame_path = path.parent / config["frames"]
        frames = read_table(frame_path)
        require_columns(frames, ["frame", "primary"], name="frames")
        require_integer(frames, ["frame"], name="frames")
        require_unique(frames, ["frame"], name="frames")
        if frames.empty:
            raise ValueError("Acquisition frame manifest is empty")
        optics = config.get("optics", {})
        coefficients = optics.pop("calibration_coefficients", None)
        if isinstance(coefficients, str):
            coefficients = np.loadtxt(path.parent / coefficients)
            if coefficients.ndim == 2:
                coefficients = coefficients[:, 0]
        result = cls(
            path.parent,
            frames.set_index("frame"),
            units,
            coefficients=None if coefficients is None else np.asarray(coefficients, dtype=float),
            **optics,
        )
        result.validate()
        return result

    def validate(self, *, check_files: bool = True) -> dict:
        if self.mode not in {"dual_phase_retrieval", "single_gabor"}:
            raise ValueError("mode must be dual_phase_retrieval or single_gabor")
        if self.mode == "dual_phase_retrieval":
            require_columns(self.frames, ["secondary"], name="frames")
        if self.padding_px < self.units.image_size_px or (self.padding_px - self.units.image_size_px) % 2:
            raise ValueError("padding_px must permit a centered integer crop of image_size_px")
        if self.phase_iterations < 1 or self.wavelength_um <= 0:
            raise ValueError("phase_iterations and wavelength_um must be positive")
        if self.coefficients is not None and (
            self.coefficients.shape != (12,) or not np.isfinite(self.coefficients).all()
        ):
            raise ValueError("Calibration requires exactly 12 finite coefficients")
        if check_files:
            for col in ["primary"] + (["secondary"] if self.mode == "dual_phase_retrieval" else []):
                for value in self.frames[col]:
                    if not (self.root / str(value)).is_file():
                        raise FileNotFoundError(f"Missing {col} image: {value}")
        return {"frames": len(self.frames), "mode": self.mode, "units": vars(self.units)}


class Reconstructor:
    """Reuse a phase-retrieved FFT for all candidate depths in one frame."""

    def __init__(self, acquisition: Acquisition, device: str = "cpu"):
        self.acquisition = acquisition
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
        a = acquisition
        self.sqrt = transfer_sqrt_arr(a.padding_px, a.wavelength_um, a.units.pixel_pitch_um, self.device)
        base = transfer_sqrt_arr(a.units.image_size_px, a.wavelength_um, a.units.pixel_pitch_um, self.device)
        self.forward = transfer(a.phase_distance_um, a.units.image_size_px, a.wavelength_um, base)
        self.inverse = transfer(-a.phase_distance_um, a.units.image_size_px, a.wavelength_um, base)
        self._frame = None
        self._fft = None

    def frame_fft(self, frame: int):
        if self._frame == frame:
            return self._fft
        a = self.acquisition
        if frame not in a.frames.index:
            raise KeyError(f"Frame {frame} is absent from the acquisition manifest")
        row = a.frames.loc[frame]
        primary = load_gray2float(a.root / str(row.primary))
        if primary.shape != (a.units.image_size_px, a.units.image_size_px):
            raise ValueError(f"Unexpected image dimensions for frame {frame}: {primary.shape}")
        primary = torch.from_numpy(primary).to(self.device)
        if a.mode == "dual_phase_retrieval":
            secondary = load_gray2float(a.root / str(row.secondary))
            if tuple(secondary.shape) != tuple(primary.shape):
                raise ValueError("Primary and secondary image dimensions differ")
            if a.coefficients is not None:
                secondary = quadratic_distortion_correction(secondary, a.coefficients)
            wave = phase_retrieval_holo(
                primary,
                torch.from_numpy(secondary).to(self.device),
                self.forward,
                self.inverse,
                a.phase_iterations,
                "torch",
            )
        else:
            wave = torch.sqrt(primary).to(torch.complex64)
        self._fft = torch.fft.fft2(mean_pad_complex(wave, a.padding_px))
        self._frame = frame
        return self._fft

    def intensity(self, frame: int, z_slice: float) -> np.ndarray:
        a = self.acquisition
        spectrum = self.frame_fft(frame)
        distance = a.reconstruction_start_um + (float(z_slice) - 1) * a.units.slice_spacing_um
        tf = torch.fft.ifftshift(transfer(-distance, a.padding_px, a.wavelength_um, self.sqrt))
        field = torch.fft.ifft2(spectrum * tf)
        intensity = (field.real.square() + field.imag.square()).clamp_(0, 1)
        offset = (a.padding_px - a.units.image_size_px) // 2
        return (
            intensity[offset : offset + a.units.image_size_px, offset : offset + a.units.image_size_px]
            .cpu()
            .numpy()
        )
