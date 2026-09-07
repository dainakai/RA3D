"""Render sequential opaque-particle transmission on a padded optical field.

This is a Python/PyTorch implementation of the paper's angular-spectrum image
simulator. Original released images remain the reference for byte-level replay.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..io import read_table, write_json, require_columns


@dataclass(frozen=True)
class RenderOptions:
    image_size_px: int = 1024
    propagation_size_px: int = 1536
    pixel_pitch_um: float = 10.0
    wavelength_um: float = 0.6328
    background_gray: float = 77.5
    primary_front_um: float = 74900.0
    secondary_front_um: float = 96350.0


class HologramSimulator:
    def __init__(self, options=RenderOptions(), device="cpu"):
        self.options = options
        self.device = torch.device(device)
        n = options.propagation_size_px
        if n < options.image_size_px or (n - options.image_size_px) % 2:
            raise ValueError("Propagation size must permit a centered integer crop")
        if min(options.pixel_pitch_um, options.wavelength_um, options.background_gray) <= 0:
            raise ValueError("Optical scale and background must be positive")
        if options.primary_front_um > options.secondary_front_um:
            raise ValueError("The primary camera must be the near plane")
        # Match the Julia frequency grid: evaluate coordinates in Float64, then store Float32.
        axis = np.arange(n, dtype=np.float64) - n / 2
        frequency = (
            axis * float(np.float32(options.wavelength_um)) / n / float(np.float32(options.pixel_pitch_um))
        )
        part = (1.0 - frequency[:, None] ** 2 - frequency[None, :] ** 2).astype(np.float32)
        self.sqrt = torch.sqrt(torch.fft.ifftshift(torch.from_numpy(part).to(self.device)))
        self.offset = (n - options.image_size_px) // 2

    def propagate(self, wave, distance):
        distance = np.float32(distance)
        if abs(distance) < 1e-6:
            return wave.clone()
        scalar = np.float32(np.float32(2 * np.pi) * distance / np.float32(self.options.wavelength_um))
        phase = float(scalar) * self.sqrt
        transfer = torch.complex(torch.cos(phase), torch.sin(phase))
        return torch.fft.ifft2(torch.fft.fft2(wave) * transfer)

    def disk(self, wave, x, y, diameter):
        opt = self.options
        x = np.float32(x) + np.float32(self.offset)
        y = np.float32(y) + np.float32(self.offset)
        radius = max(np.float32(0.5), np.float32(diameter) / (np.float32(2) * np.float32(opt.pixel_pitch_um)))
        margin = int(np.ceil(radius)) + 2
        n = wave.shape[0]
        x0 = max(0, int(np.floor(x)) - margin)
        x1 = min(n, int(np.floor(x)) + margin + 1)
        y0 = max(0, int(np.floor(y)) - margin)
        y1 = min(n, int(np.floor(y)) + margin + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xx = torch.arange(x0, x1, device=self.device, dtype=torch.float32) - float(x)
        yy = torch.arange(y0, y1, device=self.device, dtype=torch.float32) - float(y)
        mask = yy[:, None].square() + xx[None, :].square() <= float(np.float32(radius * radius))
        wave[y0:y1, x0:x1].masked_fill_(mask, 0.0)

    def intensity(self, wave):
        opt = self.options
        i = self.offset
        n = opt.image_size_px
        crop = wave[i : i + n, i : i + n]
        return (
            (float(np.float32(opt.background_gray)) * (crop.real.square() + crop.imag.square()))
            .clamp(0.0, 255.0)
            .cpu()
            .numpy()
        )

    def render(self, particles):
        require_columns(particles, ["x_px", "y_px", "z_um", "diameter_um"], finite=True)
        wave = torch.ones((self.options.propagation_size_px,) * 2, dtype=torch.complex64, device=self.device)
        current_z = np.float32(0.0)
        for index, row in enumerate(
            particles.sort_values("z_um", ascending=False, kind="stable").itertuples()
        ):
            z = np.float32(row.z_um)
            if index:
                wave = self.propagate(wave, np.float32(current_z - z))
            current_z = z
            self.disk(wave, row.x_px, row.y_px, row.diameter_um)
        near = self.propagate(wave, np.float32(np.float32(self.options.primary_front_um) + current_z))
        far = self.propagate(
            near,
            np.float32(
                np.float32(self.options.secondary_front_um) - np.float32(self.options.primary_front_um)
            ),
        )
        return self.intensity(near), self.intensity(far)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", required=True, help="Simulator particles CSV/Parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.output)
    particles = read_table(args.particles)
    require_columns(particles, ["frame"])
    particles = particles.loc[particles.frame.ge(args.start_frame)]
    if args.end_frame is not None:
        particles = particles.loc[particles.frame.lt(args.end_frame)]
    engine = HologramSimulator(device=args.device)
    done = 0
    for frame, group in particles.groupby("frame", sort=True):
        paths = [root / role / f"{int(frame) + 1:06d}.png" for role in ["primary", "secondary"]]
        if args.resume and all(path.is_file() for path in paths):
            continue
        images = engine.render(group)
        for path, array in zip(paths, images, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".partial")
            Image.fromarray(np.rint(array).astype(np.uint8)).save(temp, format="PNG")
            temp.replace(path)
        done += 1
        print(f"Rendered frame {int(frame)}", flush=True)
    write_json(
        {
            "options": asdict(engine.options),
            "rendered_frames": done,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "generator": "sequential_transmission_angular_spectrum_python",
        },
        root / "render.json",
    )


if __name__ == "__main__":
    main()
