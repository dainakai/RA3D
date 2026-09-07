"""Portable table I/O, strict validation, and atomic output."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path, low_memory=False, float_precision="round_trip")


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if path.suffix == ".parquet":
        frame.to_parquet(partial, index=False, compression="zstd")
    else:
        frame.to_csv(partial, index=False, compression="gzip" if path.suffix == ".gz" else None)
    partial.replace(path)
    return path


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(value: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    partial.replace(path)
    return path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns, *, name="table", finite=False) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    if finite:
        values = frame[list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite values in {list(columns)}")


def require_integer(frame: pd.DataFrame, columns, *, name="table") -> None:
    require_columns(frame, columns, name=name, finite=True)
    for column in columns:
        values = frame[column].to_numpy(float)
        if not np.equal(values, np.rint(values)).all():
            raise ValueError(f"{name}.{column} must contain integers")


def require_unique(frame: pd.DataFrame, columns, *, name="table") -> None:
    require_columns(frame, columns, name=name)
    if frame.duplicated(list(columns)).any():
        raise ValueError(f"{name} contains duplicate keys {list(columns)}")


def contained_path(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Expected a relative path contained in the selected directory")
    return path
