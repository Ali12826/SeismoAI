"""
loader.py — Core loading and normalization functions for seismoai_io.
"""
from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Dict, Literal, Union

import numpy as np
import numpy.typing as npt

try:
    import segyio
    _SEGYIO_AVAILABLE = True
except ImportError:
    _SEGYIO_AVAILABLE = False


def _read_sgy_manual(filepath: Path, dtype: npt.DTypeLike) -> np.ndarray:
    """Manual little-endian SEG-Y reader for non-standard INOVA Hawk files."""
    file_size = filepath.stat().st_size
    with open(filepath, "rb") as f:
        text_hdr = f.read(3200).decode("ascii", errors="replace")
        f.read(400)
        trace_hdr = f.read(240)
        n_samples_hdr = struct.unpack("<h", trace_hdr[114:116])[0]

        n_samples = None
        for line in text_hdr.split("C"):
            if "SAMPLES" in line.upper() and "TRACE" in line.upper():
                for token in line.split():
                    if token.isdigit() and int(token) > 100:
                        n_samples = int(token)
                        break

        if not n_samples or n_samples <= 0:
            n_samples = n_samples_hdr if n_samples_hdr > 0 else 4001

        trace_size = 240 + n_samples * 4
        remainder = file_size - 3600
        n_traces = remainder // trace_size

        f.seek(3600)
        rows = []
        for _ in range(n_traces):
            f.seek(240, 1)
            raw = f.read(n_samples * 4)
            rows.append(np.frombuffer(raw, dtype="<f4").astype(dtype))

    return np.array(rows)


def load_sgy(
    filepath: Union[str, Path],
    *,
    ignore_geometry: bool = True,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """Load a single SEG-Y file. Returns ndarray shape (n_traces, n_samples).

    Tries segyio first; falls back to manual little-endian reader for
    non-standard INOVA Hawk files that segyio cannot parse.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {filepath}")

    if _SEGYIO_AVAILABLE:
        try:
            with segyio.open(str(filepath), "r", ignore_geometry=ignore_geometry) as f:
                return segyio.tools.collect(f.trace[:]).astype(dtype)
        except (RuntimeError, OSError) as exc:
            warnings.warn(
                f"segyio could not open '{filepath.name}' ({exc}). "
                "Falling back to manual binary reader.",
                stacklevel=2,
            )

    try:
        return _read_sgy_manual(filepath, dtype)
    except Exception as exc:
        raise ValueError(
            f"Could not read SEG-Y file '{filepath}': {exc}"
        ) from exc


def load_sgy_folder(
    folder: Union[str, Path],
    *,
    extensions: tuple[str, ...] = (".sgy", ".segy"),
    ignore_geometry: bool = True,
    dtype: npt.DTypeLike = np.float32,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Load every SEG-Y file in a directory. Returns dict[stem -> ndarray]."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    extensions_lower = {ext.lower() for ext in extensions}
    sgy_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions_lower
    )

    if not sgy_files:
        warnings.warn(f"No SEG-Y files found in '{folder}'.", stacklevel=2)
        return {}

    dataset: Dict[str, np.ndarray] = {}
    for path in sgy_files:
        try:
            traces = load_sgy(path, ignore_geometry=ignore_geometry, dtype=dtype)
            dataset[path.stem] = traces
            if verbose:
                print(f"  ✓ {path.name:50s}  shape={traces.shape}")
        except (ValueError, OSError) as exc:
            warnings.warn(f"Skipping '{path.name}': {exc}", stacklevel=2)

    if verbose:
        print(f"\nLoaded {len(dataset)} / {len(sgy_files)} files.")
    return dataset


def normalize_traces(
    traces: np.ndarray,
    method: Literal["minmax", "zscore", "maxabs"] = "zscore",
    *,
    per_trace: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """Normalize seismic trace amplitudes.

    method: 'zscore' (default), 'minmax', or 'maxabs'
    """
    if traces.ndim != 2:
        raise ValueError(f"traces must be 2-D, got shape {traces.shape}")
    valid = {"minmax", "zscore", "maxabs"}
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'")

    out = traces.astype(np.float32, copy=True)
    axis: int | None = 1 if per_trace else None

    if method == "zscore":
        mu = out.mean(axis=axis, keepdims=True)
        sigma = out.std(axis=axis, keepdims=True)
        out = (out - mu) / (sigma + eps)
    elif method == "minmax":
        t_min = out.min(axis=axis, keepdims=True)
        t_max = out.max(axis=axis, keepdims=True)
        out = (out - t_min) / (t_max - t_min + eps)
    elif method == "maxabs":
        t_maxabs = np.abs(out).max(axis=axis, keepdims=True)
        out = out / (t_maxabs + eps)

    return out
