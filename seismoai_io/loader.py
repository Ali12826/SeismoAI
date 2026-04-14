"""
loader.py — Core loading and normalization functions for seismoai_io.

All functions return standard NumPy arrays so downstream modules (viz, qc,
model, xai) can consume them without additional conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Union

import numpy as np

try:
    import segyio
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "segyio is required by seismoai_io. Install it with: pip install segyio"
    ) from exc


# ---------------------------------------------------------------------------
# Function 1 — load a single SGY file
# ---------------------------------------------------------------------------


def load_sgy(
    filepath: Union[str, Path],
    *,
    ignore_geometry: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load a single SEG-Y file and return its traces as a 2-D NumPy array.

    The file is opened in read-only mode, so the original data on disk is
    never modified.  Both big-endian (standard SEG-Y Rev 1) and
    little-endian files (e.g. INOVA Hawk) are handled automatically by
    *segyio*.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Absolute or relative path to the ``.sgy`` / ``.segy`` file.
    ignore_geometry : bool, optional
        When *True* (default) segyio does not try to infer inline / crossline
        geometry, which is required for 2-D shot-gather files that have no
        such sorting.  Set to *False* only for 3-D post-stack data with a
        valid geometry header.
    dtype : numpy.dtype, optional
        Output array dtype.  Defaults to ``np.float32``, which matches the
        on-disk IEEE-754 format used by most modern recording systems.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_traces, n_samples)``.  Each row is one recorded trace
        (receiver channel); each column is one time sample.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file cannot be opened as a valid SEG-Y stream.

    Examples
    --------
    >>> traces = load_sgy("data/record_027.sgy")
    >>> traces.shape
    (167, 4001)
    >>> traces.dtype
    dtype('float32')
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {filepath}")

    try:
        with segyio.open(str(filepath), "r", ignore_geometry=ignore_geometry) as f:
            traces = segyio.tools.collect(f.trace[:]).astype(dtype)
    except Exception as exc:
        # segyio raises RuntimeError for malformed files; wrap with context.
        raise ValueError(f"Could not read SEG-Y file '{filepath}': {exc}") from exc

    return traces  # shape: (n_traces, n_samples)


# ---------------------------------------------------------------------------
# Function 2 — load a folder of SGY files
# ---------------------------------------------------------------------------


def load_sgy_folder(
    folder: Union[str, Path],
    *,
    extensions: tuple[str, ...] = (".sgy", ".segy"),
    ignore_geometry: bool = True,
    dtype: np.dtype = np.float32,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Load every SEG-Y file in a directory and return them as a dictionary.

    The function recurses one level deep — it does **not** walk sub-folders —
    so that data from different survey lines stored in sub-directories is not
    accidentally mixed.  Files that cannot be parsed are skipped with a
    warning rather than raising, so a single corrupt file does not abort an
    entire batch job.

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory that contains the ``.sgy`` / ``.segy`` files.
    extensions : tuple of str, optional
        File extensions to recognise as SEG-Y.  Case-insensitive.
        Defaults to ``('.sgy', '.segy')``.
    ignore_geometry : bool, optional
        Forwarded to :func:`load_sgy`.  Default is *True*.
    dtype : numpy.dtype, optional
        Output array dtype.  Forwarded to :func:`load_sgy`.
    verbose : bool, optional
        When *True* (default) prints a summary line for each file loaded.

    Returns
    -------
    dict[str, numpy.ndarray]
        Keys are the file **stem** (basename without extension); values are
        2-D arrays of shape ``(n_traces, n_samples)``.  The dictionary is
        ordered by filename so results are reproducible.

    Raises
    ------
    NotADirectoryError
        If *folder* does not exist or is not a directory.

    Examples
    --------
    >>> dataset = load_sgy_folder("data/forge_2d/")
    >>> list(dataset.keys())[:3]
    ['record_027', 'record_028', 'record_029']
    >>> dataset['record_027'].shape
    (167, 4001)
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found or is not a directory: {folder}")

    extensions_lower = {ext.lower() for ext in extensions}
    sgy_files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions_lower
    )

    if not sgy_files:
        import warnings

        warnings.warn(
            f"No SEG-Y files found in '{folder}' with extensions {extensions}.",
            stacklevel=2,
        )
        return {}

    dataset: Dict[str, np.ndarray] = {}
    for path in sgy_files:
        try:
            traces = load_sgy(path, ignore_geometry=ignore_geometry, dtype=dtype)
            dataset[path.stem] = traces
            if verbose:
                print(f"  ✓ {path.name:40s}  shape={traces.shape}")
        except (ValueError, OSError) as exc:
            import warnings

            warnings.warn(f"Skipping '{path.name}': {exc}", stacklevel=2)

    if verbose:
        print(f"\nLoaded {len(dataset)} / {len(sgy_files)} files from '{folder}'.")

    return dataset


# ---------------------------------------------------------------------------
# Function 3 — normalize traces
# ---------------------------------------------------------------------------


def normalize_traces(
    traces: np.ndarray,
    method: Literal["minmax", "zscore", "maxabs"] = "zscore",
    *,
    per_trace: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """Normalize seismic trace amplitudes to a common scale.

    Seismic gathers often contain traces with very different amplitude levels
    (dead channels, noisy channels, far-offset geometric spreading).
    Normalizing before ML feature extraction prevents high-amplitude outliers
    from dominating the model.

    Three normalization strategies are provided:

    ``'minmax'``
        Rescales each trace to the range **[0, 1]** using
        ``(x - min) / (max - min)``.  Preserves the shape of the amplitude
        distribution but is sensitive to spikes.

    ``'zscore'``  *(default)*
        Zero-means and unit-variance scaling: ``(x - μ) / σ``.
        Robust for Gaussian-like seismic noise and compatible with most
        scikit-learn estimators.

    ``'maxabs'``
        Divides by the maximum absolute value so the range becomes
        **[-1, 1]**.  Preserves the sign of the amplitudes — important when
        polarity information is meaningful (e.g. bright spots).

    Parameters
    ----------
    traces : numpy.ndarray
        Input array of shape ``(n_traces, n_samples)``.
    method : {'minmax', 'zscore', 'maxabs'}, optional
        Normalization strategy.  Default is ``'zscore'``.
    per_trace : bool, optional
        When *True* (default) statistics are computed **per trace** (each row
        independently).  When *False* the statistics are computed over the
        entire gather, which is useful when relative amplitude differences
        between traces should be preserved (AVO analysis).
    eps : float, optional
        Small constant added to denominators to avoid division by zero on
        dead (zero-amplitude) traces.  Default is ``1e-10``.

    Returns
    -------
    numpy.ndarray
        Normalized array with the same shape and dtype as *traces*.

    Raises
    ------
    ValueError
        If *traces* is not a 2-D array, or *method* is unrecognised.

    Examples
    --------
    >>> raw = load_sgy("data/record_027.sgy")   # (167, 4001)
    >>> normed = normalize_traces(raw, method='zscore')
    >>> normed.mean(axis=1)                      # ≈ 0 for each trace
    >>> normed.std(axis=1)                       # ≈ 1 for each trace
    """
    if traces.ndim != 2:
        raise ValueError(
            f"traces must be a 2-D array (n_traces × n_samples), got shape {traces.shape}"
        )

    valid_methods = {"minmax", "zscore", "maxabs"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    out = traces.astype(np.float32, copy=True)
    axis = 1 if per_trace else None  # axis=1 → row-wise; None → global

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
