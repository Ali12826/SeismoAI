"""
plotter.py — Core visualization functions for seismoai_viz.

All three functions return a ``matplotlib.figure.Figure`` so the caller
has full control over display (plt.show), file export (fig.savefig), or
embedding in Jupyter / Colab notebooks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.figure
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required by seismoai_viz. Install it with: pip install matplotlib"
    ) from exc


# ---------------------------------------------------------------------------
# Function 1 — plot seismic gather as a 2-D image
# ---------------------------------------------------------------------------


def plot_gather(
    traces: np.ndarray,
    *,
    dt_ms: float = 1.0,
    title: str = "Seismic Gather",
    cmap: str = "seismic",
    clip_percentile: float = 99.0,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Plot a seismic gather as a 2-D wiggle / colour-fill image.

    The image uses a diverging colour map centred on zero amplitude so
    positive and negative half-cycles are visually distinct.  Amplitudes are
    clipped at a user-defined percentile to prevent a small number of
    high-amplitude spikes from washing out the colour scale.

    Parameters
    ----------
    traces : numpy.ndarray
        2-D array of shape ``(n_traces, n_samples)``.  Rows are traces;
        columns are time samples.
    dt_ms : float, optional
        Sample interval in **milliseconds**.  Used to label the time axis.
        Default is ``1.0`` ms (1 000 µs), which matches the Forge 2D dataset.
    title : str, optional
        Figure title.  Default is ``'Seismic Gather'``.
    cmap : str, optional
        Matplotlib colour map name.  ``'seismic'`` (default) and ``'RdBu_r'``
        work well for signed seismic amplitudes.
    clip_percentile : float, optional
        Both the positive and negative colour scale limits are set to the
        *clip_percentile*-th percentile of ``|traces|``.  Values outside this
        range are clipped to the colour-scale extremes.  Default is ``99.0``.
    figsize : tuple of int, optional
        Figure size in inches ``(width, height)``.  Default is ``(14, 8)``.
    save_path : str or None, optional
        If provided, the figure is saved to this path (format inferred from
        the file extension, e.g. ``.png``, ``.pdf``).  Default is *None*
        (figure is not saved automatically).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.  Call ``plt.show()`` or ``fig.show()`` to
        display it, or ``fig.savefig(path)`` to export.

    Raises
    ------
    ValueError
        If *traces* is not a 2-D array.

    Examples
    --------
    >>> from seismoai_io import load_sgy
    >>> from seismoai_viz import plot_gather
    >>> traces = load_sgy("data/record_027.sgy")
    >>> fig = plot_gather(traces, dt_ms=1.0, title="Record 27 — Forge 2D")
    >>> fig.savefig("gather.png", dpi=150)
    """
    if traces.ndim != 2:
        raise ValueError(
            f"traces must be 2-D (n_traces × n_samples), got shape {traces.shape}"
        )

    n_traces, n_samples = traces.shape
    time_axis_ms = np.arange(n_samples) * dt_ms  # 0 … record_length ms

    clip = float(np.percentile(np.abs(traces), clip_percentile))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        traces.T,  # transpose → time on Y, trace on X
        aspect="auto",
        cmap=cmap,
        vmin=-clip,
        vmax=clip,
        extent=(1, n_traces, float(time_axis_ms[-1]), float(time_axis_ms[0])),
        interpolation="bilinear",
    )
    ax.set_xlabel("Trace Number", fontsize=12)
    ax.set_ylabel("Two-Way Travel Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Amplitude", fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Function 2 — plot a single trace as a waveform
# ---------------------------------------------------------------------------


def plot_trace(
    traces: np.ndarray,
    trace_index: int = 0,
    *,
    dt_ms: float = 1.0,
    title: Optional[str] = None,
    color: str = "#1f77b4",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Plot a single seismic trace as a 1-D amplitude-vs-time waveform.

    This view is complementary to :func:`plot_gather`: it lets you inspect
    the fine-grained amplitude structure of one channel, verify normalization
    results, or diagnose an anomalous trace flagged by the QC module.

    Parameters
    ----------
    traces : numpy.ndarray
        2-D array of shape ``(n_traces, n_samples)``.
    trace_index : int, optional
        Zero-based index of the trace to plot.  Default is ``0`` (first
        trace).
    dt_ms : float, optional
        Sample interval in milliseconds.  Default is ``1.0``.
    title : str or None, optional
        Figure title.  If *None*, defaults to
        ``'Trace <trace_index> Waveform'``.
    color : str, optional
        Line colour as a Matplotlib colour string.  Default is ``'#1f77b4'``
        (Matplotlib blue).
    figsize : tuple of int, optional
        Figure size in inches.  Default is ``(12, 4)``.
    save_path : str or None, optional
        If provided, saves the figure to this path.  Default is *None*.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If *traces* is not 2-D or *trace_index* is out of range.

    Examples
    --------
    >>> from seismoai_io import load_sgy, normalize_traces
    >>> from seismoai_viz import plot_trace
    >>> traces = load_sgy("data/record_027.sgy")
    >>> normed = normalize_traces(traces, method='zscore')
    >>> fig = plot_trace(normed, trace_index=83, title="Channel 84 — z-score")
    """
    if traces.ndim != 2:
        raise ValueError(
            f"traces must be 2-D (n_traces × n_samples), got shape {traces.shape}"
        )
    n_traces, n_samples = traces.shape
    if not (0 <= trace_index < n_traces):
        raise ValueError(
            f"trace_index {trace_index} is out of range for {n_traces} traces."
        )

    waveform = traces[trace_index]
    time_ms = np.arange(n_samples) * dt_ms

    if title is None:
        title = f"Trace {trace_index} — Waveform"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_ms, waveform, color=color, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Two-Way Travel Time (ms)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(time_ms[0], time_ms[-1])

    # Annotate basic stats
    stats_text = (
        f"min={waveform.min():.3f}  "
        f"max={waveform.max():.3f}  "
        f"μ={waveform.mean():.4f}  "
        f"σ={waveform.std():.4f}"
    )
    ax.text(
        0.01,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Function 3 — plot the frequency spectrum of a trace
# ---------------------------------------------------------------------------


def plot_spectrum(
    traces: np.ndarray,
    trace_index: int = 0,
    *,
    dt_ms: float = 1.0,
    title: Optional[str] = None,
    color: str = "#d62728",
    max_freq_hz: Optional[float] = None,
    db_scale: bool = True,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Plot the amplitude frequency spectrum of a single trace using the FFT.

    The spectrum reveals the dominant frequency content of the trace.
    Typical surface-seismic reflection data has usable energy in the
    10–120 Hz band.  Dead or noisy traces will have a flat or irregular
    spectrum.  This function is useful for filter design and QC.

    Parameters
    ----------
    traces : numpy.ndarray
        2-D array of shape ``(n_traces, n_samples)``.
    trace_index : int, optional
        Zero-based index of the trace to analyse.  Default is ``0``.
    dt_ms : float, optional
        Sample interval in milliseconds.  Used to compute the Nyquist
        frequency ``f_Nyquist = 1 / (2 × dt_s)`` Hz.  Default is ``1.0`` ms,
        giving a Nyquist of 500 Hz.
    title : str or None, optional
        Figure title.  Defaults to ``'Trace <i> — Frequency Spectrum'``.
    color : str, optional
        Line colour.  Default is Matplotlib red (``'#d62728'``).
    max_freq_hz : float or None, optional
        Upper frequency limit for the X-axis in Hz.  If *None* (default),
        uses the Nyquist frequency.  Set to e.g. ``150`` to zoom in on the
        seismic band of interest.
    db_scale : bool, optional
        When *True* (default) the Y-axis is displayed in **decibels**
        (``20 × log10(amplitude)``), which is the industry-standard way to
        inspect spectral shape.  When *False*, raw amplitude is shown.
    figsize : tuple of int, optional
        Figure size in inches.  Default is ``(10, 4)``.
    save_path : str or None, optional
        If provided, saves the figure to this path.  Default is *None*.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If *traces* is not 2-D or *trace_index* is out of range.

    Examples
    --------
    >>> from seismoai_io import load_sgy
    >>> from seismoai_viz import plot_spectrum
    >>> traces = load_sgy("data/record_027.sgy")
    >>> fig = plot_spectrum(traces, trace_index=0, max_freq_hz=200, db_scale=True)
    >>> fig.savefig("spectrum.png", dpi=150)
    """
    if traces.ndim != 2:
        raise ValueError(
            f"traces must be 2-D (n_traces × n_samples), got shape {traces.shape}"
        )
    n_traces, n_samples = traces.shape
    if not (0 <= trace_index < n_traces):
        raise ValueError(
            f"trace_index {trace_index} is out of range for {n_traces} traces."
        )

    waveform = traces[trace_index].astype(np.float64)
    dt_s = dt_ms / 1000.0  # convert ms → seconds
    nyquist_hz = 1.0 / (2.0 * dt_s)  # Nyquist frequency

    # Apply a Hann window to reduce spectral leakage
    window = np.hanning(n_samples)
    fft_vals = np.fft.rfft(waveform * window)
    freqs_hz = np.fft.rfftfreq(n_samples, d=dt_s)

    amplitude = np.abs(fft_vals)
    if db_scale:
        # Floor at a small positive value to avoid log(0)
        amplitude = 20.0 * np.log10(np.maximum(amplitude, 1e-10))
        ylabel = "Amplitude (dB)"
    else:
        ylabel = "Amplitude"

    if max_freq_hz is None:
        max_freq_hz = nyquist_hz
    mask = freqs_hz <= max_freq_hz

    if title is None:
        title = f"Trace {trace_index} — Frequency Spectrum"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freqs_hz[mask], amplitude[mask], color=color, linewidth=1.2)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, max_freq_hz)
    ax.grid(True, alpha=0.3)

    # Mark dominant frequency
    dominant_freq = freqs_hz[mask][np.argmax(amplitude[mask])]
    ax.axvline(
        dominant_freq,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label=f"Peak: {dominant_freq:.1f} Hz",
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
