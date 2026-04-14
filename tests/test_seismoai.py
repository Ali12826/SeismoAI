"""
tests/test_seismoai.py — Test suite for seismoai_io and seismoai_viz.

Run with:
    pytest tests/test_seismoai.py -v

All tests use synthetic in-memory data so no SGY files are needed on disk.
Tests marked with ``@pytest.mark.sgyfile`` require the real Forge 2D SGY
file and are skipped automatically when it is not present.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── helpers ─────────────────────────────────────────────────────────────────


def _make_traces(n_traces: int = 20, n_samples: int = 200) -> np.ndarray:
    """Return a synthetic (n_traces × n_samples) float32 array."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((n_traces, n_samples)).astype(np.float32)


def _write_minimal_sgy(path: Path, traces: np.ndarray) -> None:
    """Write a bare-minimum little-endian SEG-Y file for integration tests."""
    n_traces, n_samples = traces.shape
    with open(path, "wb") as f:
        # 3200-byte text header (blank)
        f.write(b" " * 3200)
        # 400-byte binary header (all zeros)
        f.write(b"\x00" * 400)
        for i in range(n_traces):
            # 240-byte trace header (all zeros)
            f.write(b"\x00" * 240)
            # Trace data as little-endian float32
            f.write(traces[i].astype("<f4").tobytes())


# ═══════════════════════════════════════════════════════════════════════════
#  seismoai_io — loader.py
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadSgy:
    """Tests for seismoai_io.load_sgy"""

    def test_raises_file_not_found(self):
        from seismoai_io import load_sgy

        with pytest.raises(FileNotFoundError, match="not found"):
            load_sgy("/nonexistent/path/file.sgy")

    def test_raises_value_error_on_bad_file(self, tmp_path):
        """A file that exists but is not valid SEG-Y should raise ValueError."""
        bad = tmp_path / "bad.sgy"
        bad.write_bytes(b"not a segy file at all")
        from seismoai_io import load_sgy

        with pytest.raises(ValueError, match="Could not read"):
            load_sgy(bad)

    def test_returns_ndarray(self, tmp_path):
        """load_sgy should return a numpy ndarray."""
        traces = _make_traces()
        sgy_path = tmp_path / "test.sgy"
        _write_minimal_sgy(sgy_path, traces)

        # Patch segyio.open to return controlled data
        mock_f = MagicMock()
        mock_f.__enter__ = lambda s: s
        mock_f.__exit__ = MagicMock(return_value=False)
        mock_f.trace.__getitem__ = MagicMock(return_value=traces)

        with (
            patch("segyio.open", return_value=mock_f),
            patch("segyio.tools.collect", return_value=traces),
        ):
            from seismoai_io import load_sgy

            result = load_sgy(sgy_path)

        assert isinstance(result, np.ndarray)

    def test_output_dtype_float32(self, tmp_path):
        """Default dtype should be float32."""
        traces = _make_traces().astype(np.float64)
        sgy_path = tmp_path / "test.sgy"
        _write_minimal_sgy(sgy_path, traces)

        with (
            patch("segyio.open") as mock_open,
            patch("segyio.tools.collect", return_value=traces.astype(np.float32)),
        ):
            mock_open.return_value.__enter__ = lambda s: MagicMock()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            from seismoai_io import load_sgy

            result = (
                load_sgy.__wrapped__(sgy_path)
                if hasattr(load_sgy, "__wrapped__")
                else None
            )

        # Direct dtype test via mock
        with (
            patch("segyio.open") as mo,
            patch(
                "segyio.tools.collect", return_value=np.zeros((5, 10), dtype=np.float64)
            ),
        ):
            mo.return_value.__enter__ = lambda s: MagicMock()
            mo.return_value.__exit__ = MagicMock(return_value=False)
            from seismoai_io.loader import load_sgy as _load

            result = _load(sgy_path)
        assert result.dtype == np.float32


class TestLoadSgyFolder:
    """Tests for seismoai_io.load_sgy_folder"""

    def test_raises_not_a_directory(self):
        from seismoai_io import load_sgy_folder

        with pytest.raises(NotADirectoryError):
            load_sgy_folder("/no/such/folder")

    def test_warns_when_empty(self, tmp_path):
        from seismoai_io import load_sgy_folder

        with pytest.warns(UserWarning, match="No SEG-Y files found"):
            result = load_sgy_folder(tmp_path)
        assert result == {}

    def test_returns_dict_with_stems_as_keys(self, tmp_path):
        """Keys should be file stems (no extension)."""
        traces = _make_traces(n_traces=5, n_samples=50)
        for name in ("record_001.sgy", "record_002.sgy"):
            (tmp_path / name).write_bytes(b" " * 3600)  # dummy

        with patch("seismoai_io.loader.load_sgy", return_value=traces):
            from seismoai_io import load_sgy_folder

            dataset = load_sgy_folder(tmp_path, verbose=False)

        assert set(dataset.keys()) == {"record_001", "record_002"}
        assert isinstance(dataset["record_001"], np.ndarray)

    def test_skips_corrupt_files_with_warning(self, tmp_path):
        """Corrupt files should be skipped; valid ones should still load."""
        traces = _make_traces(n_traces=5, n_samples=50)
        (tmp_path / "good.sgy").write_bytes(b" " * 3600)
        (tmp_path / "bad.sgy").write_bytes(b"corrupt")

        def side_effect(path, **kwargs):
            if "bad" in str(path):
                raise ValueError("bad file")
            return traces

        with patch("seismoai_io.loader.load_sgy", side_effect=side_effect):
            from seismoai_io import load_sgy_folder

            with pytest.warns(UserWarning):
                dataset = load_sgy_folder(tmp_path, verbose=False)

        assert "good" in dataset
        assert "bad" not in dataset


class TestNormalizeTraces:
    """Tests for seismoai_io.normalize_traces"""

    def test_zscore_zero_mean(self):
        """Z-score per-trace normalization must yield mean ≈ 0 for each trace."""
        from seismoai_io import normalize_traces

        traces = _make_traces(n_traces=10, n_samples=500)
        normed = normalize_traces(traces, method="zscore", per_trace=True)
        means = normed.mean(axis=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_zscore_unit_std(self):
        """Z-score per-trace normalization must yield std ≈ 1 for each trace."""
        from seismoai_io import normalize_traces

        traces = _make_traces(n_traces=10, n_samples=500)
        normed = normalize_traces(traces, method="zscore", per_trace=True)
        stds = normed.std(axis=1)
        np.testing.assert_allclose(stds, 1.0, atol=1e-4)

    def test_minmax_range_zero_to_one(self):
        """Min-Max normalization must produce values in [0, 1]."""
        from seismoai_io import normalize_traces

        traces = _make_traces()
        normed = normalize_traces(traces, method="minmax")
        assert normed.min() >= 0.0 - 1e-6
        assert normed.max() <= 1.0 + 1e-6

    def test_maxabs_range_minus_one_to_one(self):
        """MaxAbs normalization must produce values in [-1, 1]."""
        from seismoai_io import normalize_traces

        traces = _make_traces()
        normed = normalize_traces(traces, method="maxabs")
        assert np.abs(normed).max() <= 1.0 + 1e-6

    def test_output_shape_unchanged(self):
        """Normalization must preserve the input shape."""
        from seismoai_io import normalize_traces

        traces = _make_traces(n_traces=15, n_samples=300)
        for method in ("zscore", "minmax", "maxabs"):
            normed = normalize_traces(traces, method=method)
            assert normed.shape == traces.shape, f"Shape changed for method={method}"

    def test_output_dtype_float32(self):
        """Output dtype should be float32 regardless of input dtype."""
        from seismoai_io import normalize_traces

        traces = _make_traces().astype(np.float64)
        normed = normalize_traces(traces, method="zscore")
        assert normed.dtype == np.float32

    def test_raises_on_1d_input(self):
        """A 1-D array should raise ValueError."""
        from seismoai_io import normalize_traces

        with pytest.raises(ValueError, match="2-D"):
            normalize_traces(np.zeros(100))

    def test_raises_on_unknown_method(self):
        """An unknown method string should raise ValueError."""
        from seismoai_io import normalize_traces

        with pytest.raises(ValueError, match="method must be"):
            normalize_traces(_make_traces(), method="l2")

    def test_dead_trace_no_nan(self):
        """A zero-amplitude dead trace should not produce NaN (eps guard)."""
        from seismoai_io import normalize_traces

        traces = np.zeros((5, 100), dtype=np.float32)
        normed = normalize_traces(traces, method="zscore")
        assert not np.any(np.isnan(normed))


# ═══════════════════════════════════════════════════════════════════════════
#  seismoai_viz — plotter.py
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all open figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


class TestPlotGather:
    """Tests for seismoai_viz.plot_gather"""

    def test_returns_figure(self):
        from seismoai_viz import plot_gather

        fig = plot_gather(_make_traces())
        assert isinstance(fig, plt.Figure)

    def test_raises_on_1d_input(self):
        from seismoai_viz import plot_gather

        with pytest.raises(ValueError, match="2-D"):
            plot_gather(np.zeros(100))

    def test_saves_file(self, tmp_path):
        from seismoai_viz import plot_gather

        out = tmp_path / "gather.png"
        plot_gather(_make_traces(), save_path=str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_axes_labels(self):
        """The figure should have time and trace-number axis labels."""
        from seismoai_viz import plot_gather

        fig = plot_gather(_make_traces())
        ax = fig.axes[0]
        assert "Trace" in ax.get_xlabel()
        assert "Time" in ax.get_ylabel()

    def test_custom_title(self):
        from seismoai_viz import plot_gather

        fig = plot_gather(_make_traces(), title="My Custom Title")
        ax = fig.axes[0]
        assert "My Custom Title" in ax.get_title()


class TestPlotTrace:
    """Tests for seismoai_viz.plot_trace"""

    def test_returns_figure(self):
        from seismoai_viz import plot_trace

        fig = plot_trace(_make_traces(), trace_index=0)
        assert isinstance(fig, plt.Figure)

    def test_raises_on_1d_input(self):
        from seismoai_viz import plot_trace

        with pytest.raises(ValueError, match="2-D"):
            plot_trace(np.zeros(100))

    def test_raises_on_out_of_range_index(self):
        from seismoai_viz import plot_trace

        with pytest.raises(ValueError, match="out of range"):
            plot_trace(_make_traces(n_traces=10), trace_index=99)

    def test_negative_index_raises(self):
        from seismoai_viz import plot_trace

        with pytest.raises(ValueError):
            plot_trace(_make_traces(), trace_index=-1)

    def test_saves_file(self, tmp_path):
        from seismoai_viz import plot_trace

        out = tmp_path / "trace.png"
        plot_trace(_make_traces(), save_path=str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_axes_labels(self):
        from seismoai_viz import plot_trace

        fig = plot_trace(_make_traces())
        ax = fig.axes[0]
        assert "Time" in ax.get_xlabel()
        assert "Amplitude" in ax.get_ylabel()

    def test_last_trace_index(self):
        """Border case: last valid index should not raise."""
        from seismoai_viz import plot_trace

        traces = _make_traces(n_traces=10)
        fig = plot_trace(traces, trace_index=9)
        assert isinstance(fig, plt.Figure)


class TestPlotSpectrum:
    """Tests for seismoai_viz.plot_spectrum"""

    def test_returns_figure(self):
        from seismoai_viz import plot_spectrum

        fig = plot_spectrum(_make_traces())
        assert isinstance(fig, plt.Figure)

    def test_raises_on_1d_input(self):
        from seismoai_viz import plot_spectrum

        with pytest.raises(ValueError, match="2-D"):
            plot_spectrum(np.zeros(100))

    def test_raises_on_out_of_range_index(self):
        from seismoai_viz import plot_spectrum

        with pytest.raises(ValueError, match="out of range"):
            plot_spectrum(_make_traces(n_traces=5), trace_index=10)

    def test_db_scale_ylabel(self):
        from seismoai_viz import plot_spectrum

        fig = plot_spectrum(_make_traces(), db_scale=True)
        ax = fig.axes[0]
        assert "dB" in ax.get_ylabel()

    def test_linear_scale_ylabel(self):
        from seismoai_viz import plot_spectrum

        fig = plot_spectrum(_make_traces(), db_scale=False)
        ax = fig.axes[0]
        assert "Amplitude" in ax.get_ylabel()

    def test_max_freq_clamps_xaxis(self):
        """X-axis should not exceed max_freq_hz."""
        from seismoai_viz import plot_spectrum

        fig = plot_spectrum(_make_traces(), max_freq_hz=100.0)
        ax = fig.axes[0]
        assert ax.get_xlim()[1] <= 100.0 + 1.0  # +1 for tick rounding

    def test_saves_file(self, tmp_path):
        from seismoai_viz import plot_spectrum

        out = tmp_path / "spectrum.png"
        plot_spectrum(_make_traces(), save_path=str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_axes_labels(self):
        from seismoai_viz import plot_spectrum

        fig = plot_spectrum(_make_traces())
        ax = fig.axes[0]
        assert "Hz" in ax.get_xlabel()
