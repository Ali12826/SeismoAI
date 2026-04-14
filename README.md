# seismoai — Seismic Data Processing Library

A real Python library for AI-powered seismic data analysis built on the **Forge 2D Survey (2017)** dataset.

## Modules

| Package | PyPI name | Description |
|---|---|---|
| `seismoai_io` | `seismoai-io` | Load SGY files, batch loading, normalization |
| `seismoai_viz` | `seismoai-viz` | Gather images, trace waveforms, frequency spectra |
| `seismoai_qc` | `seismoai-qc` | Dead / noisy trace detection, QC reports |
| `seismoai_model` | `seismoai-model` | Feature extraction, noise classifier |
| `seismoai_xai` | `seismoai-xai` | SHAP values, feature importance, trace explanation |

---

## Installation

```bash
pip install seismoai-io seismoai-viz
```

---

## Quick Start

```python
from seismoai_io import load_sgy, normalize_traces
from seismoai_viz import plot_gather, plot_trace, plot_spectrum

# Load a real SGY file
traces = load_sgy("data/27_1511546140_30100_50100_20171127_150416_752.sgy")
print(traces.shape)   # (167, 4001)

# Normalize
normed = normalize_traces(traces, method="zscore")

# Visualize
fig1 = plot_gather(normed, dt_ms=1.0, title="Record 27 — Forge 2D")
fig1.savefig("gather.png", dpi=150)

fig2 = plot_trace(normed, trace_index=83)
fig2.savefig("trace_83.png", dpi=150)

fig3 = plot_spectrum(traces, trace_index=0, max_freq_hz=200)
fig3.savefig("spectrum.png", dpi=150)
```

---

## seismoai_io API

### `load_sgy(filepath, *, ignore_geometry=True, dtype=np.float32)`
Load a single SEG-Y file → `ndarray` shape `(n_traces, n_samples)`.

### `load_sgy_folder(folder, *, extensions=('.sgy','.segy'), verbose=True)`
Load all SEG-Y files in a directory → `dict[stem → ndarray]`.

### `normalize_traces(traces, method='zscore', *, per_trace=True)`
Normalize trace amplitudes.

| `method` | Description | Output range |
|---|---|---|
| `'zscore'` | Zero-mean, unit-variance | (−∞, +∞) |
| `'minmax'` | Min-Max scaling | [0, 1] |
| `'maxabs'` | Maximum absolute value | [−1, 1] |

---

## seismoai_viz API

### `plot_gather(traces, *, dt_ms=1.0, title=..., cmap='seismic', clip_percentile=99.0)`
2-D colour-fill gather image. Returns `Figure`.

### `plot_trace(traces, trace_index=0, *, dt_ms=1.0, title=...)`
1-D amplitude-vs-time waveform for one channel. Returns `Figure`.

### `plot_spectrum(traces, trace_index=0, *, dt_ms=1.0, max_freq_hz=None, db_scale=True)`
FFT amplitude spectrum with Hann window. Returns `Figure`.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v --tb=short
```

---

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/seismoai.git
cd seismoai
pip install -e ".[dev]"
```

---

## Dataset

The SGY files are from the **Forge 2D seismic survey** (Utah, USA, 2017),
recorded with an INOVA Geophysical Hawk system.
- 167 traces per record
- 4001 samples per trace
- 1 ms sample interval (4-second record length)
- IEEE Float32, Little-Endian encoding

---

## License
MIT
