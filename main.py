import matplotlib.pyplot as plt

from seismoai_io import load_sgy, normalize_traces
from seismoai_viz import plot_gather, plot_spectrum, plot_trace

# ── 1. LOAD ──────────────────────────────────────────────────────────────
print("Loading SGY file...")
traces = load_sgy("data/27_1511546140_30100_50100_20171127_150416_752.sgy")
print(f"✓ Loaded: {traces.shape[0]} traces × {traces.shape[1]} samples")
print(f"  Min amplitude : {traces.min():.4f}")
print(f"  Max amplitude : {traces.max():.4f}")

# ── 2. NORMALIZE ─────────────────────────────────────────────────────────
print("\nNormalizing traces...")
normed = normalize_traces(traces, method="zscore")
print(f"✓ Normalized  — mean={normed.mean():.6f}  std={normed.std():.4f}")

# ── 3. VISUALIZE — Seismic Gather ────────────────────────────────────────
print("\nPlotting gather...")
fig1 = plot_gather(normed, dt_ms=1.0, title="Record 27 — Forge 2D Survey")
fig1.savefig("gather.png", dpi=150)
print("✓ Saved → gather.png")

# ── 4. VISUALIZE — Single Trace Waveform ─────────────────────────────────
print("\nPlotting single trace...")
fig2 = plot_trace(normed, trace_index=83, title="Channel 84 — Waveform")
fig2.savefig("trace_83.png", dpi=150)
print("✓ Saved → trace_83.png")

# ── 5. VISUALIZE — Frequency Spectrum ────────────────────────────────────
print("\nPlotting spectrum...")
fig3 = plot_spectrum(traces, trace_index=0, max_freq_hz=200, db_scale=True)
fig3.savefig("spectrum.png", dpi=150)
print("✓ Saved → spectrum.png")

# ── 6. SHOW ALL PLOTS in VS Code ─────────────────────────────────────────
plt.show()
print("\n✅ Done! Check gather.png, trace_83.png, spectrum.png in your folder.")
