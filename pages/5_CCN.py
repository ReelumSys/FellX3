"""
XRD Full-Profile CNN Fitter — Streamlit App
============================================
Supports: .txt, .csv, .dat, .xy, .xye, .asc, .ras, .raw (Bruker ASCII), .fxye
Run: streamlit run xrd_app.py
"""

import io
import re
import os
import tempfile
import traceback

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.optimize import least_squares

# ── optional torch import ──────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="XRD Profile Fitter",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Header */
.xrd-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 0.5rem;
}
.xrd-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: -0.04em;
}
.xrd-sub {
    font-size: 0.85rem;
    color: #888;
    font-family: 'Space Mono', monospace;
}

/* Upload zone styling */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #00d4ff55 !important;
    border-radius: 12px !important;
    background: #0a0a1600 !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00d4ff !important;
}

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 1rem 0;
}
.metric-card {
    background: #0f1623;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #557;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-unit {
    font-size: 0.7rem;
    color: #557;
    margin-top: 2px;
}

/* Peak table */
.peak-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
.peak-table th {
    background: #0f1623;
    color: #00d4ff;
    padding: 8px 10px;
    text-align: right;
    font-weight: 700;
    border-bottom: 1px solid #1e2d3d;
}
.peak-table th:first-child { text-align: left; }
.peak-table td {
    padding: 7px 10px;
    text-align: right;
    color: #c0c8d8;
    border-bottom: 1px solid #111a22;
}
.peak-table td:first-child { text-align: left; color: #ffd700; }
.peak-table tr:hover td { background: #0f1c2a; }

/* Badge */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    background: #0d2233;
    color: #00d4ff;
    border: 1px solid #00d4ff44;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080d14;
    border-right: 1px solid #1a2433;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PROFILE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def gaussian(x, pos, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((x - pos) / sigma) ** 2)

def lorentzian(x, pos, fwhm):
    gamma = fwhm / 2.0
    return 1.0 / (1.0 + ((x - pos) / gamma) ** 2)

def pseudo_voigt(x, pos, intensity, fwhm, eta):
    eta = np.clip(eta, 0.0, 1.0)
    return intensity * (eta * lorentzian(x, pos, fwhm) +
                        (1 - eta) * gaussian(x, pos, fwhm))

def polynomial_bg(x, a0, a1, a2):
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return a0 + a1 * xn + a2 * xn**2

def full_profile(x, params, n_peaks):
    y = np.zeros_like(x, dtype=np.float64)
    for k in range(n_peaks):
        pos, intensity, fwhm, eta = params[4*k:4*k+4]
        y += pseudo_voigt(x, pos, intensity, fwhm, eta)
    a0, a1, a2 = params[4*n_peaks:]
    y += polynomial_bg(x, a0, a1, a2)
    return y

# ══════════════════════════════════════════════════════════════
#  FILE PARSERS
# ══════════════════════════════════════════════════════════════

def _try_loadtxt(content_bytes):
    """Robust two-column loader: skip comment/header lines."""
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            text = content_bytes.decode(enc)
            break
        except Exception:
            continue
    else:
        raise ValueError("Cannot decode file (tried utf-8, latin-1, cp1252)")

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '!', ';', "'", '"')):
            continue
        # skip lines that start with letters (header keys like "TITLE = ...")
        if re.match(r'^[A-Za-z_]', line):
            continue
        parts = line.split()
        # accept lines with exactly 2 or 3 numeric tokens (3rd = sigma / weight)
        try:
            nums = [float(p) for p in parts[:3]]
            if len(nums) >= 2:
                rows.append((nums[0], nums[1]))
        except ValueError:
            continue
    if len(rows) < 10:
        raise ValueError(f"Only {len(rows)} numeric rows found — check file format")
    arr = np.array(rows, dtype=np.float32)
    return arr[:, 0], arr[:, 1]

PARSERS = {
    ".txt":  _try_loadtxt,
    ".dat":  _try_loadtxt,
    ".csv":  _try_loadtxt,
    ".xy":   _try_loadtxt,
    ".xye":  _try_loadtxt,
    ".asc":  _try_loadtxt,
    ".fxye": _try_loadtxt,
    ".ras":  _try_loadtxt,   # Rigaku ASCII — same structure after header
    ".raw":  _try_loadtxt,   # Bruker ASCII export
    ".gsas": _try_loadtxt,
    ".cpi":  _try_loadtxt,
}

def parse_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    parser = PARSERS.get(ext, _try_loadtxt)
    content = uploaded_file.read()
    x, y = parser(content)
    # sanity checks
    if x.max() > 180 or x.min() < 0:
        raise ValueError(f"2θ range looks wrong: [{x.min():.1f}, {x.max():.1f}]°")
    if y.min() < 0:
        y = y - y.min()  # shift negative baseline
    return x, y

# ══════════════════════════════════════════════════════════════
#  PEAK DETECTION
# ══════════════════════════════════════════════════════════════

def detect_peaks_auto(x, y, min_height_factor=5, min_dist_deg=0.4):
    step = float(x[1] - x[0]) if len(x) > 1 else 0.02
    min_dist_pts = max(1, int(min_dist_deg / step))
    bg = np.percentile(y, 15)
    threshold = max(bg * min_height_factor, y.max() * 0.03)
    idx, props = find_peaks(y, height=threshold,
                            distance=min_dist_pts,
                            prominence=threshold * 0.4)
    return idx, x[idx], y[idx]

# ══════════════════════════════════════════════════════════════
#  CNN (if torch available)
# ══════════════════════════════════════════════════════════════

if TORCH_OK:
    class ResBlock1d(nn.Module):
        def __init__(self, ch, k=5):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(ch, ch, k, padding=k//2),
                nn.BatchNorm1d(ch), nn.GELU(),
                nn.Conv1d(ch, ch, k, padding=k//2),
                nn.BatchNorm1d(ch),
            )
            self.act = nn.GELU()
        def forward(self, x):
            return self.act(x + self.block(x))

    class XRDProfileCNN(nn.Module):
        def __init__(self, n_pts, n_params, base=64):
            super().__init__()
            c = base
            self.stem = nn.Sequential(
                nn.Conv1d(1, c, 15, padding=7), nn.BatchNorm1d(c), nn.GELU())
            self.s1 = nn.Sequential(ResBlock1d(c),   nn.MaxPool1d(2))
            self.s2 = nn.Sequential(nn.Conv1d(c, c*2, 1), ResBlock1d(c*2), nn.MaxPool1d(2))
            self.s3 = nn.Sequential(nn.Conv1d(c*2, c*4, 1), ResBlock1d(c*4), nn.MaxPool1d(2))
            self.pool = nn.AdaptiveAvgPool1d(32)
            flat = c * 4 * 32
            self.reg = nn.Sequential(
                nn.Flatten(), nn.Linear(flat, 512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, n_params), nn.Sigmoid())
        def forward(self, x):
            return self.reg(self.pool(self.s3(self.s2(self.s1(self.stem(x.unsqueeze(1)))))))

    class ParamScaler:
        def fit(self, p):
            self.lo = p.min(0); self.hi = p.max(0)
            self.rng = np.where(self.hi - self.lo > 0, self.hi - self.lo, 1.0)
            return self
        def transform(self, p):
            return ((p - self.lo) / self.rng).astype(np.float32)
        def inverse(self, p):
            return p * self.rng + self.lo

    def make_synthetic(x, peak_pos, n=40_000, seed=42):
        rng = np.random.default_rng(seed)
        n_pts, n_pk = len(x), len(peak_pos)
        n_par = 4 * n_pk + 3
        P = np.zeros((n, n_pts), np.float32)
        Q = np.zeros((n, n_par), np.float32)
        for i in range(n):
            p = np.empty(n_par)
            for k, pr in enumerate(peak_pos):
                p[4*k]   = pr + rng.uniform(-0.12, 0.12)
                p[4*k+1] = rng.uniform(50, 3000)
                p[4*k+2] = rng.uniform(0.05, 0.4)
                p[4*k+3] = rng.uniform(0, 1)
            a0 = rng.uniform(20, 120)
            p[4*n_pk:] = [a0, rng.uniform(-20,20), rng.uniform(-10,10)]
            y = full_profile(x.astype(np.float64), p, n_pk)
            P[i] = (y + rng.normal(0, 8, n_pts)).astype(np.float32)
            Q[i] = p
        return P, Q

    def train_cnn(x, peak_pos, n_samples, epochs, batch, device, progress_cb):
        n_par = 4 * len(peak_pos) + 3
        P, Q = make_synthetic(x, peak_pos, n=n_samples)
        pm = P.mean(1, keepdims=True); ps = P.std(1, keepdims=True) + 1e-6
        Pn = (P - pm) / ps
        scaler = ParamScaler().fit(Q)
        Qn = scaler.transform(Q)
        sp = int(0.9 * n_samples)
        tr = DataLoader(TensorDataset(torch.tensor(Pn[:sp]), torch.tensor(Qn[:sp])),
                        batch_size=batch, shuffle=True)
        va = DataLoader(TensorDataset(torch.tensor(Pn[sp:]), torch.tensor(Qn[sp:])),
                        batch_size=batch)
        model = XRDProfileCNN(len(x), n_par).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit  = nn.HuberLoss(delta=0.1)
        hist  = {'train': [], 'val': []}
        for ep in range(1, epochs + 1):
            model.train()
            tl = sum(crit(model(xb.to(device)), yb.to(device)).item() * len(xb)
                     for xb, yb in tr) / sp
            model.eval()
            with torch.no_grad():
                vl = sum(crit(model(xb.to(device)), yb.to(device)).item() * len(xb)
                         for xb, yb in va) / (n_samples - sp)
            hist['train'].append(tl); hist['val'].append(vl)
            sched.step()
            progress_cb(ep / epochs, f"Epoch {ep}/{epochs}  train={tl:.5f}  val={vl:.5f}")
        return model, scaler, hist

# ══════════════════════════════════════════════════════════════
#  SCIPY-ONLY QUICK FIT
# ══════════════════════════════════════════════════════════════

def scipy_fit(x, y_obs, peak_pos, peak_int):
    """Direct least-squares fit without CNN — fast fallback."""
    n_pk  = len(peak_pos)
    n_par = 4 * n_pk + 3
    bg    = np.percentile(y_obs, 10)
    p0    = np.empty(n_par)
    lo    = np.empty(n_par)
    hi    = np.empty(n_par)
    for k in range(n_pk):
        p0[4*k]   = peak_pos[k]; lo[4*k]   = peak_pos[k] - 1.0; hi[4*k]   = peak_pos[k] + 1.0
        p0[4*k+1] = peak_int[k]; lo[4*k+1] = 0;                  hi[4*k+1] = y_obs.max() * 5
        p0[4*k+2] = 0.12;        lo[4*k+2] = 0.02;               hi[4*k+2] = 3.0
        p0[4*k+3] = 0.5;         lo[4*k+3] = 0.0;                hi[4*k+3] = 1.0
    p0[4*n_pk:] = [bg, 0, 0]
    lo[4*n_pk:] = [-500, -500, -500]
    hi[4*n_pk:] = [ 500,  500,  500]
    res = least_squares(lambda p: full_profile(x, p, n_pk) - y_obs,
                        x0=p0, bounds=(lo, hi), method='trf',
                        ftol=1e-10, xtol=1e-10, max_nfev=20_000)
    return res.x, full_profile(x, res.x, n_pk)

def refine_from_cnn(x, y_obs, cnn_params, n_peaks):
    n_p = 4 * n_peaks + 3
    lo = np.empty(n_p); hi = np.empty(n_p)
    for k in range(n_peaks):
        lo[4*k]   = cnn_params[4*k] - 1.0;   hi[4*k]   = cnn_params[4*k] + 1.0
        lo[4*k+1] = 0;                         hi[4*k+1] = y_obs.max() * 5
        lo[4*k+2] = 0.02;                      hi[4*k+2] = 3.0
        lo[4*k+3] = 0.0;                       hi[4*k+3] = 1.0
    lo[4*n_peaks:] = [-500, -500, -500]
    hi[4*n_peaks:] = [ 500,  500,  500]
    res = least_squares(lambda p: full_profile(x, p, n_peaks) - y_obs,
                        x0=cnn_params, bounds=(lo, hi), method='trf',
                        ftol=1e-12, xtol=1e-12, max_nfev=30_000)
    return res.x, full_profile(x, res.x, n_peaks)

# ══════════════════════════════════════════════════════════════
#  PLOTTING (dark theme)
# ══════════════════════════════════════════════════════════════

COLORS = ['#00d4ff','#ffd700','#a8ff78','#ff6b6b','#a29bfe',
          '#fd79a8','#74b9ff','#ff9f43','#55efc4','#e17055']
TEXT  = '#c8d0e0'
GRID  = '#1a2233'
BG    = '#080d14'
BG2   = '#0d1520'

def make_plot(x, y_obs, y_fit, params, n_peaks, history=None):
    plt.rcParams.update({'text.color': TEXT, 'axes.labelcolor': TEXT,
                         'xtick.color': TEXT, 'ytick.color': TEXT,
                         'font.family': 'monospace'})
    has_hist = history is not None and len(history.get('train', [])) > 0
    rows = 3 if has_hist else 2
    fig = plt.figure(figsize=(14, 4 * rows))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(rows, 2, figure=fig, hspace=0.5, wspace=0.35,
                            left=0.07, right=0.97, top=0.93, bottom=0.06)

    def ax_style(ax, title):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        ax.set_title(title, color=TEXT, fontsize=9, pad=5)

    # panel 1: full fit
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, y_obs,  color='#445566', lw=0.7, label='Observed', zorder=1)
    ax1.plot(x, y_fit,  color='#00d4ff', lw=1.4, label='Calc',     zorder=3)
    for k in range(n_peaks):
        pos, intensity, fwhm, eta = params[4*k:4*k+4]
        yp = pseudo_voigt(x, pos, intensity, fwhm, eta)
        ax1.fill_between(x, yp, alpha=0.22, color=COLORS[k % len(COLORS)])
        ax1.axvline(pos, color=COLORS[k % len(COLORS)], lw=0.6, alpha=0.55, ls='--')
    ax1.set_xlabel('2θ (°)'); ax1.set_ylabel('Intensity (cts)')
    ax1.legend(framealpha=0.15, labelcolor=TEXT, fontsize=8,
               facecolor='#111', edgecolor=GRID)
    ax_style(ax1, 'Full Profile Fit')

    # panel 2: residuals
    ax2 = fig.add_subplot(gs[1, :])
    residual = y_obs - y_fit
    r_wp = np.sqrt(np.sum(residual**2) / np.sum(y_obs**2)) * 100
    ax2.plot(x, residual, color='#ff6b6b', lw=0.7)
    ax2.fill_between(x, residual, alpha=0.25, color='#ff6b6b')
    ax2.axhline(0, color=TEXT, lw=0.8, ls='--', alpha=0.4)
    ax_style(ax2, f'Residuals (Obs − Calc)   Rwp = {r_wp:.2f}%')
    ax2.set_xlabel('2θ (°)'); ax2.set_ylabel('ΔI (cts)')

    if has_hist:
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(history['train'], color='#ffd700', lw=1.2, label='Train')
        ax3.plot(history['val'],   color='#00d4ff', lw=1.2, label='Val', ls='--')
        ax3.set_yscale('log')
        ax3.set_xlabel('Epoch'); ax3.set_ylabel('Huber loss')
        ax3.legend(framealpha=0.15, labelcolor=TEXT, fontsize=8,
                   facecolor='#111', edgecolor=GRID)
        ax_style(ax3, 'CNN Training Loss')

        ax4 = fig.add_subplot(gs[2, 1])
        ww = 0.5
        xk = np.arange(n_peaks)
        pos_vals = [params[4*k] for k in range(n_peaks)]
        ax4.bar(xk, pos_vals, ww, color='#00d4ff', alpha=0.8)
        ax4.set_xticks(xk); ax4.set_xticklabels([f'P{k+1}' for k in range(n_peaks)],
                                                  color=TEXT, fontsize=8)
        ax4.set_ylabel('Position 2θ (°)')
        ax_style(ax4, 'Peak Positions')

    fig.suptitle('XRD Full-Profile Fit', color=TEXT, fontsize=11, fontweight='bold', y=0.97)
    return fig

# ══════════════════════════════════════════════════════════════
#  SIDEBAR SETTINGS
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    fit_mode = st.radio(
        "Fitting mode",
        ["scipy only (fast)", "CNN + scipy (accurate)"],
        help="scipy only: direct least-squares from auto-detected peak positions.\n"
             "CNN + scipy: train a neural network first, use it as warm start."
    )
    use_cnn = fit_mode.startswith("CNN") and TORCH_OK

    if not TORCH_OK and fit_mode.startswith("CNN"):
        st.warning("PyTorch not installed — falling back to scipy only.")

    st.divider()
    st.markdown("**Peak detection**")
    min_height = st.slider("Min height factor", 2, 15, 5,
                            help="Peaks must be N× the 15th-percentile background")
    min_dist   = st.slider("Min distance (°)", 0.1, 2.0, 0.4, step=0.05)

    if use_cnn:
        st.divider()
        st.markdown("**CNN training**")
        n_samples = st.select_slider("Synthetic samples", [5_000, 10_000, 20_000,
                                                            40_000, 60_000], value=20_000)
        n_epochs  = st.slider("Epochs", 50, 500, 150, step=25)
        batch     = st.select_slider("Batch size", [128, 256, 512, 1024], value=256)
        device_str = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"
        st.caption(f"Device: `{device_str}`")

    st.divider()
    st.markdown("**Export**")
    export_csv  = st.checkbox("Export fit as CSV", value=True)
    export_png  = st.checkbox("Export plot as PNG", value=True)

# ══════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="xrd-header">
  <span class="xrd-title">⚛ XRD Profile Fitter</span>
  <span class="xrd-sub">pseudo-Voigt · CNN + scipy · full-profile</span>
</div>
""", unsafe_allow_html=True)

SUPPORTED_EXTS = [".txt", ".csv", ".dat", ".xy", ".xye",
                  ".asc", ".ras", ".raw", ".fxye", ".gsas", ".cpi"]

uploaded = st.file_uploader(
    "Drop your diffraction file here",
    type=[e.lstrip('.') for e in SUPPORTED_EXTS],
    help="Two-column (2θ  Intensity) text format.\n"
         "Supported: " + "  ".join(SUPPORTED_EXTS)
)

if uploaded is None:
    st.markdown("""
    <div style='margin-top:2rem;padding:2rem;border:1px dashed #1a2d3d;border-radius:12px;
                background:#060b10;text-align:center;color:#445;'>
        <div style='font-size:2.5rem;margin-bottom:.5rem'>📂</div>
        <div style='font-family:monospace;font-size:.9rem'>
            Supported formats: .txt &nbsp;·&nbsp; .csv &nbsp;·&nbsp; .dat &nbsp;·&nbsp;
            .xy &nbsp;·&nbsp; .xye &nbsp;·&nbsp; .asc &nbsp;·&nbsp; .ras &nbsp;·&nbsp;
            .raw &nbsp;·&nbsp; .fxye &nbsp;·&nbsp; .gsas &nbsp;·&nbsp; .cpi
        </div>
        <div style='font-size:.75rem;margin-top:.5rem;color:#334'>
            Two-column  (2θ [°]  &nbsp; Intensity [counts])  &nbsp;—&nbsp;
            comment lines starting with #  !  ;  are skipped automatically
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Parse ──
try:
    x_raw, y_raw = parse_file(uploaded)
    st.success(f"✓ Loaded **{uploaded.name}**  —  "
               f"{len(x_raw):,} points  ·  "
               f"2θ = [{x_raw.min():.2f}°, {x_raw.max():.2f}°]  ·  "
               f"I_max = {y_raw.max():.0f} cts")
except Exception as e:
    st.error(f"**Parse error:** {e}")
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
    st.stop()

x  = x_raw.astype(np.float64)
y  = y_raw.astype(np.float64)

# ── Preview ──
with st.expander("📊 Raw pattern preview", expanded=True):
    fig_pre, ax_pre = plt.subplots(figsize=(12, 2.8))
    fig_pre.patch.set_facecolor(BG)
    ax_pre.set_facecolor(BG2)
    ax_pre.plot(x, y, color='#445577', lw=0.6, label='Raw data')
    for sp in ax_pre.spines.values(): sp.set_color(GRID)
    ax_pre.grid(True, color=GRID, lw=0.4, alpha=0.6)
    ax_pre.set_xlabel('2θ (°)', color=TEXT); ax_pre.set_ylabel('I (cts)', color=TEXT)
    ax_pre.tick_params(colors=TEXT)
    fig_pre.tight_layout()
    st.pyplot(fig_pre, use_container_width=True)
    plt.close(fig_pre)

# ── Peak detection ──
peaks_idx, peak_pos, peak_int = detect_peaks_auto(x, y, min_height, min_dist)
n_peaks = len(peaks_idx)

if n_peaks == 0:
    st.error("No peaks detected — try lowering the **Min height factor** in the sidebar.")
    st.stop()

st.markdown(f"**{n_peaks} peaks detected** &nbsp;"
            + "  ".join(f'<span class="badge">{p:.3f}°</span>' for p in peak_pos),
            unsafe_allow_html=True)

# ── Fitting ──
run_btn = st.button("▶  Run Full-Profile Fit", type="primary", use_container_width=True)

if run_btn:
    history = None

    with st.spinner("Fitting…"):
        if use_cnn:
            # ── CNN training ──
            prog_bar  = st.progress(0.0)
            prog_text = st.empty()

            def cb(frac, msg):
                prog_bar.progress(frac)
                prog_text.text(msg)

            device = torch.device(device_str)
            model, scaler, history = train_cnn(
                x, peak_pos, n_samples, n_epochs, batch, device, cb)
            prog_bar.empty(); prog_text.empty()

            # predict
            model.eval()
            yn = ((y - y.mean()) / (y.std() + 1e-6)).astype(np.float32)
            with torch.no_grad():
                pn = model(torch.tensor(yn).unsqueeze(0).to(device)).cpu().numpy()[0]
            cnn_params = scaler.inverse(pn[np.newaxis])[0]

            # refine
            with st.spinner("Refining with scipy…"):
                final_params, y_fit = refine_from_cnn(x, y, cnn_params, n_peaks)
        else:
            # scipy only
            final_params, y_fit = scipy_fit(x, y, peak_pos, peak_int)

    # ── Metrics ──
    residual = y - y_fit
    r_wp  = np.sqrt(np.sum(residual**2) / np.sum(y**2)) * 100
    r_p   = np.sum(np.abs(residual)) / np.sum(y) * 100
    r_rms = np.sqrt(np.mean(residual**2))
    gof   = r_wp / r_p if r_p > 0 else 0

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">Rwp</div>
        <div class="metric-value">{r_wp:.2f}</div>
        <div class="metric-unit">%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Rp</div>
        <div class="metric-value">{r_p:.2f}</div>
        <div class="metric-unit">%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">RMS</div>
        <div class="metric-value">{r_rms:.1f}</div>
        <div class="metric-unit">counts</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">GoF</div>
        <div class="metric-value">{gof:.3f}</div>
        <div class="metric-unit">Rwp / Rp</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Plot ──
    fig = make_plot(x, y, y_fit, final_params, n_peaks, history)
    st.pyplot(fig, use_container_width=True)

    if export_png:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        st.download_button("⬇  Download plot (PNG)", buf,
                           file_name="xrd_fit.png", mime="image/png")
    plt.close(fig)

    # ── Peak table ──
    st.markdown("#### Refined peak parameters")
    rows_html = ""
    for k in range(n_peaks):
        pos, intensity, fwhm, eta = final_params[4*k:4*k+4]
        rows_html += (f"<tr><td>Peak {k+1}</td>"
                      f"<td>{pos:.4f}</td><td>{intensity:.1f}</td>"
                      f"<td>{fwhm:.4f}</td><td>{eta:.4f}</td></tr>")
    a0, a1, a2 = final_params[4*n_peaks:]
    st.markdown(f"""
    <table class="peak-table">
      <thead><tr>
        <th>#</th><th>2θ (°)</th><th>Intensity</th><th>FWHM (°)</th><th>η</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div style='font-size:.75rem;color:#445;margin-top:.5rem;font-family:monospace'>
      Background: a₀={a0:.2f}  a₁={a1:.2f}  a₂={a2:.2f}
    </div>
    """, unsafe_allow_html=True)

    # ── CSV export ──
    if export_csv:
        # full fit table
        csv_lines = ["2theta,obs,calc,residual"]
        for xi, yo, yc in zip(x, y, y_fit):
            csv_lines.append(f"{xi:.4f},{yo:.1f},{yc:.4f},{yo-yc:.4f}")
        csv_str = "\n".join(csv_lines)
        st.download_button("⬇  Download fit (CSV)", csv_str,
                           file_name="xrd_fit.csv", mime="text/csv")

        # peak params
        pk_lines = ["peak,pos_2theta,intensity,fwhm,eta"]
        for k in range(n_peaks):
            pos, intensity, fwhm, eta = final_params[4*k:4*k+4]
            pk_lines.append(f"{k+1},{pos:.6f},{intensity:.2f},{fwhm:.6f},{eta:.6f}")
        st.download_button("⬇  Download peak params (CSV)",
                           "\n".join(pk_lines),
                           file_name="xrd_peaks.csv", mime="text/csv")