"""
FWHM Diffractogram Analyzer
Loads First.csv automatically (space-separated 2theta, intensity).
Fit Gaussian / Pseudo-Voigt / Lorentzian profiles, report FWHM,
crystallite size (Scherrer), Williamson-Hall plot, CSV export.

Install:  pip install streamlit plotly numpy pandas scipy
Run:      streamlit run fwhm_analyzer.py
          → place First.csv in the same folder
"""

import os, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import streamlit as st

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="FWHM Analyzer", page_icon="📐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');
html,body,[data-testid="stApp"]{background:#06090f!important;color:#c0d4f0!important;font-family:'Exo 2',sans-serif!important}
[data-testid="stSidebar"]{background:#090d18!important;border-right:1px solid #1a3050}
h1,h2,h3{font-family:'Rajdhani',sans-serif!important;letter-spacing:2px!important}
h1{color:#6db3ff!important;font-weight:700!important}
h2,h3{color:#90c4ff!important;font-weight:600!important}
.stTabs [data-baseweb="tab-list"]{background:#090d18;border-bottom:1px solid #1a3555}
.stTabs [data-baseweb="tab"]{font-family:'Rajdhani',sans-serif!important;font-size:0.95rem;letter-spacing:1px;color:#3a6888;border-radius:6px 6px 0 0;padding:8px 20px}
.stTabs [aria-selected="true"]{color:#6db3ff!important;background:#0e1930!important;border-bottom:2px solid #6db3ff!important}
[data-testid="stMetricValue"]{font-family:'Share Tech Mono',monospace!important;color:#6db3ff!important}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PEAK PROFILE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def gaussian(x, A, mu, sigma, bg):
    return bg + A * np.exp(-((x - mu)**2) / (2 * sigma**2))

def lorentzian(x, A, mu, gamma, bg):
    return bg + A * (gamma**2 / ((x - mu)**2 + gamma**2))

def pseudo_voigt(x, A, mu, sigma, eta, bg):
    g = np.exp(-((x - mu)**2) / (2 * sigma**2))
    l = (sigma**2) / ((x - mu)**2 + sigma**2)
    return bg + A * (eta * l + (1 - eta) * g)

PROFILES = {
    "Gaussian":     {"fn": gaussian,     "fwhm": lambda p: 2*np.sqrt(2*np.log(2))*abs(p[2])},
    "Lorentzian":   {"fn": lorentzian,   "fwhm": lambda p: 2*abs(p[2])},
    "Pseudo-Voigt": {"fn": pseudo_voigt, "fwhm": lambda p: 2*np.sqrt(2*np.log(2))*abs(p[2])},
}

# ══════════════════════════════════════════════════════════════
#  SCHERRER
# ══════════════════════════════════════════════════════════════

def scherrer_size(fwhm_deg, two_theta_deg, wavelength, K=0.9):
    beta  = np.radians(fwhm_deg)
    theta = np.radians(two_theta_deg / 2)
    if beta <= 0 or np.cos(theta) == 0:
        return np.nan
    return (K * wavelength) / (beta * np.cos(theta)) / 10   # Å → nm

# ══════════════════════════════════════════════════════════════
#  LOAD First.csv  (space-separated, no header)
# ══════════════════════════════════════════════════════════════

DATA_FILE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "First.csv"))

@st.cache_data
def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    if not rows:
        st.error(f"First.csv loaded but contained no valid numeric rows. Path: {path}")
        st.stop()
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]

if not os.path.exists(DATA_FILE):
    st.error(f"**First.csv not found.** Place it next to `fwhm_analyzer.py`.\nExpected: `{DATA_FILE}`")
    st.stop()

two_theta, intensity = load_data(DATA_FILE)

# ══════════════════════════════════════════════════════════════
#  PEAK FITTING ENGINE
# ══════════════════════════════════════════════════════════════

def fit_peaks(two_theta, intensity, profile_name,
              min_prominence_pct, min_distance_deg, window_deg, wavelength, K):
    step        = two_theta[1] - two_theta[0]
    min_dist_pt = max(3, int(min_distance_deg / step))
    smooth      = gaussian_filter1d(intensity, sigma=2)

    peak_idx, _ = find_peaks(
        smooth,
        prominence=min_prominence_pct * smooth.max() / 100,
        distance=min_dist_pt,
    )

    fn      = PROFILES[profile_name]["fn"]
    fwhm_fn = PROFILES[profile_name]["fwhm"]
    results, fit_curves = [], []

    for i, idx in enumerate(peak_idx):
        pos     = two_theta[idx]
        amp_est = float(intensity[idx])
        bg_est  = float(np.percentile(intensity, 10))
        half_w  = max(int(window_deg / step), 5)
        lo, hi  = max(0, idx - half_w), min(len(two_theta)-1, idx + half_w)
        xw, yw  = two_theta[lo:hi+1], intensity[lo:hi+1]
        if len(xw) < 5:
            continue
        try:
            s0 = 0.15
            if profile_name == "Gaussian":
                p0 = [amp_est-bg_est, pos, s0, bg_est]
                bounds = ([0, pos-2, 0.01, 0], [amp_est*3, pos+2, 5, amp_est])
            elif profile_name == "Lorentzian":
                p0 = [amp_est-bg_est, pos, s0, bg_est]
                bounds = ([0, pos-2, 0.01, 0], [amp_est*3, pos+2, 5, amp_est])
            else:
                p0 = [amp_est-bg_est, pos, s0, 0.5, bg_est]
                bounds = ([0, pos-2, 0.01, 0, 0], [amp_est*3, pos+2, 5, 1, amp_est])

            popt, pcov = curve_fit(fn, xw, yw, p0=p0, bounds=bounds, maxfev=6000)
            fwhm = fwhm_fn(popt)
            mu   = popt[1]
            bg   = popt[-1]
            A    = popt[0]
            size = scherrer_size(fwhm, mu, wavelength, K)

            yfit = fn(xw, *popt)
            ss_r = np.sum((yw - yfit)**2)
            ss_t = np.sum((yw - yw.mean())**2)
            r2   = 1 - ss_r/ss_t if ss_t > 0 else np.nan

            results.append({
                "Peak #":              i + 1,
                "2θ_fit (°)":         round(mu, 4),
                "Intensity":          round(A + bg, 1),
                "Background":         round(bg, 1),
                "FWHM (°)":          round(fwhm, 5),
                "FWHM (rad)":        round(np.radians(fwhm), 6),
                "σ (°)":             round(abs(popt[2]), 5),
                "Crystallite D (nm)": round(size, 2) if not np.isnan(size) else "—",
                "R²":                round(r2, 4),
                "lo": lo, "hi": hi, "popt": popt,
            })

            xd = np.linspace(xw[0], xw[-1], 300)
            fit_curves.append((xd, fn(xd, *popt), mu, fwhm, A + bg, bg))

        except Exception:
            continue

    return results, fit_curves

# ══════════════════════════════════════════════════════════════
#  SIDEBAR — settings only
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;margin-bottom:16px'>
      <span style='font-family:Rajdhani,sans-serif;font-size:1.4rem;
                   color:#6db3ff;letter-spacing:3px;font-weight:700'>
        📐 FWHM ANALYZER
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:#0d1826;border:1px solid #1a3555;border-radius:8px;
                padding:10px 14px;margin-bottom:12px;
                font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#4a8abf'>
      📂 &nbsp;<b style='color:#6db3ff'>First.csv</b><br>
      {len(two_theta):,} points &nbsp;·&nbsp;
      2θ: {two_theta[0]:.2f}° – {two_theta[-1]:.2f}°
    </div>
    """, unsafe_allow_html=True)

    st.subheader("X-ray Settings")
    wavelength = st.number_input("λ (Å)", value=1.5406, format="%.4f",
                                 help="Cu Kα = 1.5406 Å")
    scherrer_K = st.number_input("Scherrer K", value=0.9, format="%.2f")

    st.divider()
    st.subheader("Peak Detection")
    min_prom = st.slider("Min prominence (% of max)", 1, 40, 4)
    min_dist = st.slider("Min peak separation (°)", 0.1, 5.0, 0.3, 0.05)
    win_deg  = st.slider("Fit window ± (°)", 0.2, 3.0, 0.8, 0.1)

    st.divider()
    st.subheader("Peak Profile")
    profile_name = st.selectbox("Fit function", list(PROFILES.keys()))

# ══════════════════════════════════════════════════════════════
#  RUN FITTING
# ══════════════════════════════════════════════════════════════

with st.spinner("Fitting peaks…"):
    peak_results, fit_curves = fit_peaks(
        two_theta, intensity, profile_name,
        min_prom, min_dist, win_deg, wavelength, scherrer_K
    )

n_peaks = len(peak_results)

# ══════════════════════════════════════════════════════════════
#  HEADER + METRICS
# ══════════════════════════════════════════════════════════════

st.title("📐 FWHM Diffractogram Analyzer")
st.markdown(
    "<p style='color:#3a6080;font-family:Share Tech Mono,monospace;"
    "font-size:0.8rem;letter-spacing:1px;margin-top:-10px'>"
    "NH₄/K-SODALITE SYNTHESIS 7  ·  Cu Kα  ·  POWDER XRD</p>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Peaks fitted", n_peaks)
if peak_results:
    fwhm_vals = [r["FWHM (°)"] for r in peak_results]
    c2.metric("Mean FWHM (°)", f"{np.mean(fwhm_vals):.4f}")
    c3.metric("Min FWHM (°)",  f"{np.min(fwhm_vals):.4f}")
    c4.metric("Max FWHM (°)",  f"{np.max(fwhm_vals):.4f}")

st.divider()

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Diffractogram + Fits",
    "📊 FWHM Results",
    "📉 Scherrer / Williamson-Hall",
    "🔬 Peak Inspector",
])

COLORS = [f"hsl({int(i*360/max(n_peaks,1))},85%,62%)" for i in range(n_peaks)]

# ── Tab 1: full pattern ───────────────────────────────────────
with tab1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=two_theta, y=intensity,
        mode="lines", name="Measured",
        line=dict(color="#4a90d9", width=1.0),
    ))

    for i, (xf, yf, mu, fwhm, peak_top, bg) in enumerate(fit_curves):
        col = COLORS[i]
        fig.add_trace(go.Scatter(
            x=xf, y=yf, mode="lines",
            name=f"Peak {i+1}",
            line=dict(color=col, width=2, dash="dash"),
        ))
        half_max = bg + (peak_top - bg) / 2
        fig.add_shape(type="line",
            x0=mu-fwhm/2, x1=mu+fwhm/2, y0=half_max, y1=half_max,
            line=dict(color=col, width=2, dash="dot"))
        fig.add_annotation(
            x=mu, y=half_max * 1.04,
            text=f"{fwhm:.4f}°",
            showarrow=False,
            font=dict(size=8, color=col),
        )

    if peak_results:
        fig.add_trace(go.Scatter(
            x=[r["2θ_fit (°)"] for r in peak_results],
            y=[r["Intensity"]   for r in peak_results],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=9, color="yellow"),
            text=[f"#{r['Peak #']}" for r in peak_results],
            textposition="top center",
            textfont=dict(size=8),
            name="Positions",
        ))

    fig.update_layout(
        xaxis_title="2θ (°)", yaxis_title="Intensity (counts)",
        height=500,
        paper_bgcolor="#06090f", plot_bgcolor="#06090f",
        font=dict(color="#c0d4f0"),
        xaxis=dict(gridcolor="#14243a"),
        yaxis=dict(gridcolor="#14243a"),
        legend=dict(bgcolor="rgba(6,9,15,0.8)", bordercolor="#1a3555",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: results table + CSV download ──────────────────────
with tab2:
    st.subheader("Peak Fitting Results")
    if peak_results:
        display_cols = ["Peak #","2θ_fit (°)","Intensity","Background",
                        "FWHM (°)","FWHM (rad)","σ (°)",
                        "Crystallite D (nm)","R²"]
        df = pd.DataFrame(peak_results)[display_cols]

        st.dataframe(
            df.style.background_gradient(subset=["FWHM (°)","R²"], cmap="plasma"),
            use_container_width=True, height=440, hide_index=True,
        )

        buf = io.StringIO()
        buf.write(f"# FWHM Analysis — First.csv  |  Profile: {profile_name}\n")
        buf.write(f"# lambda = {wavelength} A  |  K = {scherrer_K}\n")
        df.to_csv(buf, index=False)

        st.download_button(
            "⬇️  Download FWHM Results as CSV",
            data=buf.getvalue(),
            file_name="fwhm_results.csv",
            mime="text/csv",
            type="primary",
        )

        df_pat = pd.DataFrame({"2theta_deg": two_theta, "intensity": intensity})
        st.download_button(
            "⬇️  Download Raw Pattern as CSV",
            data=df_pat.to_csv(index=False),
            file_name="pattern_First.csv",
            mime="text/csv",
        )
    else:
        st.warning("No peaks fitted — try lowering the prominence threshold in the sidebar.")

# ── Tab 3: Scherrer / Williamson-Hall ────────────────────────
with tab3:
    st.subheader("Scherrer Crystallite Size Analysis")
    st.latex(r"D = \frac{K\lambda}{\beta\cos\theta}")

    numeric = [r for r in peak_results if r["Crystallite D (nm)"] != "—"]
    if numeric:
        sizes  = [float(r["Crystallite D (nm)"]) for r in numeric]
        angles = [r["2θ_fit (°)"] for r in numeric]
        fwhms  = [r["FWHM (°)"] for r in numeric]

        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Crystallite size D vs 2θ",
                            "Williamson-Hall  (β·cosθ vs 4·sinθ)"],
        )

        fig2.add_trace(go.Scatter(
            x=angles, y=sizes, mode="markers+text",
            marker=dict(size=11, color=sizes, colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=dict(text="D (nm)", font=dict(color="#c0d4f0")), x=0.44,
                                      tickfont=dict(color="#c0d4f0"))),
            text=[f"#{r['Peak #']}" for r in numeric],
            textposition="top center",
            textfont=dict(size=9, color="#c0d4f0"),
            showlegend=False,
        ), row=1, col=1)

        cos_t    = [np.cos(np.radians(a/2)) for a in angles]
        sin_t    = [np.sin(np.radians(a/2)) for a in angles]
        beta_cos = [np.radians(f)*c for f,c in zip(fwhms, cos_t)]
        four_sin = [4*s for s in sin_t]

        fig2.add_trace(go.Scatter(
            x=four_sin, y=beta_cos, mode="markers+text",
            marker=dict(size=10, color="#4fc3f7"),
            text=[f"#{r['Peak #']}" for r in numeric],
            textposition="top center",
            textfont=dict(size=9, color="#c0d4f0"),
            showlegend=False,
        ), row=1, col=2)

        if len(four_sin) >= 2:
            m_wh, b_wh = np.polyfit(four_sin, beta_cos, 1)
            xl = np.linspace(min(four_sin), max(four_sin), 60)
            fig2.add_trace(go.Scatter(
                x=xl, y=m_wh*xl + b_wh, mode="lines",
                line=dict(color="#f0a500", dash="dash", width=2),
                showlegend=False,
            ), row=1, col=2)
            D_wh = (scherrer_K * wavelength) / (b_wh * 1e10) if b_wh > 0 else np.nan
            st.info(
                f"**Williamson-Hall** → D ≈ {D_wh:.1f} nm  |  "
                f"micro-strain ε ≈ {m_wh:.4f}"
            )

        fig2.update_xaxes(title_text="2θ (°)",       gridcolor="#14243a", row=1, col=1)
        fig2.update_yaxes(title_text="D (nm)",        gridcolor="#14243a", row=1, col=1)
        fig2.update_xaxes(title_text="4·sinθ",        gridcolor="#14243a", row=1, col=2)
        fig2.update_yaxes(title_text="β·cosθ (rad)",  gridcolor="#14243a", row=1, col=2)

        fig2.update_layout(
            height=400,
            paper_bgcolor="#06090f", plot_bgcolor="#06090f",
            font=dict(color="#c0d4f0"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("Mean crystallite size", f"{np.mean(sizes):.1f} nm",
                  delta=f"σ = {np.std(sizes):.1f} nm")
        m2.metric("Peaks used", len(numeric))
    else:
        st.info("No crystallite sizes computed — check FWHM values or lower the prominence threshold.")

# ── Tab 4: individual peak zoom ──────────────────────────────
with tab4:
    st.subheader("Individual Peak Inspector")
    if peak_results and fit_curves:
        sel = st.selectbox(
            "Select peak",
            range(n_peaks),
            format_func=lambda i: (
                f"Peak {peak_results[i]['Peak #']}  —  "
                f"2θ = {peak_results[i]['2θ_fit (°)']:.3f}°  |  "
                f"FWHM = {peak_results[i]['FWHM (°)']:.4f}°  |  "
                f"D = {peak_results[i]['Crystallite D (nm)']} nm"
            ),
        )
        r = peak_results[sel]
        xf, yf, mu, fwhm, peak_top, bg = fit_curves[sel]
        lo, hi = r["lo"], r["hi"]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=two_theta[lo:hi+1], y=intensity[lo:hi+1],
            mode="lines+markers", name="Data",
            line=dict(color="#4a90d9", width=1.5),
            marker=dict(size=4),
        ))
        fig3.add_trace(go.Scatter(
            x=xf, y=yf, mode="lines",
            name=f"{profile_name} fit",
            line=dict(color="#f0a500", width=2.5),
        ))
        half_max = bg + (peak_top - bg) / 2
        fig3.add_shape(type="line",
            x0=mu-fwhm/2, x1=mu+fwhm/2, y0=half_max, y1=half_max,
            line=dict(color="#ff4e4e", width=2.5))
        fig3.add_annotation(
            x=mu, y=half_max * 1.06,
            text=f"FWHM = {fwhm:.5f}°",
            showarrow=False,
            font=dict(size=14, color="#ff4e4e", family="Share Tech Mono"),
        )
        fig3.update_layout(
            xaxis_title="2θ (°)", yaxis_title="Intensity (counts)",
            height=380,
            paper_bgcolor="#06090f", plot_bgcolor="#06090f",
            font=dict(color="#c0d4f0"),
            xaxis=dict(gridcolor="#14243a"),
            yaxis=dict(gridcolor="#14243a"),
            legend=dict(bgcolor="rgba(6,9,15,0.8)", bordercolor="#1a3555", borderwidth=1),
        )
        st.plotly_chart(fig3, use_container_width=True)

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("2θ (°)",     f"{r['2θ_fit (°)']:.4f}")
        s2.metric("FWHM (°)",   f"{r['FWHM (°)']:.5f}")
        s3.metric("FWHM (rad)", f"{r['FWHM (rad)']:.6f}")
        s4.metric("R²",         f"{r['R²']:.4f}")
        s5.metric("D (nm)",     f"{r['Crystallite D (nm)']}")
    else:
        st.info("No fitted peaks available.")

st.divider()
st.caption(
    "scipy.optimize.curve_fit  ·  scipy.signal.find_peaks  ·  "
    "Scherrer: D = Kλ/(β·cosθ)  ·  Williamson-Hall: β·cosθ = Kλ/D + 4ε·sinθ"
)