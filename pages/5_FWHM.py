"""
FWHM Diffractogram Analyzer
- Load measured diffractogram (CSV/XY) OR simulate one from mineral structure factors
- Auto-detect peaks, fit Gaussian / Pseudo-Voigt / Lorentzian profiles
- Report FWHM, peak position, intensity, crystallite size (Scherrer)
- Export full results to CSV
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import io

# ─────────────────────────────────────────────────────────────────────────────
# Peak profile functions
# ─────────────────────────────────────────────────────────────────────────────

def gaussian(x, A, mu, sigma, bg):
    return bg + A * np.exp(-((x - mu)**2) / (2 * sigma**2))

def lorentzian(x, A, mu, gamma, bg):
    return bg + A * (gamma**2 / ((x - mu)**2 + gamma**2))

def pseudo_voigt(x, A, mu, sigma, eta, bg):
    """Linear combination: eta*Lorentzian + (1-eta)*Gaussian"""
    gauss = np.exp(-((x - mu)**2) / (2 * sigma**2))
    lor   = (sigma**2) / ((x - mu)**2 + sigma**2)
    return bg + A * (eta * lor + (1 - eta) * gauss)

PROFILES = {
    "Gaussian":      {"fn": gaussian,     "params": ["A", "μ", "σ", "bg"],        "fwhm": lambda p: 2*np.sqrt(2*np.log(2))*abs(p[2])},
    "Lorentzian":    {"fn": lorentzian,   "params": ["A", "μ", "γ", "bg"],        "fwhm": lambda p: 2*abs(p[2])},
    "Pseudo-Voigt":  {"fn": pseudo_voigt, "params": ["A", "μ", "σ", "η", "bg"],  "fwhm": lambda p: 2*np.sqrt(2*np.log(2))*abs(p[2])},
}

# ─────────────────────────────────────────────────────────────────────────────
# Scherrer equation: D = Kλ / (β·cosθ)
# ─────────────────────────────────────────────────────────────────────────────

def scherrer_size(fwhm_deg, two_theta_deg, wavelength_angstrom, K=0.9):
    """Return crystallite size in nm."""
    beta  = np.radians(fwhm_deg)
    theta = np.radians(two_theta_deg / 2)
    if beta <= 0 or np.cos(theta) == 0:
        return np.nan
    return (K * wavelength_angstrom) / (beta * np.cos(theta)) / 10  # Å → nm

# ─────────────────────────────────────────────────────────────────────────────
# Mineral simulation (reuse from hkl_phases logic)
# ─────────────────────────────────────────────────────────────────────────────

SCATTERING_FACTORS = {
    "Si": ([6.2915, 3.0353, 1.9891, 0.5399, 1.1410], [2.4386, 32.3337, 0.6785, 81.6937, 0.0], 1.1407),
    "O":  ([3.0485, 2.2868, 1.0624, 0.1156, 0.0],    [13.2771, 5.7011, 0.3239, 32.9089, 0.0], 0.3006),
    "Al": ([6.4202, 1.9002, 1.5936, 1.9646, 0.0],    [3.0387, 0.7426, 31.5472, 85.0886, 0.0], 1.1151),
    "Ca": ([8.6266, 7.3873, 1.5899, 1.0211, 0.0],    [10.4421, 0.6599, 85.7484, 178.437, 0.0], 1.3751),
    "Fe": ([11.7695, 7.3573, 3.5222, 2.3045, 0.0],   [4.7611, 0.3072, 15.3535, 76.8805, 0.0], 1.0369),
    "Mg": ([5.4204, 2.1735, 1.2269, 2.3073, 0.0],    [2.8275, 79.2611, 0.3808, 7.1937, 0.0], 0.8584),
    "Na": ([6.4202, 1.9002, 1.5936, 1.9646, 0.0],    [3.0387, 0.7426, 31.5472, 85.0886, 0.0], 0.4655),
    "C":  ([2.3100, 1.0200, 1.5886, 0.8650, 0.0],    [20.8439, 10.2075, 0.5687, 51.6512, 0.0], 0.2156),
}

MINERALS = {
    "Quartz (SiO₂)": {
        "system": "Hexagonal", "a": 4.9133, "b": 4.9133, "c": 5.4053,
        "alpha": 90, "beta": 90, "gamma": 120,
        "atoms": [
            {"element": "Si", "x": 0.4697, "y": 0.0000, "z": 0.0000, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.0000, "y": 0.4697, "z": 0.6667, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.5303, "y": 0.5303, "z": 0.3333, "occ": 1.0, "Biso": 0.5},
            {"element": "O",  "x": 0.4135, "y": 0.2669, "z": 0.1188, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.2669, "y": 0.4135, "z": 0.8812, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.7331, "y": 0.1466, "z": 0.4521, "occ": 1.0, "Biso": 0.8},
        ],
    },
    "Calcite (CaCO₃)": {
        "system": "Trigonal", "a": 4.9896, "b": 4.9896, "c": 17.0610,
        "alpha": 90, "beta": 90, "gamma": 120,
        "atoms": [
            {"element": "Ca", "x": 0.0000, "y": 0.0000, "z": 0.0000, "occ": 1.0, "Biso": 0.6},
            {"element": "C",  "x": 0.0000, "y": 0.0000, "z": 0.2500, "occ": 1.0, "Biso": 0.5},
            {"element": "O",  "x": 0.2573, "y": 0.0000, "z": 0.2500, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.0000, "y": 0.2573, "z": 0.2500, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.7427, "y": 0.7427, "z": 0.2500, "occ": 1.0, "Biso": 1.0},
        ],
    },
    "Forsterite (Mg₂SiO₄)": {
        "system": "Orthorhombic", "a": 4.7540, "b": 10.1971, "c": 5.9806,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [
            {"element": "Mg", "x": 0.0000, "y": 0.0000, "z": 0.0000, "occ": 1.0, "Biso": 0.5},
            {"element": "Mg", "x": 0.5000, "y": 0.5000, "z": 0.0000, "occ": 1.0, "Biso": 0.5},
            {"element": "Mg", "x": 0.0000, "y": 0.2211, "z": 0.5000, "occ": 1.0, "Biso": 0.5},
            {"element": "Mg", "x": 0.5000, "y": 0.7789, "z": 0.5000, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.0000, "y": 0.0940, "z": 0.4232, "occ": 1.0, "Biso": 0.4},
            {"element": "Si", "x": 0.5000, "y": 0.4060, "z": 0.4232, "occ": 1.0, "Biso": 0.4},
            {"element": "O",  "x": 0.0000, "y": 0.0926, "z": 0.7656, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.5000, "y": 0.4074, "z": 0.7656, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.0000, "y": 0.4512, "z": 0.2199, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.5000, "y": 0.0488, "z": 0.2199, "occ": 1.0, "Biso": 0.8},
        ],
    },
    "Albite (NaAlSi₃O₈)": {
        "system": "Triclinic", "a": 8.1360, "b": 12.7870, "c": 7.1582,
        "alpha": 94.253, "beta": 116.605, "gamma": 87.756,
        "atoms": [
            {"element": "Na", "x": 0.2690, "y": 0.9890, "z": 0.1470, "occ": 1.0, "Biso": 1.5},
            {"element": "Al", "x": 0.0088, "y": 0.1680, "z": 0.2082, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.0036, "y": 0.8200, "z": 0.2390, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.6900, "y": 0.1120, "z": 0.3150, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.6813, "y": 0.8820, "z": 0.3610, "occ": 1.0, "Biso": 0.5},
            {"element": "O",  "x": 0.0055, "y": 0.1310, "z": 0.9680, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.5934, "y": 0.9970, "z": 0.2800, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.8194, "y": 0.1085, "z": 0.1902, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.0203, "y": 0.3027, "z": 0.2700, "occ": 1.0, "Biso": 1.0},
        ],
    },
}

def deg2rad(d): return np.radians(d)

def compute_metric_tensor(a, b, c, alpha, beta, gamma):
    ca, cb, cg = np.cos(deg2rad(alpha)), np.cos(deg2rad(beta)), np.cos(deg2rad(gamma))
    return np.array([[a*a, a*b*cg, a*c*cb],
                     [a*b*cg, b*b, b*c*ca],
                     [a*c*cb, b*c*ca, c*c]])

def d_spacing(h, k, l, a, b, c, alpha, beta, gamma):
    G = compute_metric_tensor(a, b, c, alpha, beta, gamma)
    Ginv = np.linalg.inv(G)
    q2 = np.array([h, k, l]) @ Ginv @ np.array([h, k, l])
    return 1.0 / np.sqrt(q2) if q2 > 0 else np.inf

def atomic_scattering_factor(element, s):
    if element not in SCATTERING_FACTORS:
        return 1.0
    a_c, b_c, c = SCATTERING_FACTORS[element]
    return c + sum(ai * np.exp(-bi * s**2) for ai, bi in zip(a_c, b_c))

def structure_factor_amplitude(h, k, l, atoms, a, b, c, alpha, beta, gamma, wavelength):
    d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma)
    s = wavelength / (2 * d) if d > 0 else 0
    F = 0+0j
    for atom in atoms:
        f  = atomic_scattering_factor(atom["element"], s)
        DW = np.exp(-atom.get("Biso", 0.5) * s**2)
        ph = 2 * np.pi * (h*atom["x"] + k*atom["y"] + l*atom["z"])
        F += atom.get("occ", 1.0) * f * DW * np.exp(1j * ph)
    return abs(F)

def simulate_diffractogram(mineral_name, wavelength, two_theta_range, n_pts,
                            peak_fwhm_deg, noise_level, crystallite_nm):
    """Generate a synthetic powder diffractogram with Gaussian peaks + noise."""
    mineral = MINERALS[mineral_name]
    a, b, c = mineral["a"], mineral["b"], mineral["c"]
    alpha, beta, gamma = mineral["alpha"], mineral["beta"], mineral["gamma"]
    atoms = mineral["atoms"]

    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], n_pts)
    pattern = np.zeros(n_pts)

    # Scherrer broadening contribution
    # fwhm_inst = instrument broadening; fwhm_size = size broadening (approx)
    sigma_inst = peak_fwhm_deg / (2 * np.sqrt(2 * np.log(2)))

    reflections = []
    for h in range(-5, 6):
        for k in range(-5, 6):
            for l in range(-5, 6):
                if h == 0 and k == 0 and l == 0:
                    continue
                d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma)
                st_ = wavelength / (2 * d) if d > 0 else 999
                if st_ > 1.0:
                    continue
                tt = np.degrees(2 * np.arcsin(st_))
                if not (two_theta_range[0] < tt < two_theta_range[1]):
                    continue
                amp = structure_factor_amplitude(h, k, l, atoms, a, b, c, alpha, beta, gamma, wavelength)
                intensity = amp**2
                if intensity < 1:
                    continue

                # Size broadening (Scherrer): β_size = Kλ/(D·cosθ)
                theta_rad = np.radians(tt / 2)
                beta_size = np.degrees(0.9 * wavelength / (crystallite_nm * 10 * np.cos(theta_rad)))
                fwhm_total = np.sqrt(peak_fwhm_deg**2 + beta_size**2)
                sigma_total = fwhm_total / (2 * np.sqrt(2 * np.log(2)))

                pattern += intensity * np.exp(-((two_theta - tt)**2) / (2 * sigma_total**2))
                reflections.append((h, k, l, tt, intensity))

    # Lorentz-polarization correction (simplified)
    theta_arr = np.radians(two_theta / 2)
    lp = (1 + np.cos(2*theta_arr)**2) / (np.sin(theta_arr)**2 * np.cos(theta_arr) + 1e-9)
    lp = lp / lp.max()
    pattern *= lp

    # Background + noise
    bg = 50 + 200 * np.exp(-two_theta / 40)
    pattern += bg
    if noise_level > 0:
        pattern += np.random.normal(0, noise_level * pattern.max() / 100, n_pts)

    pattern = np.clip(pattern, 0, None)
    return two_theta, pattern, reflections

# ─────────────────────────────────────────────────────────────────────────────
# Peak fitting engine
# ─────────────────────────────────────────────────────────────────────────────

def fit_peaks(two_theta, intensity, profile_name, min_prominence, min_distance_deg,
              window_factor, wavelength):
    """
    1. Find peaks via scipy
    2. Fit each with chosen profile within a local window
    3. Return dataframe of results
    """
    step = two_theta[1] - two_theta[0]
    min_dist_pts = max(3, int(min_distance_deg / step))

    # Smooth slightly for peak detection only
    smooth = gaussian_filter1d(intensity, sigma=2)
    peak_idx, props = find_peaks(
        smooth,
        prominence=min_prominence * smooth.max() / 100,
        distance=min_dist_pts,
    )

    profile_info = PROFILES[profile_name]
    fn = profile_info["fn"]
    fwhm_fn = profile_info["fwhm"]

    results = []
    fit_curves = []

    for i, idx in enumerate(peak_idx):
        pos = two_theta[idx]
        amp_est = intensity[idx]
        bg_est  = np.percentile(intensity, 10)

        # Window around peak
        half_win = max(int(window_factor * 2 / step), 5)
        lo = max(0, idx - half_win)
        hi = min(len(two_theta) - 1, idx + half_win)

        x_win = two_theta[lo:hi+1]
        y_win = intensity[lo:hi+1]

        if len(x_win) < 5:
            continue

        try:
            sigma_est = 0.15
            if profile_name == "Gaussian":
                p0     = [amp_est - bg_est, pos, sigma_est, bg_est]
                bounds = ([0, pos-2, 0.01, 0], [amp_est*3, pos+2, 5, amp_est])
            elif profile_name == "Lorentzian":
                p0     = [amp_est - bg_est, pos, sigma_est, bg_est]
                bounds = ([0, pos-2, 0.01, 0], [amp_est*3, pos+2, 5, amp_est])
            else:  # Pseudo-Voigt
                p0     = [amp_est - bg_est, pos, sigma_est, 0.5, bg_est]
                bounds = ([0, pos-2, 0.01, 0, 0], [amp_est*3, pos+2, 5, 1, amp_est])

            popt, pcov = curve_fit(fn, x_win, y_win, p0=p0, bounds=bounds, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))

            fwhm = fwhm_fn(popt)
            mu   = popt[1]
            A    = popt[0]
            bg   = popt[-1]

            # Scherrer crystallite size
            size_nm = scherrer_size(fwhm, mu, wavelength)

            # R² on window
            y_fit_win = fn(x_win, *popt)
            ss_res = np.sum((y_win - y_fit_win)**2)
            ss_tot = np.sum((y_win - np.mean(y_win))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            results.append({
                "Peak #":        i + 1,
                "2θ_fit (°)":   round(mu, 4),
                "Intensity":     round(A + bg, 1),
                "Background":    round(bg, 1),
                "FWHM (°)":     round(fwhm, 5),
                "FWHM (rad)":   round(np.radians(fwhm), 6),
                f"σ_{profile_name[:3]} (°)": round(abs(popt[2]), 5),
                "Crystallite D (nm)": round(size_nm, 2) if not np.isnan(size_nm) else "—",
                "R²":           round(r2, 4),
                "Profile":      profile_name,
                "Fit_lo":       lo, "Fit_hi": hi,
                "popt":         popt,
            })

            # Dense fit curve for plotting
            x_dense = np.linspace(x_win[0], x_win[-1], 300)
            y_dense = fn(x_dense, *popt)
            fit_curves.append((x_dense, y_dense, mu, fwhm, popt[0]+popt[-1]))

        except Exception:
            pass

    return results, fit_curves

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="FWHM Diffractogram Analyzer", page_icon="📐", layout="wide")

st.title("📐 FWHM Diffractogram Analyzer")
st.markdown(
    "Fit diffraction peaks, extract **FWHM** values, estimate crystallite sizes via the "
    "**Scherrer equation**, and export results to **CSV**."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    data_source = st.radio("Data source", ["📁 Upload file", "🔬 Simulate mineral"])

    if data_source == "📁 Upload file":
        uploaded = st.file_uploader(
            "Upload diffractogram (CSV or XY)",
            type=["csv", "txt", "xy", "dat"],
            help="Two-column file: 2θ [°] , Intensity. Delimiter: comma, tab, or space."
        )
        col_sep = st.selectbox("Delimiter", ["auto", ",", ";", "\\t", "space"])

    else:
        mineral_sel = st.selectbox("Mineral", list(MINERALS.keys()))
        cryst_nm    = st.slider("Crystallite size (nm)", 5, 200, 50,
                                help="Affects synthetic peak broadening via Scherrer")
        noise_pct   = st.slider("Noise level (%)", 0, 20, 3)
        sim_fwhm    = st.slider("Instrument FWHM (°)", 0.05, 1.0, 0.15, 0.01)
        tt_min      = st.number_input("2θ min (°)", value=5.0)
        tt_max      = st.number_input("2θ max (°)", value=70.0)
        n_pts       = st.select_slider("Data points", [1000, 2000, 5000, 10000], value=5000)

    st.divider()
    st.subheader("X-ray Settings")
    wavelength = st.number_input("λ (Å)", value=1.5406, format="%.4f",
                                 help="Cu Kα = 1.5406 Å")

    st.divider()
    st.subheader("Peak Detection")
    min_prom   = st.slider("Min prominence (% of max)", 1, 40, 5)
    min_dist   = st.slider("Min peak separation (°)", 0.1, 5.0, 0.5, 0.1)
    win_factor = st.slider("Fit window ±(°)", 0.2, 3.0, 0.8, 0.1)

    st.divider()
    st.subheader("Peak Profile")
    profile_name = st.selectbox("Fit function", list(PROFILES.keys()))

    st.divider()
    scherrer_K = st.number_input("Scherrer K factor", value=0.9, format="%.2f")

# ─────────────────────────────────────────────────────────────────────────────
# Load or simulate data
# ─────────────────────────────────────────────────────────────────────────────

two_theta = None
intensity = None
data_label = ""

if data_source == "📁 Upload file":
    if uploaded is not None:
        try:
            content = uploaded.read().decode("utf-8", errors="replace")
            sep_map = {"auto": None, ",": ",", ";": ";", "\\t": "\t", "space": r"\s+"}
            sep = sep_map[col_sep]
            df_raw = pd.read_csv(
                io.StringIO(content),
                sep=sep, engine="python",
                comment="#", header=None,
                on_bad_lines="skip",
            )
            df_raw = df_raw.apply(pd.to_numeric, errors="coerce").dropna()
            two_theta = df_raw.iloc[:, 0].values
            intensity = df_raw.iloc[:, 1].values
            data_label = uploaded.name
            st.sidebar.success(f"Loaded {len(two_theta)} data points")
        except Exception as e:
            st.error(f"Could not parse file: {e}")
    else:
        st.info("👈 Upload a diffractogram file or switch to simulation mode.")

else:
    with st.spinner("Simulating diffractogram…"):
        two_theta, intensity, reflections = simulate_diffractogram(
            mineral_sel, wavelength, (tt_min, tt_max), n_pts,
            sim_fwhm, noise_pct, cryst_nm
        )
    data_label = f"{mineral_sel} (simulated)"

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

if two_theta is not None and intensity is not None:
    # Fit peaks
    with st.spinner("Fitting peaks…"):
        peak_results, fit_curves = fit_peaks(
            two_theta, intensity, profile_name,
            min_prom, min_dist, win_factor, wavelength
        )

    n_peaks = len(peak_results)

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peaks found & fitted", n_peaks)
    if peak_results:
        fwhm_vals = [r["FWHM (°)"] for r in peak_results]
        m2.metric("Mean FWHM (°)", f"{np.mean(fwhm_vals):.4f}")
        m3.metric("Min FWHM (°)",  f"{np.min(fwhm_vals):.4f}")
        m4.metric("Max FWHM (°)",  f"{np.max(fwhm_vals):.4f}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Diffractogram + Fits", "📊 FWHM Results", "📉 Scherrer Analysis", "🔬 Individual Peaks"]
    )

    # ── Tab 1: full pattern with overlaid fits ────────────────────────────────
    with tab1:
        st.subheader(f"Diffractogram — {data_label}")
        fig = go.Figure()

        # Raw pattern
        fig.add_trace(go.Scatter(
            x=two_theta, y=intensity,
            mode="lines", name="Measured",
            line=dict(color="#90caf9", width=1.2),
        ))

        # Fit curves
        colors = [f"hsl({int(i*360/max(n_peaks,1))},90%,60%)" for i in range(n_peaks)]
        for i, (xf, yf, mu, fwhm, peak_top) in enumerate(fit_curves):
            fig.add_trace(go.Scatter(
                x=xf, y=yf,
                mode="lines", name=f"Peak {i+1} fit",
                line=dict(color=colors[i], width=2, dash="dash"),
            ))
            # FWHM bracket
            half_max = (peak_top + peak_results[i]["Background"]) / 2
            fig.add_shape(type="line",
                x0=mu - fwhm/2, x1=mu + fwhm/2, y0=half_max, y1=half_max,
                line=dict(color=colors[i], width=2, dash="dot"))
            fig.add_annotation(
                x=mu, y=half_max * 1.03,
                text=f"FWHM={fwhm:.4f}°",
                showarrow=False, font=dict(size=9, color=colors[i]),
            )

        # Peak markers
        if peak_results:
            fig.add_trace(go.Scatter(
                x=[r["2θ_fit (°)"] for r in peak_results],
                y=[r["Intensity"] for r in peak_results],
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=10, color="yellow"),
                text=[f"#{r['Peak #']}" for r in peak_results],
                textposition="top center",
                name="Peak positions",
            ))

        fig.update_layout(
            xaxis_title="2θ (°)", yaxis_title="Intensity (counts)",
            legend=dict(orientation="h", y=-0.2),
            height=480,
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: FWHM results table ─────────────────────────────────────────────
    with tab2:
        st.subheader("Peak Fitting Results")
        if peak_results:
            display_cols = [
                "Peak #", "2θ_fit (°)", "Intensity", "Background",
                "FWHM (°)", "FWHM (rad)",
                f"σ_{profile_name[:3]} (°)",
                "Crystallite D (nm)", "R²", "Profile"
            ]
            df_results = pd.DataFrame(peak_results)[display_cols]

            st.dataframe(
                df_results.style.background_gradient(subset=["FWHM (°)", "R²"], cmap="plasma"),
                use_container_width=True, height=420
            )

            # ── CSV export ────────────────────────────────────────────────────
            csv_buf = io.StringIO()
            # Add metadata header
            csv_buf.write(f"# FWHM Analysis — {data_label}\n")
            csv_buf.write(f"# Profile: {profile_name}  |  λ = {wavelength} Å  |  K = {scherrer_K}\n")
            csv_buf.write(f"# Generated by FWHM Diffractogram Analyzer\n")
            df_results.to_csv(csv_buf, index=False)
            csv_str = csv_buf.getvalue()

            st.download_button(
                label="⬇️ Download FWHM Results as CSV",
                data=csv_str,
                file_name=f"fwhm_{data_label.replace(' ','_')}.csv",
                mime="text/csv",
                type="primary",
            )

            # Also offer the full raw pattern
            df_pattern = pd.DataFrame({"2theta_deg": two_theta, "intensity": intensity})
            st.download_button(
                label="⬇️ Download Raw Pattern as CSV",
                data=df_pattern.to_csv(index=False),
                file_name=f"pattern_{data_label.replace(' ','_')}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No peaks were successfully fitted. Try lowering the prominence threshold.")

    # ── Tab 3: Scherrer analysis ──────────────────────────────────────────────
    with tab3:
        st.subheader("Scherrer Crystallite Size Analysis")
        st.latex(r"D = \frac{K\lambda}{\beta\cos\theta}")
        st.markdown(f"K = {scherrer_K},  λ = {wavelength} Å")

        if peak_results:
            numeric_results = [r for r in peak_results if r["Crystallite D (nm)"] != "—"]
            if numeric_results:
                sizes  = [float(r["Crystallite D (nm)"]) for r in numeric_results]
                angles = [r["2θ_fit (°)"] for r in numeric_results]
                fwhms  = [r["FWHM (°)"] for r in numeric_results]

                fig2 = make_subplots(rows=1, cols=2,
                    subplot_titles=["Crystallite Size vs 2θ", "FWHM vs 2θ (Williamson-Hall style)"])

                fig2.add_trace(go.Scatter(
                    x=angles, y=sizes, mode="markers+text",
                    marker=dict(size=12, color=sizes, colorscale="Viridis", showscale=True,
                                colorbar=dict(title="D (nm)", x=0.45)),
                    text=[f"#{r['Peak #']}" for r in numeric_results],
                    textposition="top center",
                    name="D (nm)",
                ), row=1, col=1)

                # Williamson-Hall: β·cosθ vs 4·sinθ
                cos_t = [np.cos(np.radians(a/2)) for a in angles]
                sin_t = [np.sin(np.radians(a/2)) for a in angles]
                beta_cos = [np.radians(f)*c for f, c in zip(fwhms, cos_t)]
                four_sin = [4*s for s in sin_t]

                fig2.add_trace(go.Scatter(
                    x=four_sin, y=beta_cos, mode="markers+text",
                    marker=dict(size=10, color="#4fc3f7"),
                    text=[f"#{r['Peak #']}" for r in numeric_results],
                    textposition="top center",
                    name="β·cosθ",
                ), row=1, col=2)

                # Linear fit (WH)
                if len(four_sin) >= 2:
                    m, b_wh = np.polyfit(four_sin, beta_cos, 1)
                    x_line = np.linspace(min(four_sin), max(four_sin), 50)
                    fig2.add_trace(go.Scatter(
                        x=x_line, y=m*x_line + b_wh,
                        mode="lines", line=dict(color="orange", dash="dash"),
                        name=f"WH fit (ε={m:.2e})",
                    ), row=1, col=2)
                    D_wh = (scherrer_K * wavelength) / (b_wh * 1e10) if b_wh > 0 else np.nan
                    st.info(f"Williamson-Hall intercept → D ≈ {D_wh:.1f} nm  |  Micro-strain ε ≈ {m:.4f}")

                fig2.update_xaxes(title_text="2θ (°)", row=1, col=1, gridcolor="#333")
                fig2.update_xaxes(title_text="4·sinθ", row=1, col=2, gridcolor="#333")
                fig2.update_yaxes(title_text="D (nm)", row=1, col=1, gridcolor="#333")
                fig2.update_yaxes(title_text="β·cosθ (rad)", row=1, col=2, gridcolor="#333")
                fig2.update_layout(
                    height=400, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font=dict(color="white"), showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Mean crystallite size
                st.metric("Mean crystallite size", f"{np.mean(sizes):.1f} nm",
                          delta=f"±{np.std(sizes):.1f} nm")
            else:
                st.info("No numeric crystallite sizes computed (check FWHM values).")

    # ── Tab 4: individual peak zoom ───────────────────────────────────────────
    with tab4:
        st.subheader("Individual Peak Inspector")
        if peak_results and fit_curves:
            peak_idx_sel = st.selectbox(
                "Select peak",
                options=list(range(n_peaks)),
                format_func=lambda i: f"Peak {peak_results[i]['Peak #']}  —  2θ={peak_results[i]['2θ_fit (°)']:.3f}°  FWHM={peak_results[i]['FWHM (°)']:.4f}°"
            )
            r   = peak_results[peak_idx_sel]
            xf, yf, mu, fwhm, peak_top = fit_curves[peak_idx_sel]
            lo, hi = r["Fit_lo"], r["Fit_hi"]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=two_theta[lo:hi+1], y=intensity[lo:hi+1],
                mode="lines+markers", name="Data",
                line=dict(color="#90caf9"), marker=dict(size=4),
            ))
            fig3.add_trace(go.Scatter(
                x=xf, y=yf, mode="lines", name=f"{profile_name} fit",
                line=dict(color="orange", width=2.5),
            ))
            half_max = (peak_top + r["Background"]) / 2
            fig3.add_shape(type="line",
                x0=mu - fwhm/2, x1=mu + fwhm/2, y0=half_max, y1=half_max,
                line=dict(color="red", width=2))
            fig3.add_annotation(
                x=mu, y=half_max * 1.05,
                text=f"FWHM = {fwhm:.5f}°",
                showarrow=False, font=dict(size=13, color="red"),
            )
            fig3.update_layout(
                xaxis_title="2θ (°)", yaxis_title="Intensity",
                height=380, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="white"),
                xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Stats card
            stat_cols = st.columns(4)
            stat_cols[0].metric("2θ (°)",        f"{r['2θ_fit (°)']:.4f}")
            stat_cols[1].metric("FWHM (°)",       f"{r['FWHM (°)']:.5f}")
            stat_cols[2].metric("R²",             f"{r['R²']:.4f}")
            stat_cols[3].metric("Crystallite D",  f"{r['Crystallite D (nm)']} nm")
        else:
            st.info("No fitted peaks to display.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Peak fitting: scipy.optimize.curve_fit  |  "
    "Peak detection: scipy.signal.find_peaks  |  "
    "Crystallite size: Scherrer equation D=Kλ/(β·cosθ)  |  "
    "Williamson-Hall: β·cosθ = Kλ/D + 4ε·sinθ"
)