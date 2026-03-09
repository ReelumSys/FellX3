"""
Rietveld Refinement App
=======================
Streamlit-basierte Anwendung zur Rietveld-Verfeinerung von Pulverdiffraktogrammen.

Abhängigkeiten installieren:
    pip install streamlit numpy scipy matplotlib gemmi pandas

Starten:
    streamlit run rietveld_app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import re
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rietveld Refinement",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg: #0d0f14;
    --card: #13161d;
    --border: #1e2330;
    --accent: #00e5ff;
    --accent2: #ff6b35;
    --text: #e8eaf0;
    --muted: #6b7080;
    --success: #00c896;
    --warn: #ffcc00;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    border-radius: 6px;
    padding: 6px 16px;
}

.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #000 !important;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 4px 0;
}

.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--accent);
}

.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.status-ok  { color: var(--success); }
.status-warn{ color: var(--warn);    }
.status-err { color: var(--accent2); }

.stSlider > div > div > div > div { background: var(--accent) !important; }
.stNumberInput input { background: var(--card) !important; color: var(--text) !important;
                       border: 1px solid var(--border) !important; font-family: 'JetBrains Mono', monospace; }
.stButton > button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    border-radius: 6px;
    transition: all 0.2s;
}
.stButton > button:hover { background: var(--accent); color: #000; }

.run-btn > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 700;
    font-size: 1rem;
    padding: 12px 32px;
    border: none !important;
    border-radius: 8px;
    width: 100%;
}

.info-box {
    background: var(--card);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; }
div[data-testid="stMetric"] div   { color: var(--accent) !important; font-family: 'JetBrains Mono'; }

[data-testid="stFileUploader"] {
    background: var(--card);
    border: 1px dashed var(--border);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: CIF Parser (light-weight, no external dep)
# ─────────────────────────────────────────────────────────────────────────────
def parse_cif(content: str) -> dict:
    """Minimal CIF parser – extracts cell parameters, space group, atom sites."""
    data = {}

    def extract(key):
        pattern = rf"_{key}\s+([\S]+)"
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            val = m.group(1).replace("(", "").replace(")", "")
            try:
                return float(val)
            except ValueError:
                return val
        return None

    data["a"]     = extract("cell_length_a")     or 5.0
    data["b"]     = extract("cell_length_b")     or 5.0
    data["c"]     = extract("cell_length_c")     or 5.0
    data["alpha"] = extract("cell_angle_alpha")  or 90.0
    data["beta"]  = extract("cell_angle_beta")   or 90.0
    data["gamma"] = extract("cell_angle_gamma")  or 90.0
    data["volume"]= extract("cell_volume")

    sg_name = extract("symmetry_space_group_name_H-M") or extract("space_group_name_H-M_alt")
    sg_num  = extract("symmetry_Int_Tables_number") or extract("space_group_IT_number")
    data["space_group"] = str(sg_name) if sg_name else (f"No. {int(sg_num)}" if sg_num else "P 1")
    data["sg_number"]   = int(sg_num) if sg_num else 1

    # Atom sites
    atoms = []
    loop_match = re.search(
        r"loop_.*?_atom_site_label.*?(?=loop_|\Z)", content, re.DOTALL | re.IGNORECASE
    )
    if loop_match:
        block = loop_match.group(0)
        headers = re.findall(r"_atom_site_(\w+)", block, re.IGNORECASE)
        rows = re.findall(
            r"^\s+([A-Za-z][A-Za-z0-9]*\d*\s+.*?)$", block, re.MULTILINE
        )
        for row in rows:
            parts = row.split()
            if len(parts) >= 5:
                atom = {"label": parts[0], "type": re.sub(r"\d", "", parts[0])}
                try:
                    idx_x = next(i for i, h in enumerate(headers) if "fract_x" in h.lower() or h.lower() == "x")
                    atom["x"] = float(parts[idx_x + 1].split("(")[0])
                    atom["y"] = float(parts[idx_x + 2].split("(")[0])
                    atom["z"] = float(parts[idx_x + 3].split("(")[0])
                except Exception:
                    atom["x"], atom["y"], atom["z"] = 0.0, 0.0, 0.0
                try:
                    atom["occ"] = float(parts[-2].split("(")[0])
                except Exception:
                    atom["occ"] = 1.0
                try:
                    atom["Biso"] = float(parts[-1].split("(")[0])
                except Exception:
                    atom["Biso"] = 1.0
                atoms.append(atom)
    data["atoms"] = atoms
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Diffractogram Parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_diffractogram(content: str) -> tuple:
    """Parse 2θ / intensity columns from xy, dat, or xye files."""
    two_theta, intensity, error = [], [], []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        parts = line.split()
        try:
            tt = float(parts[0])
            ii = float(parts[1])
            two_theta.append(tt)
            intensity.append(ii)
            if len(parts) >= 3:
                error.append(float(parts[2]))
            else:
                error.append(np.sqrt(max(ii, 1.0)))
        except (ValueError, IndexError):
            continue
    return np.array(two_theta), np.array(intensity), np.array(error)


# ─────────────────────────────────────────────────────────────────────────────
# Physics: d-spacing, 2θ, reflection conditions
# ─────────────────────────────────────────────────────────────────────────────
def d_cubic(h, k, l, a):
    return a / np.sqrt(h**2 + k**2 + l**2)

def d_tetragonal(h, k, l, a, c):
    return 1.0 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)

def d_orthorhombic(h, k, l, a, b, c):
    return 1.0 / np.sqrt(h**2/a**2 + k**2/b**2 + l**2/c**2)

def d_hexagonal(h, k, l, a, c):
    return 1.0 / np.sqrt(4/3 * (h**2 + h*k + k**2)/a**2 + l**2/c**2)

def d_monoclinic(h, k, l, a, b, c, beta_deg):
    beta = np.radians(beta_deg)
    sin_b, cos_b = np.sin(beta), np.cos(beta)
    return 1.0 / np.sqrt(
        h**2/(a*sin_b)**2 + k**2/b**2 + l**2/(c*sin_b)**2
        - 2*h*l*cos_b/(a*c*sin_b**2)
    )

def get_crystal_system(sg_number: int) -> str:
    if sg_number <= 2:   return "triclinic"
    if sg_number <= 15:  return "monoclinic"
    if sg_number <= 74:  return "orthorhombic"
    if sg_number <= 142: return "tetragonal"
    if sg_number <= 167: return "trigonal/hexagonal"
    if sg_number <= 194: return "hexagonal"
    return "cubic"

def compute_d_spacing(h, k, l, cell: dict) -> float:
    a, b, c = cell["a"], cell["b"], cell["c"]
    alpha, beta, gamma = cell["alpha"], cell["beta"], cell["gamma"]
    sg = cell.get("sg_number", 1)
    system = get_crystal_system(sg)
    try:
        if system == "cubic":
            return d_cubic(h, k, l, a)
        elif system == "tetragonal":
            return d_tetragonal(h, k, l, a, c)
        elif system == "orthorhombic":
            return d_orthorhombic(h, k, l, a, b, c)
        elif system in ("hexagonal", "trigonal/hexagonal"):
            return d_hexagonal(h, k, l, a, c)
        elif system == "monoclinic":
            return d_monoclinic(h, k, l, a, b, c, beta)
        else:
            # triclinic general
            al, be, ga = np.radians(alpha), np.radians(beta), np.radians(gamma)
            V = a*b*c*np.sqrt(
                1 - np.cos(al)**2 - np.cos(be)**2 - np.cos(ga)**2
                + 2*np.cos(al)*np.cos(be)*np.cos(ga)
            )
            s11 = (b*c*np.sin(al))**2
            s22 = (a*c*np.sin(be))**2
            s33 = (a*b*np.sin(ga))**2
            s12 = a*b*c**2*(np.cos(al)*np.cos(be) - np.cos(ga))
            s23 = a**2*b*c*(np.cos(be)*np.cos(ga) - np.cos(al))
            s13 = a*b**2*c*(np.cos(ga)*np.cos(al) - np.cos(be))
            return V / np.sqrt(s11*h**2 + s22*k**2 + s33*l**2
                               + 2*s12*h*k + 2*s23*k*l + 2*s13*h*l)
    except Exception:
        return 0.0

def generate_reflections(cell: dict, wavelength: float, two_theta_max: float = 90.0,
                         hkl_max: int = 8):
    """Generate hkl list with 2θ positions."""
    reflections = []
    for h in range(-hkl_max, hkl_max+1):
        for k in range(-hkl_max, hkl_max+1):
            for l in range(-hkl_max, hkl_max+1):
                if h == 0 and k == 0 and l == 0:
                    continue
                d = compute_d_spacing(h, k, l, cell)
                if d <= 0:
                    continue
                sin_theta = wavelength / (2 * d)
                if abs(sin_theta) > 1:
                    continue
                two_theta = np.degrees(2 * np.arcsin(sin_theta))
                if 0 < two_theta <= two_theta_max:
                    reflections.append({
                        "h": h, "k": k, "l": l,
                        "d": d, "two_theta": two_theta,
                        "multiplicity": 1
                    })
    # Deduplicate (group close 2θ within 0.001°) and accumulate multiplicity
    if not reflections:
        return reflections
    reflections.sort(key=lambda x: x["two_theta"])
    merged = [reflections[0]]
    for r in reflections[1:]:
        if abs(r["two_theta"] - merged[-1]["two_theta"]) < 0.005:
            merged[-1]["multiplicity"] += 1
        else:
            merged.append(r)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Peak profile: pseudo-Voigt
# ─────────────────────────────────────────────────────────────────────────────
def pseudo_voigt(two_theta_arr, two_theta_k, Fwhm, eta=0.5):
    """eta: mixing parameter (0=Gauss, 1=Lorentz)."""
    x = two_theta_arr - two_theta_k
    sigma = Fwhm / (2 * np.sqrt(2 * np.log(2)))
    gauss = np.exp(-x**2 / (2 * sigma**2))
    lorentz = 1 / (1 + (x / (Fwhm/2))**2)
    return eta * lorentz + (1 - eta) * gauss

def caglioti_fwhm(two_theta_deg, U, V, W):
    """Caglioti FWHM formula: FWHM² = U·tan²θ + V·tanθ + W"""
    theta = np.radians(two_theta_deg / 2)
    tan_t = np.tan(theta)
    val = U * tan_t**2 + V * tan_t + W
    return np.sqrt(max(val, 1e-8))

def lorentz_polarization(two_theta_deg: float) -> float:
    theta = np.radians(two_theta_deg / 2)
    cos2t = np.cos(np.radians(two_theta_deg))
    return (1 + cos2t**2) / (np.sin(theta)**2 * np.cos(theta) + 1e-12)

# ─────────────────────────────────────────────────────────────────────────────
# Background model: Chebyshev polynomial
# ─────────────────────────────────────────────────────────────────────────────
def chebyshev_background(two_theta_arr, coeffs):
    tt_norm = 2 * (two_theta_arr - two_theta_arr.min()) / (two_theta_arr.max() - two_theta_arr.min()) - 1
    result = np.zeros_like(two_theta_arr)
    for i, c in enumerate(coeffs):
        if i == 0:
            result += c
        elif i == 1:
            result += c * tt_norm
        else:
            result += c * np.polynomial.chebyshev.chebval(tt_norm, [0]*i + [1])
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Full pattern calculation
# ─────────────────────────────────────────────────────────────────────────────
def calc_pattern(two_theta_arr, reflections, params: dict) -> np.ndarray:
    """Calculate full powder pattern (no structure factors – intensity ∝ Lp × multiplicity)."""
    U = params.get("U", 0.01)
    V = params.get("V", -0.001)
    W = params.get("W", 0.005)
    eta = params.get("eta", 0.5)
    scale = params.get("scale", 1.0)
    zero_shift = params.get("zero_shift", 0.0)
    Biso_global = params.get("Biso", 1.0)
    bg_coeffs = params.get("bg_coeffs", [0.0] * 6)

    pattern = np.zeros_like(two_theta_arr)
    for r in reflections:
        tt_k = r["two_theta"] + zero_shift
        fwhm = caglioti_fwhm(tt_k, U, V, W)
        lp = lorentz_polarization(tt_k)
        theta_k = np.radians(tt_k / 2)
        dw = np.exp(-2 * Biso_global * (np.sin(theta_k) / params.get("wavelength", 1.54056))**2)
        peak_intensity = scale * r["multiplicity"] * lp * dw
        profile = pseudo_voigt(two_theta_arr, tt_k, fwhm, eta)
        pattern += peak_intensity * profile

    background = chebyshev_background(two_theta_arr, bg_coeffs)
    return pattern + background

# ─────────────────────────────────────────────────────────────────────────────
# R-factors
# ─────────────────────────────────────────────────────────────────────────────
def calc_rfactors(obs, calc, weights=None):
    if weights is None:
        weights = 1.0 / np.maximum(obs, 1.0)
    diff = obs - calc
    Rp  = 100 * np.sum(np.abs(diff)) / np.sum(np.abs(obs))
    Rwp = 100 * np.sqrt(np.sum(weights * diff**2) / np.sum(weights * obs**2))
    chi2 = np.sum(weights * diff**2) / max(len(obs) - 10, 1)
    return Rp, Rwp, chi2

# ─────────────────────────────────────────────────────────────────────────────
# Refinement engine
# ─────────────────────────────────────────────────────────────────────────────
def refine(two_theta, obs, reflections, params_init: dict, refine_flags: dict,
           wavelength: float, n_cycles: int = 5) -> dict:
    """Least-squares refinement using scipy."""
    # Build parameter vector from flags
    param_names = []
    p0 = []
    bounds_lo, bounds_hi = [], []

    def add(name, val, lo, hi):
        if refine_flags.get(name, False):
            param_names.append(name)
            p0.append(val)
            bounds_lo.append(lo)
            bounds_hi.append(hi)

    add("scale",      params_init.get("scale",      1.0),   0.0,   1e8)
    add("zero_shift", params_init.get("zero_shift", 0.0),  -1.0,   1.0)
    add("U",          params_init.get("U",          0.01),  0.0,   5.0)
    add("V",          params_init.get("V",         -0.001), -5.0,  0.0)
    add("W",          params_init.get("W",          0.005), 1e-6,  5.0)
    add("eta",        params_init.get("eta",        0.5),   0.0,   1.0)
    add("Biso",       params_init.get("Biso",       1.0),   0.0,   30.0)
    add("a",          params_init.get("a",          5.0),   0.1,  100.0)
    add("b",          params_init.get("b",          5.0),   0.1,  100.0)
    add("c",          params_init.get("c",          5.0),   0.1,  100.0)
    # Background coefficients
    for i in range(6):
        add(f"bg_{i}", params_init.get("bg_coeffs", [0]*6)[i], -1e6, 1e6)

    if not param_names:
        return {"params": params_init, "Rp": 99.9, "Rwp": 99.9, "chi2": 99.9,
                "message": "No parameters selected for refinement."}

    weights = 1.0 / np.maximum(obs, 1.0)

    def residuals(p):
        params = dict(params_init)  # copy
        params["wavelength"] = wavelength
        for name, val in zip(param_names, p):
            if name.startswith("bg_"):
                idx = int(name.split("_")[1])
                if "bg_coeffs" not in params:
                    params["bg_coeffs"] = [0.0]*6
                params["bg_coeffs"][idx] = val
            elif name in ("a", "b", "c"):
                params[name] = val
                # Update reflections based on new cell (simplified: rebuild would be slow)
            else:
                params[name] = val
        calc = calc_pattern(two_theta, reflections, params)
        return np.sqrt(weights) * (obs - calc)

    result = least_squares(
        residuals, p0,
        bounds=(bounds_lo, bounds_hi),
        method="trf",
        max_nfev=n_cycles * 200,
        ftol=1e-8, xtol=1e-8, gtol=1e-8
    )

    # Update params
    params_out = dict(params_init)
    params_out["wavelength"] = wavelength
    for name, val in zip(param_names, result.x):
        if name.startswith("bg_"):
            idx = int(name.split("_")[1])
            if "bg_coeffs" not in params_out:
                params_out["bg_coeffs"] = [0.0]*6
            params_out["bg_coeffs"][idx] = val
        else:
            params_out[name] = val

    calc_final = calc_pattern(two_theta, reflections, params_out)
    Rp, Rwp, chi2 = calc_rfactors(obs, calc_final, weights)
    return {"params": params_out, "Rp": Rp, "Rwp": Rwp, "chi2": chi2,
            "message": result.message, "calc": calc_final}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d0f14"
CARD_BG   = "#13161d"
ACCENT    = "#00e5ff"
ACCENT2   = "#ff6b35"
SUCCESS   = "#00c896"
MUTED     = "#6b7080"
TEXT      = "#e8eaf0"

def plot_rietveld(two_theta, obs, calc, reflections, title="Rietveld Plot"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [4, 1]},
                                    facecolor=DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
        ax.tick_params(colors=MUTED)

    ax1.plot(two_theta, obs,  color=TEXT,    lw=0.9, label="Observed",   zorder=2)
    ax1.plot(two_theta, calc, color=ACCENT,  lw=1.4, label="Calculated", zorder=3)
    ax1.fill_between(two_theta, obs, calc, alpha=0.15, color=ACCENT2, label="Difference (fill)")

    # Tick marks
    if reflections:
        tt_refl = [r["two_theta"] for r in reflections
                   if two_theta.min() <= r["two_theta"] <= two_theta.max()]
        ax1.vlines(tt_refl, ymin=obs.min()*0.95, ymax=obs.min()*0.85,
                   color=SUCCESS, lw=0.8, alpha=0.7, label="Reflections")

    ax1.set_ylabel("Intensity (a.u.)", color=MUTED)
    ax1.set_title(title, color=TEXT, fontweight="bold", fontsize=13)
    ax1.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)
    ax1.set_xlim(two_theta.min(), two_theta.max())
    ax1.xaxis.set_visible(False)

    diff = obs - calc
    ax2.plot(two_theta, diff, color=ACCENT2, lw=0.8, label="Δ (obs−calc)")
    ax2.axhline(0, color=MUTED, lw=0.6, ls="--")
    ax2.set_xlabel("2θ (°)", color=MUTED)
    ax2.set_ylabel("Δ", color=MUTED)
    ax2.set_xlim(two_theta.min(), two_theta.max())
    ax2.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)

    fig.tight_layout(h_pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_diffractogram_only(two_theta, obs):
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.plot(two_theta, obs, color=ACCENT, lw=0.9)
    ax.set_xlabel("2θ (°)", color=MUTED)
    ax.set_ylabel("Intensity", color=MUTED)
    ax.set_title("Loaded Diffractogram", color=TEXT, fontweight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Session State initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "cell" not in st.session_state:
    st.session_state.cell = {}
if "atoms" not in st.session_state:
    st.session_state.atoms = []
if "two_theta" not in st.session_state:
    st.session_state.two_theta = None
if "obs" not in st.session_state:
    st.session_state.obs = None
if "error" not in st.session_state:
    st.session_state.error = None
if "reflections" not in st.session_state:
    st.session_state.reflections = []
if "params" not in st.session_state:
    st.session_state.params = {
        "scale": 1.0, "zero_shift": 0.0,
        "U": 0.01, "V": -0.001, "W": 0.005,
        "eta": 0.5, "Biso": 1.0,
        "bg_coeffs": [0.0]*6,
    }
if "calc" not in st.session_state:
    st.session_state.calc = None
if "R_vals" not in st.session_state:
    st.session_state.R_vals = None
if "cif_raw" not in st.session_state:
    st.session_state.cif_raw = ""


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 20px 0;'>
  <span style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800; color:#00e5ff; letter-spacing:-1px;'>
    Rietveld Refinement
  </span>
  <span style='font-family:JetBrains Mono,monospace; font-size:0.85rem; color:#6b7080; margin-left:16px;'>
    v1.0 · Powder Diffraction Analysis
  </span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Datei-Import")

    cif_file  = st.file_uploader("CIF-Datei (.cif)", type=["cif"])
    diff_file = st.file_uploader("Diffraktogramm (.xy .dat .xye .txt)", type=["xy","dat","xye","txt","csv"])

    if cif_file:
        content = cif_file.read().decode("utf-8", errors="ignore")
        st.session_state.cif_raw = content
        parsed = parse_cif(content)
        st.session_state.cell = parsed
        st.session_state.atoms = parsed.get("atoms", [])
        # Copy cell params into params
        for k in ("a","b","c","alpha","beta","gamma"):
            st.session_state.params[k] = parsed.get(k, 5.0)
        st.success(f"✓ CIF geladen · {parsed.get('space_group','?')}")

    if diff_file:
        content = diff_file.read().decode("utf-8", errors="ignore")
        tt, obs, err = parse_diffractogram(content)
        if len(tt) > 5:
            st.session_state.two_theta = tt
            st.session_state.obs = obs
            st.session_state.error = err
            # Auto scale
            st.session_state.params["scale"] = float(obs.max()) * 0.01
            st.success(f"✓ Diffraktogramm geladen · {len(tt)} Punkte")
        else:
            st.error("Datei konnte nicht geparsed werden.")

    st.divider()
    st.markdown("### ⚙️ Messparameter")
    wavelength = st.number_input("Wellenlänge λ (Å)", value=1.54056, step=0.00001,
                                  format="%.5f", help="CuKα1: 1.54056, MoKα1: 0.70930")
    st.session_state.params["wavelength"] = wavelength

    hkl_max = st.slider("hkl-Maximum", 2, 15, 8)
    tt_max   = st.slider("2θ-Maximum (°)", 30.0, 150.0, 90.0, 5.0)

    if st.button("🔄 Reflexliste erzeugen"):
        if st.session_state.cell:
            cell = {**st.session_state.cell,
                    "a": st.session_state.params.get("a", st.session_state.cell.get("a",5)),
                    "b": st.session_state.params.get("b", st.session_state.cell.get("b",5)),
                    "c": st.session_state.params.get("c", st.session_state.cell.get("c",5))}
            with st.spinner("Berechne Reflexionen…"):
                refs = generate_reflections(cell, wavelength, tt_max, hkl_max)
            st.session_state.reflections = refs
            st.success(f"{len(refs)} Reflexe erzeugt")
        else:
            st.warning("Bitte zuerst CIF-Datei laden.")

    st.divider()
    st.markdown("### 🎛️ Verfeinerungsoptionen")
    n_cycles = st.slider("Zyklen", 1, 20, 5)
    refine_flags = {}
    cols = st.columns(2)
    flag_items = [
        ("scale",      "Skalierung"),
        ("zero_shift", "Nullpunkt"),
        ("U","U (FWHM)"), ("V","V (FWHM)"), ("W","W (FWHM)"),
        ("eta","η (PV)"),
        ("Biso","B_iso"),
        ("a","a"), ("b","b"), ("c","c"),
    ]
    for i, (key, label) in enumerate(flag_items):
        with cols[i % 2]:
            refine_flags[key] = st.checkbox(label, value=(key in ("scale","zero_shift","W","Biso")),
                                             key=f"flag_{key}")
    refine_bg = st.checkbox("Hintergrund (Cheby.)", value=True)
    for i in range(6):
        refine_flags[f"bg_{i}"] = refine_bg


# ══════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Diffraktogramm", "🔬 Struktur", "⚙️ Parameter", "▶ Verfeinerung", "📋 Ergebnisse"
])

# ────────────────────────────────────────
# TAB 1: Diffraktogramm Vorschau
# ────────────────────────────────────────
with tab1:
    if st.session_state.obs is not None:
        tt = st.session_state.two_theta
        obs = st.session_state.obs

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Punkte",      f"{len(tt)}")
        c2.metric("2θ-Bereich",  f"{tt.min():.2f}° – {tt.max():.2f}°")
        c3.metric("I_max",       f"{obs.max():.0f}")
        c4.metric("I_min",       f"{obs.min():.0f}")

        # Optional: trim range
        col_a, col_b = st.columns(2)
        with col_a:
            tt_lo = st.number_input("2θ von (°)", value=float(tt.min()), step=0.1)
        with col_b:
            tt_hi = st.number_input("2θ bis (°)", value=float(tt.max()), step=0.1)

        mask = (tt >= tt_lo) & (tt <= tt_hi)
        if mask.sum() > 5:
            st.session_state.two_theta = tt[mask]
            st.session_state.obs = obs[mask]
            if st.session_state.error is not None:
                st.session_state.error = st.session_state.error[mask]

        buf = plot_diffractogram_only(st.session_state.two_theta, st.session_state.obs)
        st.image(buf, use_container_width=True)

        # Peak finder
        if st.checkbox("🔍 Peaks automatisch finden"):
            height_thr = st.slider("Min. Intensität (%)", 1, 50, 5) / 100 * obs.max()
            peaks, props = find_peaks(st.session_state.obs, height=height_thr, distance=5)
            peak_tt = st.session_state.two_theta[peaks]
            df_peaks = pd.DataFrame({
                "2θ (°)": peak_tt,
                "d (Å)": wavelength / (2 * np.sin(np.radians(peak_tt / 2)))
            })
            st.dataframe(df_peaks.round(4), use_container_width=True)
    else:
        st.info("Lade ein Diffraktogramm in der Seitenleiste.")

# ────────────────────────────────────────
# TAB 2: Strukturparameter
# ────────────────────────────────────────
with tab2:
    if st.session_state.cell:
        cell = st.session_state.cell
        st.markdown(f"""
        <div class='info-box'>
          Raumgruppe: <b>{cell.get('space_group','?')}</b> &nbsp;|&nbsp;
          Nr. {cell.get('sg_number','?')} &nbsp;|&nbsp;
          System: {get_crystal_system(cell.get('sg_number',1)).upper()}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Gitterparameter")
        col1, col2 = st.columns(2)
        with col1:
            a_val = st.number_input("a (Å)", value=float(st.session_state.params.get("a", cell.get("a",5.0))), step=0.001, format="%.5f")
            b_val = st.number_input("b (Å)", value=float(st.session_state.params.get("b", cell.get("b",5.0))), step=0.001, format="%.5f")
            c_val = st.number_input("c (Å)", value=float(st.session_state.params.get("c", cell.get("c",5.0))), step=0.001, format="%.5f")
        with col2:
            alpha_v = st.number_input("α (°)", value=float(cell.get("alpha",90.0)), step=0.01, format="%.4f")
            beta_v  = st.number_input("β (°)", value=float(cell.get("beta", 90.0)), step=0.01, format="%.4f")
            gamma_v = st.number_input("γ (°)", value=float(cell.get("gamma",90.0)), step=0.01, format="%.4f")

        for k, v in [("a",a_val),("b",b_val),("c",c_val),
                      ("alpha",alpha_v),("beta",beta_v),("gamma",gamma_v)]:
            st.session_state.params[k] = v
            st.session_state.cell[k] = v

        # Volume
        a,b,c = a_val, b_val, c_val
        al,be,ga = np.radians(alpha_v), np.radians(beta_v), np.radians(gamma_v)
        vol = a*b*c*np.sqrt(1 - np.cos(al)**2 - np.cos(be)**2 - np.cos(ga)**2
                             + 2*np.cos(al)*np.cos(be)*np.cos(ga))
        st.metric("Volumen (ų)", f"{vol:.3f}")

        if st.session_state.atoms:
            st.markdown("#### Atompositionen")
            atom_data = []
            for at in st.session_state.atoms:
                atom_data.append({
                    "Label": at.get("label","?"), "Typ": at.get("type","?"),
                    "x": at.get("x",0.0), "y": at.get("y",0.0), "z": at.get("z",0.0),
                    "Occ": at.get("occ",1.0), "B_iso": at.get("Biso",1.0)
                })
            df_atoms = pd.DataFrame(atom_data)
            edited = st.data_editor(df_atoms, use_container_width=True, num_rows="dynamic")
            # Write back
            atoms_new = []
            for _, row in edited.iterrows():
                atoms_new.append({
                    "label": row["Label"], "type": row["Typ"],
                    "x": row["x"], "y": row["y"], "z": row["z"],
                    "occ": row["Occ"], "Biso": row["B_iso"]
                })
            st.session_state.atoms = atoms_new

        if st.session_state.reflections:
            st.markdown("#### Reflexliste")
            ref_df = pd.DataFrame([{
                "h": r["h"], "k": r["k"], "l": r["l"],
                "d (Å)": round(r["d"],4), "2θ (°)": round(r["two_theta"],4),
                "m": r["multiplicity"]
            } for r in st.session_state.reflections])
            st.dataframe(ref_df, use_container_width=True, height=300)
    else:
        st.info("Lade eine CIF-Datei in der Seitenleiste.")

# ────────────────────────────────────────
# TAB 3: Instrument & Profil Parameter
# ────────────────────────────────────────
with tab3:
    st.markdown("#### Profilparameter (Caglioti / Pseudo-Voigt)")
    col1, col2, col3 = st.columns(3)
    with col1:
        U_v = st.number_input("U", value=float(st.session_state.params.get("U",0.01)),
                               step=0.001, format="%.4f",
                               help="FWHM² = U·tan²θ + V·tanθ + W")
    with col2:
        V_v = st.number_input("V", value=float(st.session_state.params.get("V",-0.001)),
                               step=0.001, format="%.4f")
    with col3:
        W_v = st.number_input("W", value=float(st.session_state.params.get("W",0.005)),
                               step=0.0001, format="%.5f")

    eta_v = st.slider("η (Pseudo-Voigt-Mischung)", 0.0, 1.0,
                       float(st.session_state.params.get("eta",0.5)), 0.01,
                       help="0 = rein Gauß, 1 = rein Lorentz")

    for k,v in [("U",U_v),("V",V_v),("W",W_v),("eta",eta_v)]:
        st.session_state.params[k] = v

    # FWHM preview
    tt_arr = np.linspace(5, 90, 500)
    fwhm_arr = np.array([caglioti_fwhm(tt, U_v, V_v, W_v) for tt in tt_arr])
    fig2, ax2 = plt.subplots(figsize=(8,2.5), facecolor=DARK_BG)
    ax2.set_facecolor(DARK_BG)
    for sp in ax2.spines.values(): sp.set_color(MUTED)
    ax2.tick_params(colors=MUTED)
    ax2.plot(tt_arr, fwhm_arr, color=ACCENT, lw=1.5)
    ax2.set_xlabel("2θ (°)", color=MUTED)
    ax2.set_ylabel("FWHM (°)", color=MUTED)
    ax2.set_title("Caglioti-FWHM-Verlauf", color=TEXT, fontsize=10)
    fig2.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=100, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig2)
    buf2.seek(0)
    st.image(buf2, use_container_width=True)

    st.markdown("#### Skalierung & Nullpunktkorrektur")
    col_s, col_z = st.columns(2)
    with col_s:
        sc_v = st.number_input("Skalierungsfaktor", value=float(st.session_state.params.get("scale",1.0)),
                                step=0.01, format="%.4f")
    with col_z:
        zs_v = st.number_input("Nullpunktverschiebung (°)", value=float(st.session_state.params.get("zero_shift",0.0)),
                                step=0.001, format="%.4f")
    st.session_state.params["scale"]      = sc_v
    st.session_state.params["zero_shift"] = zs_v

    st.markdown("#### Globaler Debye-Waller-Faktor")
    biso_v = st.number_input("B_iso (Å²)", value=float(st.session_state.params.get("Biso",1.0)),
                              step=0.1, format="%.3f")
    st.session_state.params["Biso"] = biso_v

    st.markdown("#### Hintergrund (Chebyshev-Polynome)")
    bg_coeffs = st.session_state.params.get("bg_coeffs", [0.0]*6)
    new_bg = []
    bg_cols = st.columns(3)
    for i in range(6):
        with bg_cols[i % 3]:
            v = st.number_input(f"c_{i}", value=float(bg_coeffs[i]), step=1.0, format="%.2f",
                                 key=f"bg_coeff_{i}")
            new_bg.append(v)
    st.session_state.params["bg_coeffs"] = new_bg

# ────────────────────────────────────────
# TAB 4: Verfeinerung
# ────────────────────────────────────────
with tab4:
    st.markdown("### Verfeinerung starten")

    ready = (
        st.session_state.obs is not None and
        len(st.session_state.reflections) > 0
    )

    if not ready:
        st.warning("Bitte zuerst: (1) CIF laden, (2) Diffraktogramm laden, (3) Reflexliste erzeugen.")
    else:
        st.markdown(f"""
        <div class='info-box'>
          ✓ {len(st.session_state.two_theta)} Messpunkte &nbsp;|&nbsp;
          ✓ {len(st.session_state.reflections)} Reflexe &nbsp;|&nbsp;
          Wellenlänge: {wavelength:.5f} Å
        </div>
        """, unsafe_allow_html=True)

        # Preview before refinement
        with st.expander("👁 Vorschau aktuelles Muster", expanded=True):
            params_prev = dict(st.session_state.params)
            params_prev["wavelength"] = wavelength
            calc_prev = calc_pattern(st.session_state.two_theta, st.session_state.reflections, params_prev)
            Rp0, Rwp0, chi0 = calc_rfactors(st.session_state.obs, calc_prev)
            st.markdown(f"**R_p = {Rp0:.2f}%  |  R_wp = {Rwp0:.2f}%  |  χ² = {chi0:.3f}**")
            buf_prev = plot_rietveld(st.session_state.two_theta, st.session_state.obs,
                                     calc_prev, st.session_state.reflections, "Vorschau")
            st.image(buf_prev, use_container_width=True)

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_btn = st.button("▶ Verfeinerung starten", key="run_refine")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_btn:
            params_in = dict(st.session_state.params)
            params_in["wavelength"] = wavelength
            with st.spinner(f"Verfeinere ({n_cycles} Zyklen)…"):
                result = refine(
                    st.session_state.two_theta,
                    st.session_state.obs,
                    st.session_state.reflections,
                    params_in,
                    refine_flags,
                    wavelength,
                    n_cycles
                )
            st.session_state.params.update(result["params"])
            st.session_state.R_vals = (result["Rp"], result["Rwp"], result["chi2"])
            if "calc" in result:
                st.session_state.calc = result["calc"]
            st.success(f"✓ Verfeinerung abgeschlossen — {result['message']}")

        if st.session_state.calc is not None:
            Rp, Rwp, chi2 = st.session_state.R_vals
            m1, m2, m3 = st.columns(3)
            m1.metric("R_p (%)",  f"{Rp:.3f}")
            m2.metric("R_wp (%)", f"{Rwp:.3f}")
            m3.metric("χ²",       f"{chi2:.4f}")
            buf_ref = plot_rietveld(st.session_state.two_theta, st.session_state.obs,
                                    st.session_state.calc, st.session_state.reflections,
                                    "Rietveld-Verfeinerung")
            st.image(buf_ref, use_container_width=True)

# ────────────────────────────────────────
# TAB 5: Ergebnisse & Export
# ────────────────────────────────────────
with tab5:
    st.markdown("### Verfeinerungsergebnisse")

    if st.session_state.R_vals:
        Rp, Rwp, chi2 = st.session_state.R_vals
        col1, col2, col3 = st.columns(3)
        status = "status-ok" if Rwp < 10 else ("status-warn" if Rwp < 20 else "status-err")
        for col, label, val in [(col1,"R_p (%)",f"{Rp:.4f}"),
                                 (col2,"R_wp (%)",f"{Rwp:.4f}"),
                                 (col3,"χ²",f"{chi2:.5f}")]:
            col.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>{label}</div>
              <div class='metric-val {status}'>{val}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        p = st.session_state.params
        st.markdown("#### Verfeinerte Strukturparameter")
        st.markdown(f"""
        | Parameter | Wert |
        |-----------|------|
        | a (Å) | `{p.get('a','-'):.5f}` |
        | b (Å) | `{p.get('b','-'):.5f}` |
        | c (Å) | `{p.get('c','-'):.5f}` |
        | α (°) | `{p.get('alpha','-'):.4f}` |
        | β (°) | `{p.get('beta','-'):.4f}` |
        | γ (°) | `{p.get('gamma','-'):.4f}` |
        | Skalierung | `{p.get('scale',0):.5f}` |
        | Nullpunkt (°) | `{p.get('zero_shift',0):.5f}` |
        | U | `{p.get('U',0):.5f}` |
        | V | `{p.get('V',0):.5f}` |
        | W | `{p.get('W',0):.5f}` |
        | η | `{p.get('eta',0):.4f}` |
        | B_iso (Å²) | `{p.get('Biso',0):.4f}` |
        """)

        # Export
        st.divider()
        st.markdown("#### Export")
        if st.session_state.calc is not None:
            out_df = pd.DataFrame({
                "2theta": st.session_state.two_theta,
                "obs":    st.session_state.obs,
                "calc":   st.session_state.calc,
                "diff":   st.session_state.obs - st.session_state.calc,
            })
            csv = out_df.to_csv(index=False).encode()
            st.download_button("⬇ Muster als CSV", csv, "rietveld_pattern.csv", "text/csv")

        # Summary text
        summary = f"""Rietveld Refinement Summary
============================
Raumgruppe : {st.session_state.cell.get('space_group','?')}
Wellenlänge: {wavelength:.5f} Å

Gitterparameter:
  a = {p.get('a',0):.5f} Å
  b = {p.get('b',0):.5f} Å
  c = {p.get('c',0):.5f} Å
  α = {p.get('alpha',0):.4f}°
  β = {p.get('beta',0):.4f}°
  γ = {p.get('gamma',0):.4f}°

Gütefaktoren:
  Rp  = {Rp:.4f} %
  Rwp = {Rwp:.4f} %
  χ²  = {chi2:.5f}
"""
        st.download_button("⬇ Summary als TXT", summary.encode(), "rietveld_summary.txt", "text/plain")
    else:
        st.info("Noch keine Verfeinerung durchgeführt. Gehe zu Tab ▶ Verfeinerung.")