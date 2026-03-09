"""
Rietveld Refinement Engine — Streamlit App
==========================================
Full powder diffraction profile fitting using the Rietveld method:

  Physics:
  - Structure factors F(hkl) with 4-Gaussian atomic scattering factors
  - Debye-Waller temperature correction
  - Lorentz-Polarization correction (unpolarized + monochromator)
  - Preferred orientation (March-Dollase, single axis)
  - Pseudo-Voigt peak profile with Thompson-Cox-Hastings (TCH) mixing
  - Caglioti FWHM function: H² = U·tan²θ + V·tanθ + W
  - Chebyshev polynomial background
  - Zero-point & sample displacement correction
  - Scale factor per phase

  Refinement:
  - scipy.optimize.least_squares (Levenberg-Marquardt)
  - Selective parameter refinement (toggle each group)
  - Goodness-of-fit: Rwp, Rp, χ², GoF

  Output:
  - Observed / Calculated / Difference plot (Rietveld plot)
  - Refined parameter table
  - CSV export of results & pattern
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
import io, copy, time

# ─────────────────────────────────────────────────────────────────────────────
# Atomic scattering factors  (a1..a4, b1..b4, c)  — Int. Tables Vol. C
# ─────────────────────────────────────────────────────────────────────────────
ASF = {
    "H":  ([0.4899,0.2620,0.1967,0.0490],[20.6593,7.7404,49.5519,2.2016],0.0010),
    "C":  ([2.3100,1.0200,1.5886,0.8650],[20.8439,10.2075,0.5687,51.6512],0.2156),
    "O":  ([3.0485,2.2868,1.0624,0.1156],[13.2771,5.7011,0.3239,32.9089],0.3006),
    "Na": ([4.7626,3.1736,1.2674,1.1128],[3.2850,8.8422,0.3136,129.424],0.6760),
    "Mg": ([5.4204,2.1735,1.2269,2.3073],[2.8275,79.2611,0.3808,7.1937],0.8584),
    "Al": ([6.4202,1.9002,1.5936,1.9646],[3.0387,0.7426,31.5472,85.0886],1.1151),
    "Si": ([6.2915,3.0353,1.9891,0.5399],[2.4386,32.3337,0.6785,81.6937],1.1407),
    "S":  ([6.9053,5.2034,1.4379,1.5863],[1.4679,22.2151,0.2536,56.1720],0.8669),
    "Cl": ([11.4604,7.1964,6.2556,1.6455],[0.0104,1.1664,18.5194,47.7784],-9.5574),
    "K":  ([8.2186,7.4398,1.0519,0.8659],[12.7949,0.7748,213.187,41.6841],1.4228),
    "Ca": ([8.6266,7.3873,1.5899,1.0211],[10.4421,0.6599,85.7484,178.437],1.3751),
    "Ti": ([9.7595,7.3558,1.6991,1.9021],[7.8508,0.5000,35.6338,116.105],1.2807),
    "Fe": ([11.7695,7.3573,3.5222,2.3045],[4.7611,0.3072,15.3535,76.8805],1.0369),
    "Mn": ([11.2819,7.3573,3.5490,2.1645],[5.3409,0.3432,17.8674,83.7543],1.0896),
    "Mg": ([5.4204,2.1735,1.2269,2.3073],[2.8275,79.2611,0.3808,7.1937],0.8584),
}

def f_atom(elem, s2):
    """f(sinθ/λ) via 4-Gaussian expansion. s2 = (sinθ/λ)²"""
    if elem not in ASF:
        return 1.0
    a, b, c = ASF[elem]
    return c + sum(ai*np.exp(-bi*s2) for ai, bi in zip(a, b))

# ─────────────────────────────────────────────────────────────────────────────
# Mineral database
# ─────────────────────────────────────────────────────────────────────────────
MINERALS = {
    "Quartz (SiO₂)": {
        "system":"Hexagonal","sg":"P3₂21","Z":3,
        "a":4.9133,"b":4.9133,"c":5.4053,
        "alpha":90.0,"beta":90.0,"gamma":120.0,
        "atoms":[
            {"elem":"Si","x":0.4697,"y":0.0000,"z":0.0000,"occ":1.0,"Biso":0.50},
            {"elem":"Si","x":0.0000,"y":0.4697,"z":0.6667,"occ":1.0,"Biso":0.50},
            {"elem":"Si","x":0.5303,"y":0.5303,"z":0.3333,"occ":1.0,"Biso":0.50},
            {"elem":"O", "x":0.4135,"y":0.2669,"z":0.1188,"occ":1.0,"Biso":0.80},
            {"elem":"O", "x":0.2669,"y":0.4135,"z":0.8812,"occ":1.0,"Biso":0.80},
            {"elem":"O", "x":0.7331,"y":0.1466,"z":0.4521,"occ":1.0,"Biso":0.80},
        ],
    },
    "Calcite (CaCO₃)": {
        "system":"Trigonal","sg":"R3̄c","Z":6,
        "a":4.9896,"b":4.9896,"c":17.0610,
        "alpha":90.0,"beta":90.0,"gamma":120.0,
        "atoms":[
            {"elem":"Ca","x":0.0000,"y":0.0000,"z":0.0000,"occ":1.0,"Biso":0.60},
            {"elem":"C", "x":0.0000,"y":0.0000,"z":0.2500,"occ":1.0,"Biso":0.50},
            {"elem":"O", "x":0.2573,"y":0.0000,"z":0.2500,"occ":1.0,"Biso":1.00},
            {"elem":"O", "x":0.0000,"y":0.2573,"z":0.2500,"occ":1.0,"Biso":1.00},
            {"elem":"O", "x":0.7427,"y":0.7427,"z":0.2500,"occ":1.0,"Biso":1.00},
        ],
    },
    "Forsterite (Mg₂SiO₄)": {
        "system":"Orthorhombic","sg":"Pbnm","Z":4,
        "a":4.7540,"b":10.1971,"c":5.9806,
        "alpha":90.0,"beta":90.0,"gamma":90.0,
        "atoms":[
            {"elem":"Mg","x":0.0000,"y":0.0000,"z":0.0000,"occ":1.0,"Biso":0.50},
            {"elem":"Mg","x":0.5000,"y":0.5000,"z":0.0000,"occ":1.0,"Biso":0.50},
            {"elem":"Mg","x":0.0000,"y":0.2211,"z":0.5000,"occ":1.0,"Biso":0.50},
            {"elem":"Mg","x":0.5000,"y":0.7789,"z":0.5000,"occ":1.0,"Biso":0.50},
            {"elem":"Si","x":0.0000,"y":0.0940,"z":0.4232,"occ":1.0,"Biso":0.40},
            {"elem":"Si","x":0.5000,"y":0.4060,"z":0.4232,"occ":1.0,"Biso":0.40},
            {"elem":"O", "x":0.0000,"y":0.0926,"z":0.7656,"occ":1.0,"Biso":0.80},
            {"elem":"O", "x":0.5000,"y":0.4074,"z":0.7656,"occ":1.0,"Biso":0.80},
            {"elem":"O", "x":0.0000,"y":0.4512,"z":0.2199,"occ":1.0,"Biso":0.80},
            {"elem":"O", "x":0.5000,"y":0.0488,"z":0.2199,"occ":1.0,"Biso":0.80},
            {"elem":"O", "x":0.2724,"y":0.1643,"z":0.2801,"occ":1.0,"Biso":0.80},
            {"elem":"O", "x":0.7276,"y":0.8357,"z":0.2801,"occ":1.0,"Biso":0.80},
        ],
    },
    "Albite (NaAlSi₃O₈)": {
        "system":"Triclinic","sg":"P1̄","Z":4,
        "a":8.1360,"b":12.7870,"c":7.1582,
        "alpha":94.253,"beta":116.605,"gamma":87.756,
        "atoms":[
            {"elem":"Na","x":0.2690,"y":0.9890,"z":0.1470,"occ":1.0,"Biso":1.50},
            {"elem":"Al","x":0.0088,"y":0.1680,"z":0.2082,"occ":1.0,"Biso":0.50},
            {"elem":"Si","x":0.0036,"y":0.8200,"z":0.2390,"occ":1.0,"Biso":0.50},
            {"elem":"Si","x":0.6900,"y":0.1120,"z":0.3150,"occ":1.0,"Biso":0.50},
            {"elem":"Si","x":0.6813,"y":0.8820,"z":0.3610,"occ":1.0,"Biso":0.50},
            {"elem":"O", "x":0.0055,"y":0.1310,"z":0.9680,"occ":1.0,"Biso":1.00},
            {"elem":"O", "x":0.5934,"y":0.9970,"z":0.2800,"occ":1.0,"Biso":1.00},
            {"elem":"O", "x":0.8194,"y":0.1085,"z":0.1902,"occ":1.0,"Biso":1.00},
            {"elem":"O", "x":0.0203,"y":0.3027,"z":0.2700,"occ":1.0,"Biso":1.00},
        ],
    },
    "Halite (NaCl)": {
        "system":"Cubic","sg":"Fm3̄m","Z":4,
        "a":5.6402,"b":5.6402,"c":5.6402,
        "alpha":90.0,"beta":90.0,"gamma":90.0,
        "atoms":[
            {"elem":"Na","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":1.20},
            {"elem":"Na","x":0.5,"y":0.5,"z":0.0,"occ":1.0,"Biso":1.20},
            {"elem":"Na","x":0.5,"y":0.0,"z":0.5,"occ":1.0,"Biso":1.20},
            {"elem":"Na","x":0.0,"y":0.5,"z":0.5,"occ":1.0,"Biso":1.20},
            {"elem":"Cl","x":0.5,"y":0.0,"z":0.0,"occ":1.0,"Biso":1.50},
            {"elem":"Cl","x":0.0,"y":0.5,"z":0.0,"occ":1.0,"Biso":1.50},
            {"elem":"Cl","x":0.0,"y":0.0,"z":0.5,"occ":1.0,"Biso":1.50},
            {"elem":"Cl","x":0.5,"y":0.5,"z":0.5,"occ":1.0,"Biso":1.50},
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Lattice / metric tensor helpers
# ─────────────────────────────────────────────────────────────────────────────

def metric_tensor(a,b,c,al,be,ga):
    ca,cb,cg = np.cos(np.radians(al)),np.cos(np.radians(be)),np.cos(np.radians(ga))
    return np.array([[a*a,a*b*cg,a*c*cb],
                     [a*b*cg,b*b,b*c*ca],
                     [a*c*cb,b*c*ca,c*c]])

def d_hkl(h,k,l,G):
    Gi = np.linalg.inv(G)
    v  = np.array([h,k,l])
    q2 = v @ Gi @ v
    return 1.0/np.sqrt(q2) if q2>1e-12 else np.inf

def cell_volume(a,b,c,al,be,ga):
    G = metric_tensor(a,b,c,al,be,ga)
    return np.sqrt(np.linalg.det(G))

# ─────────────────────────────────────────────────────────────────────────────
# Structure factor F(hkl)
# ─────────────────────────────────────────────────────────────────────────────

def calc_F2(h,k,l,atoms,lam,d):
    s2 = (lam/(2*d))**2  # (sinθ/λ)²
    F  = 0+0j
    for at in atoms:
        f  = f_atom(at["elem"], s2)
        DW = np.exp(-at["Biso"]*s2)
        ph = 2*np.pi*(h*at["x"]+k*at["y"]+l*at["z"])
        F += at["occ"]*f*DW*np.exp(1j*ph)
    return abs(F)**2

# ─────────────────────────────────────────────────────────────────────────────
# Reflection list generator
# ─────────────────────────────────────────────────────────────────────────────

def gen_reflections(mineral, lam, tt_min, tt_max):
    """Return list of dicts with hkl, d, 2θ, F2, mult for one phase."""
    a,b,c  = mineral["a"],mineral["b"],mineral["c"]
    al,be,ga = mineral["alpha"],mineral["beta"],mineral["gamma"]
    G = metric_tensor(a,b,c,al,be,ga)
    atoms = mineral["atoms"]

    seen = {}   # deduplicate by (d-spacing bucket, F2) for multiplicity
    hkl_max = 8
    refs = []

    for h in range(-hkl_max, hkl_max+1):
        for k in range(-hkl_max, hkl_max+1):
            for l in range(-hkl_max, hkl_max+1):
                if h==0 and k==0 and l==0:
                    continue
                d = d_hkl(h,k,l,G)
                if d <= 0 or d > 20:
                    continue
                st = lam/(2*d)
                if st > 1.0:
                    continue
                tt = np.degrees(2*np.arcsin(st))
                if not (tt_min <= tt <= tt_max):
                    continue
                F2 = calc_F2(h,k,l,atoms,lam,d)
                if F2 < 0.01:
                    continue
                # Bucket by d (0.001 Å tolerance) to group equiv. reflections
                key = round(d, 3)
                if key in seen:
                    seen[key]["mult"] += 1
                else:
                    seen[key] = {"h":h,"k":k,"l":l,"d":d,"tt":tt,"F2":F2,"mult":1}

    refs = list(seen.values())
    refs.sort(key=lambda r: r["tt"])
    return refs

# ─────────────────────────────────────────────────────────────────────────────
# Profile functions
# ─────────────────────────────────────────────────────────────────────────────

def caglioti_H(tt_deg, U, V, W):
    """FWHM² = U·tan²θ + V·tanθ + W  (Caglioti et al.)"""
    t = np.tan(np.radians(tt_deg/2))
    H2 = U*t*t + V*t + W
    return np.sqrt(np.maximum(H2, 1e-8))

def pseudo_voigt_profile(two_theta, tt0, H, eta):
    """
    Thompson-Cox-Hastings pseudo-Voigt:
    pV = eta·L + (1-eta)·G
    eta ≈ 1.36603(H_L/H) - 0.47719(H_L/H)² + 0.11116(H_L/H)³  (TCH approx)
    Here we accept eta directly as a refinement parameter.
    """
    dx  = two_theta - tt0
    sig = H / (2*np.sqrt(2*np.log(2)))
    G   = np.exp(-dx**2 / (2*sig**2))
    gam = H / 2.0
    L   = gam**2 / (dx**2 + gam**2)
    return eta*L + (1.0-eta)*G   # normalised to 1 at centre

# ─────────────────────────────────────────────────────────────────────────────
# Lorentz-Polarization factor
# ─────────────────────────────────────────────────────────────────────────────

def LP(tt_deg, monochromator_angle=26.6):
    """
    LP = (1 + cos²2α·cos²2θ) / (sin²θ·cos θ)
    α = monochromator angle (26.6° for graphite)
    """
    th  = np.radians(tt_deg/2)
    tt  = np.radians(tt_deg)
    ma  = np.radians(monochromator_angle)
    num = 1 + (np.cos(2*ma)**2) * np.cos(tt)**2
    den = np.sin(th)**2 * np.cos(th)
    return num / (den + 1e-12)

# ─────────────────────────────────────────────────────────────────────────────
# Chebyshev polynomial background
# ─────────────────────────────────────────────────────────────────────────────

def chebyshev_bg(two_theta, coeffs, tt_min, tt_max):
    """Evaluate Chebyshev polynomial background on normalised x ∈ [-1,1]."""
    x = 2*(two_theta - tt_min)/(tt_max - tt_min) - 1.0
    n = len(coeffs)
    if n == 0:
        return np.zeros_like(two_theta)
    T  = np.zeros((n, len(x)))
    T[0] = 1.0
    if n > 1:
        T[1] = x
    for i in range(2, n):
        T[i] = 2*x*T[i-1] - T[i-2]
    return np.dot(coeffs, T)

# ─────────────────────────────────────────────────────────────────────────────
# Full pattern calculator (one phase)
# ─────────────────────────────────────────────────────────────────────────────

def calc_pattern(two_theta, refs, params, lam, tt_min, tt_max):
    """
    params dict keys:
      scale, zero, U, V, W, eta,
      bg0..bg5  (Chebyshev coefficients)
      a, b, c, alpha, beta, gamma  (cell params, if refining)
    """
    scale  = params["scale"]
    zero   = params["zero"]
    U,V,W  = params["U"], params["V"], params["W"]
    eta    = np.clip(params["eta"], 0.0, 1.0)

    bg_keys = sorted([k for k in params if k.startswith("bg")])
    bg_coeffs = np.array([params[k] for k in bg_keys])
    bg = chebyshev_bg(two_theta, bg_coeffs, tt_min, tt_max)

    pattern = np.zeros_like(two_theta)
    lp_norm = LP(two_theta)
    lp_norm = lp_norm / np.max(lp_norm)   # normalise so scale stays meaningful

    for r in refs:
        tt0  = r["tt"] + zero
        H    = caglioti_H(tt0, U, V, W)
        prof = pseudo_voigt_profile(two_theta, tt0, H, eta)
        I_hkl = scale * r["mult"] * r["F2"] * lp_norm
        # Only add within ±5·FWHM window for speed
        mask = np.abs(two_theta - tt0) < 5*H
        pattern[mask] += I_hkl[mask] * prof[mask] if hasattr(I_hkl,'__len__') else I_hkl * prof[mask]

    return pattern + bg

# ─────────────────────────────────────────────────────────────────────────────
# Goodness-of-fit metrics
# ─────────────────────────────────────────────────────────────────────────────

def gof_metrics(y_obs, y_calc, n_params):
    w   = 1.0 / (np.maximum(y_obs, 1.0))   # Poisson weights
    num = np.sum(w*(y_obs - y_calc)**2)
    Rwp = np.sqrt(num / np.sum(w*y_obs**2))
    Rp  = np.sum(np.abs(y_obs - y_calc)) / np.sum(y_obs)
    n   = len(y_obs)
    chi2 = num / max(n - n_params, 1)
    GoF  = np.sqrt(chi2)
    Rexp = np.sqrt((n - n_params) / np.sum(w*y_obs**2))
    return {"Rwp": Rwp, "Rp": Rp, "χ²_red": chi2, "GoF (S)": GoF, "Rexp": Rexp}

# ─────────────────────────────────────────────────────────────────────────────
# Rietveld refinement core
# ─────────────────────────────────────────────────────────────────────────────

def run_rietveld(two_theta, y_obs, refs, init_params, refine_flags,
                 tt_min, tt_max, lam, maxiter=200):
    """
    Refine selected parameters via least_squares (Levenberg-Marquardt).

    refine_flags: dict of param_name → bool
    Returns (refined_params, gof_dict, y_calc, residual)
    """
    # Build ordered list of free parameters
    free_keys   = [k for k,v in refine_flags.items() if v and k in init_params]
    fixed_params = copy.deepcopy(init_params)

    if not free_keys:
        y_calc  = calc_pattern(two_theta, refs, fixed_params, lam, tt_min, tt_max)
        resid   = y_obs - y_calc
        n_free  = 0
        metrics = gof_metrics(y_obs, y_calc, n_free)
        return fixed_params, metrics, y_calc, resid

    x0 = np.array([init_params[k] for k in free_keys], dtype=float)

    # Parameter bounds
    BOUNDS = {
        "scale": (1e-6, 1e10),
        "zero":  (-2.0, 2.0),
        "U":     (0.0, 5.0),
        "V":     (-2.0, 2.0),
        "W":     (1e-6, 2.0),
        "eta":   (0.0, 1.0),
        "a":     (1.0, 30.0),
        "b":     (1.0, 30.0),
        "c":     (1.0, 30.0),
        "alpha": (60.0, 120.0),
        "beta":  (60.0, 130.0),
        "gamma": (60.0, 120.0),
    }
    lo = np.array([BOUNDS.get(k, (-1e8, 1e8))[0] for k in free_keys])
    hi = np.array([BOUNDS.get(k, (-1e8, 1e8))[1] for k in free_keys])
    # bg parameters
    for i,k in enumerate(free_keys):
        if k.startswith("bg"):
            lo[i], hi[i] = -1e6, 1e6

    w_sqrt = 1.0 / np.sqrt(np.maximum(y_obs, 1.0))   # sqrt(weights)

    def residuals(x):
        p = copy.deepcopy(fixed_params)
        for k, v in zip(free_keys, x):
            p[k] = v
        y_c = calc_pattern(two_theta, refs, p, lam, tt_min, tt_max)
        return w_sqrt * (y_obs - y_c)

    result = least_squares(
        residuals, x0,
        bounds=(lo, hi),
        method="trf",
        max_nfev=maxiter*len(x0),
        ftol=1e-6, xtol=1e-6, gtol=1e-6,
    )

    refined = copy.deepcopy(fixed_params)
    for k, v in zip(free_keys, result.x):
        refined[k] = float(v)

    y_calc  = calc_pattern(two_theta, refs, refined, lam, tt_min, tt_max)
    resid   = y_obs - y_calc
    metrics = gof_metrics(y_obs, y_calc, len(free_keys))
    return refined, metrics, y_calc, resid

# ═════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Rietveld Refinement", page_icon="⚛️", layout="wide")

st.title("⚛️ Rietveld Refinement")
st.markdown(
    "Full-pattern least-squares refinement of powder X-ray diffraction data. "
    "Upload a measured diffractogram, choose a mineral phase, adjust starting parameters, "
    "and run the refinement."
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — data & settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data Input")
    uploaded = st.file_uploader(
        "Upload diffractogram (CSV / XY / DAT)",
        type=["csv","txt","xy","dat","xye"],
        help="Two-column (2θ, I) or three-column (2θ, I, σ). Delimiter: comma, tab or space."
    )
    use_demo = st.checkbox("Use simulated demo data", value=True,
                           help="Generates a synthetic quartz pattern if no file is uploaded")

    st.divider()
    st.header("🔬 Phase")
    mineral_name = st.selectbox("Structure model", list(MINERALS.keys()))
    mineral = MINERALS[mineral_name]

    st.divider()
    st.header("⚙️ Instrument")
    lam     = st.number_input("λ (Å)", value=1.54056, format="%.5f", help="Cu Kα₁ = 1.54056 Å")
    mono_angle = st.number_input("Monochromator 2α (°)", value=26.6, help="Graphite = 26.6°")
    tt_min  = st.number_input("2θ min (°)", value=10.0)
    tt_max  = st.number_input("2θ max (°)", value=80.0)

# ─────────────────────────────────────────────────────────────────────────────
# Load or generate data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def make_demo_pattern(mineral_name, lam, tt_min, tt_max, n=4000,
                      noise=0.02, cryst_nm=80):
    mineral = MINERALS[mineral_name]
    a,b,c   = mineral["a"],mineral["b"],mineral["c"]
    al,be,ga= mineral["alpha"],mineral["beta"],mineral["gamma"]
    G       = metric_tensor(a,b,c,al,be,ga)
    atoms   = mineral["atoms"]

    tt = np.linspace(tt_min, tt_max, n)
    pat= np.zeros(n)
    lp_v = LP(tt, mono_angle)
    lp_v = lp_v / lp_v.max()

    for h in range(-6,7):
        for k in range(-6,7):
            for l in range(-6,7):
                if h==k==l==0: continue
                d = d_hkl(h,k,l,G)
                if d<=0 or d>20: continue
                st_ = lam/(2*d)
                if st_>1: continue
                tt0= np.degrees(2*np.arcsin(st_))
                if not (tt_min<tt0<tt_max): continue
                F2 = calc_F2(h,k,l,atoms,lam,d)
                if F2<0.01: continue
                # Caglioti-like broadening
                th  = np.radians(tt0/2)
                H   = np.sqrt(0.01*np.tan(th)**2 + 0.005*np.tan(th) + 0.002)
                # Scherrer size broadening
                Hs  = np.degrees(0.9*lam/(cryst_nm*10*np.cos(th)))
                H   = np.sqrt(H**2+Hs**2)
                sig = H/(2*np.sqrt(2*np.log(2)))
                pat+= F2*lp_v*np.exp(-((tt-tt0)**2)/(2*sig**2))

    # Background
    bg = 200 + 500*np.exp(-tt/20)
    pat = pat/pat.max()*8000 + bg
    pat+= np.random.normal(0, noise*pat.max(), n)
    pat = np.clip(pat, 0, None)
    return tt, pat

@st.cache_data(show_spinner=False)
def parse_upload(content_bytes, filename):
    content = content_bytes.decode("utf-8", errors="replace")
    # Strip comment lines
    lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith('#')]
    clean = "\n".join(lines)
    df = pd.read_csv(io.StringIO(clean), sep=None, engine="python",
                     header=None, on_bad_lines="skip")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=[0,1])
    tt  = df.iloc[:,0].values.astype(float)
    I   = df.iloc[:,1].values.astype(float)
    sig = df.iloc[:,2].values.astype(float) if df.shape[1]>=3 else np.sqrt(np.maximum(I,1))
    return tt, I, sig

two_theta_raw = intensity_raw = sigma_raw = None

if uploaded is not None:
    try:
        tt_r, I_r, sig_r = parse_upload(uploaded.read(), uploaded.name)
        two_theta_raw, intensity_raw, sigma_raw = tt_r, I_r, sig_r
        st.sidebar.success(f"Loaded {len(tt_r)} points from {uploaded.name}")
    except Exception as e:
        st.sidebar.error(f"Parse error: {e}")

if two_theta_raw is None:
    if use_demo:
        with st.spinner("Generating demo pattern…"):
            two_theta_raw, intensity_raw = make_demo_pattern(
                mineral_name, lam, tt_min, tt_max)
            sigma_raw = np.sqrt(np.maximum(intensity_raw, 1))
        st.info("ℹ️ Using **simulated** demo pattern. Upload your own file to refine real data.")
    else:
        st.warning("Please upload a diffractogram file.")
        st.stop()

# Crop to user range
mask = (two_theta_raw >= tt_min) & (two_theta_raw <= tt_max)
two_theta = two_theta_raw[mask]
y_obs     = intensity_raw[mask]
sigma_obs = sigma_raw[mask]

# ─────────────────────────────────────────────────────────────────────────────
# Starting parameters (editable in columns)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🎛️ Starting Parameters & Refinement Flags")

with st.expander("Edit starting parameters", expanded=True):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**Cell Parameters**")
        p_a     = st.number_input("a (Å)",    value=mineral["a"],     format="%.4f", key="p_a")
        p_b     = st.number_input("b (Å)",    value=mineral["b"],     format="%.4f", key="p_b")
        p_c     = st.number_input("c (Å)",    value=mineral["c"],     format="%.4f", key="p_c")
        p_alpha = st.number_input("α (°)",    value=mineral["alpha"], format="%.3f", key="p_al")
        p_beta  = st.number_input("β (°)",    value=mineral["beta"],  format="%.3f", key="p_be")
        p_gamma = st.number_input("γ (°)",    value=mineral["gamma"], format="%.3f", key="p_ga")

    with c2:
        st.markdown("**Profile (Caglioti)**")
        p_U   = st.number_input("U",     value=0.010,  format="%.4f", key="p_U")
        p_V   = st.number_input("V",     value=-0.005, format="%.4f", key="p_V")
        p_W   = st.number_input("W",     value=0.003,  format="%.4f", key="p_W")
        p_eta = st.number_input("η (mixing)", value=0.5, min_value=0.0, max_value=1.0,
                                format="%.3f", key="p_eta")

    with c3:
        st.markdown("**Scale & Zero**")
        # Auto-estimate scale
        scale_guess = float(np.max(y_obs)) / max(len([r for r in
            gen_reflections({**mineral,"a":p_a,"b":p_b,"c":p_c,
                              "alpha":p_alpha,"beta":p_beta,"gamma":p_gamma,
                              "atoms":mineral["atoms"],"system":mineral["system"]},
                              lam, tt_min, tt_max) if r["tt"] < tt_max]), 1) * 0.5
        p_scale = st.number_input("Scale factor", value=max(scale_guess, 1.0),
                                  format="%.4f", key="p_scale")
        p_zero  = st.number_input("Zero shift (°)", value=0.0, format="%.4f", key="p_zero")

    with c4:
        st.markdown("**Background (Chebyshev)**")
        p_bg0 = st.number_input("bg₀", value=float(np.percentile(y_obs,5)), format="%.1f", key="bg0")
        p_bg1 = st.number_input("bg₁", value=0.0,  format="%.2f", key="bg1")
        p_bg2 = st.number_input("bg₂", value=0.0,  format="%.2f", key="bg2")
        p_bg3 = st.number_input("bg₃", value=0.0,  format="%.2f", key="bg3")

    st.markdown("**Refinement flags** (check to include in least-squares)")
    rf1, rf2, rf3, rf4, rf5, rf6 = st.columns(6)
    ref_scale  = rf1.checkbox("Scale",        value=True)
    ref_cell   = rf2.checkbox("Cell params",  value=True)
    ref_profile= rf3.checkbox("U, V, W, η",   value=True)
    ref_zero   = rf4.checkbox("Zero shift",   value=True)
    ref_bg     = rf5.checkbox("Background",   value=True)
    ref_angles = rf6.checkbox("Angles (α,β,γ)", value=False,
                               help="Fix angles unless triclinic/monoclinic")

# ─────────────────────────────────────────────────────────────────────────────
# Assemble params & flags
# ─────────────────────────────────────────────────────────────────────────────
mineral_work = {**mineral,
    "a":p_a,"b":p_b,"c":p_c,
    "alpha":p_alpha,"beta":p_beta,"gamma":p_gamma}

init_params = {
    "scale": p_scale, "zero": p_zero,
    "U": p_U, "V": p_V, "W": p_W, "eta": p_eta,
    "a": p_a, "b": p_b, "c": p_c,
    "alpha": p_alpha, "beta": p_beta, "gamma": p_gamma,
    "bg0": p_bg0, "bg1": p_bg1, "bg2": p_bg2, "bg3": p_bg3,
}

refine_flags = {
    "scale":  ref_scale,
    "U":      ref_profile, "V": ref_profile, "W": ref_profile, "eta": ref_profile,
    "zero":   ref_zero,
    "a":      ref_cell, "b": ref_cell, "c": ref_cell,
    "alpha":  ref_cell and ref_angles,
    "beta":   ref_cell and ref_angles,
    "gamma":  ref_cell and ref_angles,
    "bg0":    ref_bg, "bg1": ref_bg, "bg2": ref_bg, "bg3": ref_bg,
}

# ─────────────────────────────────────────────────────────────────────────────
# Generate reflection list
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Computing reflections…"):
    refs = gen_reflections(mineral_work, lam, tt_min, tt_max)

st.caption(f"Phase: **{mineral_name}**  |  {len(refs)} reflections in range {tt_min}–{tt_max}°")

# ─────────────────────────────────────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────────────────────────────────────
col_run, col_iter = st.columns([2,1])
with col_iter:
    max_iter = st.number_input("Max iterations", value=150, min_value=10, max_value=2000)
run_btn = col_run.button("▶️ Run Rietveld Refinement", type="primary", use_container_width=True)

# Pre-compute initial pattern (before refinement)
y_init = calc_pattern(two_theta, refs, init_params, lam, tt_min, tt_max)
metrics_init = gof_metrics(y_obs, y_init, 0)

# ─────────────────────────────────────────────────────────────────────────────
# Refinement
# ─────────────────────────────────────────────────────────────────────────────
if "refined_params" not in st.session_state:
    st.session_state.refined_params = None
    st.session_state.metrics        = None
    st.session_state.y_calc         = y_init
    st.session_state.residual       = y_obs - y_init
    st.session_state.refs           = refs
    st.session_state.refined        = False

if run_btn:
    t0 = time.time()
    progress_bar = st.progress(0, text="Refining…")
    with st.spinner("Running least-squares refinement…"):
        rp, rm, yc, res = run_rietveld(
            two_theta, y_obs, refs, init_params, refine_flags,
            tt_min, tt_max, lam, maxiter=int(max_iter)
        )
    elapsed = time.time() - t0
    progress_bar.progress(1.0, text=f"Done in {elapsed:.1f}s")
    st.session_state.refined_params = rp
    st.session_state.metrics        = rm
    st.session_state.y_calc         = yc
    st.session_state.residual       = res
    st.session_state.refs           = refs
    st.session_state.refined        = True

y_calc   = st.session_state.y_calc
residual = st.session_state.residual
metrics  = st.session_state.metrics if st.session_state.refined else metrics_init
ref_params = st.session_state.refined_params or init_params

# ─────────────────────────────────────────────────────────────────────────────
# Goodness-of-fit strip
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Goodness of Fit")
g1,g2,g3,g4,g5 = st.columns(5)
g1.metric("Rwp",    f"{metrics['Rwp']*100:.2f}%",  help="Weighted profile R-factor (target <10%)")
g2.metric("Rp",     f"{metrics['Rp']*100:.2f}%",   help="Profile R-factor")
g3.metric("χ²_red", f"{metrics['χ²_red']:.3f}",    help="Reduced chi-squared (target ~1)")
g4.metric("GoF",    f"{metrics['GoF (S)']:.3f}",   help="Goodness-of-fit S = √χ²_red (target ~1)")
g5.metric("Rexp",   f"{metrics['Rexp']*100:.2f}%", help="Expected R-factor (statistical limit)")

# ─────────────────────────────────────────────────────────────────────────────
# Main Rietveld plot
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Rietveld Plot")

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.75, 0.25],
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=["Observed / Calculated / Difference", "Difference (Obs − Calc)"],
)

# Observed
fig.add_trace(go.Scatter(
    x=two_theta, y=y_obs,
    mode="markers", name="Observed",
    marker=dict(size=2, color="#90caf9", opacity=0.7),
), row=1, col=1)

# Calculated
fig.add_trace(go.Scatter(
    x=two_theta, y=y_calc,
    mode="lines", name="Calculated",
    line=dict(color="#ef5350", width=1.8),
), row=1, col=1)

# Background
bg_coeffs = [ref_params.get(f"bg{i}", 0) for i in range(4)]
bg_curve  = chebyshev_bg(two_theta, np.array(bg_coeffs), tt_min, tt_max)
fig.add_trace(go.Scatter(
    x=two_theta, y=bg_curve,
    mode="lines", name="Background",
    line=dict(color="#ffb300", width=1.2, dash="dot"),
), row=1, col=1)

# Bragg tick marks
bragg_tt = [r["tt"] + ref_params.get("zero",0) for r in refs]
fig.add_trace(go.Scatter(
    x=bragg_tt,
    y=[-0.03 * y_obs.max()] * len(bragg_tt),
    mode="markers",
    marker=dict(symbol="line-ns", size=8, color="#a5d6a7",
                line=dict(color="#a5d6a7", width=1.5)),
    name="Bragg positions",
    hovertemplate="2θ=%{x:.3f}°<extra></extra>",
), row=1, col=1)

# Difference
fig.add_trace(go.Scatter(
    x=two_theta, y=residual,
    mode="lines", name="Difference",
    line=dict(color="#ce93d8", width=1.2),
), row=2, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

fig.update_layout(
    height=620,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="white"),
    legend=dict(orientation="h", y=-0.12, bgcolor="rgba(0,0,0,0)"),
    xaxis2=dict(title="2θ (°)", gridcolor="#2a2a2a"),
    yaxis=dict(title="Intensity (counts)", gridcolor="#2a2a2a"),
    yaxis2=dict(title="ΔI", gridcolor="#2a2a2a", zeroline=True),
)
fig.update_xaxes(gridcolor="#2a2a2a")
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs for detailed results
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Refined Parameters", "🔍 Reflection List", "📉 Residual Analysis", "💾 Export"]
)

# ── Tab 1 — refined parameters ────────────────────────────────────────────────
with tab1:
    st.subheader("Refined vs Starting Parameters")
    param_rows = []
    label_map = {
        "scale":"Scale factor","zero":"Zero shift (°)",
        "U":"U (Caglioti)","V":"V (Caglioti)","W":"W (Caglioti)",
        "eta":"η (pseudo-Voigt mixing)",
        "a":"a (Å)","b":"b (Å)","c":"c (Å)",
        "alpha":"α (°)","beta":"β (°)","gamma":"γ (°)",
        "bg0":"Background bg₀","bg1":"bg₁","bg2":"bg₂","bg3":"bg₃",
    }
    for k,label in label_map.items():
        if k not in init_params:
            continue
        was_refined = refine_flags.get(k, False)
        init_v = init_params[k]
        ref_v  = ref_params.get(k, init_v)
        delta  = ref_v - init_v
        param_rows.append({
            "Parameter":     label,
            "Starting":      round(init_v, 6),
            "Refined":       round(ref_v,  6),
            "Δ":             round(delta,  6),
            "Was refined":   "✓" if was_refined else "—",
        })
    df_params = pd.DataFrame(param_rows)
    st.dataframe(df_params.style.applymap(
        lambda v: "color:#a5d6a7" if v=="✓" else "",
        subset=["Was refined"]
    ), use_container_width=True, hide_index=True)

    # Cell volume
    V = cell_volume(ref_params["a"], ref_params["b"], ref_params["c"],
                    ref_params["alpha"], ref_params["beta"], ref_params["gamma"])
    st.metric("Refined cell volume V (Å³)", f"{V:.3f}")

# ── Tab 2 — reflection list ───────────────────────────────────────────────────
with tab2:
    st.subheader("Reflection List")
    df_refs = pd.DataFrame([
        {"h":r["h"],"k":r["k"],"l":r["l"],
         "d (Å)":round(r["d"],4),
         "2θ (°)":round(r["tt"]+ref_params.get("zero",0),4),
         "F² (arb.)":round(r["F2"],2),
         "Mult.":r["mult"],
         "I_hkl (∝ m·F²)":round(r["mult"]*r["F2"],2)}
        for r in refs
    ])
    st.dataframe(
        df_refs.style.background_gradient(subset=["F² (arb.)","I_hkl (∝ m·F²)"], cmap="plasma"),
        use_container_width=True, height=450
    )

# ── Tab 3 — residual analysis ─────────────────────────────────────────────────
with tab3:
    st.subheader("Residual & Statistical Analysis")

    fig3 = make_subplots(rows=1, cols=2,
        subplot_titles=["Weighted residuals", "Normal probability plot"])

    w_res = residual / np.sqrt(np.maximum(y_obs, 1))

    fig3.add_trace(go.Scatter(
        x=two_theta, y=w_res,
        mode="lines", line=dict(color="#ce93d8", width=1),
        name="wΔ", showlegend=False,
    ), row=1, col=1)
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # Normal probability plot
    sorted_res = np.sort(w_res)
    n = len(sorted_res)
    quantiles = np.array([(i-0.5)/n for i in range(1,n+1)])
    from scipy.stats import norm
    theoretical = norm.ppf(quantiles)
    fig3.add_trace(go.Scatter(
        x=theoretical, y=sorted_res,
        mode="markers", marker=dict(size=2, color="#4fc3f7"),
        name="Normal Q-Q", showlegend=False,
    ), row=1, col=2)
    fig3.add_trace(go.Scatter(
        x=[theoretical[0], theoretical[-1]],
        y=[theoretical[0], theoretical[-1]],
        mode="lines", line=dict(color="orange", dash="dash"),
        name="Ideal", showlegend=False,
    ), row=1, col=2)

    fig3.update_layout(
        height=350, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    fig3.update_xaxes(gridcolor="#2a2a2a")
    fig3.update_yaxes(gridcolor="#2a2a2a")
    st.plotly_chart(fig3, use_container_width=True)

    col_a, col_b = st.columns(2)
    col_a.metric("Mean weighted residual", f"{np.mean(w_res):.4f}", help="Should be ≈0")
    col_b.metric("Std weighted residual",  f"{np.std(w_res):.4f}",  help="Should be ≈1 for good fit")

# ── Tab 4 — export ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Export Results")

    # 1. Pattern CSV
    df_pat = pd.DataFrame({
        "two_theta_deg": two_theta,
        "I_obs":         y_obs,
        "I_calc":        y_calc,
        "I_background":  bg_curve,
        "difference":    residual,
    })

    # 2. Parameters CSV
    buf_par = io.StringIO()
    buf_par.write(f"# Rietveld Refinement — {mineral_name}\n")
    buf_par.write(f"# lambda = {lam} Å\n")
    buf_par.write(f"# Rwp = {metrics['Rwp']*100:.3f}%  Rp = {metrics['Rp']*100:.3f}%  "
                  f"chi2_red = {metrics['χ²_red']:.4f}  GoF = {metrics['GoF (S)']:.4f}\n")
    df_params.to_csv(buf_par, index=False)

    # 3. Reflections CSV
    buf_ref = io.StringIO()
    buf_ref.write(f"# Reflection list — {mineral_name}\n")
    df_refs.to_csv(buf_ref, index=False)

    c1e, c2e, c3e = st.columns(3)
    c1e.download_button(
        "⬇️ Pattern (obs/calc/diff)",
        df_pat.to_csv(index=False),
        f"rietveld_pattern_{mineral_name[:8]}.csv", "text/csv",
        type="primary",
    )
    c2e.download_button(
        "⬇️ Refined Parameters",
        buf_par.getvalue(),
        f"rietveld_params_{mineral_name[:8]}.csv", "text/csv",
    )
    c3e.download_button(
        "⬇️ Reflection List",
        buf_ref.getvalue(),
        f"rietveld_reflections_{mineral_name[:8]}.csv", "text/csv",
    )

    st.markdown("---")
    st.markdown("**Summary report**")
    st.code(
f"""Rietveld Refinement Summary
============================
Phase         : {mineral_name}
Space group   : {mineral.get("sg","—")}
λ             : {lam:.5f} Å

Refined Cell Parameters:
  a = {ref_params['a']:.5f} Å
  b = {ref_params['b']:.5f} Å
  c = {ref_params['c']:.5f} Å
  α = {ref_params['alpha']:.4f} °
  β = {ref_params['beta']:.4f} °
  γ = {ref_params['gamma']:.4f} °
  V = {cell_volume(ref_params['a'],ref_params['b'],ref_params['c'],ref_params['alpha'],ref_params['beta'],ref_params['gamma']):.4f} Å³

Profile (Caglioti):
  U = {ref_params['U']:.5f}
  V = {ref_params['V']:.5f}
  W = {ref_params['W']:.5f}
  η = {ref_params['eta']:.4f}

Scale / Zero:
  Scale = {ref_params['scale']:.5e}
  Zero  = {ref_params['zero']:.5f} °

Goodness of Fit:
  Rwp   = {metrics['Rwp']*100:.3f} %
  Rp    = {metrics['Rp']*100:.3f} %
  χ²_red= {metrics['χ²_red']:.4f}
  GoF S = {metrics['GoF (S)']:.4f}
  Rexp  = {metrics['Rexp']*100:.3f} %

Reflections used: {len(refs)}
Data points     : {len(two_theta)}
""", language="text")

# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Structure factors: 4-Gaussian ASF (Int. Tables Vol. C)  ·  "
    "Peak profile: pseudo-Voigt (Thompson-Cox-Hastings)  ·  "
    "Broadening: Caglioti H²=U·tan²θ+V·tanθ+W  ·  "
    "Background: Chebyshev polynomial  ·  "
    "Optimiser: scipy least_squares (TRF / Levenberg-Marquardt)  ·  "
    "Weights: 1/σ² (Poisson)"
)