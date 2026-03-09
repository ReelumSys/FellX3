"""
Rietveld Refinement App – Streamlit
=====================================
Inputs:
  • Diffraktogramm  (.xy / .xye / .dat / .csv  – Spalten: 2θ, Intensity [, Sigma])
  • CIF-Datei       (.cif)

Was das Programm macht:
  1. CIF parsen  → Gitterparameter, Raumgruppe, Atompositionen
  2. Strukturfaktor  F_hkl  berechnen  (geometrische + atomare Streuung)
  3. Profilform  pseudo-Voigt  (U, V, W, η)
  4. Untergrund  via Chebyshev-Polynome
  5. Scipy least_squares  Optimierung
  6. Ausgabe:  Plot, R-Werte, verfeinerte Parameter

Abhängigkeiten:
  pip install streamlit gemmi numpy scipy matplotlib pandas
"""

import io
import re
import math
import textwrap
from itertools import product as iproduct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import least_squares
from scipy.special import wofz

# ── optional: gemmi für CIF-Parsing ──────────────────────────────────────────
try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False

# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

SCATTERING_FACTORS = {
    # element: [a1,b1,a2,b2,a3,b3,a4,b4,c]  (Cromer–Mann)
    "H":  [0.493,10.511,0.323,26.126,0.140,3.142,0.041,57.800,0.003],
    "C":  [2.310, 20.844,1.020,10.208,1.589, 0.569,0.865,51.651,0.216],
    "N":  [12.213,0.006,3.132,9.893,2.013,28.997,1.166,0.583,-11.529],
    "O":  [3.049,13.277,2.287,5.701,1.546,0.324,0.867,32.909,0.251],
    "Si": [6.292, 2.439,3.035,32.334,1.989, 0.678,1.541,81.694,1.141],
    "Al": [6.420, 3.039,1.900,0.743,1.594,31.547,1.965,85.088,1.115],
    "Fe": [11.769,4.761,7.357,0.307,3.522,15.354,2.304,76.881,1.037],
    "Ca": [8.626, 10.442,7.387,0.660,1.590,85.748,1.022,178.437,1.375],
    "Mg": [5.420, 2.828,2.174,79.261,1.226,0.381,2.307,7.194,0.858],
    "Ti": [9.759, 7.851,7.357,0.500,1.699,35.634,1.902,116.105,1.281],
    "Zr": [17.876,1.276,10.948,11.916,5.418,0.118,3.657,87.663,2.070],
    "La": [20.578,0.589,19.599,7.449,11.373,22.743,3.287,117.020,2.146],
    "Ba": [20.336,3.216,19.297,0.275,10.888,20.207,2.696,167.202,2.774],
    "Sr": [17.566,1.556,9.818,14.099,5.422,0.166,2.669,132.376,2.507],
    "K":  [8.219, 12.795,7.440,0.775,1.052,213.187,0.866,41.684,1.423],
    "Na": [4.763, 3.285,3.174,8.842,1.267,0.314,1.113,129.424,0.676],
    "Cl": [11.460,0.010,7.196,1.166,6.256,18.519,4.073,47.778,-9.557],
    "S":  [6.905, 1.468,5.203,22.215,1.438,0.254,1.586,56.172,0.867],
    "P":  [6.435, 1.907,4.179,27.157,1.780,0.526,1.491,68.164,1.115],
    "Zn": [14.074,3.266,7.032,0.233,5.165,10.316,2.410,58.710,1.304],
    "Cu": [13.338,3.583,7.168,0.247,5.616,11.397,1.673,64.812,1.191],
    "Ni": [12.838,3.879,7.292,0.254,4.444,12.176,2.380,66.342,1.034],
    "Mn": [11.282,5.341,7.357,0.343,3.020,17.867,2.244,83.754,1.089],
    "Cr": [10.641,6.104,7.354,0.392,3.324,20.267,1.492,98.740,1.183],
    "Co": [12.284,4.279,7.341,0.279,4.003,13.536,2.348,71.169,1.012],
    "Bi": [33.369,0.704,19.282,1.698,12.013,10.305,2.770,68.070,3.770],
    "Pb": [31.061,0.690,13.064,2.358,18.442,8.618,5.961,47.258,13.412],
    "W":  [29.082,1.720,15.430,9.615,14.433,0.321,5.119,52.416,1.305],
    "Mo": [3.702,0.277,17.236,1.096,12.888,11.004,3.742,61.658,4.387],
}

def scattering_factor(element, s2):
    """Atomarer Streuformfaktor f(s) mit Cromer-Mann Koeffizienten. s2 = (sin θ / λ)²"""
    el = element.capitalize()
    if el not in SCATTERING_FACTORS:
        return np.ones_like(s2) * 6.0  # Fallback
    c = SCATTERING_FACTORS[el]
    a = c[0::2][:4]
    b = c[1::2][:4]
    c0 = c[8]
    f = c0 + sum(ai * np.exp(-bi * s2) for ai, bi in zip(a, b))
    return f

# ─────────────────────────────────────────────────────────────────────────────
#  CIF Parser (minimal, ohne gemmi)
# ─────────────────────────────────────────────────────────────────────────────

def parse_cif_manual(text):
    """Einfacher CIF-Parser für die wichtigsten Felder."""
    def get_val(key):
        m = re.search(rf"^\s*{re.escape(key)}\s+([^\s#]+)", text, re.MULTILINE | re.IGNORECASE)
        return m.group(1).replace("(", "").replace(")", "") if m else None

    a = float(get_val("_cell_length_a") or 5.0)
    b = float(get_val("_cell_length_b") or a)
    c = float(get_val("_cell_length_c") or a)
    alpha = float(get_val("_cell_angle_alpha") or 90.0)
    beta  = float(get_val("_cell_angle_beta")  or 90.0)
    gamma = float(get_val("_cell_angle_gamma") or 90.0)
    sg    = get_val("_symmetry_space_group_name_H-M") or \
            get_val("_space_group_name_H-M_alt") or "P 1"

    # Atome aus _atom_site Loop
    atoms = []
    loop_pattern = re.compile(
        r"_atom_site_type_symbol\s+.*?(?=loop_|\Z)", re.DOTALL | re.IGNORECASE
    )
    # Einfacheres Muster: alle Zeilen mit Element x y z
    atom_line = re.compile(
        r"^\s*(\w+)\s+\w+\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)",
        re.MULTILINE,
    )
    for m in atom_line.finditer(text):
        el = re.sub(r"[0-9+\-]", "", m.group(1)).capitalize()
        if el and el[0].isupper():
            atoms.append({
                "element": el,
                "x": float(m.group(2)),
                "y": float(m.group(3)),
                "z": float(m.group(4)),
                "Uiso": 0.01,
            })

    return dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                sg=sg, atoms=atoms)


def parse_cif_gemmi(text):
    doc = gemmi.cif.read_string(text)
    block = doc.sole_block()
    st = gemmi.make_small_structure_from_block(block)
    cell = st.cell
    atoms = []
    for site in st.sites:
        atoms.append({
            "element": site.element.name,
            "x": site.fract.x,
            "y": site.fract.y,
            "z": site.fract.z,
            "Uiso": site.u_iso if site.u_iso > 0 else 0.01,
        })
    sg = str(st.spacegroup_hm) if st.spacegroup_hm else "P 1"
    return dict(a=cell.a, b=cell.b, c=cell.c,
                alpha=cell.alpha, beta=cell.beta, gamma=cell.gamma,
                sg=sg, atoms=atoms)


def parse_cif(text):
    if HAS_GEMMI:
        try:
            return parse_cif_gemmi(text)
        except Exception:
            pass
    return parse_cif_manual(text)

# ─────────────────────────────────────────────────────────────────────────────
#  Kristallographie
# ─────────────────────────────────────────────────────────────────────────────

def metric_tensor(a, b, c, alpha, beta, gamma):
    """Metriktensor G der Elementarzelle."""
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    G = np.array([
        [a*a,     a*b*cg,  a*c*cb],
        [a*b*cg,  b*b,     b*c*ca],
        [a*c*cb,  b*c*ca,  c*c  ],
    ])
    return G


def cell_volume(a, b, c, alpha, beta, gamma):
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    return a*b*c*math.sqrt(
        1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg
    )


def d_spacing(h, k, l, a, b, c, alpha, beta, gamma):
    """d_hkl für beliebige Kristallsysteme."""
    G = metric_tensor(a, b, c, alpha, beta, gamma)
    Ginv = np.linalg.inv(G)
    hkl = np.array([h, k, l])
    inv_d2 = hkl @ Ginv @ hkl
    if inv_d2 <= 0:
        return np.inf
    return 1.0 / math.sqrt(inv_d2)


def generate_hkl(a, b, c, alpha, beta, gamma, two_theta_max, wavelength):
    """Generiere alle (hkl) Reflexe bis 2θ_max."""
    lam = wavelength
    sin_th_max = math.sin(math.radians(two_theta_max / 2))
    d_min = lam / (2 * sin_th_max) if sin_th_max > 0 else 0.5

    hmax = int(a / d_min) + 1
    kmax = int(b / d_min) + 1
    lmax = int(c / d_min) + 1

    reflections = []
    for h, k, l in iproduct(range(-hmax, hmax+1),
                             range(-kmax, kmax+1),
                             range(-lmax, lmax+1)):
        if h == k == l == 0:
            continue
        d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma)
        if d < d_min or d > 1e5:
            continue
        sinth_lam = 1 / (2 * d)
        if sinth_lam > 1:
            continue
        two_th = math.degrees(2 * math.asin(lam * sinth_lam))
        if 0 < two_th < two_theta_max:
            reflections.append((h, k, l, d, two_th))
    return reflections


def structure_factor(hkl_list, atoms, wavelength):
    """F_hkl als numpy-Array für alle Reflexe."""
    F = np.zeros(len(hkl_list), dtype=complex)
    for i, (h, k, l, d, _) in enumerate(hkl_list):
        s2 = 1 / (4 * d**2)   # (sinθ/λ)²
        Fhkl = 0+0j
        for at in atoms:
            f = scattering_factor(at["element"], np.array([s2]))[0]
            dw = math.exp(-8 * math.pi**2 * at["Uiso"] * s2)
            phase = 2 * math.pi * (h*at["x"] + k*at["y"] + l*at["z"])
            Fhkl += f * dw * complex(math.cos(phase), math.sin(phase))
        F[i] = Fhkl
    return F

# ─────────────────────────────────────────────────────────────────────────────
#  Profilform  (pseudo-Voigt)
# ─────────────────────────────────────────────────────────────────────────────

def pseudo_voigt(two_th_grid, two_th0, U, V, W, eta):
    """Pseudo-Voigt Profil (Thompson-Cox-Hastings)."""
    tan_th = math.tan(math.radians(two_th0 / 2))
    fwhm2 = U * tan_th**2 + V * tan_th + W
    fwhm2 = max(fwhm2, 1e-8)
    fwhm = math.sqrt(fwhm2)
    dt = two_th_grid - two_th0
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    gaussian = np.exp(-dt**2 / (2 * sigma**2))
    lorentzian = 1 / (1 + (dt / (fwhm/2))**2)
    eta_c = min(max(eta, 0), 1)
    return eta_c * lorentzian + (1 - eta_c) * gaussian


def lorentz_polarization(two_th):
    """LP-Korrektur für Pulverdiffraktometrie."""
    th = np.radians(two_th / 2)
    sin_th = np.sin(th)
    cos_2th = np.cos(np.radians(two_th))
    lp = (1 + cos_2th**2) / (sin_th**2 * np.cos(th) + 1e-12)
    return lp

# ─────────────────────────────────────────────────────────────────────────────
#  Untergrundmodell  (Chebyshev)
# ─────────────────────────────────────────────────────────────────────────────

def chebyshev_background(two_th, coeffs):
    x = 2 * (two_th - two_th.min()) / (two_th.max() - two_th.min()) - 1
    result = np.zeros_like(two_th)
    n = len(coeffs)
    if n >= 1: result += coeffs[0]
    if n >= 2: result += coeffs[1] * x
    if n >= 3: result += coeffs[2] * (2*x**2 - 1)
    if n >= 4: result += coeffs[3] * (4*x**3 - 3*x)
    if n >= 5: result += coeffs[4] * (8*x**4 - 8*x**2 + 1)
    if n >= 6: result += coeffs[5] * (16*x**5 - 20*x**3 + 5*x)
    return result

# ─────────────────────────────────────────────────────────────────────────────
#  Berechnetes Diffraktogramm
# ─────────────────────────────────────────────────────────────────────────────

def calc_pattern(two_th_grid, hkl_list, F_hkl, scale, U, V, W, eta,
                 bg_coeffs, wavelength):
    lp = lorentz_polarization(two_th_grid)
    pattern = np.zeros_like(two_th_grid)
    for i, (h, k, l, d, two_th0) in enumerate(hkl_list):
        I_hkl = abs(F_hkl[i])**2
        peak = pseudo_voigt(two_th_grid, two_th0, U, V, W, eta)
        pattern += I_hkl * peak
    pattern *= scale * lp
    pattern += chebyshev_background(two_th_grid, bg_coeffs)
    return pattern

# ─────────────────────────────────────────────────────────────────────────────
#  Verfeinerung
# ─────────────────────────────────────────────────────────────────────────────

def run_refinement(two_th, obs, sig, cell, atoms, wavelength,
                   refine_cell, refine_profile, refine_bg, n_bg):

    # Startparameter
    a0, b0, c0 = cell["a"], cell["b"], cell["c"]
    alpha0, beta0, gamma0 = cell["alpha"], cell["beta"], cell["gamma"]
    scale0 = obs.max() / max(obs.sum() * 1e-4, 1)
    U0, V0, W0, eta0 = 0.01, -0.001, 0.005, 0.5
    bg0 = [obs.min()] + [0.0] * (n_bg - 1)

    def pack(a, b, c, al, be, ga, sc, U, V, W, et, bg):
        p = [sc, U, V, W, et] + list(bg)
        if refine_cell:
            p = [a, b, c, al, be, ga] + p
        return np.array(p, dtype=float)

    def unpack(p):
        idx = 0
        if refine_cell:
            a, b, c = p[0], p[1], p[2]
            al, be, ga = p[3], p[4], p[5]
            idx = 6
        else:
            a, b, c = a0, b0, c0
            al, be, ga = alpha0, beta0, gamma0
        sc = p[idx]; U = p[idx+1]; V = p[idx+2]; W = p[idx+3]; et = p[idx+4]
        bg = list(p[idx+5:idx+5+n_bg])
        return a, b, c, al, be, ga, sc, U, V, W, et, bg

    x0 = pack(a0, b0, c0, alpha0, beta0, gamma0, scale0, U0, V0, W0, eta0, bg0)

    best_hkl = [None]
    best_F   = [None]

    def residuals(p):
        a, b, c, al, be, ga, sc, U, V, W, et, bg = unpack(p)
        try:
            hkl_list = generate_hkl(a, b, c, al, be, ga, two_th.max(), wavelength)
        except Exception:
            return np.ones_like(obs) * 1e6
        if len(hkl_list) == 0:
            return (obs - chebyshev_background(two_th, bg)) / sig
        F_hkl = structure_factor(hkl_list, atoms, wavelength)
        calc = calc_pattern(two_th, hkl_list, F_hkl, sc, U, V, W, et, bg, wavelength)
        best_hkl[0] = hkl_list
        best_F[0]   = F_hkl
        return (obs - calc) / sig

    result = least_squares(residuals, x0, method="lm", max_nfev=500,
                           ftol=1e-6, xtol=1e-6, gtol=1e-6)

    a, b, c, al, be, ga, sc, U, V, W, et, bg = unpack(result.x)

    hkl_list = best_hkl[0] or []
    F_hkl    = best_F[0]   if best_F[0] is not None else np.array([])

    calc = calc_pattern(two_th, hkl_list, F_hkl, sc, U, V, W, et, bg, wavelength)
    diff = obs - calc

    # R-Werte
    Rp  = np.sum(np.abs(diff)) / np.sum(np.abs(obs))
    Rwp = np.sqrt(np.sum((diff/sig)**2) / np.sum((obs/sig)**2))
    chi2 = np.sum((diff/sig)**2) / max(len(obs) - len(result.x), 1)

    refined = dict(a=a, b=b, c=c, alpha=al, beta=be, gamma=ga,
                   scale=sc, U=U, V=V, W=W, eta=et, bg=bg,
                   Rp=Rp, Rwp=Rwp, chi2=chi2)
    return refined, calc, diff, hkl_list

# ─────────────────────────────────────────────────────────────────────────────
#  Diffraktogramm-Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_diffractogram(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().startswith("#")]
    rows = []
    for l in lines:
        parts = re.split(r"[,\s]+", l)
        try:
            nums = [float(x) for x in parts if x]
            if len(nums) >= 2:
                rows.append(nums[:3])
        except ValueError:
            continue
    arr = np.array(rows)
    two_th = arr[:, 0]
    intensity = arr[:, 1]
    sigma = arr[:, 2] if arr.shape[1] > 2 else np.sqrt(np.maximum(intensity, 1))
    # Sortieren
    idx = np.argsort(two_th)
    return two_th[idx], intensity[idx], sigma[idx]

# ─────────────────────────────────────────────────────────────────────────────
#  Demo-Daten
# ─────────────────────────────────────────────────────────────────────────────

def demo_diffractogram():
    two_th = np.linspace(10, 80, 700)
    # Rutile-ähnliche Peaks
    peaks = [(25.3, 1000), (36.1, 500), (41.2, 300), (54.3, 400),
             (56.6, 200), (62.7, 350), (68.0, 150)]
    intensity = np.ones_like(two_th) * 50
    np.random.seed(42)
    intensity += np.random.normal(0, 5, len(two_th))
    for p, I in peaks:
        fwhm = 0.3
        intensity += I * np.exp(-4*np.log(2)*(two_th-p)**2/fwhm**2)
    sigma = np.sqrt(np.maximum(intensity, 1))
    return two_th, intensity, sigma


DEMO_CIF = textwrap.dedent("""\
    data_TiO2_rutile
    _cell_length_a   4.5937
    _cell_length_b   4.5937
    _cell_length_c   2.9587
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _symmetry_space_group_name_H-M 'P 42/m n m'
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Ti1  Ti  0.000  0.000  0.000
    O1   O   0.305  0.305  0.000
""")

# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Rietveld Refinement", layout="wide",
                   page_icon="⚗️")

st.title("⚗️ Rietveld Refinement")
st.caption("Powder X-ray diffraction profile refinement – powered by Python / scipy")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Daten")
    use_demo = st.checkbox("Demo-Daten verwenden (TiO₂ Rutil)", value=True)

    cif_file  = st.file_uploader("CIF-Datei (.cif)", type=["cif"])
    diff_file = st.file_uploader("Diffraktogramm (.xy .xye .dat .csv)",
                                 type=["xy","xye","dat","csv","txt"])

    st.header("⚙️ Messparameter")
    wavelength = st.number_input("Wellenlänge λ (Å)", value=1.5406,
                                 min_value=0.1, max_value=3.0, step=0.0001,
                                 format="%.4f")

    st.header("🔧 Verfeinerungsoptionen")
    refine_cell    = st.checkbox("Gitterparameter verfeinern", value=True)
    refine_profile = st.checkbox("Profilparameter (U,V,W,η)", value=True)
    n_bg = st.slider("Untergrundterme (Chebyshev)", 1, 6, 4)

    th_min = st.number_input("2θ min (°)", value=10.0)
    th_max = st.number_input("2θ max (°)", value=80.0)

    run_btn = st.button("▶ Verfeinerung starten", type="primary",
                        use_container_width=True)

# ── Daten laden ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

if use_demo:
    two_th_raw, obs_raw, sig_raw = demo_diffractogram()
    cif_text = DEMO_CIF
    with col1:
        st.info("Demo: TiO₂ Rutil (synthetisches Diffraktogramm)")
    with col2:
        st.info("Demo: TiO₂ Rutil CIF")
else:
    two_th_raw = obs_raw = sig_raw = cif_text = None
    if diff_file:
        two_th_raw, obs_raw, sig_raw = load_diffractogram(
            diff_file.read(), diff_file.name)
        with col1:
            st.success(f"Diffraktogramm geladen: {len(two_th_raw)} Punkte")
    if cif_file:
        cif_text = cif_file.read().decode("utf-8", errors="replace")
        with col2:
            st.success("CIF geladen")

# Vorschau des Diffraktogramms
if two_th_raw is not None:
    mask = (two_th_raw >= th_min) & (two_th_raw <= th_max)
    two_th = two_th_raw[mask]
    obs    = obs_raw[mask]
    sig    = sig_raw[mask]

    fig0, ax0 = plt.subplots(figsize=(10, 3))
    ax0.plot(two_th, obs, "k-", lw=0.8, label="Beobachtet")
    ax0.set_xlabel("2θ (°)")
    ax0.set_ylabel("Intensität")
    ax0.set_title("Eingabe-Diffraktogramm")
    ax0.legend()
    st.pyplot(fig0, use_container_width=True)

# ── Verfeinerung ──────────────────────────────────────────────────────────────
if run_btn:
    if two_th_raw is None or cif_text is None:
        st.error("Bitte Diffraktogramm und CIF-Datei laden oder Demo aktivieren.")
        st.stop()

    with st.spinner("Struktur aus CIF parsen …"):
        cell = parse_cif(cif_text)

    # Atome anzeigen
    with st.expander("📋 Kristallstruktur aus CIF", expanded=False):
        st.write(f"**Raumgruppe:** {cell['sg']}")
        st.write(f"**a** = {cell['a']:.4f} Å  |  **b** = {cell['b']:.4f} Å  |  **c** = {cell['c']:.4f} Å")
        st.write(f"**α** = {cell['alpha']:.2f}°  |  **β** = {cell['beta']:.2f}°  |  **γ** = {cell['gamma']:.2f}°")
        if cell["atoms"]:
            df_atoms = pd.DataFrame(cell["atoms"])
            st.dataframe(df_atoms, use_container_width=True)
        else:
            st.warning("Keine Atome im CIF gefunden – nur Profilverfeinerung möglich.")

    if not cell["atoms"]:
        st.error("CIF enthält keine Atompositionen. Bitte vollständige CIF-Datei verwenden.")
        st.stop()

    mask = (two_th_raw >= th_min) & (two_th_raw <= th_max)
    two_th = two_th_raw[mask]
    obs    = obs_raw[mask]
    sig    = sig_raw[mask]

    with st.spinner("Verfeinerung läuft … (kann einige Sekunden dauern)"):
        try:
            refined, calc, diff_arr, hkl_list = run_refinement(
                two_th, obs, sig, cell, cell["atoms"], wavelength,
                refine_cell, refine_profile, True, n_bg
            )
        except Exception as e:
            st.error(f"Fehler bei der Verfeinerung: {e}")
            st.stop()

    st.success("✅ Verfeinerung abgeschlossen!")

    # ── R-Werte ──────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Rₚ", f"{refined['Rp']*100:.2f} %")
    m2.metric("Rwₚ", f"{refined['Rwp']*100:.2f} %")
    m3.metric("χ²", f"{refined['chi2']:.3f}")

    # ── Verfeinerte Parameter ─────────────────────────────────────────────────
    with st.expander("📊 Verfeinerte Parameter", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Gitterparameter")
            gp = pd.DataFrame({
                "Parameter": ["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)"],
                "Ausgangswert": [cell["a"], cell["b"], cell["c"],
                                 cell["alpha"], cell["beta"], cell["gamma"]],
                "Verfeinert":   [refined["a"], refined["b"], refined["c"],
                                 refined["alpha"], refined["beta"], refined["gamma"]],
            })
            st.dataframe(gp.style.format({"Ausgangswert": "{:.4f}",
                                          "Verfeinert": "{:.4f}"}),
                         use_container_width=True)
        with c2:
            st.subheader("Profilparameter")
            pp = pd.DataFrame({
                "Parameter": ["U", "V", "W", "η (Lorentz-Anteil)", "Scale"],
                "Wert": [refined["U"], refined["V"], refined["W"],
                         refined["eta"], refined["scale"]],
            })
            st.dataframe(pp.style.format({"Wert": "{:.6f}"}),
                         use_container_width=True)

    # ── Rietveld-Plot ─────────────────────────────────────────────────────────
    st.subheader("📈 Rietveld-Plot")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    ax1.plot(two_th, obs,  "k-",  lw=0.8, label="Beobachtet (Iₒ)")
    ax1.plot(two_th, calc, "r-",  lw=1.2, label="Berechnet (Iᶜ)")
    bg = chebyshev_background(two_th, refined["bg"])
    ax1.plot(two_th, bg, "b--", lw=0.8, label="Untergrund")

    # Reflex-Marker
    if hkl_list:
        two_th_hkl = [r[4] for r in hkl_list]
        ax1.vlines(two_th_hkl, ymin=ax1.get_ylim()[0],
                   ymax=obs.max()*0.05 + obs.min(),
                   colors="green", lw=0.5, alpha=0.7, label="Reflexe")

    ax1.set_ylabel("Intensität")
    ax1.legend(fontsize=8)
    ax1.set_title(f"Rietveld-Verfeinerung  |  Rwp = {refined['Rwp']*100:.2f} %  "
                  f"|  χ² = {refined['chi2']:.3f}")

    ax2.plot(two_th, diff_arr, "g-", lw=0.7, label="Differenz Iₒ − Iᶜ")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("2θ (°)")
    ax2.set_ylabel("ΔI")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Download Plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    st.download_button("💾 Plot als PNG herunterladen", buf,
                       file_name="rietveld_plot.png", mime="image/png")

    # ── Reflexliste ───────────────────────────────────────────────────────────
    if hkl_list:
        with st.expander("📋 Reflexliste (hkl)", expanded=False):
            F_hkl = structure_factor(hkl_list, cell["atoms"], wavelength)
            rows = []
            for i, (h, k, l, d, two_th0) in enumerate(hkl_list):
                rows.append({
                    "h": h, "k": k, "l": l,
                    "d (Å)": round(d, 4),
                    "2θ (°)": round(two_th0, 3),
                    "|F|²": round(abs(F_hkl[i])**2, 1),
                })
            df_hkl = pd.DataFrame(rows).sort_values("2θ (°)")
            st.dataframe(df_hkl, use_container_width=True)

# ── Fußzeile ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Implementierung: pseudo-Voigt Profilfunktion · Cromer-Mann Atomformfaktoren · "
    "Chebyshev-Untergrund · Lorentz-Polarisationskorrektur · "
    "Optimierung via `scipy.optimize.least_squares`"
)