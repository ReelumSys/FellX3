"""
Diffraction Analyser — Full Profile Refinement (Le Bail + Rietveld)
Run with:  streamlit run diffraction_analyser.py
Requires:  pip install streamlit numpy scipy plotly pandas
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import least_squares
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Diffraction Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.3rem; }
.block-container { padding-top: 1rem; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
h1 { font-size: 1.8rem !important; }
h2 { font-size: 1.2rem !important; }
h3 { font-size: 1.05rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CRYSTALLOGRAPHY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def d_spacing(h, k, l, a, b, c, alpha_deg, beta_deg, gamma_deg, system):
    """Return d-spacing (Å) for a given reflection and crystal system."""
    ar, br, gr = (np.radians(x) for x in (alpha_deg, beta_deg, gamma_deg))
    h, k, l = float(h), float(k), float(l)

    if system == "Cubic":
        inv = (h**2 + k**2 + l**2) / a**2
    elif system == "Tetragonal":
        inv = (h**2 + k**2) / a**2 + l**2 / c**2
    elif system == "Orthorhombic":
        inv = h**2/a**2 + k**2/b**2 + l**2/c**2
    elif system == "Hexagonal":
        inv = 4/3*(h**2 + h*k + k**2)/a**2 + l**2/c**2
    elif system == "Monoclinic":
        sb = np.sin(br)
        inv = (1/sb**2)*(h**2/a**2 + k**2*sb**2/b**2 + l**2/c**2
                         - 2*h*l*np.cos(br)/(a*c))
    else:  # Triclinic
        ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
        sa, sb, sg = np.sin(ar), np.sin(br), np.sin(gr)
        V = a*b*c*np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg)
        if V < 1e-10:
            return None
        inv = (b**2*c**2*sa**2*h**2 + a**2*c**2*sb**2*k**2 + a**2*b**2*sg**2*l**2
               + 2*a*b*c**2*(ca*cb-cg)*h*k
               + 2*a**2*b*c*(cb*cg-ca)*k*l
               + 2*a*b**2*c*(ca*cg-cb)*h*l) / V**2

    return None if inv <= 1e-12 else 1.0/np.sqrt(inv)


def is_absent(h, k, l, sg):
    """Very simplified systematic absence check based on lattice centering."""
    h, k, l = int(h), int(k), int(l)
    sg = sg.upper().replace(" ", "")
    if sg.startswith("I") and (h+k+l) % 2 != 0:
        return True
    if sg.startswith("F"):
        parities = {h%2, k%2, l%2}
        if len(parities) > 1:
            return True
    if sg.startswith("C") and (h+k) % 2 != 0:
        return True
    if sg.startswith("A") and (k+l) % 2 != 0:
        return True
    if sg.startswith("B") and (h+l) % 2 != 0:
        return True
    # FD screw axes (very simplified)
    if "FD" in sg or "Fd" in sg.replace("-",""):
        if h == 0 and k == 0 and l % 4 != 0:
            return True
    return False


def gen_reflections(a, b, c, alpha, beta, gamma, system, sg, wl, tt_min, tt_max):
    """Return list of (h,k,l,d,2theta,I_lb) for all allowed reflections."""
    d_min = wl / (2*np.sin(np.radians(tt_max/2)))
    d_max = wl / (2*np.sin(np.radians(max(tt_min, 0.5)/2)))
    mh = int(2*a/d_min)+2
    mk = int(2*b/d_min)+2
    ml = int(2*c/d_min)+2

    seen = {}
    for h in range(-mh, mh+1):
        for k in range(-mk, mk+1):
            for l in range(-ml, ml+1):
                if h == k == l == 0:
                    continue
                if is_absent(h, k, l, sg):
                    continue
                d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma, system)
                if d is None or not (d_min <= d <= d_max):
                    continue
                tt = 2*np.degrees(np.arcsin(np.clip(wl/(2*d), -1, 1)))
                key = round(tt, 4)
                if key not in seen:
                    seen[key] = [h, k, l, d, tt, 1000.0]
    refs = sorted(seen.values(), key=lambda r: r[4])
    return refs


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE & BACKGROUND FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def pseudo_voigt(x, x0, fwhm, eta):
    """Normalised pseudo-Voigt profile."""
    eta = np.clip(eta, 0, 1)
    fwhm = max(fwhm, 1e-6)
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    G = np.exp(-0.5*((x-x0)/sigma)**2)
    L = 1.0 / (1 + ((x-x0)/(fwhm/2))**2)
    return eta*L + (1-eta)*G


def caglioti_fwhm(tt, U, V, W):
    th = np.radians(tt/2)
    tan_th = np.tan(th)
    return max(np.sqrt(max(U*tan_th**2 + V*tan_th + W, 1e-8)), 0.005)


def chebyshev_bg(tt, coeffs):
    x = 2*(tt - tt.min())/(tt.max()-tt.min()) - 1
    T = [np.ones_like(x), x, 2*x**2-1, 4*x**3-3*x,
         8*x**4-8*x**2+1, 16*x**5-20*x**3+5*x,
         32*x**6-48*x**4+18*x**2-1, 64*x**7-112*x**5+56*x**3-7*x]
    result = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        result += c * T[i]
    return result


def lp_factor(tt):
    th = np.radians(tt/2)
    cos2 = np.cos(np.radians(tt))**2
    return (1+cos2) / (np.sin(th)**2 * np.cos(th) + 1e-12)


def multiplicity(h, k, l, system):
    h, k, l = abs(int(h)), abs(int(k)), abs(int(l))
    zeros = sum(x == 0 for x in [h, k, l])
    if system == "Cubic":
        eq = len({h, k, l})
        if eq == 1:    return 8
        if eq == 2:    return 24
        return 48
    elif system == "Tetragonal":
        if h == 0 and k == 0: return 2
        base = 4 if h == k else 8
        return base if l == 0 else base*2
    elif system == "Hexagonal":
        if h == 0 and k == 0: return 2
        return 12 if l != 0 else 6
    elif system == "Orthorhombic":
        return 2**(3-zeros)*2
    return max(2**(3-zeros), 1)


# ─────────────────────────────────────────────────────────────────────────────
# ATOMIC SCATTERING FACTORS (Cromer-Mann)
# ─────────────────────────────────────────────────────────────────────────────

CM = {
    "H":  ([0.4899,0.2620,0.1968,0.0499],[20.659,7.740,49.552,2.202],0.001),
    "C":  ([2.310,1.020,1.589,0.865],[20.844,10.208,0.569,51.651],0.216),
    "N":  ([12.213,3.132,2.013,1.166],[0.006,9.893,28.997,0.583],-11.529),
    "O":  ([3.049,2.287,1.546,0.867],[13.277,5.701,0.324,32.909],0.251),
    "Na": ([4.763,3.174,1.267,1.113],[3.285,8.842,0.314,129.424],0.676),
    "MG": ([5.420,2.174,1.227,2.307],[2.828,79.261,0.381,7.194],0.858),
    "AL": ([6.420,1.900,1.594,1.965],[3.039,0.743,31.547,85.089],1.115),
    "SI": ([6.292,3.035,1.989,1.541],[2.439,32.334,0.679,81.694],1.141),
    "CA": ([8.627,7.387,1.590,1.021],[10.442,0.660,85.748,178.437],1.375),
    "TI": ([9.760,7.359,1.699,1.902],[7.851,0.500,35.634,116.105],1.281),
    "FE": ([11.770,7.357,3.522,2.305],[4.761,0.307,15.354,76.881],1.037),
    "CU": ([13.338,7.168,5.616,1.674],[3.583,0.247,11.397,64.813],1.191),
    "ZN": ([14.074,7.032,5.165,2.410],[3.266,0.233,10.316,58.710],1.304),
    "LA": ([20.578,19.599,11.373,3.287],[2.948,0.244,18.773,133.124],2.147),
    "CE": ([21.167,19.770,11.851,3.330],[2.812,0.226,17.608,127.113],1.862),
    "BA": ([20.336,19.297,10.888,5.480],[3.216,0.275,20.207,109.460],2.775),
    "ZR": ([17.876,10.948,5.418,3.657],[1.276,11.916,0.118,87.663],2.069),
}

def f_atom(element, s):
    """Atomic scattering factor, s = sin(theta)/lambda."""
    key = element.upper()
    if key not in CM:
        return max(1.0, float(key[0].isalpha()))
    a4, b4, c = CM[key]
    s2 = s*s
    return c + sum(ai*np.exp(-bi*s2) for ai, bi in zip(a4, b4))


def structure_factor_sq(h, k, l, atoms, wl, tt):
    """|F_hkl|^2 including Debye-Waller."""
    theta = np.radians(tt/2)
    s = np.sin(theta)/wl
    Fr = Fi = 0.0
    for at in atoms:
        f   = f_atom(at["element"], s)
        DW  = np.exp(-at["Biso"]*s*s)
        phi = 2*np.pi*(h*at["x"] + k*at["y"] + l*at["z"])
        Fr += at["occ"]*f*DW*np.cos(phi)
        Fi += at["occ"]*f*DW*np.sin(phi)
    return Fr*Fr + Fi*Fi


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_pattern(tt_arr, refs, pr, bg_c, atoms=None, mode="lebail"):
    """Return (calculated_pattern, background_array)."""
    bg   = chebyshev_bg(tt_arr, bg_c)
    patt = np.zeros_like(tt_arr, dtype=float)
    U, V, W = pr["U"], pr["V"], pr["W"]
    eta0  = pr.get("eta0", 0.3)
    scale = pr["scale"]
    wl    = pr["wl"]
    system= pr.get("system", "Cubic")

    for ref in refs:
        h, k, l, d, tt_pk = ref[0], ref[1], ref[2], ref[3], ref[4]
        if not (tt_arr[0] <= tt_pk <= tt_arr[-1]):
            continue
        fwhm = caglioti_fwhm(tt_pk, U, V, W)
        eta  = np.clip(eta0, 0, 1)
        lp   = lp_factor(tt_pk)
        mult = multiplicity(h, k, l, system)

        if mode == "rietveld" and atoms:
            F2 = structure_factor_sq(h, k, l, atoms, wl, tt_pk)
        else:
            F2 = ref[5]

        prof   = pseudo_voigt(tt_arr, tt_pk, fwhm, eta)
        patt  += scale * mult * lp * F2 * prof

    return patt + bg, bg


# ─────────────────────────────────────────────────────────────────────────────
# R-FACTOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def r_factors(obs, calc, n_params=0):
    w      = 1.0 / np.maximum(obs, 1)
    Rwp    = 100*np.sqrt(np.sum(w*(obs-calc)**2) / np.sum(w*obs**2))
    Rp     = 100*np.sum(np.abs(obs-calc)) / np.sum(obs)
    chi2   = np.sum(w*(obs-calc)**2) / max(len(obs)-n_params, 1)
    GoF    = np.sqrt(chi2)
    return Rwp, Rp, chi2, GoF


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "Si  (cubic Fd-3m, a=5.431 Å)":   dict(system="Cubic",  sg="Fd-3m",  a=5.4309, b=5.4309, c=5.4309, al=90,be=90,ga=90,
                                            atoms=[{"element":"Si","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.46},
                                                   {"element":"Si","x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.46}]),
    "LaB6 (cubic Pm-3m, a=4.157 Å)":  dict(system="Cubic",  sg="Pm-3m",  a=4.1569, b=4.1569, c=4.1569, al=90,be=90,ga=90,
                                            atoms=[{"element":"La","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.20},
                                                   {"element":"B", "x":0.5,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.50},
                                                   {"element":"B", "x":0.0,"y":0.5,"z":0.0,"occ":1.0,"Biso":0.50},
                                                   {"element":"B", "x":0.0,"y":0.0,"z":0.5,"occ":1.0,"Biso":0.50}]),
    "CeO2 (cubic Fm-3m, a=5.411 Å)":  dict(system="Cubic",  sg="Fm-3m",  a=5.4124, b=5.4124, c=5.4124, al=90,be=90,ga=90,
                                            atoms=[{"element":"Ce","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.40},
                                                   {"element":"O", "x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.60}]),
    "Custom":                          dict(system="Cubic",  sg="P-1",    a=4.0, b=4.0, c=4.0, al=90,be=90,ga=90, atoms=[]),
}


# ─────────────────────────────────────────────────────────────────────────────
# SESSION-STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = dict(
        refs=None, obs_tt=None, obs_I=None,
        lb_result=None, rv_result=None,
        atoms=[{"element":"Si","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.5},
               {"element":"Si","x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.5}],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Experiment Setup")

    wl = st.number_input("Wavelength λ (Å)", 0.5, 3.0, 1.54056, 0.00001, "%.5f",
                         help="CuKα1 = 1.54056 Å, MoKα1 = 0.70930 Å")

    st.markdown("---")
    st.markdown("### 📂 Data")
    data_mode = st.radio("Source", ["Synthetic (preset)", "Upload XY file"], label_visibility="collapsed")

    if data_mode == "Upload XY file":
        uploaded = st.file_uploader("XY file (2θ  I per line)", type=["xy","dat","txt","csv"])
        if uploaded:
            lines = uploaded.read().decode().splitlines()
            pts = []
            for ln in lines:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) >= 2:
                    try:
                        pts.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
            if pts:
                arr = np.array(pts)
                st.session_state.obs_tt = arr[:, 0]
                st.session_state.obs_I  = arr[:, 1]
                st.success(f"Loaded {len(pts)} points")
    else:
        preset_key = st.selectbox("Preset material", list(PRESETS.keys()))
        preset = PRESETS[preset_key]

    st.markdown("---")
    st.markdown("### 🔷 Unit Cell")

    system_options = ["Cubic","Tetragonal","Orthorhombic","Hexagonal","Monoclinic","Triclinic"]
    system = st.selectbox("Crystal system",
                          system_options,
                          index=system_options.index(preset.get("system","Cubic") if data_mode=="Synthetic (preset)" else "Cubic"))

    c1, c2 = st.columns(2)
    a = c1.number_input("a (Å)", 0.5, 30.0,
                         preset.get("a",5.43) if data_mode=="Synthetic (preset)" else 5.43,
                         0.0001, "%.4f")
    if system == "Cubic":
        b = a; c = a
        c2.markdown(f"**b = c = a**")
    elif system in ("Tetragonal","Hexagonal"):
        b = a
        c = c2.number_input("c (Å)", 0.5, 30.0,
                              preset.get("c",5.43) if data_mode=="Synthetic (preset)" else 5.43,
                              0.0001, "%.4f")
        if system == "Tetragonal":
            st.markdown("b = a")
    else:
        b = c2.number_input("b (Å)", 0.5, 30.0,
                              preset.get("b",5.43) if data_mode=="Synthetic (preset)" else 5.43,
                              0.0001, "%.4f")
        c = c1.number_input("c (Å)", 0.5, 30.0,
                              preset.get("c",5.43) if data_mode=="Synthetic (preset)" else 5.43,
                              0.0001, "%.4f")

    if system in ("Monoclinic","Triclinic"):
        c3, c4, c5 = st.columns(3)
        al = c3.number_input("α°", 1.0, 179.0,
                              preset.get("al",90.0) if data_mode=="Synthetic (preset)" else 90.0,
                              0.01, "%.2f")
        be = c4.number_input("β°", 1.0, 179.0,
                              preset.get("be",90.0) if data_mode=="Synthetic (preset)" else 90.0,
                              0.01, "%.2f")
        ga = c5.number_input("γ°", 1.0, 179.0,
                              preset.get("ga",90.0) if data_mode=="Synthetic (preset)" else 90.0,
                              0.01, "%.2f")
    elif system == "Hexagonal":
        al, be, ga = 90.0, 90.0, 120.0
    else:
        al = be = ga = 90.0

    sg = st.text_input("Space group", preset.get("sg","P1") if data_mode=="Synthetic (preset)" else "P1")

    st.markdown("---")
    st.markdown("### 📐 2θ Range & Grid")
    c1, c2 = st.columns(2)
    tt_min = c1.number_input("Min 2θ (°)", 1.0, 170.0, 10.0, 0.5)
    tt_max = c2.number_input("Max 2θ (°)", 10.0, 170.0, 100.0, 0.5)
    n_pts  = st.slider("Grid points", 500, 5000, 2000, 100)

    st.markdown("---")
    st.markdown("### 📊 Profile Parameters")
    U     = st.number_input("U (Caglioti)",  0.0,  5.0,   0.010, 0.001, "%.4f")
    V     = st.number_input("V (Caglioti)", -1.0,  0.0,  -0.001, 0.001, "%.4f")
    W     = st.number_input("W (Caglioti)",  1e-4, 5.0,   0.005, 0.001, "%.4f")
    eta0  = st.number_input("η₀ (Lorentzian frac.)", 0.0, 1.0, 0.3, 0.01)
    scale = st.number_input("Scale factor", 0.001, 1e9, 1000.0, 100.0)

    st.markdown("---")
    st.markdown("### 🌐 Background")
    n_bg = st.slider("Chebyshev polynomial terms", 2, 8, 5)

    st.markdown("---")
    if st.button("🔄 Generate Reflections & Data", type="primary", use_container_width=True):
        refs = gen_reflections(a, b, c, al, be, ga, system, sg, wl, tt_min, tt_max)
        st.session_state.refs = refs
        st.session_state.lb_result = None
        st.session_state.rv_result = None

        # Build base profile params
        pr0 = dict(U=U, V=V, W=W, eta0=eta0, scale=scale, wl=wl, system=system)
        bg0 = np.zeros(n_bg); bg0[0] = 80.0; bg0[1] = -20.0

        tt_arr = np.linspace(tt_min, tt_max, n_pts)

        if data_mode == "Synthetic (preset)":
            # Use preset atoms
            atoms_pr = PRESETS[preset_key].get("atoms", [])
            if atoms_pr:
                pat, _ = calc_pattern(tt_arr, refs, pr0, bg0,
                                       atoms=atoms_pr, mode="rietveld")
            else:
                pat, _ = calc_pattern(tt_arr, refs, pr0, bg0, mode="lebail")
            noise = np.random.default_rng(42).normal(
                0, np.sqrt(np.abs(pat)+1)*0.04)
            st.session_state.obs_tt = tt_arr
            st.session_state.obs_I  = np.maximum(pat + noise, 0)
        elif st.session_state.obs_tt is None:
            st.warning("Upload a data file first, or use Synthetic mode.")

        st.success(f"✅ {len(refs)} reflections generated")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 Diffraction Analyser — Full Profile Refinement")

# Convenience references
refs   = st.session_state.refs
obs_tt = st.session_state.obs_tt
obs_I  = st.session_state.obs_I
have_data = obs_tt is not None and obs_I is not None and refs is not None

def bragg_ticks(refs, tt_min, tt_max, y0, dy=-0.04, max_ticks=300):
    """Return plotly shapes + trace for tick marks."""
    shapes, xs, ys = [], [], []
    for ref in refs[:max_ticks]:
        tt_pk = ref[4]
        if tt_min <= tt_pk <= tt_max:
            xs += [tt_pk, tt_pk, None]
            ys += [y0, y0+dy, None]
    return xs, ys

# ─── TABS ───────────────────────────────────────────────────────────────────
tab_data, tab_lb, tab_rv, tab_results = st.tabs(
    ["📈 Pattern", "⚗️ Le Bail Fit", "🔬 Rietveld Fit", "📋 Results"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PATTERN VIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    if not have_data:
        st.info("👈 Configure the sidebar and click **Generate Reflections & Data** to start.")
    else:
        ymax = obs_I.max()
        tick_x, tick_y = bragg_ticks(refs, obs_tt[0], obs_tt[-1],
                                      y0=ymax, dy=-ymax*0.04)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=obs_tt, y=obs_I, mode="lines",
                                  name="Observed", line=dict(color="#1f77b4", width=1.2)))
        if tick_x:
            fig.add_trace(go.Scatter(x=tick_x, y=tick_y, mode="lines",
                                      name="Bragg positions",
                                      line=dict(color="red", width=1), showlegend=True,
                                      hoverinfo="skip"))
        fig.update_layout(
            xaxis_title="2θ (°)", yaxis_title="Intensity (counts)",
            template="plotly_white", height=480,
            title=f"Observed Pattern — {len(refs)} reflections  |  λ = {wl:.5f} Å",
            legend=dict(x=0.75, y=0.95))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Data points",     f"{len(obs_tt):,}")
        col2.metric("Reflections",     f"{len(refs)}")
        col3.metric("2θ range",        f"{obs_tt[0]:.1f}° – {obs_tt[-1]:.1f}°")

        # HKL table
        with st.expander("📄 Reflection list (first 60)"):
            rows = [{"h":int(r[0]),"k":int(r[1]),"l":int(r[2]),
                     "d (Å)":f"{r[3]:.4f}","2θ (°)":f"{r[4]:.3f}"}
                    for r in refs[:60]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — LE BAIL
# ════════════════════════════════════════════════════════════════════════════
with tab_lb:
    st.markdown("""
    **Le Bail method** (Armel Le Bail, 1988) extracts integrated intensities |F_hkl|² without
    an atomic model, via iterative profile fitting. Only the unit-cell, profile shape, and
    background are refined.
    """)

    if not have_data:
        st.warning("Generate data first (sidebar).")
    else:
        ctrl_col, plot_col = st.columns([1, 2.5])

        with ctrl_col:
            st.markdown("#### Refinement switches")
            lb_scale   = st.checkbox("Scale",           True,  key="lb_sc")
            lb_profile = st.checkbox("Profile U, V, W", True,  key="lb_prf")
            lb_eta     = st.checkbox("Mixing η₀",       True,  key="lb_eta")
            lb_bg      = st.checkbox("Background",      True,  key="lb_bg")
            lb_iters   = st.number_input("Le Bail cycles", 20, 500, 100, 10)

            run_lb = st.button("▶ Run Le Bail", type="primary", use_container_width=True)

        # ── Le Bail algorithm ──────────────────────────────────────────────
        if run_lb:
            tt_arr  = obs_tt
            obs     = obs_I
            refs_lb = [list(r) for r in refs]  # local mutable copy

            # Init intensities
            for r in refs_lb:
                r[5] = float(scale)

            pr = dict(U=U, V=V, W=W, eta0=eta0, scale=scale, wl=wl, system=system)
            bg = np.zeros(n_bg); bg[0] = float(np.percentile(obs, 3))

            bar = st.progress(0, text="Running Le Bail…")

            for cycle in range(int(lb_iters)):
                # ① Current calculated pattern
                calc, bgv = calc_pattern(tt_arr, refs_lb, pr, bg, mode="lebail")

                # ② Distribute observed to each reflection (Le Bail formula)
                for i, ref in enumerate(refs_lb):
                    tt_pk = ref[4]
                    fwhm  = caglioti_fwhm(tt_pk, pr["U"], pr["V"], pr["W"])
                    p_k   = pseudo_voigt(tt_arr, tt_pk, fwhm, pr["eta0"])
                    p_sum = p_k.sum()
                    if p_sum < 1e-12:
                        continue
                    # share of calculated (excl. background)
                    I_old  = ref[5]
                    calc_nb = np.maximum(calc - bgv, 0)
                    total_at_peak = np.dot(calc_nb, p_k) / (p_sum + 1e-12)
                    obs_nb = np.maximum(obs - bgv, 0)
                    I_new  = I_old * (np.dot(obs_nb, p_k) / (np.dot(calc_nb, p_k) + 1e-6))
                    refs_lb[i][5] = max(I_new, 1e-3)

                # ③ Every 5 cycles refine profile/BG with least-squares
                if cycle % 5 == 4:
                    x0_ls, lo_ls, hi_ls, keys_ls = [], [], [], []
                    if lb_scale:
                        x0_ls.append(pr["scale"]); lo_ls.append(1e-3); hi_ls.append(1e10); keys_ls.append("scale")
                    if lb_profile:
                        x0_ls += [pr["U"], pr["V"], pr["W"]]
                        lo_ls  += [0.0, -2.0, 1e-4]; hi_ls += [20.0, 0.0, 20.0]
                        keys_ls += ["U", "V", "W"]
                    if lb_eta:
                        x0_ls.append(pr["eta0"]); lo_ls.append(0.0); hi_ls.append(1.0); keys_ls.append("eta0")
                    if lb_bg:
                        x0_ls += list(bg)
                        lo_ls  += [-1e6]*n_bg; hi_ls += [1e6]*n_bg
                        keys_ls += [f"bg{j}" for j in range(n_bg)]

                    if x0_ls:
                        def _res(p):
                            pr_t = pr.copy(); bg_t = bg.copy(); idx = 0
                            for key in keys_ls:
                                if key.startswith("bg"):
                                    j = int(key[2:])
                                    bg_t[j] = p[idx]
                                else:
                                    pr_t[key] = p[idx]
                                idx += 1
                            c, _ = calc_pattern(tt_arr, refs_lb, pr_t, bg_t, mode="lebail")
                            w = 1.0/np.maximum(obs, 1)
                            return np.sqrt(w)*(obs - c)

                        try:
                            res = least_squares(_res, x0_ls,
                                                bounds=(lo_ls, hi_ls),
                                                max_nfev=30, method="trf")
                            idx = 0
                            for key in keys_ls:
                                if key.startswith("bg"):
                                    bg[int(key[2:])] = res.x[idx]
                                else:
                                    pr[key] = res.x[idx]
                                idx += 1
                        except Exception:
                            pass

                bar.progress((cycle+1)/int(lb_iters),
                             text=f"Le Bail cycle {cycle+1}/{lb_iters}")

            bar.empty()

            calc_f, bg_f = calc_pattern(tt_arr, refs_lb, pr, bg, mode="lebail")
            Rwp, Rp, chi2, GoF = r_factors(obs, calc_f, len(x0_ls) if x0_ls else 0)

            st.session_state.lb_result = dict(
                refs=refs_lb, pr=pr, bg=bg, calc=calc_f, bgv=bg_f,
                Rwp=Rwp, Rp=Rp, chi2=chi2, GoF=GoF)

        # ── Plot ─────────────────────────────────────────────────────────────
        with plot_col:
            res = st.session_state.lb_result
            if res:
                tt_arr = obs_tt; obs = obs_I
                calc   = res["calc"]; bgv = res["bgv"]
                diff   = obs - calc

                fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28],
                                    shared_xaxes=True, vertical_spacing=0.03)

                fig.add_trace(go.Scatter(x=tt_arr, y=obs,  mode="lines", name="Observed",
                                         line=dict(color="#1f77b4",width=1.2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=tt_arr, y=calc, mode="lines", name="Calculated",
                                         line=dict(color="#ff7f0e",width=1.8)), row=1, col=1)
                fig.add_trace(go.Scatter(x=tt_arr, y=bgv,  mode="lines", name="Background",
                                         line=dict(color="#9467bd",width=1,dash="dash")), row=1, col=1)

                ymax_ = obs.max()
                tx, ty = bragg_ticks(res["refs"], tt_arr[0], tt_arr[-1],
                                      ymax_*1.02, -ymax_*0.04)
                if tx:
                    fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines",
                                              name="Bragg pos.",
                                              line=dict(color="#2ca02c",width=1),
                                              hoverinfo="skip"), row=1, col=1)

                fig.add_trace(go.Scatter(x=tt_arr, y=diff, mode="lines", name="Difference",
                                         line=dict(color="#d62728",width=1)), row=2, col=1)
                fig.add_hline(y=0, line_color="gray", line_dash="dot", row=2, col=1)

                fig.update_layout(
                    height=520, template="plotly_white",
                    title=f"Le Bail  |  Rwp = {res['Rwp']:.2f}%  Rp = {res['Rp']:.2f}%  χ² = {res['chi2']:.3f}",
                    xaxis2_title="2θ (°)",
                    yaxis_title="Intensity",
                    yaxis2_title="Δ",
                    legend=dict(x=0.75, y=0.95),
                    margin=dict(t=50))
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rwp", f"{res['Rwp']:.2f}%")
                m2.metric("Rp",  f"{res['Rp']:.2f}%")
                m3.metric("χ²",  f"{res['chi2']:.3f}")
                m4.metric("GoF", f"{res['GoF']:.3f}")

                pr_ = res["pr"]
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("Scale", f"{pr_['scale']:.1f}")
                p2.metric("U",     f"{pr_['U']:.5f}")
                p3.metric("V",     f"{pr_['V']:.5f}")
                p4.metric("W",     f"{pr_['W']:.5f}")
                p5.metric("η₀",    f"{pr_['eta0']:.4f}")
            else:
                st.info("← Configure and click **Run Le Bail**")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — RIETVELD
# ════════════════════════════════════════════════════════════════════════════
with tab_rv:
    st.markdown("""
    **Rietveld refinement** fits the full crystal structure — atomic positions, occupancies,
    and Debye-Waller factors — together with profile and background parameters.
    """)

    if not have_data:
        st.warning("Generate data first (sidebar).")
    else:
        ctrl_col, plot_col = st.columns([1, 2.5])

        with ctrl_col:
            st.markdown("#### Atomic Structure")

            # Atom table editor
            if data_mode == "Synthetic (preset)" and st.button("Load preset atoms", use_container_width=True):
                preset_atoms = PRESETS[preset_key].get("atoms", [])
                if preset_atoms:
                    st.session_state.atoms = [dict(a) for a in preset_atoms]

            atoms_in = st.session_state.atoms
            updated_atoms = []
            for i, at in enumerate(atoms_in):
                with st.expander(f"Atom {i+1}: {at['element']}", expanded=i < 3):
                    el  = st.text_input("Element", at["element"], key=f"el_{i}")
                    c1o, c2o, c3o = st.columns(3)
                    x   = c1o.number_input("x", 0.0, 1.0, at["x"],   0.001, "%.4f", key=f"ax_{i}")
                    y   = c2o.number_input("y", 0.0, 1.0, at["y"],   0.001, "%.4f", key=f"ay_{i}")
                    z   = c3o.number_input("z", 0.0, 1.0, at["z"],   0.001, "%.4f", key=f"az_{i}")
                    o1, o2 = st.columns(2)
                    occ  = o1.number_input("Occ", 0.0, 1.0, at["occ"], 0.01,  "%.3f", key=f"oc_{i}")
                    Biso = o2.number_input("Biso", 0.0, 20.0, at["Biso"], 0.01, "%.3f", key=f"Bs_{i}")
                    updated_atoms.append({"element":el,"x":x,"y":y,"z":z,"occ":occ,"Biso":Biso})
            st.session_state.atoms = updated_atoms

            ca, cr = st.columns(2)
            if ca.button("➕ Atom", use_container_width=True):
                st.session_state.atoms.append({"element":"O","x":0.5,"y":0.5,"z":0.5,"occ":1.0,"Biso":1.0})
                st.rerun()
            if cr.button("➖ Last", use_container_width=True) and len(st.session_state.atoms) > 1:
                st.session_state.atoms.pop()
                st.rerun()

            st.markdown("#### Refinement switches")
            rv_scale   = st.checkbox("Scale",           True,  key="rv_sc")
            rv_profile = st.checkbox("Profile U, V, W", True,  key="rv_prf")
            rv_eta     = st.checkbox("Mixing η₀",       False, key="rv_eta")
            rv_bg      = st.checkbox("Background",      True,  key="rv_bg")
            rv_pos     = st.checkbox("Atomic x, y, z",  False, key="rv_pos",
                                     help="Refine fractional coordinates")
            rv_Biso    = st.checkbox("Biso",            True,  key="rv_Biso")
            rv_occ     = st.checkbox("Occupancies",     False, key="rv_occ")

            run_rv = st.button("▶ Run Rietveld", type="primary", use_container_width=True)

        # ── Rietveld algorithm ────────────────────────────────────────────
        if run_rv:
            atoms_work = [dict(a) for a in st.session_state.atoms]
            tt_arr = obs_tt; obs = obs_I

            # Seed from Le Bail if available
            if st.session_state.lb_result:
                pr = dict(st.session_state.lb_result["pr"])
                bg = st.session_state.lb_result["bg"].copy()
            else:
                pr = dict(U=U, V=V, W=W, eta0=eta0, scale=scale, wl=wl, system=system)
                bg = np.zeros(n_bg); bg[0] = float(np.percentile(obs, 3))

            # Pack / unpack helpers
            def pack(pr_, bg_, atl):
                p = []
                if rv_scale:   p.append(pr_["scale"])
                if rv_profile: p += [pr_["U"], pr_["V"], pr_["W"]]
                if rv_eta:     p.append(pr_["eta0"])
                if rv_bg:      p += list(bg_)
                for at in atl:
                    if rv_pos:  p += [at["x"], at["y"], at["z"]]
                    if rv_Biso: p.append(at["Biso"])
                    if rv_occ:  p.append(at["occ"])
                return np.array(p, dtype=float)

            def unpack(p, pr_, bg_, atl):
                pr_  = dict(pr_); bg_ = bg_.copy()
                atl  = [dict(a) for a in atl]
                idx  = 0
                if rv_scale:   pr_["scale"] = max(p[idx], 1e-6); idx+=1
                if rv_profile: pr_["U"]=max(p[idx],0); idx+=1; pr_["V"]=p[idx]; idx+=1; pr_["W"]=max(p[idx],1e-4); idx+=1
                if rv_eta:     pr_["eta0"] = np.clip(p[idx],0,1); idx+=1
                if rv_bg:      bg_[:]=p[idx:idx+n_bg]; idx+=n_bg
                for j in range(len(atl)):
                    if rv_pos:  atl[j]["x"]=p[idx]%1; idx+=1; atl[j]["y"]=p[idx]%1; idx+=1; atl[j]["z"]=p[idx]%1; idx+=1
                    if rv_Biso: atl[j]["Biso"]=max(p[idx],0.01); idx+=1
                    if rv_occ:  atl[j]["occ"]=np.clip(p[idx],0.01,1); idx+=1
                return pr_, bg_, atl

            def residuals_rv(p):
                pr_t, bg_t, at_t = unpack(p, pr, bg, atoms_work)
                cal, _ = calc_pattern(tt_arr, refs, pr_t, bg_t, atoms=at_t, mode="rietveld")
                w = 1.0/np.maximum(obs, 1)
                return np.sqrt(w)*(obs - cal)

            x0 = pack(pr, bg, atoms_work)
            lo, hi = [], []
            if rv_scale:   lo.append(1e-6);  hi.append(1e10)
            if rv_profile: lo+=[0,-2,1e-4];  hi+=[20,0,20]
            if rv_eta:     lo.append(0);     hi.append(1)
            if rv_bg:      lo+=[-1e6]*n_bg;  hi+=[1e6]*n_bg
            for _ in atoms_work:
                if rv_pos:  lo+=[0,0,0];    hi+=[1,1,1]
                if rv_Biso: lo.append(0.01); hi.append(30)
                if rv_occ:  lo.append(0.01); hi.append(1)

            with st.spinner("Running Rietveld refinement…"):
                try:
                    res_ls = least_squares(residuals_rv, x0, bounds=(lo, hi),
                                           max_nfev=1000, ftol=1e-10, xtol=1e-10,
                                           method="trf")
                    pr_f, bg_f, at_f = unpack(res_ls.x, pr, bg, atoms_work)
                except Exception as e:
                    st.error(f"Refinement error: {e}")
                    pr_f, bg_f, at_f = pr, bg, atoms_work

            calc_f, bgv_f = calc_pattern(tt_arr, refs, pr_f, bg_f,
                                          atoms=at_f, mode="rietveld")
            Rwp, Rp, chi2, GoF = r_factors(obs, calc_f, len(x0))

            st.session_state.rv_result = dict(
                pr=pr_f, bg=bg_f, atoms=at_f,
                calc=calc_f, bgv=bgv_f,
                Rwp=Rwp, Rp=Rp, chi2=chi2, GoF=GoF)

        # ── Plot ─────────────────────────────────────────────────────────────
        with plot_col:
            res = st.session_state.rv_result
            if res:
                tt_arr = obs_tt; obs = obs_I
                calc   = res["calc"]; bgv = res["bgv"]
                diff   = obs - calc

                fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28],
                                    shared_xaxes=True, vertical_spacing=0.03)

                fig.add_trace(go.Scatter(x=tt_arr, y=obs,  mode="lines", name="Observed",
                                         line=dict(color="#1f77b4",width=1.2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=tt_arr, y=calc, mode="lines", name="Calculated",
                                         line=dict(color="#ff7f0e",width=1.8)), row=1, col=1)
                fig.add_trace(go.Scatter(x=tt_arr, y=bgv,  mode="lines", name="Background",
                                         line=dict(color="#9467bd",width=1,dash="dash")), row=1, col=1)

                ymax_ = obs.max()
                tx, ty = bragg_ticks(refs, tt_arr[0], tt_arr[-1],
                                      ymax_*1.02, -ymax_*0.04)
                if tx:
                    fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines",
                                              name="Bragg pos.",
                                              line=dict(color="#2ca02c",width=1),
                                              hoverinfo="skip"), row=1, col=1)

                fig.add_trace(go.Scatter(x=tt_arr, y=diff, mode="lines", name="Difference",
                                         line=dict(color="#d62728",width=1)), row=2, col=1)
                fig.add_hline(y=0, line_color="gray", line_dash="dot", row=2, col=1)

                fig.update_layout(
                    height=520, template="plotly_white",
                    title=(f"Rietveld  |  Rwp = {res['Rwp']:.2f}%  "
                           f"Rp = {res['Rp']:.2f}%  χ² = {res['chi2']:.3f}  GoF = {res['GoF']:.3f}"),
                    xaxis2_title="2θ (°)",
                    yaxis_title="Intensity",
                    yaxis2_title="Δ",
                    legend=dict(x=0.75, y=0.95),
                    margin=dict(t=50))
                st.plotly_chart(fig, use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rwp", f"{res['Rwp']:.2f}%")
                m2.metric("Rp",  f"{res['Rp']:.2f}%")
                m3.metric("χ²",  f"{res['chi2']:.3f}")
                m4.metric("GoF", f"{res['GoF']:.3f}")

                pr_ = res["pr"]
                p1,p2,p3,p4,p5 = st.columns(5)
                p1.metric("Scale", f"{pr_['scale']:.2f}")
                p2.metric("U",     f"{pr_['U']:.5f}")
                p3.metric("V",     f"{pr_['V']:.5f}")
                p4.metric("W",     f"{pr_['W']:.5f}")
                p5.metric("η₀",    f"{pr_['eta0']:.4f}")

                st.markdown("**Refined atoms:**")
                st.dataframe(pd.DataFrame(res["atoms"]), use_container_width=True)
            else:
                st.info("← Add atoms and click **Run Rietveld**")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS SUMMARY
# ════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown("## Results Summary")

    lb = st.session_state.lb_result
    rv = st.session_state.rv_result

    # ── R-factor comparison ───────────────────────────────────────────────
    methods, Rwps, Rps, chi2s, GoFs = [], [], [], [], []
    if lb:
        methods.append("Le Bail")
        Rwps.append(lb["Rwp"]); Rps.append(lb["Rp"])
        chi2s.append(lb["chi2"]); GoFs.append(lb["GoF"])
    if rv:
        methods.append("Rietveld")
        Rwps.append(rv["Rwp"]); Rps.append(rv["Rp"])
        chi2s.append(rv["chi2"]); GoFs.append(rv["GoF"])

    if methods:
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(x=methods, y=Rwps, name="Rwp (%)", marker_color="#1f77b4"))
        fig_r.add_trace(go.Bar(x=methods, y=Rps,  name="Rp (%)",  marker_color="#ff7f0e"))
        fig_r.update_layout(barmode="group", template="plotly_white",
                             title="R-factor comparison", height=300,
                             yaxis_title="R (%)", legend=dict(x=0.8, y=0.95))
        st.plotly_chart(fig_r, use_container_width=True)

        summary_df = pd.DataFrame({
            "Method": methods, "Rwp (%)": [f"{v:.3f}" for v in Rwps],
            "Rp (%)":  [f"{v:.3f}" for v in Rps],
            "χ²":      [f"{v:.4f}" for v in chi2s],
            "GoF":     [f"{v:.4f}" for v in GoFs],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run Le Bail and/or Rietveld to see results here.")

    # ── Extracted Le Bail intensities ─────────────────────────────────────
    if lb and lb.get("refs"):
        st.markdown("### Le Bail — Extracted |F²| (first 80 reflections)")
        rows = []
        for r in lb["refs"][:80]:
            rows.append({"h":int(r[0]),"k":int(r[1]),"l":int(r[2]),
                         "d (Å)":round(r[3],4),"2θ (°)":round(r[4],3),
                         "|F²| (arb.)":round(r[5],2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

    # ── Refined atom table ────────────────────────────────────────────────
    if rv and rv.get("atoms"):
        st.markdown("### Rietveld — Refined Atomic Parameters")
        st.dataframe(pd.DataFrame(rv["atoms"]).round(5), use_container_width=True)

    # ── Exports ──────────────────────────────────────────────────────────
    st.markdown("### 💾 Export")
    ec1, ec2, ec3 = st.columns(3)

    if ec1.button("📥 Export Pattern CSV") and have_data:
        df_exp = pd.DataFrame({"2theta_deg": obs_tt, "observed": obs_I})
        if lb:
            df_exp["LeBail_calc"]  = lb["calc"]
            df_exp["LeBail_bg"]    = lb["bgv"]
            df_exp["LeBail_diff"]  = obs_I - lb["calc"]
        if rv:
            df_exp["Rietveld_calc"] = rv["calc"]
            df_exp["Rietveld_bg"]   = rv["bgv"]
            df_exp["Rietveld_diff"] = obs_I - rv["calc"]
        ec1.download_button("⬇ Download", df_exp.to_csv(index=False),
                            "diffraction_pattern.csv", "text/csv")

    if ec2.button("📥 Export Reflections CSV") and refs is not None:
        rows_e = [{"h":int(r[0]),"k":int(r[1]),"l":int(r[2]),
                   "d_Ang":round(r[3],5),"2theta_deg":round(r[4],4),
                   "LeBail_I2":round(r[5],3) if len(r)>5 else None}
                  for r in (lb["refs"] if lb else refs)]
        ec2.download_button("⬇ Download", pd.DataFrame(rows_e).to_csv(index=False),
                            "reflections.csv", "text/csv")

    if ec3.button("📥 Export Atoms CSV") and rv and rv.get("atoms"):
        ec3.download_button("⬇ Download", pd.DataFrame(rv["atoms"]).to_csv(index=False),
                            "refined_atoms.csv", "text/csv")