
import streamlit as st
import pandas as pd

# Data from Main Page
main_df = st.session_state.get('main_df')
comp_df = st.session_state.get('comp_df')
cif_data = st.session_state.get('cif_data')

if main_df is None:
    st.warning("Main XRD pattern missing. Please upload it on the Main Page.")
    st.stop()

     1|﻿"""
     2|Diffraction Analyser — Full Profile Refinement (Le Bail + Rietveld)
     3|Run with:  streamlit run diffraction_analyser.py
     4|Requires:  pip install streamlit numpy scipy plotly pandas
     5|"""
     6|
     7|import streamlit as st
     8|import numpy as np
     9|import plotly.graph_objects as go
    10|from plotly.subplots import make_subplots
    11|from scipy.optimize import least_squares
    12|import pandas as pd
    13|import warnings
    14|
    15|warnings.filterwarnings("ignore")
    16|
    17|st.set_page_config(
    18|    page_title="Diffraction Analyser",
    19|    page_icon="🔬",
    20|    layout="wide",
    21|    initial_sidebar_state="expanded",
    22|)
    23|
    24|# ─────────────────────────────────────────────────────────────────────────────
    25|# CSS
    26|# ─────────────────────────────────────────────────────────────────────────────
    27|st.markdown("""
    28|<style>
    29|[data-testid="stMetricValue"] { font-size: 1.3rem; }
    30|.block-container { padding-top: 1rem; }
    31|.stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    32|h1 { font-size: 1.8rem !important; }
    33|h2 { font-size: 1.2rem !important; }
    34|h3 { font-size: 1.05rem !important; }
    35|</style>
    36|""", unsafe_allow_html=True)
    37|
    38|# ─────────────────────────────────────────────────────────────────────────────
    39|# CRYSTALLOGRAPHY UTILITIES
    40|# ─────────────────────────────────────────────────────────────────────────────
    41|
    42|def d_spacing(h, k, l, a, b, c, alpha_deg, beta_deg, gamma_deg, system):
    43|    """Return d-spacing (Å) for a given reflection and crystal system."""
    44|    ar, br, gr = (np.radians(x) for x in (alpha_deg, beta_deg, gamma_deg))
    45|    h, k, l = float(h), float(k), float(l)
    46|
    47|    if system == "Cubic":
    48|        inv = (h**2 + k**2 + l**2) / a**2
    49|    elif system == "Tetragonal":
    50|        inv = (h**2 + k**2) / a**2 + l**2 / c**2
    51|    elif system == "Orthorhombic":
    52|        inv = h**2/a**2 + k**2/b**2 + l**2/c**2
    53|    elif system == "Hexagonal":
    54|        inv = 4/3*(h**2 + h*k + k**2)/a**2 + l**2/c**2
    55|    elif system == "Monoclinic":
    56|        sb = np.sin(br)
    57|        inv = (1/sb**2)*(h**2/a**2 + k**2*sb**2/b**2 + l**2/c**2
    58|                         - 2*h*l*np.cos(br)/(a*c))
    59|    else:  # Triclinic
    60|        ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
    61|        sa, sb, sg = np.sin(ar), np.sin(br), np.sin(gr)
    62|        V = a*b*c*np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg)
    63|        if V < 1e-10:
    64|            return None
    65|        inv = (b**2*c**2*sa**2*h**2 + a**2*c**2*sb**2*k**2 + a**2*b**2*sg**2*l**2
    66|               + 2*a*b*c**2*(ca*cb-cg)*h*k
    67|               + 2*a**2*b*c*(cb*cg-ca)*k*l
    68|               + 2*a*b**2*c*(ca*cg-cb)*h*l) / V**2
    69|
    70|    return None if inv <= 1e-12 else 1.0/np.sqrt(inv)
    71|
    72|
    73|def is_absent(h, k, l, sg):
    74|    """Very simplified systematic absence check based on lattice centering."""
    75|    h, k, l = int(h), int(k), int(l)
    76|    sg = sg.upper().replace(" ", "")
    77|    if sg.startswith("I") and (h+k+l) % 2 != 0:
    78|        return True
    79|    if sg.startswith("F"):
    80|        parities = {h%2, k%2, l%2}
    81|        if len(parities) > 1:
    82|            return True
    83|    if sg.startswith("C") and (h+k) % 2 != 0:
    84|        return True
    85|    if sg.startswith("A") and (k+l) % 2 != 0:
    86|        return True
    87|    if sg.startswith("B") and (h+l) % 2 != 0:
    88|        return True
    89|    # FD screw axes (very simplified)
    90|    if "FD" in sg or "Fd" in sg.replace("-",""):
    91|        if h == 0 and k == 0 and l % 4 != 0:
    92|            return True
    93|    return False
    94|
    95|
    96|def gen_reflections(a, b, c, alpha, beta, gamma, system, sg, wl, tt_min, tt_max):
    97|    """Return list of (h,k,l,d,2theta,I_lb) for all allowed reflections."""
    98|    d_min = wl / (2*np.sin(np.radians(tt_max/2)))
    99|    d_max = wl / (2*np.sin(np.radians(max(tt_min, 0.5)/2)))
   100|    mh = int(2*a/d_min)+2
   101|    mk = int(2*b/d_min)+2
   102|    ml = int(2*c/d_min)+2
   103|
   104|    seen = {}
   105|    for h in range(-mh, mh+1):
   106|        for k in range(-mk, mk+1):
   107|            for l in range(-ml, ml+1):
   108|                if h == k == l == 0:
   109|                    continue
   110|                if is_absent(h, k, l, sg):
   111|                    continue
   112|                d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma, system)
   113|                if d is None or not (d_min <= d <= d_max):
   114|                    continue
   115|                tt = 2*np.degrees(np.arcsin(np.clip(wl/(2*d), -1, 1)))
   116|                key = round(tt, 4)
   117|                if key not in seen:
   118|                    seen[key] = [h, k, l, d, tt, 1000.0]
   119|    refs = sorted(seen.values(), key=lambda r: r[4])
   120|    return refs
   121|
   122|
   123|# ─────────────────────────────────────────────────────────────────────────────
   124|# PROFILE & BACKGROUND FUNCTIONS
   125|# ─────────────────────────────────────────────────────────────────────────────
   126|
   127|def pseudo_voigt(x, x0, fwhm, eta):
   128|    """Normalised pseudo-Voigt profile."""
   129|    eta = np.clip(eta, 0, 1)
   130|    fwhm = max(fwhm, 1e-6)
   131|    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
   132|    G = np.exp(-0.5*((x-x0)/sigma)**2)
   133|    L = 1.0 / (1 + ((x-x0)/(fwhm/2))**2)
   134|    return eta*L + (1-eta)*G
   135|
   136|
   137|def caglioti_fwhm(tt, U, V, W):
   138|    th = np.radians(tt/2)
   139|    tan_th = np.tan(th)
   140|    return max(np.sqrt(max(U*tan_th**2 + V*tan_th + W, 1e-8)), 0.005)
   141|
   142|
   143|def chebyshev_bg(tt, coeffs):
   144|    x = 2*(tt - tt.min())/(tt.max()-tt.min()) - 1
   145|    T = [np.ones_like(x), x, 2*x**2-1, 4*x**3-3*x,
   146|         8*x**4-8*x**2+1, 16*x**5-20*x**3+5*x,
   147|         32*x**6-48*x**4+18*x**2-1, 64*x**7-112*x**5+56*x**3-7*x]
   148|    result = np.zeros_like(x)
   149|    for i, c in enumerate(coeffs):
   150|        result += c * T[i]
   151|    return result
   152|
   153|
   154|def lp_factor(tt):
   155|    th = np.radians(tt/2)
   156|    cos2 = np.cos(np.radians(tt))**2
   157|    return (1+cos2) / (np.sin(th)**2 * np.cos(th) + 1e-12)
   158|
   159|
   160|def multiplicity(h, k, l, system):
   161|    h, k, l = abs(int(h)), abs(int(k)), abs(int(l))
   162|    zeros = sum(x == 0 for x in [h, k, l])
   163|    if system == "Cubic":
   164|        eq = len({h, k, l})
   165|        if eq == 1:    return 8
   166|        if eq == 2:    return 24
   167|        return 48
   168|    elif system == "Tetragonal":
   169|        if h == 0 and k == 0: return 2
   170|        base = 4 if h == k else 8
   171|        return base if l == 0 else base*2
   172|    elif system == "Hexagonal":
   173|        if h == 0 and k == 0: return 2
   174|        return 12 if l != 0 else 6
   175|    elif system == "Orthorhombic":
   176|        return 2**(3-zeros)*2
   177|    return max(2**(3-zeros), 1)
   178|
   179|
   180|# ─────────────────────────────────────────────────────────────────────────────
   181|# ATOMIC SCATTERING FACTORS (Cromer-Mann)
   182|# ─────────────────────────────────────────────────────────────────────────────
   183|
   184|CM = {
   185|    "H":  ([0.4899,0.2620,0.1968,0.0499],[20.659,7.740,49.552,2.202],0.001),
   186|    "C":  ([2.310,1.020,1.589,0.865],[20.844,10.208,0.569,51.651],0.216),
   187|    "N":  ([12.213,3.132,2.013,1.166],[0.006,9.893,28.997,0.583],-11.529),
   188|    "O":  ([3.049,2.287,1.546,0.867],[13.277,5.701,0.324,32.909],0.251),
   189|    "Na": ([4.763,3.174,1.267,1.113],[3.285,8.842,0.314,129.424],0.676),
   190|    "MG": ([5.420,2.174,1.227,2.307],[2.828,79.261,0.381,7.194],0.858),
   191|    "AL": ([6.420,1.900,1.594,1.965],[3.039,0.743,31.547,85.089],1.115),
   192|    "SI": ([6.292,3.035,1.989,1.541],[2.439,32.334,0.679,81.694],1.141),
   193|    "CA": ([8.627,7.387,1.590,1.021],[10.442,0.660,85.748,178.437],1.375),
   194|    "TI": ([9.760,7.359,1.699,1.902],[7.851,0.500,35.634,116.105],1.281),
   195|    "FE": ([11.770,7.357,3.522,2.305],[4.761,0.307,15.354,76.881],1.037),
   196|    "CU": ([13.338,7.168,5.616,1.674],[3.583,0.247,11.397,64.813],1.191),
   197|    "ZN": ([14.074,7.032,5.165,2.410],[3.266,0.233,10.316,58.710],1.304),
   198|    "LA": ([20.578,19.599,11.373,3.287],[2.948,0.244,18.773,133.124],2.147),
   199|    "CE": ([21.167,19.770,11.851,3.330],[2.812,0.226,17.608,127.113],1.862),
   200|    "BA": ([20.336,19.297,10.888,5.480],[3.216,0.275,20.207,109.460],2.775),
   201|    "ZR": ([17.876,10.948,5.418,3.657],[1.276,11.916,0.118,87.663],2.069),
   202|}
   203|
   204|def f_atom(element, s):
   205|    """Atomic scattering factor, s = sin(theta)/lambda."""
   206|    key = element.upper()
   207|    if key not in CM:
   208|        return max(1.0, float(key[0].isalpha()))
   209|    a4, b4, c = CM[key]
   210|    s2 = s*s
   211|    return c + sum(ai*np.exp(-bi*s2) for ai, bi in zip(a4, b4))
   212|
   213|
   214|def structure_factor_sq(h, k, l, atoms, wl, tt):
   215|    """|F_hkl|^2 including Debye-Waller."""
   216|    theta = np.radians(tt/2)
   217|    s = np.sin(theta)/wl
   218|    Fr = Fi = 0.0
   219|    for at in atoms:
   220|        f   = f_atom(at["element"], s)
   221|        DW  = np.exp(-at["Biso"]*s*s)
   222|        phi = 2*np.pi*(h*at["x"] + k*at["y"] + l*at["z"])
   223|        Fr += at["occ"]*f*DW*np.cos(phi)
   224|        Fi += at["occ"]*f*DW*np.sin(phi)
   225|    return Fr*Fr + Fi*Fi
   226|
   227|
   228|# ─────────────────────────────────────────────────────────────────────────────
   229|# PATTERN CALCULATION
   230|# ─────────────────────────────────────────────────────────────────────────────
   231|
   232|def calc_pattern(tt_arr, refs, pr, bg_c, atoms=None, mode="lebail"):
   233|    """Return (calculated_pattern, background_array)."""
   234|    bg   = chebyshev_bg(tt_arr, bg_c)
   235|    patt = np.zeros_like(tt_arr, dtype=float)
   236|    U, V, W = pr["U"], pr["V"], pr["W"]
   237|    eta0  = pr.get("eta0", 0.3)
   238|    scale = pr["scale"]
   239|    wl    = pr["wl"]
   240|    system= pr.get("system", "Cubic")
   241|
   242|    for ref in refs:
   243|        h, k, l, d, tt_pk = ref[0], ref[1], ref[2], ref[3], ref[4]
   244|        if not (tt_arr[0] <= tt_pk <= tt_arr[-1]):
   245|            continue
   246|        fwhm = caglioti_fwhm(tt_pk, U, V, W)
   247|        eta  = np.clip(eta0, 0, 1)
   248|        lp   = lp_factor(tt_pk)
   249|        mult = multiplicity(h, k, l, system)
   250|
   251|        if mode == "rietveld" and atoms:
   252|            F2 = structure_factor_sq(h, k, l, atoms, wl, tt_pk)
   253|        else:
   254|            F2 = ref[5]
   255|
   256|        prof   = pseudo_voigt(tt_arr, tt_pk, fwhm, eta)
   257|        patt  += scale * mult * lp * F2 * prof
   258|
   259|    return patt + bg, bg
   260|
   261|
   262|# ─────────────────────────────────────────────────────────────────────────────
   263|# R-FACTOR HELPERS
   264|# ─────────────────────────────────────────────────────────────────────────────
   265|
   266|def r_factors(obs, calc, n_params=0):
   267|    w      = 1.0 / np.maximum(obs, 1)
   268|    Rwp    = 100*np.sqrt(np.sum(w*(obs-calc)**2) / np.sum(w*obs**2))
   269|    Rp     = 100*np.sum(np.abs(obs-calc)) / np.sum(obs)
   270|    chi2   = np.sum(w*(obs-calc)**2) / max(len(obs)-n_params, 1)
   271|    GoF    = np.sqrt(chi2)
   272|    return Rwp, Rp, chi2, GoF
   273|
   274|
   275|# ─────────────────────────────────────────────────────────────────────────────
   276|# SYNTHETIC DATA GENERATOR
   277|# ─────────────────────────────────────────────────────────────────────────────
   278|
   279|PRESETS = {
   280|    "Si  (cubic Fd-3m, a=5.431 Å)":   dict(system="Cubic",  sg="Fd-3m",  a=5.4309, b=5.4309, c=5.4309, al=90,be=90,ga=90,
   281|                                            atoms=[{"element":"Si","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.46},
   282|                                                   {"element":"Si","x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.46}]),
   283|    "LaB6 (cubic Pm-3m, a=4.157 Å)":  dict(system="Cubic",  sg="Pm-3m",  a=4.1569, b=4.1569, c=4.1569, al=90,be=90,ga=90,
   284|                                            atoms=[{"element":"La","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.20},
   285|                                                   {"element":"B", "x":0.5,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.50},
   286|                                                   {"element":"B", "x":0.0,"y":0.5,"z":0.0,"occ":1.0,"Biso":0.50},
   287|                                                   {"element":"B", "x":0.0,"y":0.0,"z":0.5,"occ":1.0,"Biso":0.50}]),
   288|    "CeO2 (cubic Fm-3m, a=5.411 Å)":  dict(system="Cubic",  sg="Fm-3m",  a=5.4124, b=5.4124, c=5.4124, al=90,be=90,ga=90,
   289|                                            atoms=[{"element":"Ce","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.40},
   290|                                                   {"element":"O", "x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.60}]),
   291|    "Custom":                          dict(system="Cubic",  sg="P-1",    a=4.0, b=4.0, c=4.0, al=90,be=90,ga=90, atoms=[]),
   292|}
   293|
   294|
   295|# ─────────────────────────────────────────────────────────────────────────────
   296|# SESSION-STATE DEFAULTS
   297|# ─────────────────────────────────────────────────────────────────────────────
   298|
   299|def init_state():
   300|    defaults = dict(
   301|        refs=None, obs_tt=None, obs_I=None,
   302|        lb_result=None, rv_result=None,
   303|        atoms=[{"element":"Si","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.5},
   304|               {"element":"Si","x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.5}],
   305|    )
   306|    for k, v in defaults.items():
   307|        if k not in st.session_state:
   308|            st.session_state[k] = v
   309|
   310|init_state()
   311|
   312|# ─────────────────────────────────────────────────────────────────────────────
   313|# SIDEBAR
   314|# ─────────────────────────────────────────────────────────────────────────────
   315|
   316|with st.sidebar:
   317|    st.markdown("## ⚙️ Experiment Setup")
   318|
   319|    wl = st.number_input("Wavelength λ (Å)", 0.5, 3.0, 1.54056, 0.00001, "%.5f",
   320|                         help="CuKα1 = 1.54056 Å, MoKα1 = 0.70930 Å")
   321|
   322|    st.markdown("---")
   323|    st.markdown("### 📂 Data")
   324|    data_mode = st.radio("Source", ["Synthetic (preset)", "Upload XY file"], label_visibility="collapsed")
   325|
   326|    if data_mode == "Upload XY file":
   327|        uploaded = st.file_uploader("XY file (2θ  I per line)", type=["xy","dat","txt","csv"])
   328|        if uploaded:
   329|            lines = uploaded.read().decode().splitlines()
   330|            pts = []
   331|            for ln in lines:
   332|                ln = ln.strip()
   333|                if not ln or ln.startswith("#"):
   334|                    continue
   335|                parts = ln.split()
   336|                if len(parts) >= 2:
   337|                    try:
   338|                        pts.append((float(parts[0]), float(parts[1])))
   339|                    except ValueError:
   340|                        pass
   341|            if pts:
   342|                arr = np.array(pts)
   343|                st.session_state.obs_tt = arr[:, 0]
   344|                st.session_state.obs_I  = arr[:, 1]
   345|                st.success(f"Loaded {len(pts)} points")
   346|    else:
   347|        preset_key = st.selectbox("Preset material", list(PRESETS.keys()))
   348|        preset = PRESETS[preset_key]
   349|
   350|    st.markdown("---")
   351|    st.markdown("### 🔷 Unit Cell")
   352|
   353|    system_options = ["Cubic","Tetragonal","Orthorhombic","Hexagonal","Monoclinic","Triclinic"]
   354|    system = st.selectbox("Crystal system",
   355|                          system_options,
   356|                          index=system_options.index(preset.get("system","Cubic") if data_mode=="Synthetic (preset)" else "Cubic"))
   357|
   358|    c1, c2 = st.columns(2)
   359|    a = c1.number_input("a (Å)", 0.5, 30.0,
   360|                         preset.get("a",5.43) if data_mode=="Synthetic (preset)" else 5.43,
   361|                         0.0001, "%.4f")
   362|    if system == "Cubic":
   363|        b = a; c = a
   364|        c2.markdown(f"**b = c = a**")
   365|    elif system in ("Tetragonal","Hexagonal"):
   366|        b = a
   367|        c = c2.number_input("c (Å)", 0.5, 30.0,
   368|                              preset.get("c",5.43) if data_mode=="Synthetic (preset)" else 5.43,
   369|                              0.0001, "%.4f")
   370|        if system == "Tetragonal":
   371|            st.markdown("b = a")
   372|    else:
   373|        b = c2.number_input("b (Å)", 0.5, 30.0,
   374|                              preset.get("b",5.43) if data_mode=="Synthetic (preset)" else 5.43,
   375|                              0.0001, "%.4f")
   376|        c = c1.number_input("c (Å)", 0.5, 30.0,
   377|                              preset.get("c",5.43) if data_mode=="Synthetic (preset)" else 5.43,
   378|                              0.0001, "%.4f")
   379|
   380|    if system in ("Monoclinic","Triclinic"):
   381|        c3, c4, c5 = st.columns(3)
   382|        al = c3.number_input("α°", 1.0, 179.0,
   383|                              preset.get("al",90.0) if data_mode=="Synthetic (preset)" else 90.0,
   384|                              0.01, "%.2f")
   385|        be = c4.number_input("β°", 1.0, 179.0,
   386|                              preset.get("be",90.0) if data_mode=="Synthetic (preset)" else 90.0,
   387|                              0.01, "%.2f")
   388|        ga = c5.number_input("γ°", 1.0, 179.0,
   389|                              preset.get("ga",90.0) if data_mode=="Synthetic (preset)" else 90.0,
   390|                              0.01, "%.2f")
   391|    elif system == "Hexagonal":
   392|        al, be, ga = 90.0, 90.0, 120.0
   393|    else:
   394|        al = be = ga = 90.0
   395|
   396|    sg = st.text_input("Space group", preset.get("sg","P1") if data_mode=="Synthetic (preset)" else "P1")
   397|
   398|    st.markdown("---")
   399|    st.markdown("### 📐 2θ Range & Grid")
   400|    c1, c2 = st.columns(2)
   401|    tt_min = c1.number_input("Min 2θ (°)", 1.0, 170.0, 10.0, 0.5)
   402|    tt_max = c2.number_input("Max 2θ (°)", 10.0, 170.0, 100.0, 0.5)
   403|    n_pts  = st.slider("Grid points", 500, 5000, 2000, 100)
   404|
   405|    st.markdown("---")
   406|    st.markdown("### 📊 Profile Parameters")
   407|    U     = st.number_input("U (Caglioti)",  0.0,  5.0,   0.010, 0.001, "%.4f")
   408|    V     = st.number_input("V (Caglioti)", -1.0,  0.0,  -0.001, 0.001, "%.4f")
   409|    W     = st.number_input("W (Caglioti)",  1e-4, 5.0,   0.005, 0.001, "%.4f")
   410|    eta0  = st.number_input("η₀ (Lorentzian frac.)", 0.0, 1.0, 0.3, 0.01)
   411|    scale = st.number_input("Scale factor", 0.001, 1e9, 1000.0, 100.0)
   412|
   413|    st.markdown("---")
   414|    st.markdown("### 🌐 Background")
   415|    n_bg = st.slider("Chebyshev polynomial terms", 2, 8, 5)
   416|
   417|    st.markdown("---")
   418|    if st.button("🔄 Generate Reflections & Data", type="primary", use_container_width=True):
   419|        refs = gen_reflections(a, b, c, al, be, ga, system, sg, wl, tt_min, tt_max)
   420|        st.session_state.refs = refs
   421|        st.session_state.lb_result = None
   422|        st.session_state.rv_result = None
   423|
   424|        # Build base profile params
   425|        pr0 = dict(U=U, V=V, W=W, eta0=eta0, scale=scale, wl=wl, system=system)
   426|        bg0 = np.zeros(n_bg); bg0[0] = 80.0; bg0[1] = -20.0
   427|
   428|        tt_arr = np.linspace(tt_min, tt_max, n_pts)
   429|
   430|        if data_mode == "Synthetic (preset)":
   431|            # Use preset atoms
   432|            atoms_pr = PRESETS[preset_key].get("atoms", [])
   433|            if atoms_pr:
   434|                pat, _ = calc_pattern(tt_arr, refs, pr0, bg0,
   435|                                       atoms=atoms_pr, mode="rietveld")
   436|            else:
   437|                pat, _ = calc_pattern(tt_arr, refs, pr0, bg0, mode="lebail")
   438|            noise = np.random.default_rng(42).normal(
   439|                0, np.sqrt(np.abs(pat)+1)*0.04)
   440|            st.session_state.obs_tt = tt_arr
   441|            st.session_state.obs_I  = np.maximum(pat + noise, 0)
   442|        elif st.session_state.obs_tt is None:
   443|            st.warning("Upload a data file first, or use Synthetic mode.")
   444|
   445|        st.success(f"✅ {len(refs)} reflections generated")
   446|
   447|# ─────────────────────────────────────────────────────────────────────────────
   448|# MAIN PANEL
   449|# ─────────────────────────────────────────────────────────────────────────────
   450|
   451|st.title("🔬 Diffraction Analyser — Full Profile Refinement")
   452|
   453|# Convenience references
   454|refs   = st.session_state.refs
   455|obs_tt = st.session_state.obs_tt
   456|obs_I  = st.session_state.obs_I
   457|have_data = obs_tt is not None and obs_I is not None and refs is not None
   458|
   459|def bragg_ticks(refs, tt_min, tt_max, y0, dy=-0.04, max_ticks=300):
   460|    """Return plotly shapes + trace for tick marks."""
   461|    shapes, xs, ys = [], [], []
   462|    for ref in refs[:max_ticks]:
   463|        tt_pk = ref[4]
   464|        if tt_min <= tt_pk <= tt_max:
   465|            xs += [tt_pk, tt_pk, None]
   466|            ys += [y0, y0+dy, None]
   467|    return xs, ys
   468|
   469|# ─── TABS ───────────────────────────────────────────────────────────────────
   470|tab_data, tab_lb, tab_rv, tab_results = st.tabs(
   471|    ["📈 Pattern", "⚗️ Le Bail Fit", "🔬 Rietveld Fit", "📋 Results"])
   472|
   473|# ════════════════════════════════════════════════════════════════════════════
   474|# TAB 1 — PATTERN VIEW
   475|# ════════════════════════════════════════════════════════════════════════════
   476|with tab_data:
   477|    if not have_data:
   478|        st.info("👈 Configure the sidebar and click **Generate Reflections & Data** to start.")
   479|    else:
   480|        ymax = obs_I.max()
   481|        tick_x, tick_y = bragg_ticks(refs, obs_tt[0], obs_tt[-1],
   482|                                      y0=ymax, dy=-ymax*0.04)
   483|        fig = go.Figure()
   484|        fig.add_trace(go.Scatter(x=obs_tt, y=obs_I, mode="lines",
   485|                                  name="Observed", line=dict(color="#1f77b4", width=1.2)))
   486|        if tick_x:
   487|            fig.add_trace(go.Scatter(x=tick_x, y=tick_y, mode="lines",
   488|                                      name="Bragg positions",
   489|                                      line=dict(color="red", width=1), showlegend=True,
   490|                                      hoverinfo="skip"))
   491|        fig.update_layout(
   492|            xaxis_title="2θ (°)", yaxis_title="Intensity (counts)",
   493|            template="plotly_white", height=480,
   494|            title=f"Observed Pattern — {len(refs)} reflections  |  λ = {wl:.5f} Å",
   495|            legend=dict(x=0.75, y=0.95))
   496|        st.plotly_chart(fig, use_container_width=True)
   497|
   498|        col1, col2, col3 = st.columns(3)
   499|        col1.metric("Data points",     f"{len(obs_tt):,}")
   500|        col2.metric("Reflections",     f"{len(refs)}")
   501|