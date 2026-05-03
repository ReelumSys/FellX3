
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
    13|import re
    14|import warnings
    15|
    16|warnings.filterwarnings("ignore")
    17|
    18|st.set_page_config(
    19|    page_title="Diffraction Analyser",
    20|    page_icon="🔬",
    21|    layout="wide",
    22|    initial_sidebar_state="expanded",
    23|)
    24|
    25|st.markdown("""
    26|<style>
    27|[data-testid="stMetricValue"] { font-size: 1.3rem; }
    28|.block-container { padding-top: 1rem; }
    29|.stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    30|h1  { font-size: 1.8rem !important; }
    31|h2  { font-size: 1.2rem !important; }
    32|h3  { font-size: 1.05rem !important; }
    33|</style>
    34|""", unsafe_allow_html=True)
    35|
    36|
    37|# ═══════════════════════════════════════════════════════════════════════════
    38|# CIF PARSER  (zero external dependencies)
    39|# ═══════════════════════════════════════════════════════════════════════════
    40|
    41|def _cif_val(raw):
    42|    """Strip ESD parentheses and quotes; coerce to float or str."""
    43|    if raw in (".", "?", "", None):
    44|        return None
    45|    raw = re.sub(r"\(\d+\)", "", str(raw))   # remove (esd)
    46|    raw = raw.strip("'\"")
    47|    try:
    48|        return float(raw)
    49|    except ValueError:
    50|        return raw
    51|
    52|
    53|def _tokenise(text):
    54|    """Yield CIF tokens, handling semicolon text blocks and inline comments."""
    55|    lines = text.splitlines()
    56|    i = 0
    57|    while i < len(lines):
    58|        line = lines[i]
    59|        # ── semicolon text block ──────────────────────────────────────────
    60|        if line.startswith(";"):
    61|            block = []
    62|            i += 1
    63|            while i < len(lines) and not lines[i].startswith(";"):
    64|                block.append(lines[i])
    65|                i += 1
    66|            yield "\n".join(block)
    67|            i += 1
    68|            continue
    69|        # strip inline comment
    70|        line = re.sub(r"(?<!['\"])#.*", "", line)
    71|        # tokenise respecting single/double quotes
    72|        for tok in re.findall(r"'[^']*'|\"[^\"]*\"|[^\s]+", line):
    73|            yield tok
    74|        i += 1
    75|
    76|
    77|def parse_cif(text):
    78|    """
    79|    Parse a CIF text and return:
    80|      cell    : {a, b, c, alpha, beta, gamma}
    81|      sg      : H-M space-group string
    82|      sg_no   : space-group number (int or None)
    83|      system  : crystal system string
    84|      atoms   : list of {element, label, x, y, z, occ, Biso}
    85|      formula : chemical formula string or None
    86|    """
    87|    result = dict(cell={}, sg="P1", sg_no=None,
    88|                  system="Triclinic", atoms=[], formula=None)
    89|    tokens    = list(_tokenise(text))
    90|    n         = len(tokens)
    91|
    92|    i = 0
    93|    while i < n:
    94|        tok = tokens[i]
    95|
    96|        # skip data_ block headers
    97|        if tok.lower().startswith("data_"):
    98|            i += 1; continue
    99|
   100|        # ── loop_ ─────────────────────────────────────────────────────────
   101|        if tok.lower() == "loop_":
   102|            i += 1
   103|            loop_keys = []
   104|            while i < n and tokens[i].startswith("_"):
   105|                loop_keys.append(tokens[i].lower())
   106|                i += 1
   107|            lk_set = set(loop_keys)
   108|            is_atom_loop = (
   109|                "_atom_site_fract_x" in lk_set or
   110|                "_atom_site_x"       in lk_set
   111|            )
   112|            # consume rows until we hit a keyword or new block
   113|            while i < n:
   114|                t = tokens[i]
   115|                if t.startswith("_") or t.lower() in ("loop_",) or t.lower().startswith("data_"):
   116|                    break
   117|                row = {}
   118|                for key in loop_keys:
   119|                    if i >= n: break
   120|                    row[key] = _cif_val(tokens[i])
   121|                    i += 1
   122|                if is_atom_loop:
   123|                    # --- element symbol ---
   124|                    el = (row.get("_atom_site_type_symbol") or
   125|                          row.get("_atom_site_element_symbol") or
   126|                          row.get("_atom_site_label") or "X")
   127|                    if isinstance(el, str):
   128|                        # keep only letters, capitalise first
   129|                        el = re.sub(r"[^A-Za-z]", "", el)
   130|                        el = el[0].upper() + el[1:].lower() if el else "X"
   131|                    # --- label ---
   132|                    lbl = row.get("_atom_site_label", el)
   133|                    if not isinstance(lbl, str): lbl = str(lbl)
   134|                    # --- coordinates ---
   135|                    x = row.get("_atom_site_fract_x") or row.get("_atom_site_x") or 0.0
   136|                    y = row.get("_atom_site_fract_y") or row.get("_atom_site_y") or 0.0
   137|                    z = row.get("_atom_site_fract_z") or row.get("_atom_site_z") or 0.0
   138|                    # --- occupancy ---
   139|                    occ = row.get("_atom_site_occupancy", 1.0)
   140|                    if occ is None: occ = 1.0
   141|                    # --- Biso / Uiso ---
   142|                    Biso = row.get("_atom_site_b_iso_or_equiv") or \
   143|                           row.get("_atom_site_b_equiv_geom_mean")
   144|                    if Biso is None:
   145|                        Uiso = row.get("_atom_site_u_iso_or_equiv") or \
   146|                               row.get("_atom_site_u_equiv_geom_mean")
   147|                        Biso = 8 * np.pi**2 * float(Uiso) if Uiso else 0.5
   148|                    try:
   149|                        result["atoms"].append(dict(
   150|                            element=str(el), label=str(lbl),
   151|                            x=float(x),  y=float(y),  z=float(z),
   152|                            occ=float(occ), Biso=float(Biso)))
   153|                    except (TypeError, ValueError):
   154|                        pass
   155|            continue
   156|
   157|        # ── scalar key-value pairs ─────────────────────────────────────────
   158|        if tok.startswith("_"):
   159|            key = tok.lower()
   160|            i  += 1
   161|            val = _cif_val(tokens[i]) if i < n else None
   162|            i  += 1
   163|
   164|            # Cell parameters
   165|            cell_map = {
   166|                "_cell_length_a":    "a",
   167|                "_cell_length_b":    "b",
   168|                "_cell_length_c":    "c",
   169|                "_cell_angle_alpha": "alpha",
   170|                "_cell_angle_beta":  "beta",
   171|                "_cell_angle_gamma": "gamma",
   172|            }
   173|            if key in cell_map and val is not None:
   174|                result["cell"][cell_map[key]] = float(val)
   175|
   176|            # Space group
   177|            if key in ("_symmetry_space_group_name_h-m",
   178|                       "_space_group_name_h-m_alt",
   179|                       "_symmetry_space_group_name_h-m_alt",) and isinstance(val, str):
   180|                result["sg"] = val.strip()
   181|            if key in ("_symmetry_int_tables_number",
   182|                       "_space_group_it_number") and val:
   183|                try: result["sg_no"] = int(float(val))
   184|                except (TypeError, ValueError): pass
   185|
   186|            # Formula
   187|            if key == "_chemical_formula_sum" and isinstance(val, str):
   188|                result["formula"] = val.strip()
   189|            continue
   190|
   191|        i += 1
   192|
   193|    # ── infer crystal system ───────────────────────────────────────────────
   194|    n_sg = result["sg_no"]
   195|    if n_sg:
   196|        if   n_sg <= 2:   result["system"] = "Triclinic"
   197|        elif n_sg <= 15:  result["system"] = "Monoclinic"
   198|        elif n_sg <= 74:  result["system"] = "Orthorhombic"
   199|        elif n_sg <= 142: result["system"] = "Tetragonal"
   200|        elif n_sg <= 194: result["system"] = "Hexagonal"
   201|        else:             result["system"] = "Cubic"
   202|    else:
   203|        c  = result["cell"]
   204|        al = c.get("alpha", 90); be = c.get("beta", 90); ga = c.get("gamma", 90)
   205|        av = c.get("a", 1);      bv = c.get("b", 1);     cv = c.get("c", 1)
   206|        if abs(ga - 120) < 1 and abs(al - 90) < 1 and abs(be - 90) < 1:
   207|            result["system"] = "Hexagonal"
   208|        elif all(abs(x - 90) < 1 for x in (al, be, ga)):
   209|            if abs(av-bv) < 0.02 and abs(av-cv) < 0.02: result["system"] = "Cubic"
   210|            elif abs(av-bv) < 0.02:                       result["system"] = "Tetragonal"
   211|            else:                                          result["system"] = "Orthorhombic"
   212|        elif abs(al-90) < 1 and abs(ga-90) < 1:           result["system"] = "Monoclinic"
   213|        else:                                              result["system"] = "Triclinic"
   214|
   215|    return result
   216|
   217|
   218|# ═══════════════════════════════════════════════════════════════════════════
   219|# CRYSTALLOGRAPHY UTILITIES
   220|# ═══════════════════════════════════════════════════════════════════════════
   221|
   222|def d_spacing(h, k, l, a, b, c, alpha_deg, beta_deg, gamma_deg, system):
   223|    ar, br, gr = (np.radians(x) for x in (alpha_deg, beta_deg, gamma_deg))
   224|    h, k, l = float(h), float(k), float(l)
   225|    if system == "Cubic":
   226|        inv = (h**2 + k**2 + l**2) / a**2
   227|    elif system == "Tetragonal":
   228|        inv = (h**2 + k**2) / a**2 + l**2 / c**2
   229|    elif system == "Orthorhombic":
   230|        inv = h**2/a**2 + k**2/b**2 + l**2/c**2
   231|    elif system == "Hexagonal":
   232|        inv = 4/3*(h**2 + h*k + k**2)/a**2 + l**2/c**2
   233|    elif system == "Monoclinic":
   234|        sb = np.sin(br)
   235|        inv = (1/sb**2)*(h**2/a**2 + k**2*sb**2/b**2 + l**2/c**2
   236|                         - 2*h*l*np.cos(br)/(a*c))
   237|    else:  # Triclinic
   238|        ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
   239|        sa, sb, sg = np.sin(ar), np.sin(br), np.sin(gr)
   240|        V = a*b*c*np.sqrt(max(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg, 1e-20))
   241|        if V < 1e-10: return None
   242|        inv = (b**2*c**2*sa**2*h**2 + a**2*c**2*sb**2*k**2 + a**2*b**2*sg**2*l**2
   243|               + 2*a*b*c**2*(ca*cb-cg)*h*k
   244|               + 2*a**2*b*c*(cb*cg-ca)*k*l
   245|               + 2*a*b**2*c*(ca*cg-cb)*h*l) / V**2
   246|    return None if inv <= 1e-12 else 1.0/np.sqrt(inv)
   247|
   248|
   249|def is_absent(h, k, l, sg):
   250|    h, k, l = int(h), int(k), int(l)
   251|    su = sg.upper().replace(" ", "")
   252|    if su.startswith("I") and (h+k+l) % 2 != 0:        return True
   253|    if su.startswith("F") and len({h%2,k%2,l%2}) > 1:  return True
   254|    if su.startswith("C") and (h+k) % 2 != 0:           return True
   255|    if su.startswith("A") and (k+l) % 2 != 0:           return True
   256|    if su.startswith("B") and (h+l) % 2 != 0:           return True
   257|    if "FD" in su or ("F" in su and "D" in su):
   258|        if h == 0 and k == 0 and l % 4 != 0:            return True
   259|    return False
   260|
   261|
   262|def gen_reflections(a, b, c, alpha, beta, gamma, system, sg, wl, tt_min, tt_max):
   263|    d_min = wl / (2*np.sin(np.radians(tt_max/2)))
   264|    d_max = wl / (2*np.sin(np.radians(max(tt_min, 0.5)/2)))
   265|    mh = int(2*a/d_min)+2
   266|    mk = int(2*b/d_min)+2
   267|    ml = int(2*c/d_min)+2
   268|    seen = {}
   269|    for h in range(-mh, mh+1):
   270|        for k in range(-mk, mk+1):
   271|            for l in range(-ml, ml+1):
   272|                if h == k == l == 0: continue
   273|                if is_absent(h, k, l, sg): continue
   274|                d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma, system)
   275|                if d is None or not (d_min <= d <= d_max): continue
   276|                tt  = 2*np.degrees(np.arcsin(np.clip(wl/(2*d), -1, 1)))
   277|                key = round(tt, 4)
   278|                if key not in seen:
   279|                    seen[key] = [h, k, l, d, tt, 1000.0]
   280|    return sorted(seen.values(), key=lambda r: r[4])
   281|
   282|
   283|# ── Profile functions ──────────────────────────────────────────────────────
   284|
   285|def pseudo_voigt(x, x0, fwhm, eta):
   286|    eta  = np.clip(eta, 0, 1); fwhm = max(fwhm, 1e-6)
   287|    sig  = fwhm / (2*np.sqrt(2*np.log(2)))
   288|    G    = np.exp(-0.5*((x-x0)/sig)**2)
   289|    L    = 1.0 / (1 + ((x-x0)/(fwhm/2))**2)
   290|    return eta*L + (1-eta)*G
   291|
   292|def caglioti_fwhm(tt, U, V, W):
   293|    t = np.tan(np.radians(tt/2))
   294|    return max(np.sqrt(max(U*t**2 + V*t + W, 1e-8)), 0.005)
   295|
   296|def chebyshev_bg(tt, coeffs):
   297|    x = 2*(tt - tt.min())/(tt.max()-tt.min()) - 1
   298|    T = [np.ones_like(x), x, 2*x**2-1, 4*x**3-3*x,
   299|         8*x**4-8*x**2+1, 16*x**5-20*x**3+5*x,
   300|         32*x**6-48*x**4+18*x**2-1, 64*x**7-112*x**5+56*x**3-7*x]
   301|    out = np.zeros_like(x)
   302|    for i, c in enumerate(coeffs): out += c * T[i]
   303|    return out
   304|
   305|def lp_factor(tt):
   306|    th   = np.radians(tt/2)
   307|    cos2 = np.cos(np.radians(tt))**2
   308|    return (1+cos2) / (np.sin(th)**2 * np.cos(th) + 1e-12)
   309|
   310|def multiplicity(h, k, l, system):
   311|    h, k, l = abs(int(h)), abs(int(k)), abs(int(l))
   312|    zeros = sum(x == 0 for x in (h, k, l))
   313|    if system == "Cubic":
   314|        return {1:8, 2:24}.get(len({h,k,l}), 48)
   315|    elif system == "Tetragonal":
   316|        if h == 0 and k == 0: return 2
   317|        return (4 if h==k else 8) * (1 if l==0 else 2)
   318|    elif system == "Hexagonal":
   319|        return 2 if (h==0 and k==0) else (12 if l!=0 else 6)
   320|    elif system == "Orthorhombic":
   321|        return 2**(3-zeros)*2
   322|    return max(2**(3-zeros), 1)
   323|
   324|
   325|# ── Cromer-Mann scattering factors ─────────────────────────────────────────
   326|
   327|CM = {
   328|    "H":  ([0.4899,0.2620,0.1968,0.0499],[20.659,7.740,49.552,2.202],0.001),
   329|    "C":  ([2.310,1.020,1.589,0.865],[20.844,10.208,0.569,51.651],0.216),
   330|    "N":  ([12.213,3.132,2.013,1.166],[0.006,9.893,28.997,0.583],-11.529),
   331|    "O":  ([3.049,2.287,1.546,0.867],[13.277,5.701,0.324,32.909],0.251),
   332|    "F":  ([3.539,2.641,1.517,1.024],[10.283,4.294,0.261,26.147],0.278),
   333|    "Na": ([4.763,3.174,1.267,1.113],[3.285,8.842,0.314,129.424],0.676),
   334|    "Mg": ([5.420,2.174,1.227,2.307],[2.828,79.261,0.381,7.194],0.858),
   335|    "Al": ([6.420,1.900,1.594,1.965],[3.039,0.743,31.547,85.089],1.115),
   336|    "Si": ([6.292,3.035,1.989,1.541],[2.439,32.334,0.679,81.694],1.141),
   337|    "P":  ([6.435,4.179,1.780,1.491],[1.907,27.157,0.526,68.164],1.115),
   338|    "S":  ([6.905,5.203,1.438,1.586],[1.468,22.215,0.254,56.172],0.866),
   339|    "Cl": ([11.460,7.196,6.256,1.645],[0.010,1.166,18.519,47.778],-9.557),
   340|    "K":  ([8.219,7.439,1.052,0.866],[12.795,0.775,213.187,41.684],1.423),
   341|    "Ca": ([8.627,7.387,1.590,1.021],[10.442,0.660,85.748,178.437],1.375),
   342|    "Ti": ([9.760,7.359,1.699,1.902],[7.851,0.500,35.634,116.105],1.281),
   343|    "V":  ([10.297,7.351,2.070,2.045],[6.865,0.438,26.894,102.478],1.220),
   344|    "Cr": ([10.641,7.354,3.324,1.492],[6.104,0.392,20.263,98.740],1.183),
   345|    "Mn": ([11.282,7.357,3.019,2.244],[5.341,0.343,17.867,83.754],1.089),
   346|    "Fe": ([11.770,7.357,3.522,2.305],[4.761,0.307,15.354,76.881],1.037),
   347|    "Co": ([12.284,7.341,4.003,2.349],[4.279,0.278,13.536,71.169],1.012),
   348|    "Ni": ([12.838,7.290,4.444,2.380],[3.878,0.257,12.176,66.342],1.034),
   349|    "Cu": ([13.338,7.168,5.616,1.674],[3.583,0.247,11.397,64.813],1.191),
   350|    "Zn": ([14.074,7.032,5.165,2.410],[3.266,0.233,10.316,58.710],1.304),
   351|    "Ga": ([15.235,6.701,4.359,2.962],[3.067,0.241,10.781,61.413],1.719),
   352|    "Ge": ([16.082,6.375,3.707,3.683],[2.851,0.252,11.447,54.763],2.131),
   353|    "Zr": ([17.876,10.948,5.418,3.657],[1.276,11.916,0.118,87.663],2.069),
   354|    "Ba": ([20.336,19.297,10.888,5.480],[3.216,0.275,20.207,109.460],2.775),
   355|    "La": ([20.578,19.599,11.373,3.287],[2.948,0.244,18.773,133.124],2.147),
   356|    "Ce": ([21.167,19.770,11.851,3.330],[2.812,0.226,17.608,127.113],1.862),
   357|    "Pr": ([22.044,19.670,12.386,2.824],[2.774,0.222,16.767,143.644],2.058),
   358|    "Nd": ([22.685,19.685,12.774,2.851],[2.662,0.210,15.885,137.903],1.985),
   359|    "Sm": ([24.004,19.426,13.440,2.896],[2.473,0.196,14.400,128.007],2.209),
   360|    "Pb": ([31.062,13.064,18.442,5.970],[0.690,2.358,8.618,47.257],13.412),
   361|    "Bi": ([33.369,12.951,16.588,6.469],[0.704,2.923,8.794,48.009],13.579),
   362|}
   363|
   364|def f_atom(element, s):
   365|    # Try two-character then one-character lookup (capitalised)
   366|    key = (element[0].upper() + element[1:].lower()) if len(element) > 1 else element.upper()
   367|    if key not in CM:
   368|        key = element[0].upper()
   369|    if key not in CM:
   370|        return 1.0
   371|    a4, b4, c = CM[key]
   372|    s2 = s*s
   373|    return c + sum(ai*np.exp(-bi*s2) for ai, bi in zip(a4, b4))
   374|
   375|def structure_factor_sq(h, k, l, atoms, wl, tt):
   376|    s = np.sin(np.radians(tt/2)) / wl
   377|    Fr = Fi = 0.0
   378|    for at in atoms:
   379|        f   = f_atom(at["element"], s)
   380|        DW  = np.exp(-at["Biso"]*s*s)
   381|        phi = 2*np.pi*(h*at["x"] + k*at["y"] + l*at["z"])
   382|        Fr += at["occ"]*f*DW*np.cos(phi)
   383|        Fi += at["occ"]*f*DW*np.sin(phi)
   384|    return Fr*Fr + Fi*Fi
   385|
   386|
   387|# ── Pattern calculation ────────────────────────────────────────────────────
   388|
   389|def calc_pattern(tt_arr, refs, pr, bg_c, atoms=None, mode="lebail"):
   390|    bg   = chebyshev_bg(tt_arr, bg_c)
   391|    patt = np.zeros_like(tt_arr, dtype=float)
   392|    U, V, W = pr["U"], pr["V"], pr["W"]
   393|    eta0    = pr.get("eta0", 0.3)
   394|    scale   = pr["scale"]
   395|    wl      = pr["wl"]
   396|    system  = pr.get("system", "Cubic")
   397|    for ref in refs:
   398|        h, k, l, d, tt_pk = ref[0], ref[1], ref[2], ref[3], ref[4]
   399|        if not (tt_arr[0] <= tt_pk <= tt_arr[-1]): continue
   400|        fwhm = caglioti_fwhm(tt_pk, U, V, W)
   401|        lp   = lp_factor(tt_pk)
   402|        mult = multiplicity(h, k, l, system)
   403|        F2   = structure_factor_sq(h,k,l,atoms,wl,tt_pk) if (mode=="rietveld" and atoms) else ref[5]
   404|        patt += scale * mult * lp * F2 * pseudo_voigt(tt_arr, tt_pk, fwhm, eta0)
   405|    return patt + bg, bg
   406|
   407|def r_factors(obs, calc, n_params=0):
   408|    w    = 1.0 / np.maximum(obs, 1)
   409|    Rwp  = 100*np.sqrt(np.sum(w*(obs-calc)**2) / np.sum(w*obs**2))
   410|    Rp   = 100*np.sum(np.abs(obs-calc)) / np.sum(obs)
   411|    chi2 = np.sum(w*(obs-calc)**2) / max(len(obs)-n_params, 1)
   412|    return Rwp, Rp, chi2, np.sqrt(chi2)
   413|
   414|
   415|# ═══════════════════════════════════════════════════════════════════════════
   416|# PRESETS
   417|# ═══════════════════════════════════════════════════════════════════════════
   418|
   419|PRESETS = {
   420|    "Si  (cubic Fd-3m, a=5.431 Å)":
   421|        dict(system="Cubic", sg="Fd-3m", a=5.4309, b=5.4309, c=5.4309, al=90,be=90,ga=90,
   422|             atoms=[{"element":"Si","label":"Si1","x":0.0, "y":0.0, "z":0.0, "occ":1.0,"Biso":0.46},
   423|                    {"element":"Si","label":"Si2","x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.46}]),
   424|    "LaB6 (cubic Pm-3m, a=4.157 Å)":
   425|        dict(system="Cubic", sg="Pm-3m", a=4.1569, b=4.1569, c=4.1569, al=90,be=90,ga=90,
   426|             atoms=[{"element":"La","label":"La1","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.20},
   427|                    {"element":"B", "label":"B1", "x":0.5,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.50},
   428|                    {"element":"B", "label":"B2", "x":0.0,"y":0.5,"z":0.0,"occ":1.0,"Biso":0.50},
   429|                    {"element":"B", "label":"B3", "x":0.0,"y":0.0,"z":0.5,"occ":1.0,"Biso":0.50}]),
   430|    "CeO2 (cubic Fm-3m, a=5.411 Å)":
   431|        dict(system="Cubic", sg="Fm-3m", a=5.4124, b=5.4124, c=5.4124, al=90,be=90,ga=90,
   432|             atoms=[{"element":"Ce","label":"Ce1","x":0.0, "y":0.0, "z":0.0, "occ":1.0,"Biso":0.40},
   433|                    {"element":"O", "label":"O1", "x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.60}]),
   434|    "Custom":
   435|        dict(system="Cubic", sg="P1", a=4.0,b=4.0,c=4.0,al=90,be=90,ga=90,atoms=[]),
   436|}
   437|
   438|
   439|# ═══════════════════════════════════════════════════════════════════════════
   440|# SESSION STATE
   441|# ═══════════════════════════════════════════════════════════════════════════
   442|
   443|def _init():
   444|    defs = dict(
   445|        refs=None, obs_tt=None, obs_I=None,
   446|        lb_result=None, rv_result=None,
   447|        atoms=[{"element":"Si","label":"Si1","x":0.0, "y":0.0, "z":0.0, "occ":1.0,"Biso":0.5},
   448|               {"element":"Si","label":"Si2","x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.5}],
   449|        cif_loaded=False,
   450|        cif_a=None, cif_b=None, cif_c=None,
   451|        cif_al=None, cif_be=None, cif_ga=None,
   452|        cif_sg=None, cif_system=None, cif_formula=None,
   453|    )
   454|    for k, v in defs.items():
   455|        if k not in st.session_state:
   456|            st.session_state[k] = v
   457|
   458|_init()
   459|
   460|
   461|def bragg_ticks(refs, lo, hi, y0, dy):
   462|    xs, ys = [], []
   463|    for r in refs[:400]:
   464|        tp = r[4]
   465|        if lo <= tp <= hi:
   466|            xs += [tp, tp, None]; ys += [y0, y0+dy, None]
   467|    return xs, ys
   468|
   469|
   470|# ═══════════════════════════════════════════════════════════════════════════
   471|# SIDEBAR
   472|# ═══════════════════════════════════════════════════════════════════════════
   473|
   474|with st.sidebar:
   475|    st.markdown("## ⚙️ Experiment Setup")
   476|    wl = st.number_input("Wavelength λ (Å)", 0.5, 3.0, 1.54056, 0.00001, "%.5f",
   477|                         help="CuKα1=1.54056  MoKα1=0.70930  AgKα1=0.55941")
   478|
   479|    st.markdown("---")
   480|    st.markdown("### 📂 Diffraction Data")
   481|    data_mode = st.radio("Data source", ["Synthetic (preset)", "Upload XY file"],
   482|                         label_visibility="collapsed")
   483|    if data_mode == "Upload XY file":
   484|        uploaded_xy = st.file_uploader("XY file (2θ  I per line)",
   485|                                       type=["xy","dat","txt","csv"])
   486|        if uploaded_xy:
   487|            lines = uploaded_xy.read().decode(errors="replace").splitlines()
   488|            pts   = []
   489|            for ln in lines:
   490|                ln = ln.strip()
   491|                if not ln or ln.startswith("#"): continue
   492|                parts = ln.split()
   493|                if len(parts) >= 2:
   494|                    try: pts.append((float(parts[0]), float(parts[1])))
   495|                    except ValueError: pass
   496|            if pts:
   497|                arr = np.array(pts)
   498|                st.session_state.obs_tt = arr[:,0]
   499|                st.session_state.obs_I  = arr[:,1]
   500|                st.success(f"Loaded {len(pts):,} data points")
   501|