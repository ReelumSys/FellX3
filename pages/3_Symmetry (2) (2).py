
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
     2|Structure Factor & HKL Search
     3|==============================
     4|Liest CIF aus dem Parent-Ordner, berechnet Strukturfaktoren F(hkl)
     5|und ermöglicht interaktive HKL-Suche.
     6|
     7|Physik:
     8|  F(hkl) = Σ_j  f_j(s) · occ_j · exp(2πi (h·x_j + k·y_j + l·z_j))
     9|  |F|² = Intensität (proportional)
    10|  s = sinθ/λ = 1/(2d_hkl)
    11|
    12|Cromer-Mann Atomformfaktoren:
    13|  f(s) = Σ_i a_i · exp(-b_i · s²) + c
    14|
    15|Installation:
    16|    pip install streamlit matplotlib numpy pandas
    17|
    18|Starten:
    19|    streamlit run structure_factor.py
    20|"""
    21|
    22|import sys, re, math, itertools
    23|from pathlib import Path
    24|
    25|import numpy as np
    26|import pandas as pd
    27|import matplotlib
    28|matplotlib.use("Agg")
    29|import matplotlib.pyplot as plt
    30|import matplotlib.colors as mcolors
    31|import streamlit as st
    32|
    33|# ── CLI preload (same pattern as bravais_from_cif.py) ────────────────────────
    34|PRELOAD_PATH = None
    35|args = sys.argv[1:]
    36|for i, arg in enumerate(args):
    37|    if arg == "--cif" and i+1 < len(args):
    38|        PRELOAD_PATH = Path(args[i+1])
    39|    elif arg.endswith(".cif"):
    40|        PRELOAD_PATH = Path(arg)
    41|
    42|# ── Page config ───────────────────────────────────────────────────────────────
    43|st.set_page_config(
    44|    page_title="Structure Factor & HKL",
    45|    page_icon="🔭",
    46|    layout="wide",
    47|    initial_sidebar_state="expanded",
    48|)
    49|
    50|st.markdown("""
    51|<style>
    52|@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;600&display=swap');
    53|html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    54|h1,h2,h3 { font-family: 'Share Tech Mono', monospace !important; }
    55|.stApp { background: #07080f; color: #c8d4f8; }
    56|[data-testid="stSidebar"] { background: #0d0e1c !important; border-right:1px solid #1e2555; }
    57|[data-testid="stSidebar"] * { color: #c8d4f8 !important; }
    58|[data-testid="stSidebar"] label { color: #4466aa !important; font-size:12px; }
    59|[data-testid="metric-container"] {
    60|    background: rgba(20,26,70,0.6);
    61|    border: 1px solid rgba(50,80,200,0.3);
    62|    border-radius: 10px; padding: 12px 18px;
    63|}
    64|[data-testid="stMetricLabel"] { color:#4466aa !important; font-family:'Share Tech Mono',monospace; font-size:10px; }
    65|[data-testid="stMetricValue"] { color:#00e5c8 !important; font-size:22px; }
    66|.page-header {
    67|    background: linear-gradient(135deg, rgba(20,26,70,0.8), rgba(10,15,45,0.9));
    68|    border: 1px solid rgba(50,80,200,0.35);
    69|    border-left: 4px solid #00e5c8;
    70|    border-radius: 12px;
    71|    padding: 20px 28px; margin-bottom: 20px;
    72|}
    73|.tag {
    74|    display:inline-block; padding:3px 14px; border-radius:16px;
    75|    font-family:'Share Tech Mono',monospace; font-size:11px;
    76|    margin:3px; letter-spacing:0.08em;
    77|}
    78|.hkl-result {
    79|    background: rgba(0,229,200,0.06);
    80|    border: 1px solid rgba(0,229,200,0.25);
    81|    border-radius: 10px; padding:14px 20px; margin:6px 0;
    82|}
    83|.formula-box {
    84|    background: rgba(15,20,55,0.7);
    85|    border: 1px solid rgba(50,80,200,0.3);
    86|    border-radius: 10px; padding:16px 20px; margin:10px 0;
    87|    font-family:'Share Tech Mono',monospace; font-size:13px;
    88|    color:#aabcee; line-height:2;
    89|}
    90|</style>
    91|""", unsafe_allow_html=True)
    92|
    93|
    94|# ════════════════════════════════════════════════════════════════════════════
    95|# CIF PARSER  (identical to bravais_from_cif.py)
    96|# ════════════════════════════════════════════════════════════════════════════
    97|
    98|def parse_number(s):
    99|    if s is None or str(s).strip() in ("?", "."):
   100|        return None
   101|    try:
   102|        return float(re.sub(r"\(.*?\)", "", str(s).strip()))
   103|    except:
   104|        return None
   105|
   106|def _tokenize(line):
   107|    tokens = []
   108|    i = 0
   109|    while i < len(line):
   110|        if line[i] in ('"', "'"):
   111|            q = line[i]; j = line.find(q, i+1)
   112|            if j == -1: j = len(line)
   113|            tokens.append(line[i+1:j]); i = j+1
   114|        elif line[i] in (" ", "\t"):
   115|            i += 1
   116|        else:
   117|            j = i
   118|            while j < len(line) and line[j] not in (" ", "\t"):
   119|                j += 1
   120|            tokens.append(line[i:j]); i = j
   121|    return tokens
   122|
   123|def parse_cif_full(text: str):
   124|    """Parse CIF → (scalar_dict, list_of_loops).
   125|    Each loop: {'keys': [...], 'rows': [[...], ...]}
   126|    """
   127|    scalars = {}
   128|    loops   = []
   129|    lines   = text.splitlines()
   130|    i = 0
   131|    cur_loop_keys = []
   132|    cur_loop_rows = []
   133|    in_loop = False
   134|
   135|    def flush_loop():
   136|        if cur_loop_keys and cur_loop_rows:
   137|            loops.append({"keys": cur_loop_keys[:], "rows": [r[:] for r in cur_loop_rows]})
   138|
   139|    while i < len(lines):
   140|        raw  = lines[i]
   141|        line = raw.strip()
   142|
   143|        if not line or line.startswith("#"):
   144|            i += 1; continue
   145|
   146|        # multi-line string
   147|        if line.startswith(";"):
   148|            val_lines = []
   149|            i += 1
   150|            while i < len(lines) and not lines[i].strip().startswith(";"):
   151|                val_lines.append(lines[i])
   152|                i += 1
   153|            # attach to last scalar key
   154|            if scalars:
   155|                last = list(scalars)[-1]
   156|                if scalars[last] == "__PENDING__":
   157|                    scalars[last] = "\n".join(val_lines).strip()
   158|            i += 1; continue
   159|
   160|        if line.lower() == "loop_":
   161|            flush_loop()
   162|            cur_loop_keys = []; cur_loop_rows = []
   163|            in_loop = True; i += 1; continue
   164|
   165|        if in_loop:
   166|            if line.startswith("_"):
   167|                # still collecting keys — but if we already have rows, new key ends loop
   168|                if cur_loop_rows:
   169|                    flush_loop()
   170|                    cur_loop_keys = []; cur_loop_rows = []
   171|                    in_loop = False
   172|                    # re-process this line as scalar
   173|                    continue
   174|                cur_loop_keys.append(line.split()[0].lower())
   175|                i += 1; continue
   176|            else:
   177|                if line.lower() == "loop_":
   178|                    flush_loop()
   179|                    cur_loop_keys = []; cur_loop_rows = []
   180|                    i += 1; continue
   181|                # data row — collect tokens, package into rows of len(keys)
   182|                tokens = _tokenize(line)
   183|                n = len(cur_loop_keys)
   184|                if n:
   185|                    for t in tokens:
   186|                        if not cur_loop_rows or len(cur_loop_rows[-1]) == n:
   187|                            cur_loop_rows.append([])
   188|                        cur_loop_rows[-1].append(t)
   189|                i += 1; continue
   190|
   191|        if line.startswith("_"):
   192|            parts = _tokenize(line)
   193|            key   = parts[0].lower()
   194|            if len(parts) >= 2:
   195|                scalars[key] = parts[1]
   196|            else:
   197|                scalars[key] = "__PENDING__"
   198|            i += 1; continue
   199|
   200|        i += 1
   201|
   202|    flush_loop()
   203|    return scalars, loops
   204|
   205|def loop_as_df(loops, key: str) -> pd.DataFrame | None:
   206|    key = key.lower()
   207|    for lp in loops:
   208|        if any(key in k for k in lp["keys"]):
   209|            n = len(lp["keys"])
   210|            rows = [r for r in lp["rows"] if len(r) == n]
   211|            if rows:
   212|                return pd.DataFrame(rows, columns=lp["keys"])
   213|    return None
   214|
   215|
   216|# ════════════════════════════════════════════════════════════════════════════
   217|# CROMER-MANN ATOMIC FORM FACTORS  (International Tables Vol. C)
   218|# ════════════════════════════════════════════════════════════════════════════
   219|# Format: element → (a1,b1,a2,b2,a3,b3,a4,b4,c)
   220|CM_PARAMS = {
   221|    "H":  (0.489918,20.6593,0.262003,7.74039,0.196767,49.5519,0.049879,2.20159,0.001305),
   222|    "HE": (0.873400,9.1037,0.630900,3.3568,0.311200,22.9276,0.178000,0.9821,0.006400),
   223|    "LI": (1.128200,3.9546,0.750800,1.0524,0.617500,85.3905,0.465300,168.261,0.037700),
   224|    "BE": (1.591900,43.6427,1.127800,1.8623,0.539100,103.483,0.702900,0.5420,0.038500),
   225|    "B":  (2.054500,23.2185,1.332600,1.0210,1.097900,60.3498,0.706800,0.1403,-0.193200),
   226|    "C":  (2.310000,20.8439,1.020000,10.2075,1.588600,0.5687,0.865000,51.6512,0.215600),
   227|    "N":  (12.2126,0.0057,3.1322,9.8933,2.0125,28.9975,1.1663,0.5826,-11.529),
   228|    "O":  (3.048500,13.2771,2.286800,5.7011,1.546300,0.3239,0.867000,32.9089,0.250800),
   229|    "F":  (3.539200,10.2825,2.641200,4.2944,1.517000,0.2615,1.024300,26.1476,0.277600),
   230|    "NA": (4.762600,3.2850,3.173600,8.8422,1.267400,0.3136,1.112800,129.424,0.676000),
   231|    "MG": (5.420400,2.8275,2.173500,79.2611,1.226900,0.3808,2.307300,7.1937,0.858400),
   232|    "AL": (6.420200,3.0387,1.900200,0.7426,1.593600,31.5472,1.964600,85.0886,1.115100),
   233|    "SI": (6.291500,2.4386,3.035300,32.3337,1.989100,0.6785,1.541000,81.6937,1.140700),
   234|    "P":  (6.434500,1.9067,4.179100,27.1570,1.780000,0.5260,1.490800,68.1645,1.114900),
   235|    "S":  (6.905300,1.4679,5.203400,22.2151,1.437900,0.2536,1.586300,56.1720,0.866900),
   236|    "CL": (11.4604,0.0104,7.1964,1.1662,6.2556,18.5194,1.6455,47.7784,-9.5574),
   237|    "K":  (8.218600,12.7949,7.439800,0.7748,1.051900,213.187,0.865900,41.6841,1.422800),
   238|    "CA": (8.626600,10.4421,7.387300,0.6599,1.589900,85.7484,1.021100,178.437,1.375100),
   239|    "FE": (11.7695,4.7611,7.357300,0.3072,3.522200,15.3535,2.304500,76.8805,1.036900),
   240|    "CU": (13.3380,3.5828,7.167600,0.2470,5.615800,11.3966,1.673500,64.8126,1.191000),
   241|    "ZN": (14.0743,3.2655,7.031800,0.2333,5.165200,10.3163,2.410000,58.7097,1.304100),
   242|    "BR": (17.1789,2.1723,5.235800,16.5796,5.637700,0.2609,3.985100,41.4328,2.955700),
   243|    "RB": (17.1784,2.1995,9.643500,0.3491,5.139900,16.5596,1.529200,39.1799,3.487300),
   244|    "SR": (17.5663,1.5564,9.818400,14.0988,5.422000,0.1664,2.669400,132.376,2.506400),
   245|    "BA": (19.3491,0.2206,19.1080,5.7946,4.433000,14.9353,2.157800,0.0521,5.751400),
   246|    "PB": (31.0617,0.6902,13.0637,2.3576,18.4420,8.6180,5.969600,47.2579,13.4118),
   247|    "I":  (20.1472,4.3470,18.9949,0.3814,7.513800,27.7660,2.273500,66.8776,4.071200),
   248|}
   249|
   250|def get_cm(element: str):
   251|    """Return Cromer-Mann params for element, fallback to C."""
   252|    return CM_PARAMS.get(element.upper(), CM_PARAMS["C"])
   253|
   254|def atomic_form_factor(element: str, s: float) -> float:
   255|    """f(s) at s = sinθ/λ  (Å⁻¹)."""
   256|    a1,b1,a2,b2,a3,b3,a4,b4,c = get_cm(element)
   257|    s2 = s * s
   258|    return (a1*math.exp(-b1*s2) + a2*math.exp(-b2*s2) +
   259|            a3*math.exp(-b3*s2) + a4*math.exp(-b4*s2) + c)
   260|
   261|
   262|# ════════════════════════════════════════════════════════════════════════════
   263|# SYMMETRY OPERATIONS
   264|# ════════════════════════════════════════════════════════════════════════════
   265|
   266|def apply_symop(x, y, z, op_str: str):
   267|    """Apply symmetry operation string → (x', y', z') in [0,1)."""
   268|    op_str = op_str.strip().strip("'\"")
   269|    result = []
   270|    for part in op_str.split(","):
   271|        part = part.strip().lower()
   272|        part = re.sub(r"(\d+)/(\d+)", lambda m: str(float(m.group(1))/float(m.group(2))), part)
   273|        part = re.sub(r"(?<![a-z])x(?![a-z])", f"({x})", part)
   274|        part = re.sub(r"(?<![a-z])y(?![a-z])", f"({y})", part)
   275|        part = re.sub(r"(?<![a-z])z(?![a-z])", f"({z})", part)
   276|        try:
   277|            val = eval(part)
   278|        except:
   279|            val = 0.0
   280|        result.append(float(val) % 1.0)
   281|    return tuple(result)
   282|
   283|
   284|# ════════════════════════════════════════════════════════════════════════════
   285|# CELL GEOMETRY
   286|# ════════════════════════════════════════════════════════════════════════════
   287|
   288|def cell_volume(a,b,c,al,be,ga):
   289|    """Volume in Å³."""
   290|    ca,cb,cg = math.cos(math.radians(al)), math.cos(math.radians(be)), math.cos(math.radians(ga))
   291|    return a*b*c*math.sqrt(1-ca**2-cb**2-cg**2+2*ca*cb*cg)
   292|
   293|def d_spacing(h,k,l, a,b,c,al_deg,be_deg,ga_deg):
   294|    """d-spacing in Å for general triclinic cell (Buerger formula)."""
   295|    al = math.radians(al_deg)
   296|    be = math.radians(be_deg)
   297|    ga = math.radians(ga_deg)
   298|    ca,cb,cg = math.cos(al),math.cos(be),math.cos(ga)
   299|    sa,sb,sg = math.sin(al),math.sin(be),math.sin(ga)
   300|    V = cell_volume(a,b,c,al_deg,be_deg,ga_deg)
   301|
   302|    s11 = b**2*c**2*sa**2
   303|    s22 = a**2*c**2*sb**2
   304|    s33 = a**2*b**2*sg**2
   305|    s12 = a*b*c**2*(ca*cb-cg)
   306|    s23 = a**2*b*c*(cb*cg-ca)
   307|    s13 = a*b**2*c*(ca*cg-cb)
   308|
   309|    inv_d2 = (s11*h**2 + s22*k**2 + s33*l**2
   310|              + 2*s12*h*k + 2*s23*k*l + 2*s13*h*l) / V**2
   311|    if inv_d2 <= 0:
   312|        return None
   313|    return 1.0 / math.sqrt(inv_d2)
   314|
   315|def two_theta(d, wavelength=1.54056):
   316|    """2θ in degrees for given d-spacing and wavelength (Å)."""
   317|    arg = wavelength / (2*d)
   318|    if abs(arg) > 1:
   319|        return None
   320|    return 2 * math.degrees(math.asin(arg))
   321|
   322|
   323|# ════════════════════════════════════════════════════════════════════════════
   324|# STRUCTURE FACTOR CALCULATION
   325|# ════════════════════════════════════════════════════════════════════════════
   326|
   327|def compute_structure_factor(h, k, l,
   328|                              atoms,        # list of dicts: element,x,y,z,occ,U_iso
   329|                              wavelength=1.54056):
   330|    """
   331|    Returns F_hkl (complex), |F|, |F|², phase (deg), and per-atom contributions.
   332|    atoms: fractional coordinates after symmetry expansion.
   333|    """
   334|    d = d_spacing(h,k,l, *_cell_params_from_atoms_ctx)
   335|    if d is None:
   336|        return None
   337|
   338|    s = 1.0 / (2*d)   # sinθ/λ
   339|
   340|    F = 0+0j
   341|    contributions = []
   342|    for atom in atoms:
   343|        elem = atom["element"]
   344|        x,y,z   = atom["x"], atom["y"], atom["z"]
   345|        occ     = atom.get("occ", 1.0)
   346|        U_iso   = atom.get("U_iso", 0.02)   # Å²
   347|
   348|        f  = atomic_form_factor(elem, s)
   349|        DW = math.exp(-8*math.pi**2 * U_iso * s**2)  # Debye-Waller
   350|
   351|        phase_rad = 2*math.pi*(h*x + k*y + l*z)
   352|        contribution = occ * f * DW * cmath_exp(phase_rad)
   353|        F += contribution
   354|
   355|        contributions.append({
   356|            "element": elem,
   357|            "label":   atom.get("label", elem),
   358|            "f(s)":    round(f, 4),
   359|            "DW":      round(DW, 4),
   360|            "occ":     occ,
   361|            "|contrib|": round(abs(contribution), 4),
   362|            "phase(°)": round(math.degrees(phase_rad % (2*math.pi)), 2),
   363|        })
   364|
   365|    absF   = abs(F)
   366|    absF2  = absF**2
   367|    phase  = math.degrees(math.atan2(F.imag, F.real))
   368|    return {
   369|        "h": h, "k": k, "l": l,
   370|        "F_real": round(F.real, 4),
   371|        "F_imag": round(F.imag, 4),
   372|        "|F|":    round(absF, 4),
   373|        "|F|²":   round(absF2, 4),
   374|        "phase°": round(phase, 2),
   375|        "d(Å)":   round(d, 5),
   376|        "2θ(°)":  round(two_theta(d, wavelength) or 0, 4),
   377|        "s(Å⁻¹)": round(s, 5),
   378|        "contributions": contributions,
   379|    }
   380|
   381|def cmath_exp(phase_rad: float):
   382|    return complex(math.cos(phase_rad), math.sin(phase_rad))
   383|
   384|
   385|# global cell params injected before compute_structure_factor calls
   386|_cell_params_from_atoms_ctx = (1, 1, 1, 90, 90, 90)
   387|
   388|
   389|# ════════════════════════════════════════════════════════════════════════════
   390|# STRUCTURE EXPANSION (asymm. unit → full cell via symmetry)
   391|# ════════════════════════════════════════════════════════════════════════════
   392|
   393|def expand_atoms(asym_atoms, sym_ops):
   394|    tol = 0.005
   395|    all_atoms = []
   396|    seen = set()
   397|    for atom in asym_atoms:
   398|        for op in sym_ops:
   399|            nx,ny,nz = apply_symop(atom["x"], atom["y"], atom["z"], op)
   400|            key = (atom["element"], round(nx,3), round(ny,3), round(nz,3))
   401|            if key in seen: continue
   402|            seen.add(key)
   403|            new = dict(atom)
   404|            new["x"], new["y"], new["z"] = nx, ny, nz
   405|            all_atoms.append(new)
   406|    return all_atoms
   407|
   408|
   409|# ════════════════════════════════════════════════════════════════════════════
   410|# SYSTEMATIC ABSENCE CHECK
   411|# ════════════════════════════════════════════════════════════════════════════
   412|
   413|def is_systematic_absence(h, k, l, centering: str) -> bool:
   414|    """Check common systematic absences for lattice centering."""
   415|    if centering == "P":
   416|        return False
   417|    if centering == "I":
   418|        return (h+k+l) % 2 != 0
   419|    if centering == "F":
   420|        parities = {h%2, k%2, l%2}
   421|        return len(parities) > 1
   422|    if centering == "C":
   423|        return (h+k) % 2 != 0
   424|    if centering == "A":
   425|        return (k+l) % 2 != 0
   426|    if centering == "B":
   427|        return (h+l) % 2 != 0
   428|    if centering == "R":
   429|        return (-h+k+l) % 3 != 0
   430|    return False
   431|
   432|
   433|# ════════════════════════════════════════════════════════════════════════════
   434|# POWDER DIFFRACTION PATTERN
   435|# ════════════════════════════════════════════════════════════════════════════
   436|
   437|def compute_powder_pattern(all_hkl_results, two_theta_range=(5,80),
   438|                            fwhm=0.15, n_pts=3000):
   439|    """Gaussian-broadened powder pattern."""
   440|    tt_min, tt_max = two_theta_range
   441|    tt_arr = np.linspace(tt_min, tt_max, n_pts)
   442|    pattern = np.zeros(n_pts)
   443|
   444|    for r in all_hkl_results:
   445|        tt0 = r.get("2θ(°)")
   446|        I   = r.get("|F|²", 0)
   447|        if not tt0 or tt0 < tt_min or tt0 > tt_max:
   448|            continue
   449|        sigma = fwhm / (2*math.sqrt(2*math.log(2)))
   450|        pattern += I * np.exp(-0.5*((tt_arr - tt0)/sigma)**2)
   451|
   452|    return tt_arr, pattern
   453|
   454|
   455|# ════════════════════════════════════════════════════════════════════════════
   456|# SIDEBAR + FILE LOADING  (identical pattern to bravais_from_cif.py)
   457|# ════════════════════════════════════════════════════════════════════════════
   458|
   459|with st.sidebar:
   460|    st.markdown("## 🔭 STRUCTURE FACTOR")
   461|    st.markdown("---")
   462|
   463|    cif_text = None
   464|    cif_name = ""
   465|
   466|    # 1. CLI
   467|    if PRELOAD_PATH and PRELOAD_PATH.exists():
   468|        cif_text = PRELOAD_PATH.read_text(encoding="utf-8", errors="replace")
   469|        cif_name = PRELOAD_PATH.name
   470|
   471|    # 2. Auto-search parent folders
   472|    if cif_text is None:
   473|        script_dir = Path(__file__).resolve().parent
   474|        search_dirs = [script_dir.parent, script_dir.parent.parent, script_dir]
   475|        found_cifs  = []
   476|        for d in search_dirs:
   477|            found_cifs += sorted(d.glob("*.cif"))
   478|        found_cifs = list(dict.fromkeys(found_cifs))
   479|
   480|        if found_cifs:
   481|            cif_names    = [f.name for f in found_cifs]
   482|            selected_idx = st.selectbox("CIF-Datei", range(len(cif_names)),
   483|                                         format_func=lambda i: cif_names[i], index=0)
   484|            chosen   = found_cifs[selected_idx]
   485|            cif_text = chosen.read_text(encoding="utf-8", errors="replace")
   486|            cif_name = chosen.name
   487|            st.caption(f"📂 {chosen.parent}")
   488|
   489|    # 3. Manual upload fallback
   490|    if cif_text is None:
   491|        st.markdown("**Keine CIF im Parent-Ordner**")
   492|        up = st.file_uploader("CIF hochladen", type=["cif"])
   493|        if up:
   494|            cif_text = up.read().decode("utf-8", errors="replace")
   495|            cif_name = up.name
   496|
   497|    st.markdown("---")
   498|    st.markdown("**Berechnung**")
   499|    wavelength   = st.number_input("Wellenlänge λ (Å)", 0.5, 3.0, 1.54056, 0.00001,
   500|                                    format="%.5f", help="Cu Kα = 1.54056 Å")
   501|