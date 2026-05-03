
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
     2|XRD Full-Profile Fitter  ·  v2  ·  CIF-constrained CNN
     3|========================================================
     4|New in v2:
     5|  • CIF uploader — parses lattice params, space group, atom sites
     6|  • d-spacing calculator — Bragg's law + systematic absence filter
     7|  • Structure factor estimator — weighted by atomic scattering factors
     8|  • CIF-constrained peak positions fed into CNN warm-start
     9|  • 3-D unit cell / atom viewer (matplotlib)
    10|  • Wavelength selector (Cu Kα, Mo Kα, Co Kα, Fe Kα, custom)
    11|
    12|Supported diffraction formats: .txt .csv .dat .xy .xye .asc .ras .raw .fxye .gsas .cpi
    13|Supported structure format:    .cif
    14|
    15|Run: streamlit run xrd_app_v2.py
    16|"""
    17|
    18|import io, re, math, os, itertools, traceback, warnings
    19|import numpy as np
    20|import streamlit as st
    21|import matplotlib.pyplot as plt
    22|import matplotlib.gridspec as gridspec
    23|from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    24|from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    25|from scipy.signal import find_peaks
    26|from scipy.optimize import least_squares
    27|
    28|warnings.filterwarnings("ignore")
    29|
    30|try:
    31|    import torch
    32|    import torch.nn as nn
    33|    from torch.utils.data import DataLoader, TensorDataset
    34|    TORCH_OK = True
    35|except ImportError:
    36|    TORCH_OK = False
    37|
    38|# ══════════════════════════════════════════════════════════════════
    39|#  PAGE CONFIG
    40|# ══════════════════════════════════════════════════════════════════
    41|st.set_page_config(
    42|    page_title="XRD Profile Fitter",
    43|    page_icon="⚛",
    44|    layout="wide",
    45|    initial_sidebar_state="expanded",
    46|)
    47|
    48|# ══════════════════════════════════════════════════════════════════
    49|#  STYLING
    50|# ══════════════════════════════════════════════════════════════════
    51|st.markdown("""
    52|<style>
    53|@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
    54|html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    55|
    56|.xrd-title { font-family:'Space Mono',monospace; font-size:1.7rem; font-weight:700;
    57|             color:#00d4ff; letter-spacing:-.04em; }
    58|.xrd-sub   { font-size:.82rem; color:#557; font-family:'Space Mono',monospace; }
    59|
    60|.metric-row { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:1rem 0; }
    61|.metric-card { background:#0f1623; border:1px solid #1e2d3d; border-radius:10px;
    62|               padding:14px 16px; text-align:center; }
    63|.metric-label { font-family:'Space Mono',monospace; font-size:.62rem; color:#445;
    64|                text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
    65|.metric-value { font-family:'Space Mono',monospace; font-size:1.25rem;
    66|                font-weight:700; color:#00d4ff; }
    67|.metric-unit  { font-size:.68rem; color:#445; margin-top:2px; }
    68|
    69|.cif-card { background:#0a1420; border:1px solid #1a3040; border-radius:12px;
    70|            padding:1.2rem 1.5rem; margin-bottom:1rem; }
    71|.cif-key   { font-family:'Space Mono',monospace; font-size:.72rem; color:#446688;
    72|             text-transform:uppercase; letter-spacing:.06em; }
    73|.cif-val   { font-family:'Space Mono',monospace; font-size:1.05rem; color:#a8e0ff; }
    74|
    75|.hkl-table { width:100%; border-collapse:collapse;
    76|             font-family:'Space Mono',monospace; font-size:.75rem; }
    77|.hkl-table th { background:#0a1825; color:#00d4ff; padding:7px 10px;
    78|                text-align:right; font-weight:700; border-bottom:1px solid #1e2d3d; }
    79|.hkl-table th:first-child { text-align:left; }
    80|.hkl-table td { padding:5px 10px; text-align:right; color:#8aabb0;
    81|                border-bottom:1px solid #0d1820; }
    82|.hkl-table td:first-child { text-align:left; color:#ffd700; }
    83|.hkl-table tr:hover td { background:#0f1c2a; }
    84|
    85|.peak-table { width:100%; border-collapse:collapse;
    86|              font-family:'Space Mono',monospace; font-size:.78rem; }
    87|.peak-table th { background:#0f1623; color:#00d4ff; padding:8px 10px;
    88|                 text-align:right; font-weight:700; border-bottom:1px solid #1e2d3d; }
    89|.peak-table th:first-child { text-align:left; }
    90|.peak-table td { padding:7px 10px; text-align:right; color:#c0c8d8;
    91|                 border-bottom:1px solid #111a22; }
    92|.peak-table td:first-child { text-align:left; color:#ffd700; }
    93|.peak-table tr:hover td { background:#0f1c2a; }
    94|
    95|.badge { display:inline-block; padding:2px 8px; border-radius:4px;
    96|         font-size:.7rem; font-family:'Space Mono',monospace; font-weight:700;
    97|         background:#0d2233; color:#00d4ff; border:1px solid #00d4ff44; }
    98|.badge-green { background:#0a1f12; color:#a8ff78; border-color:#a8ff7844; }
    99|.badge-gold  { background:#1f180a; color:#ffd700; border-color:#ffd70044; }
   100|
   101|section[data-testid="stSidebar"] { background:#080d14; border-right:1px solid #1a2433; }
   102|[data-testid="stFileUploader"] { border:1.5px dashed #00d4ff55 !important;
   103|                                  border-radius:12px !important; transition:border-color .2s; }
   104|[data-testid="stFileUploader"]:hover { border-color:#00d4ff !important; }
   105|</style>
   106|""", unsafe_allow_html=True)
   107|
   108|# ══════════════════════════════════════════════════════════════════
   109|#  COLOUR CONSTANTS
   110|# ══════════════════════════════════════════════════════════════════
   111|TEXT  = '#c8d0e0'; GRID = '#1a2233'; BG = '#080d14'; BG2 = '#0d1520'
   112|COLORS = ['#00d4ff','#ffd700','#a8ff78','#ff6b6b','#a29bfe',
   113|          '#fd79a8','#74b9ff','#ff9f43','#55efc4','#e17055']
   114|
   115|# ══════════════════════════════════════════════════════════════════
   116|#  WAVELENGTHS (Å)
   117|# ══════════════════════════════════════════════════════════════════
   118|WAVELENGTHS = {
   119|    "Cu Kα  (1.5406 Å)": 1.5406,
   120|    "Mo Kα  (0.7093 Å)": 0.7093,
   121|    "Co Kα  (1.7890 Å)": 1.7890,
   122|    "Fe Kα  (1.9373 Å)": 1.9373,
   123|    "Cr Kα  (2.2909 Å)": 2.2909,
   124|    "Ag Kα  (0.5608 Å)": 0.5608,
   125|    "Custom …":           None,
   126|}
   127|
   128|# ══════════════════════════════════════════════════════════════════
   129|#  ATOMIC SCATTERING FACTORS  (f0 at sin θ/λ = 0,  Z-based approx)
   130|# ══════════════════════════════════════════════════════════════════
   131|ATOM_F0 = {
   132|    'H':1,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Na':11,
   133|    'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'K':19,'Ca':20,
   134|    'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,
   135|    'Zn':30,'Ge':32,'As':33,'Se':34,'Br':35,'Sr':38,'Y':39,'Zr':40,
   136|    'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,
   137|    'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Ba':56,'La':57,'Ce':58,
   138|    'Pr':59,'Nd':60,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,
   139|    'Er':68,'Yb':70,'Lu':71,'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,
   140|    'Ir':77,'Pt':78,'Au':79,'Hg':80,'Pb':82,'Bi':83,'U':92,
   141|}
   142|
   143|def atom_f0(symbol):
   144|    sym = re.sub(r'[^A-Za-z]', '', symbol)
   145|    sym = sym.capitalize()
   146|    return float(ATOM_F0.get(sym, 10))
   147|
   148|# ══════════════════════════════════════════════════════════════════
   149|#  SYSTEMATIC ABSENCE RULES  (partial — covers ~40 common SGs)
   150|# ══════════════════════════════════════════════════════════════════
   151|
   152|def is_systematically_absent(h, k, l, sg_num: int) -> bool:
   153|    """
   154|    Returns True if hkl is systematically absent for space group sg_num.
   155|    Covers body-centred (I), face-centred (F), primitive (P) with
   156|    common glide/screw conditions, plus a few special cases.
   157|    """
   158|    sg = int(sg_num) if sg_num else 0
   159|
   160|    # ── Bravais lattice centering ──────────────────────────────────
   161|    # I-centred (BCC): h+k+l must be even
   162|    I_centred = set(range(197,215)) | {87,88,107,108,109,110,139,140,141,142,
   163|                                        204,206,211,214,217,220,229,230}
   164|    # F-centred (FCC): h,k,l all even or all odd
   165|    F_centred = set(range(196,197)) | {22,23,42,43,69,70,196,202,203,209,210,216,219,225,226,227,228}
   166|    # A-centred: k+l even
   167|    A_centred = {38,39,40,41}
   168|    # C-centred: h+k even
   169|    C_centred = {5,8,9,12,13,14,15,20,21,35,36,37,63,64,65,66,67,68}
   170|
   171|    if sg in I_centred:
   172|        if (h + k + l) % 2 != 0: return True
   173|    if sg in F_centred:
   174|        parity = {h%2, k%2, l%2}
   175|        if len(parity) != 1: return True
   176|    if sg in A_centred:
   177|        if (k + l) % 2 != 0: return True
   178|    if sg in C_centred:
   179|        if (h + k) % 2 != 0: return True
   180|
   181|    # ── Special space groups used in zeolites/sodalites ───────────
   182|    # P m -3 n  (218) – sodalite:  0kl: k+l=2n, h00: h=2n
   183|    if sg == 218:
   184|        if h == 0 and k == 0:
   185|            if l % 2 != 0: return True
   186|        if h == 0 and l == 0:
   187|            if k % 2 != 0: return True
   188|        if k == 0 and l == 0:
   189|            if h % 2 != 0: return True
   190|
   191|    # I m -3 m (229) – BCC + extra
   192|    if sg == 229:
   193|        if (h + k + l) % 2 != 0: return True
   194|
   195|    # F d -3 m (227) – diamond/spinel
   196|    if sg == 227:
   197|        parity = {h%2, k%2, l%2}
   198|        if len(parity) != 1: return True
   199|        if all(x % 2 == 0 for x in [h,k,l]):
   200|            if (h+k+l) % 4 != 0: return True
   201|
   202|    return False
   203|
   204|
   205|# ══════════════════════════════════════════════════════════════════
   206|#  CIF PARSER
   207|# ══════════════════════════════════════════════════════════════════
   208|
   209|class CIFData:
   210|    """Lightweight CIF parser — no external dependencies."""
   211|
   212|    def __init__(self):
   213|        self.a = self.b = self.c = None
   214|        self.alpha = self.beta = self.gamma = 90.0
   215|        self.sg_name = "Unknown"
   216|        self.sg_num  = 0
   217|        self.atoms   = []   # list of dict: label, element, x, y, z, occ, Biso
   218|        self.formula = ""
   219|        self.name    = ""
   220|
   221|    @classmethod
   222|    def from_string(cls, text):
   223|        obj = cls()
   224|        # strip esd parentheses like 8.970(3)
   225|        text_clean = re.sub(r'\([\d]+\)', '', text)
   226|
   227|        def scalar(key):
   228|            m = re.search(key + r'\s+([\d.+-]+)', text_clean, re.IGNORECASE)
   229|            return float(m.group(1)) if m else None
   230|
   231|        def strval(key):
   232|            m = re.search(key + r"\s+['\"]?([^'\"\n#]+)['\"]?", text_clean, re.IGNORECASE)
   233|            return m.group(1).strip() if m else ""
   234|
   235|        obj.a     = scalar(r'_cell_length_a')
   236|        obj.b     = scalar(r'_cell_length_b') or obj.a
   237|        obj.c     = scalar(r'_cell_length_c') or obj.a
   238|        obj.alpha = scalar(r'_cell_angle_alpha') or 90.0
   239|        obj.beta  = scalar(r'_cell_angle_beta')  or 90.0
   240|        obj.gamma = scalar(r'_cell_angle_gamma') or 90.0
   241|
   242|        for key in [r'_symmetry_space_group_name_H-M',
   243|                    r'_space_group_name_H-M_alt',
   244|                    r'_symmetry_space_group_name_H.M']:
   245|            sg = strval(key)
   246|            if sg:
   247|                obj.sg_name = sg; break
   248|
   249|        for key in [r'_symmetry_Int_Tables_number',
   250|                    r'_space_group_IT_number',
   251|                    r'_space_group\.IT_number']:
   252|            n = scalar(key)
   253|            if n:
   254|                obj.sg_num = int(n); break
   255|
   256|        obj.name    = strval(r'_pd_phase_name') or strval(r'_chemical_name_mineral') \
   257|                    or strval(r'_chemical_name_common') or "Unknown phase"
   258|        obj.formula = strval(r'_chemical_formula_sum') or strval(r'_chemical_formula_structural')
   259|
   260|        # ── parse atom loop ────────────────────────────────────────
   261|        obj.atoms = obj._parse_atom_loop(text_clean)
   262|        return obj
   263|
   264|    @staticmethod
   265|    def _parse_atom_loop(text):
   266|        atoms = []
   267|        # find _atom_site loop blocks
   268|        loop_pattern = re.compile(
   269|            r'loop_\s*((?:_atom_site_\S+\s*)+)((?:[^_\n][^\n]*\n?)*)', re.MULTILINE)
   270|
   271|        for lm in loop_pattern.finditer(text):
   272|            header_block = lm.group(1)
   273|            data_block   = lm.group(2)
   274|
   275|            keys = re.findall(r'_atom_site_(\S+)', header_block)
   276|            if 'fract_x' not in keys and 'Cartn_x' not in keys:
   277|                continue
   278|
   279|            rows = []
   280|            for line in data_block.strip().split('\n'):
   281|                line = line.strip()
   282|                if not line or line.startswith('_') or line.startswith('#'):
   283|                    continue
   284|                # handle quoted strings
   285|                parts = re.split(r'\s+', line)
   286|                if len(parts) >= len(keys):
   287|                    rows.append(parts)
   288|
   289|            col = {k: i for i, k in enumerate(keys)}
   290|            for row in rows:
   291|                def g(k, default='.'):
   292|                    return row[col[k]] if k in col and col[k] < len(row) else default
   293|                try:
   294|                    label   = g('label', 'X')
   295|                    element = g('type_symbol', re.sub(r'[^A-Za-z]', '', label))
   296|                    element = re.sub(r'[^A-Za-z]', '', element)
   297|                    x = float(g('fract_x', '0').replace('?', '0'))
   298|                    y = float(g('fract_y', '0').replace('?', '0'))
   299|                    z = float(g('fract_z', '0').replace('?', '0'))
   300|                    occ  = float(g('occupancy', '1').replace('?', '1'))
   301|                    biso = float(g('B_iso_or_equiv', '1').replace('?', '1'))
   302|                    atoms.append(dict(label=label, element=element,
   303|                                      x=x, y=y, z=z, occ=occ, biso=biso))
   304|                except (ValueError, IndexError):
   305|                    continue
   306|        return atoms
   307|
   308|
   309|# ══════════════════════════════════════════════════════════════════
   310|#  RECIPROCAL LATTICE + STRUCTURE FACTOR
   311|# ══════════════════════════════════════════════════════════════════
   312|
   313|def metric_tensor(a, b, c, alpha_deg, beta_deg, gamma_deg):
   314|    al = math.radians(alpha_deg)
   315|    be = math.radians(beta_deg)
   316|    ga = math.radians(gamma_deg)
   317|    G = np.array([
   318|        [a*a,          a*b*math.cos(ga), a*c*math.cos(be)],
   319|        [a*b*math.cos(ga), b*b,          b*c*math.cos(al)],
   320|        [a*c*math.cos(be), b*c*math.cos(al), c*c          ],
   321|    ])
   322|    return G
   323|
   324|def d_spacing(h, k, l, G_inv):
   325|    """d = 1/sqrt(h k l · G* · h k l^T)"""
   326|    v = np.array([h, k, l], dtype=float)
   327|    q2 = float(v @ G_inv @ v)
   328|    if q2 <= 0: return np.inf
   329|    return 1.0 / math.sqrt(q2)
   330|
   331|def structure_factor_sq(h, k, l, atoms):
   332|    """
   333|    |F(hkl)|² using atomic form factors and fractional coordinates.
   334|    f(s) ≈ Z · exp(−b · s²)  with  s = sin(θ)/λ = 1/(2d)
   335|    """
   336|    if not atoms:
   337|        return 1.0
   338|    F_real = F_imag = 0.0
   339|    for at in atoms:
   340|        f0 = atom_f0(at['element'])
   341|        b  = at.get('biso', 1.0)
   342|        phase = 2 * math.pi * (h * at['x'] + k * at['y'] + l * at['z'])
   343|        F_real += at['occ'] * f0 * math.cos(phase)
   344|        F_imag += at['occ'] * f0 * math.sin(phase)
   345|    return F_real**2 + F_imag**2
   346|
   347|def multiplicity(h, k, l):
   348|    """Approximate multiplicity for cubic symmetry."""
   349|    vals = sorted([abs(h), abs(k), abs(l)])
   350|    if vals[0] == vals[1] == vals[2]:     # hhh
   351|        return 8
   352|    if vals[0] == 0 and vals[1] == 0:     # 00l
   353|        return 6
   354|    if vals[0] == 0 and vals[1] == vals[2]: # 0ll
   355|        return 12
   356|    if vals[0] == 0:                       # 0kl
   357|        return 24
   358|    if vals[0] == vals[1] == vals[2]:      # shouldn't reach here
   359|        return 8
   360|    if vals[1] == vals[2]:                 # hll
   361|        return 24
   362|    if vals[0] == vals[1]:                 # hhl
   363|        return 24
   364|    return 48                               # hkl
   365|
   366|
   367|def predict_reflections(cif: CIFData, wavelength: float,
   368|                         two_theta_min=4.0, two_theta_max=85.0,
   369|                         max_hkl=10, min_F2=0.01):
   370|    """
   371|    Returns list of dicts: {h,k,l, d, two_theta, F2, I_rel}
   372|    sorted by two_theta.
   373|    """
   374|    if not cif.a:
   375|        return []
   376|
   377|    G  = metric_tensor(cif.a, cif.b, cif.c, cif.alpha, cif.beta, cif.gamma)
   378|    Gi = np.linalg.inv(G)
   379|
   380|    reflections = []
   381|    seen_d = set()
   382|
   383|    for h in range(-max_hkl, max_hkl+1):
   384|        for k in range(-max_hkl, max_hkl+1):
   385|            for l in range(-max_hkl, max_hkl+1):
   386|                if h == k == l == 0: continue
   387|                # use positive hemisphere only
   388|                if (h < 0 or (h == 0 and k < 0) or
   389|                    (h == 0 and k == 0 and l < 0)):
   390|                    continue
   391|
   392|                if is_systematically_absent(h, k, l, cif.sg_num):
   393|                    continue
   394|
   395|                d = d_spacing(h, k, l, Gi)
   396|                if d == np.inf: continue
   397|
   398|                sin_th = wavelength / (2.0 * d)
   399|                if sin_th > 1.0: continue
   400|                two_th = 2.0 * math.degrees(math.asin(sin_th))
   401|                if not (two_theta_min <= two_th <= two_theta_max):
   402|                    continue
   403|
   404|                # deduplicate by d (within 0.001 Å)
   405|                d_key = round(d, 3)
   406|                if d_key in seen_d:
   407|                    continue
   408|                seen_d.add(d_key)
   409|
   410|                F2  = structure_factor_sq(h, k, l, cif.atoms)
   411|                if F2 < min_F2 and cif.atoms:
   412|                    continue
   413|                m   = multiplicity(h, k, l)
   414|                Lp  = (1 + math.cos(math.radians(two_th))**2) / \
   415|                      (math.sin(math.radians(two_th/2))**2 *
   416|                       math.cos(math.radians(two_th/2)))
   417|                I   = m * Lp * F2
   418|
   419|                reflections.append(dict(h=h, k=k, l=l, d=d,
   420|                                         two_theta=two_th, F2=F2, I_raw=I))
   421|
   422|    if not reflections:
   423|        return reflections
   424|
   425|    I_max = max(r['I_raw'] for r in reflections)
   426|    for r in reflections:
   427|        r['I_rel'] = 100.0 * r['I_raw'] / I_max if I_max > 0 else 0.0
   428|
   429|    return sorted(reflections, key=lambda r: r['two_theta'])
   430|
   431|
   432|# ══════════════════════════════════════════════════════════════════
   433|#  PROFILE FUNCTIONS
   434|# ══════════════════════════════════════════════════════════════════
   435|
   436|def gaussian(x, pos, fwhm):
   437|    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
   438|    return np.exp(-0.5 * ((x - pos) / sigma) ** 2)
   439|
   440|def lorentzian(x, pos, fwhm):
   441|    gamma = fwhm / 2.0
   442|    return 1.0 / (1.0 + ((x - pos) / gamma) ** 2)
   443|
   444|def pseudo_voigt(x, pos, intensity, fwhm, eta):
   445|    eta = np.clip(eta, 0.0, 1.0)
   446|    return intensity * (eta * lorentzian(x, pos, fwhm) +
   447|                        (1 - eta) * gaussian(x, pos, fwhm))
   448|
   449|def polynomial_bg(x, a0, a1, a2):
   450|    xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
   451|    return a0 + a1 * xn + a2 * xn**2
   452|
   453|def full_profile(x, params, n_peaks):
   454|    y = np.zeros_like(x, dtype=np.float64)
   455|    for k in range(n_peaks):
   456|        pos, intensity, fwhm, eta = params[4*k:4*k+4]
   457|        y += pseudo_voigt(x, pos, intensity, fwhm, eta)
   458|    a0, a1, a2 = params[4*n_peaks:]
   459|    y += polynomial_bg(x, a0, a1, a2)
   460|    return y
   461|
   462|
   463|# ══════════════════════════════════════════════════════════════════
   464|#  FILE PARSERS  (diffraction data)
   465|# ══════════════════════════════════════════════════════════════════
   466|
   467|def _try_loadtxt(content_bytes):
   468|    for enc in ('utf-8', 'latin-1', 'cp1252'):
   469|        try:
   470|            text = content_bytes.decode(enc); break
   471|        except Exception:
   472|            continue
   473|    else:
   474|        raise ValueError("Cannot decode file")
   475|
   476|    rows = []
   477|    for line in text.splitlines():
   478|        line = line.strip()
   479|        if not line or line.startswith(('#','!',';',"'",'"')): continue
   480|        if re.match(r'^[A-Za-z_]', line): continue
   481|        parts = line.split()
   482|        try:
   483|            nums = [float(p) for p in parts[:3]]
   484|            if len(nums) >= 2:
   485|                rows.append((nums[0], nums[1]))
   486|        except ValueError:
   487|            continue
   488|    if len(rows) < 10:
   489|        raise ValueError(f"Only {len(rows)} numeric rows found")
   490|    arr = np.array(rows, dtype=np.float32)
   491|    return arr[:,0], arr[:,1]
   492|
   493|DIFF_EXTS = [".txt",".csv",".dat",".xy",".xye",".asc",
   494|             ".ras",".raw",".fxye",".gsas",".cpi"]
   495|
   496|def parse_diff_file(f):
   497|    content = f.read()
   498|    x, y = _try_loadtxt(content)
   499|    if x.max() > 180 or x.min() < 0:
   500|        raise ValueError(f"2θ range [{x.min():.1f}, {x.max():.1f}] looks wrong")
   501|