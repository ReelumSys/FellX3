"""
Structure Factor & HKL Search
==============================
Liest CIF aus dem Parent-Ordner, berechnet Strukturfaktoren F(hkl)
und ermöglicht interaktive HKL-Suche.

Physik:
  F(hkl) = Σ_j  f_j(s) · occ_j · exp(2πi (h·x_j + k·y_j + l·z_j))
  |F|² = Intensität (proportional)
  s = sinθ/λ = 1/(2d_hkl)

Cromer-Mann Atomformfaktoren:
  f(s) = Σ_i a_i · exp(-b_i · s²) + c

Installation:
    pip install streamlit matplotlib numpy pandas

Starten:
    streamlit run structure_factor.py
"""

import sys, re, math, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# ── CLI preload (same pattern as bravais_from_cif.py) ────────────────────────
PRELOAD_PATH = None
args = sys.argv[1:]
for i, arg in enumerate(args):
    if arg == "--cif" and i+1 < len(args):
        PRELOAD_PATH = Path(args[i+1])
    elif arg.endswith(".cif"):
        PRELOAD_PATH = Path(arg)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Structure Factor & HKL",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Share Tech Mono', monospace !important; }
.stApp { background: #07080f; color: #c8d4f8; }
[data-testid="stSidebar"] { background: #0d0e1c !important; border-right:1px solid #1e2555; }
[data-testid="stSidebar"] * { color: #c8d4f8 !important; }
[data-testid="stSidebar"] label { color: #4466aa !important; font-size:12px; }
[data-testid="metric-container"] {
    background: rgba(20,26,70,0.6);
    border: 1px solid rgba(50,80,200,0.3);
    border-radius: 10px; padding: 12px 18px;
}
[data-testid="stMetricLabel"] { color:#4466aa !important; font-family:'Share Tech Mono',monospace; font-size:10px; }
[data-testid="stMetricValue"] { color:#00e5c8 !important; font-size:22px; }
.page-header {
    background: linear-gradient(135deg, rgba(20,26,70,0.8), rgba(10,15,45,0.9));
    border: 1px solid rgba(50,80,200,0.35);
    border-left: 4px solid #00e5c8;
    border-radius: 12px;
    padding: 20px 28px; margin-bottom: 20px;
}
.tag {
    display:inline-block; padding:3px 14px; border-radius:16px;
    font-family:'Share Tech Mono',monospace; font-size:11px;
    margin:3px; letter-spacing:0.08em;
}
.hkl-result {
    background: rgba(0,229,200,0.06);
    border: 1px solid rgba(0,229,200,0.25);
    border-radius: 10px; padding:14px 20px; margin:6px 0;
}
.formula-box {
    background: rgba(15,20,55,0.7);
    border: 1px solid rgba(50,80,200,0.3);
    border-radius: 10px; padding:16px 20px; margin:10px 0;
    font-family:'Share Tech Mono',monospace; font-size:13px;
    color:#aabcee; line-height:2;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CIF PARSER  (identical to bravais_from_cif.py)
# ════════════════════════════════════════════════════════════════════════════

def parse_number(s):
    if s is None or str(s).strip() in ("?", "."):
        return None
    try:
        return float(re.sub(r"\(.*?\)", "", str(s).strip()))
    except:
        return None

def _tokenize(line):
    tokens = []
    i = 0
    while i < len(line):
        if line[i] in ('"', "'"):
            q = line[i]; j = line.find(q, i+1)
            if j == -1: j = len(line)
            tokens.append(line[i+1:j]); i = j+1
        elif line[i] in (" ", "\t"):
            i += 1
        else:
            j = i
            while j < len(line) and line[j] not in (" ", "\t"):
                j += 1
            tokens.append(line[i:j]); i = j
    return tokens

def parse_cif_full(text: str):
    """Parse CIF → (scalar_dict, list_of_loops).
    Each loop: {'keys': [...], 'rows': [[...], ...]}
    """
    scalars = {}
    loops   = []
    lines   = text.splitlines()
    i = 0
    cur_loop_keys = []
    cur_loop_rows = []
    in_loop = False

    def flush_loop():
        if cur_loop_keys and cur_loop_rows:
            loops.append({"keys": cur_loop_keys[:], "rows": [r[:] for r in cur_loop_rows]})

    while i < len(lines):
        raw  = lines[i]
        line = raw.strip()

        if not line or line.startswith("#"):
            i += 1; continue

        # multi-line string
        if line.startswith(";"):
            val_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(";"):
                val_lines.append(lines[i])
                i += 1
            # attach to last scalar key
            if scalars:
                last = list(scalars)[-1]
                if scalars[last] == "__PENDING__":
                    scalars[last] = "\n".join(val_lines).strip()
            i += 1; continue

        if line.lower() == "loop_":
            flush_loop()
            cur_loop_keys = []; cur_loop_rows = []
            in_loop = True; i += 1; continue

        if in_loop:
            if line.startswith("_"):
                # still collecting keys — but if we already have rows, new key ends loop
                if cur_loop_rows:
                    flush_loop()
                    cur_loop_keys = []; cur_loop_rows = []
                    in_loop = False
                    # re-process this line as scalar
                    continue
                cur_loop_keys.append(line.split()[0].lower())
                i += 1; continue
            else:
                if line.lower() == "loop_":
                    flush_loop()
                    cur_loop_keys = []; cur_loop_rows = []
                    i += 1; continue
                # data row — collect tokens, package into rows of len(keys)
                tokens = _tokenize(line)
                n = len(cur_loop_keys)
                if n:
                    for t in tokens:
                        if not cur_loop_rows or len(cur_loop_rows[-1]) == n:
                            cur_loop_rows.append([])
                        cur_loop_rows[-1].append(t)
                i += 1; continue

        if line.startswith("_"):
            parts = _tokenize(line)
            key   = parts[0].lower()
            if len(parts) >= 2:
                scalars[key] = parts[1]
            else:
                scalars[key] = "__PENDING__"
            i += 1; continue

        i += 1

    flush_loop()
    return scalars, loops

def loop_as_df(loops, key: str) -> pd.DataFrame | None:
    key = key.lower()
    for lp in loops:
        if any(key in k for k in lp["keys"]):
            n = len(lp["keys"])
            rows = [r for r in lp["rows"] if len(r) == n]
            if rows:
                return pd.DataFrame(rows, columns=lp["keys"])
    return None


# ════════════════════════════════════════════════════════════════════════════
# CROMER-MANN ATOMIC FORM FACTORS  (International Tables Vol. C)
# ════════════════════════════════════════════════════════════════════════════
# Format: element → (a1,b1,a2,b2,a3,b3,a4,b4,c)
CM_PARAMS = {
    "H":  (0.489918,20.6593,0.262003,7.74039,0.196767,49.5519,0.049879,2.20159,0.001305),
    "HE": (0.873400,9.1037,0.630900,3.3568,0.311200,22.9276,0.178000,0.9821,0.006400),
    "LI": (1.128200,3.9546,0.750800,1.0524,0.617500,85.3905,0.465300,168.261,0.037700),
    "BE": (1.591900,43.6427,1.127800,1.8623,0.539100,103.483,0.702900,0.5420,0.038500),
    "B":  (2.054500,23.2185,1.332600,1.0210,1.097900,60.3498,0.706800,0.1403,-0.193200),
    "C":  (2.310000,20.8439,1.020000,10.2075,1.588600,0.5687,0.865000,51.6512,0.215600),
    "N":  (12.2126,0.0057,3.1322,9.8933,2.0125,28.9975,1.1663,0.5826,-11.529),
    "O":  (3.048500,13.2771,2.286800,5.7011,1.546300,0.3239,0.867000,32.9089,0.250800),
    "F":  (3.539200,10.2825,2.641200,4.2944,1.517000,0.2615,1.024300,26.1476,0.277600),
    "NA": (4.762600,3.2850,3.173600,8.8422,1.267400,0.3136,1.112800,129.424,0.676000),
    "MG": (5.420400,2.8275,2.173500,79.2611,1.226900,0.3808,2.307300,7.1937,0.858400),
    "AL": (6.420200,3.0387,1.900200,0.7426,1.593600,31.5472,1.964600,85.0886,1.115100),
    "SI": (6.291500,2.4386,3.035300,32.3337,1.989100,0.6785,1.541000,81.6937,1.140700),
    "P":  (6.434500,1.9067,4.179100,27.1570,1.780000,0.5260,1.490800,68.1645,1.114900),
    "S":  (6.905300,1.4679,5.203400,22.2151,1.437900,0.2536,1.586300,56.1720,0.866900),
    "CL": (11.4604,0.0104,7.1964,1.1662,6.2556,18.5194,1.6455,47.7784,-9.5574),
    "K":  (8.218600,12.7949,7.439800,0.7748,1.051900,213.187,0.865900,41.6841,1.422800),
    "CA": (8.626600,10.4421,7.387300,0.6599,1.589900,85.7484,1.021100,178.437,1.375100),
    "FE": (11.7695,4.7611,7.357300,0.3072,3.522200,15.3535,2.304500,76.8805,1.036900),
    "CU": (13.3380,3.5828,7.167600,0.2470,5.615800,11.3966,1.673500,64.8126,1.191000),
    "ZN": (14.0743,3.2655,7.031800,0.2333,5.165200,10.3163,2.410000,58.7097,1.304100),
    "BR": (17.1789,2.1723,5.235800,16.5796,5.637700,0.2609,3.985100,41.4328,2.955700),
    "RB": (17.1784,2.1995,9.643500,0.3491,5.139900,16.5596,1.529200,39.1799,3.487300),
    "SR": (17.5663,1.5564,9.818400,14.0988,5.422000,0.1664,2.669400,132.376,2.506400),
    "BA": (19.3491,0.2206,19.1080,5.7946,4.433000,14.9353,2.157800,0.0521,5.751400),
    "PB": (31.0617,0.6902,13.0637,2.3576,18.4420,8.6180,5.969600,47.2579,13.4118),
    "I":  (20.1472,4.3470,18.9949,0.3814,7.513800,27.7660,2.273500,66.8776,4.071200),
}

def get_cm(element: str):
    """Return Cromer-Mann params for element, fallback to C."""
    return CM_PARAMS.get(element.upper(), CM_PARAMS["C"])

def atomic_form_factor(element: str, s: float) -> float:
    """f(s) at s = sinθ/λ  (Å⁻¹)."""
    a1,b1,a2,b2,a3,b3,a4,b4,c = get_cm(element)
    s2 = s * s
    return (a1*math.exp(-b1*s2) + a2*math.exp(-b2*s2) +
            a3*math.exp(-b3*s2) + a4*math.exp(-b4*s2) + c)


# ════════════════════════════════════════════════════════════════════════════
# SYMMETRY OPERATIONS
# ════════════════════════════════════════════════════════════════════════════

def apply_symop(x, y, z, op_str: str):
    """Apply symmetry operation string → (x', y', z') in [0,1)."""
    op_str = op_str.strip().strip("'\"")
    result = []
    for part in op_str.split(","):
        part = part.strip().lower()
        part = re.sub(r"(\d+)/(\d+)", lambda m: str(float(m.group(1))/float(m.group(2))), part)
        part = re.sub(r"(?<![a-z])x(?![a-z])", f"({x})", part)
        part = re.sub(r"(?<![a-z])y(?![a-z])", f"({y})", part)
        part = re.sub(r"(?<![a-z])z(?![a-z])", f"({z})", part)
        try:
            val = eval(part)
        except:
            val = 0.0
        result.append(float(val) % 1.0)
    return tuple(result)


# ════════════════════════════════════════════════════════════════════════════
# CELL GEOMETRY
# ════════════════════════════════════════════════════════════════════════════

def cell_volume(a,b,c,al,be,ga):
    """Volume in Å³."""
    ca,cb,cg = math.cos(math.radians(al)), math.cos(math.radians(be)), math.cos(math.radians(ga))
    return a*b*c*math.sqrt(1-ca**2-cb**2-cg**2+2*ca*cb*cg)

def d_spacing(h,k,l, a,b,c,al_deg,be_deg,ga_deg):
    """d-spacing in Å for general triclinic cell (Buerger formula)."""
    al = math.radians(al_deg)
    be = math.radians(be_deg)
    ga = math.radians(ga_deg)
    ca,cb,cg = math.cos(al),math.cos(be),math.cos(ga)
    sa,sb,sg = math.sin(al),math.sin(be),math.sin(ga)
    V = cell_volume(a,b,c,al_deg,be_deg,ga_deg)

    s11 = b**2*c**2*sa**2
    s22 = a**2*c**2*sb**2
    s33 = a**2*b**2*sg**2
    s12 = a*b*c**2*(ca*cb-cg)
    s23 = a**2*b*c*(cb*cg-ca)
    s13 = a*b**2*c*(ca*cg-cb)

    inv_d2 = (s11*h**2 + s22*k**2 + s33*l**2
              + 2*s12*h*k + 2*s23*k*l + 2*s13*h*l) / V**2
    if inv_d2 <= 0:
        return None
    return 1.0 / math.sqrt(inv_d2)

def two_theta(d, wavelength=1.54056):
    """2θ in degrees for given d-spacing and wavelength (Å)."""
    arg = wavelength / (2*d)
    if abs(arg) > 1:
        return None
    return 2 * math.degrees(math.asin(arg))


# ════════════════════════════════════════════════════════════════════════════
# STRUCTURE FACTOR CALCULATION
# ════════════════════════════════════════════════════════════════════════════

def compute_structure_factor(h, k, l,
                              atoms,        # list of dicts: element,x,y,z,occ,U_iso
                              wavelength=1.54056):
    """
    Returns F_hkl (complex), |F|, |F|², phase (deg), and per-atom contributions.
    atoms: fractional coordinates after symmetry expansion.
    """
    d = d_spacing(h,k,l, *_cell_params_from_atoms_ctx)
    if d is None:
        return None

    s = 1.0 / (2*d)   # sinθ/λ

    F = 0+0j
    contributions = []
    for atom in atoms:
        elem = atom["element"]
        x,y,z   = atom["x"], atom["y"], atom["z"]
        occ     = atom.get("occ", 1.0)
        U_iso   = atom.get("U_iso", 0.02)   # Å²

        f  = atomic_form_factor(elem, s)
        DW = math.exp(-8*math.pi**2 * U_iso * s**2)  # Debye-Waller

        phase_rad = 2*math.pi*(h*x + k*y + l*z)
        contribution = occ * f * DW * cmath_exp(phase_rad)
        F += contribution

        contributions.append({
            "element": elem,
            "label":   atom.get("label", elem),
            "f(s)":    round(f, 4),
            "DW":      round(DW, 4),
            "occ":     occ,
            "|contrib|": round(abs(contribution), 4),
            "phase(°)": round(math.degrees(phase_rad % (2*math.pi)), 2),
        })

    absF   = abs(F)
    absF2  = absF**2
    phase  = math.degrees(math.atan2(F.imag, F.real))
    return {
        "h": h, "k": k, "l": l,
        "F_real": round(F.real, 4),
        "F_imag": round(F.imag, 4),
        "|F|":    round(absF, 4),
        "|F|²":   round(absF2, 4),
        "phase°": round(phase, 2),
        "d(Å)":   round(d, 5),
        "2θ(°)":  round(two_theta(d, wavelength) or 0, 4),
        "s(Å⁻¹)": round(s, 5),
        "contributions": contributions,
    }

def cmath_exp(phase_rad: float):
    return complex(math.cos(phase_rad), math.sin(phase_rad))


# global cell params injected before compute_structure_factor calls
_cell_params_from_atoms_ctx = (1, 1, 1, 90, 90, 90)


# ════════════════════════════════════════════════════════════════════════════
# STRUCTURE EXPANSION (asymm. unit → full cell via symmetry)
# ════════════════════════════════════════════════════════════════════════════

def expand_atoms(asym_atoms, sym_ops):
    tol = 0.005
    all_atoms = []
    seen = set()
    for atom in asym_atoms:
        for op in sym_ops:
            nx,ny,nz = apply_symop(atom["x"], atom["y"], atom["z"], op)
            key = (atom["element"], round(nx,3), round(ny,3), round(nz,3))
            if key in seen: continue
            seen.add(key)
            new = dict(atom)
            new["x"], new["y"], new["z"] = nx, ny, nz
            all_atoms.append(new)
    return all_atoms


# ════════════════════════════════════════════════════════════════════════════
# SYSTEMATIC ABSENCE CHECK
# ════════════════════════════════════════════════════════════════════════════

def is_systematic_absence(h, k, l, centering: str) -> bool:
    """Check common systematic absences for lattice centering."""
    if centering == "P":
        return False
    if centering == "I":
        return (h+k+l) % 2 != 0
    if centering == "F":
        parities = {h%2, k%2, l%2}
        return len(parities) > 1
    if centering == "C":
        return (h+k) % 2 != 0
    if centering == "A":
        return (k+l) % 2 != 0
    if centering == "B":
        return (h+l) % 2 != 0
    if centering == "R":
        return (-h+k+l) % 3 != 0
    return False


# ════════════════════════════════════════════════════════════════════════════
# POWDER DIFFRACTION PATTERN
# ════════════════════════════════════════════════════════════════════════════

def compute_powder_pattern(all_hkl_results, two_theta_range=(5,80),
                            fwhm=0.15, n_pts=3000):
    """Gaussian-broadened powder pattern."""
    tt_min, tt_max = two_theta_range
    tt_arr = np.linspace(tt_min, tt_max, n_pts)
    pattern = np.zeros(n_pts)

    for r in all_hkl_results:
        tt0 = r.get("2θ(°)")
        I   = r.get("|F|²", 0)
        if not tt0 or tt0 < tt_min or tt0 > tt_max:
            continue
        sigma = fwhm / (2*math.sqrt(2*math.log(2)))
        pattern += I * np.exp(-0.5*((tt_arr - tt0)/sigma)**2)

    return tt_arr, pattern


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR + FILE LOADING  (identical pattern to bravais_from_cif.py)
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔭 STRUCTURE FACTOR")
    st.markdown("---")

    cif_text = None
    cif_name = ""

    # 1. CLI
    if PRELOAD_PATH and PRELOAD_PATH.exists():
        cif_text = PRELOAD_PATH.read_text(encoding="utf-8", errors="replace")
        cif_name = PRELOAD_PATH.name

    # 2. Auto-search parent folders
    if cif_text is None:
        script_dir = Path(__file__).resolve().parent
        search_dirs = [script_dir.parent, script_dir.parent.parent, script_dir]
        found_cifs  = []
        for d in search_dirs:
            found_cifs += sorted(d.glob("*.cif"))
        found_cifs = list(dict.fromkeys(found_cifs))

        if found_cifs:
            cif_names    = [f.name for f in found_cifs]
            selected_idx = st.selectbox("CIF-Datei", range(len(cif_names)),
                                         format_func=lambda i: cif_names[i], index=0)
            chosen   = found_cifs[selected_idx]
            cif_text = chosen.read_text(encoding="utf-8", errors="replace")
            cif_name = chosen.name
            st.caption(f"📂 {chosen.parent}")

    # 3. Manual upload fallback
    if cif_text is None:
        st.markdown("**Keine CIF im Parent-Ordner**")
        up = st.file_uploader("CIF hochladen", type=["cif"])
        if up:
            cif_text = up.read().decode("utf-8", errors="replace")
            cif_name = up.name

    st.markdown("---")
    st.markdown("**Berechnung**")
    wavelength   = st.number_input("Wellenlänge λ (Å)", 0.5, 3.0, 1.54056, 0.00001,
                                    format="%.5f", help="Cu Kα = 1.54056 Å")
    hkl_max      = st.slider("max |h|,|k|,|l|", 1, 10, 5)
    min_I        = st.slider("Min. |F|² Schwelle", 0, 100, 0)
    show_extinct = st.checkbox("Ausgelöschte Reflexe zeigen", False)

    st.markdown("---")
    st.markdown("**Pulverdiffraktogramm**")
    tt_min = st.number_input("2θ min (°)", 1.0, 30.0,  5.0, 1.0)
    tt_max = st.number_input("2θ max (°)", 30.0, 180.0, 80.0, 1.0)
    fwhm   = st.slider("FWHM (°)", 0.05, 1.0, 0.15, 0.01)

    st.markdown("---")
    st.markdown("**HKL Einzelsuche**")
    h_in = st.number_input("h", -10, 10, 1, 1)
    k_in = st.number_input("k", -10, 10, 1, 1)
    l_in = st.number_input("l", -10, 10, 1, 1)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if cif_text is None:
    st.markdown("""
    <div style='text-align:center;padding:100px 40px;'>
      <div style='font-size:80px;margin-bottom:28px;filter:drop-shadow(0 0 30px #00e5c8);'>🔭</div>
      <h1 style='font-family:"Share Tech Mono",monospace;font-size:28px;
                 letter-spacing:0.15em;color:#c8d4f8;'>STRUCTURE FACTOR & HKL</h1>
      <p style='color:#334488;font-size:16px;margin-top:14px;'>
        Lege eine CIF-Datei in den Parent-Ordner oder lade sie hoch.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Parse CIF ─────────────────────────────────────────────────────────────────
scalars, loops = parse_cif_full(cif_text)

def g(key, default=None):
    return scalars.get(key.lower(), default)

a_val = parse_number(g("_cell_length_a")) or 1.0
b_val = parse_number(g("_cell_length_b")) or a_val
c_val = parse_number(g("_cell_length_c")) or a_val
alpha = parse_number(g("_cell_angle_alpha")) or 90.0
beta  = parse_number(g("_cell_angle_beta"))  or 90.0
gamma = parse_number(g("_cell_angle_gamma")) or 90.0
vol   = parse_number(g("_cell_volume")) or cell_volume(a_val,b_val,c_val,alpha,beta,gamma)

# inject cell params into module-level context
_cell_params_from_atoms_ctx = (a_val, b_val, c_val, alpha, beta, gamma)

crystal_name = (g("_chemical_name_common") or g("_pd_block_id") or
                cif_name.replace(".cif","")).strip().strip("'\"")
if crystal_name in ("?","."): crystal_name = cif_name.replace(".cif","")

formula = (g("_chemical_formula_sum") or g("_chemical_formula_iupac") or "–").strip().strip("'\"")
spacegroup = (g("_symmetry_space_group_name_h-m") or
              g("_space_group_name_h-m_alt") or "–").strip().strip("'\"")
cell_setting = (g("_symmetry_cell_setting") or "–").capitalize()

# centering from H-M symbol
centering = "P"
hm_clean = spacegroup.strip("'\"")
if hm_clean:
    first = hm_clean[0].upper()
    if first in ("P","I","F","C","A","B","R"):
        centering = first

# ── Symmetry operations ───────────────────────────────────────────────────────
sym_df = loop_as_df(loops, "_symmetry_equiv_pos_as_xyz")
if sym_df is None:
    sym_df = loop_as_df(loops, "_space_group_symop_operation_xyz")

sym_ops = ["x,y,z"]
if sym_df is not None:
    col = next((c for c in sym_df.columns if "xyz" in c), None)
    if col:
        sym_ops = sym_df[col].tolist()

# ── Cromer-Mann scattering factors from CIF (if present) ─────────────────────
cm_from_cif = {}
cm_df = loop_as_df(loops, "_atom_type_scat_Cromer_Mann_a1")
if cm_df is not None:
    sym_col = next((c for c in cm_df.columns if "symbol" in c), None)
    if sym_col:
        keys = ["a1","b1","a2","b2","a3","b3","a4","b4","c"]
        col_map = {}
        for k in keys:
            found = next((c for c in cm_df.columns if k in c.lower()), None)
            col_map[k] = found
        for _, row in cm_df.iterrows():
            elem = str(row[sym_col]).strip().upper()
            try:
                params = tuple(float(row[col_map[k]]) for k in keys if col_map[k])
                if len(params) == 9:
                    cm_from_cif[elem] = params
            except:
                pass

# Override global CM_PARAMS with CIF values
CM_PARAMS.update(cm_from_cif)

# ── Atom sites ────────────────────────────────────────────────────────────────
atom_df = loop_as_df(loops, "_atom_site_label")
asym_atoms = []
if atom_df is not None:
    def fc(df, *names):
        for n in names:
            m = next((c for c in df.columns if n.lower() in c.lower()), None)
            if m: return m
        return None
    lbl = fc(atom_df, "label")
    xc  = fc(atom_df, "fract_x")
    yc  = fc(atom_df, "fract_y")
    zc  = fc(atom_df, "fract_z")
    oc  = fc(atom_df, "occupancy")
    uc  = fc(atom_df, "u_iso", "U_iso")
    tc  = fc(atom_df, "type_symbol")
    for _, row in atom_df.iterrows():
        label = str(row[lbl]) if lbl else "X"
        fx = parse_number(row[xc]) if xc else 0.0
        fy = parse_number(row[yc]) if yc else 0.0
        fz = parse_number(row[zc]) if zc else 0.0
        occ = parse_number(row[oc]) if oc else 1.0
        Uiso = parse_number(row[uc]) if uc else 0.01
        if tc and str(row[tc]) not in ("?","."):
            elem = re.sub(r"[^A-Za-z]","",str(row[tc])).capitalize()
        else:
            elem = re.sub(r"[^A-Za-z]","",label).capitalize()
        asym_atoms.append({
            "label": label, "element": elem,
            "x": fx or 0.0, "y": fy or 0.0, "z": fz or 0.0,
            "occ": occ or 1.0, "U_iso": Uiso or 0.01,
        })

# expand to full unit cell
all_atoms = expand_atoms(asym_atoms, sym_ops)

# ── Compute all HKL ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_all_hkl(_atoms_key, atoms, hkl_max, wavelength,
                     a,b,c,al,be,ga, min_I, show_extinct, centering):
    global _cell_params_from_atoms_ctx
    _cell_params_from_atoms_ctx = (a,b,c,al,be,ga)
    results = []
    seen_d  = {}
    for h in range(-hkl_max, hkl_max+1):
        for k in range(-hkl_max, hkl_max+1):
            for l in range(-hkl_max, hkl_max+1):
                if h==0 and k==0 and l==0: continue
                # Friedel equivalence — only positive hemisphere
                if (h < 0 or (h==0 and k<0) or (h==0 and k==0 and l<0)):
                    continue
                extinct = is_systematic_absence(h,k,l,centering)
                if extinct and not show_extinct: continue

                r = compute_structure_factor(h,k,l,atoms,wavelength)
                if r is None: continue
                if r["|F|²"] < min_I and not extinct: continue
                tt = r["2θ(°)"]
                if tt <= 0 or tt >= 180: continue
                r["extinct"]   = extinct
                r["centering"] = centering
                results.append(r)
    results.sort(key=lambda x: x["d(Å)"], reverse=True)
    return results

atoms_key = str([(a["label"],a["x"],a["y"],a["z"]) for a in all_atoms])

with st.spinner("Berechne Strukturfaktoren …"):
    all_results = compute_all_hkl(
        atoms_key, all_atoms, hkl_max, wavelength,
        a_val,b_val,c_val,alpha,beta,gamma,
        min_I, show_extinct, centering
    )

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='page-header'>
  <span style='font-family:"Share Tech Mono",monospace;font-size:11px;
               color:#334488;letter-spacing:0.2em;'>STRUKTURFAKTOR & HKL</span>
  <h1 style='font-size:30px;font-weight:600;color:#e8f0ff;margin:6px 0 4px;
             font-family:"DM Sans",sans-serif;'>{crystal_name}</h1>
  <div>
    <span class='tag' style='background:#00e5c822;border:1px solid #00e5c855;color:#00e5c8;'>
      {formula}
    </span>
    <span class='tag' style='background:rgba(30,40,90,0.6);border:1px solid #2a3880;color:#6688cc;'>
      {spacegroup}
    </span>
    <span class='tag' style='background:rgba(30,40,90,0.4);border:1px solid #1e2555;color:#445588;'>
      {cell_setting}
    </span>
    <span class='tag' style='background:rgba(30,40,90,0.4);border:1px solid #1e2555;color:#445588;'>
      λ = {wavelength:.5f} Å
    </span>
    <span class='tag' style='background:rgba(30,40,90,0.4);border:1px solid #1e2555;color:#445588;'>
      {len(all_results)} Reflexe
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_single, tab_powder, tab_table, tab_reciprocal, tab_info = st.tabs([
    "🎯 HKL Einzelsuche",
    "📈 Pulverdiffraktogramm",
    "📋 HKL Tabelle",
    "⬡ Reziprokes Gitter",
    "ℹ️ Struktur & Formel",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1: HKL SINGLE SEARCH
# ────────────────────────────────────────────────────────────────────────────
with tab_single:
    h, k, l = int(h_in), int(k_in), int(l_in)

    _cell_params_from_atoms_ctx = (a_val,b_val,c_val,alpha,beta,gamma)
    r = compute_structure_factor(h, k, l, all_atoms, wavelength)
    extinct = is_systematic_absence(h, k, l, centering)

    st.markdown(f"### Reflex ({h} {k} {l})")

    if r is None:
        st.error(f"({h} {k} {l}) — kein gültiger d-Abstand (d ≤ 0).")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("|F(hkl)|",  f"{r['|F|']:.4f}")
        col2.metric("|F|²",      f"{r['|F|²']:.2f}")
        col3.metric("Phase φ",   f"{r['phase°']:.2f}°")
        col4.metric("d-Abstand", f"{r['d(Å)']:.5f} Å")
        col5.metric("2θ",        f"{r['2θ(°)']:.4f}°")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='hkl-result'>
              <div style='font-family:"Share Tech Mono",monospace;font-size:13px;color:#6688cc;margin-bottom:8px;'>
                KOMPLEXER STRUKTURFAKTOR
              </div>
              <div style='font-size:20px;color:#00e5c8;font-family:"Share Tech Mono",monospace;'>
                F = {r['F_real']:+.4f} {'+' if r['F_imag']>=0 else ''}{r['F_imag']:.4f}i
              </div>
              <div style='margin-top:8px;font-size:13px;color:#aabcee;'>
                s = sinθ/λ = {r['s(Å⁻¹)']:.5f} Å⁻¹
              </div>
              {'<div style="margin-top:8px;color:#ff6666;font-size:12px;">⚠ Systematisch ausgelöscht</div>' if extinct else ''}
            </div>
            """, unsafe_allow_html=True)

        with c2:
            # Argand diagram
            fig, ax = plt.subplots(figsize=(4,4), facecolor="#07080f")
            ax.set_facecolor("#07080f")
            ax.set_aspect("equal")

            # unit circle reference
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta)*abs(r["|F|"]), np.sin(theta)*abs(r["|F|"]),
                    color="#1e2555", lw=0.8, ls="--")

            # per-atom contributions
            cx, cy = 0.0, 0.0
            for contrib in r["contributions"]:
                dx = contrib["|contrib|"] * math.cos(math.radians(contrib["phase(°)"]))
                dy = contrib["|contrib|"] * math.sin(math.radians(contrib["phase(°)"]))
                ax.annotate("", xy=(cx+dx, cy+dy), xytext=(cx,cy),
                            arrowprops=dict(arrowstyle="->", color="#4466ff", lw=1.2))
                cx += dx; cy += dy

            # total F
            ax.annotate("", xy=(r["F_real"],r["F_imag"]), xytext=(0,0),
                        arrowprops=dict(arrowstyle="->", color="#00e5c8", lw=2.5))
            ax.plot(r["F_real"], r["F_imag"], "o", color="#00e5c8", ms=8)

            ax.axhline(0, color="#2a3880", lw=0.6)
            ax.axvline(0, color="#2a3880", lw=0.6)
            ax.set_xlabel("Re(F)", color="#445588", fontsize=10)
            ax.set_ylabel("Im(F)", color="#445588", fontsize=10)
            ax.tick_params(colors="#334466")
            ax.set_title("Argand-Diagramm", color="#6688cc", fontsize=11,
                         fontfamily="monospace")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2555")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # per-atom contributions table
        st.markdown("**Beiträge der Atome**")
        contrib_df = pd.DataFrame(r["contributions"])
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        # formula display
        st.markdown("""
        <div class='formula-box'>
          F(hkl) = Σ<sub>j</sub>  f<sub>j</sub>(s) · occ<sub>j</sub> · DW<sub>j</sub> · exp[2πi(hx<sub>j</sub> + ky<sub>j</sub> + lz<sub>j</sub>)]<br>
          s = sinθ/λ = 1/(2d) &nbsp;·&nbsp;
          DW = exp(−8π²·U<sub>iso</sub>·s²) &nbsp;·&nbsp;
          f(s) = Σ a<sub>i</sub>·exp(−b<sub>i</sub>·s²) + c
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2: POWDER DIFFRACTOGRAM
# ────────────────────────────────────────────────────────────────────────────
with tab_powder:
    visible = [r for r in all_results if not r.get("extinct", False)]

    tt_arr, pattern = compute_powder_pattern(
        visible, (tt_min, tt_max), fwhm=fwhm
    )

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#07080f")
    ax.set_facecolor("#07080f")

    # pattern
    ax.fill_between(tt_arr, pattern, color="#00e5c8", alpha=0.15)
    ax.plot(tt_arr, pattern, color="#00e5c8", lw=1.2)

    # peak markers
    max_I = max((r["|F|²"] for r in visible), default=1)
    for r in visible:
        tt0 = r["2θ(°)"]
        if tt_min <= tt0 <= tt_max:
            ht = r["|F|²"] / max_I * pattern.max() if pattern.max() > 0 else 0
            ax.plot([tt0, tt0], [0, ht * 0.85],
                    color="#4466ff", lw=0.8, alpha=0.5)

    ax.set_xlabel("2θ (°)", color="#445588", fontsize=11)
    ax.set_ylabel("|F|²  (normiert)", color="#445588", fontsize=11)
    ax.set_xlim(tt_min, tt_max)
    ax.tick_params(colors="#445588")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2555")
    ax.set_title(f"Simuliertes Pulverdiffraktogramm — {crystal_name}",
                 color="#c8d4f8", fontsize=12, fontfamily="monospace")
    ax.grid(axis="x", color="#1e2555", lw=0.5, ls=":")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # strongest peaks table
    st.markdown("**Stärkste Reflexe**")
    top = sorted(visible, key=lambda x: x["|F|²"], reverse=True)[:20]
    df_top = pd.DataFrame([{
        "hkl": f"({r['h']} {r['k']} {r['l']})",
        "2θ (°)": r["2θ(°)"],
        "d (Å)": r["d(Å)"],
        "|F|": r["|F|"],
        "|F|²": r["|F|²"],
        "Phase (°)": r["phase°"],
    } for r in top])
    st.dataframe(df_top, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3: FULL HKL TABLE
# ────────────────────────────────────────────────────────────────────────────
with tab_table:
    col_s1, col_s2 = st.columns([2,1])
    with col_s1:
        search = st.text_input("Suche (z.B. '1 1 0' oder Element)", "")
    with col_s2:
        sort_by = st.selectbox("Sortieren nach",
            ["|F|²", "d(Å)", "2θ(°)", "|F|", "h", "k", "l"], index=0)

    df_all = pd.DataFrame([{
        "hkl":     f"({r['h']} {r['k']} {r['l']})",
        "h": r["h"], "k": r["k"], "l": r["l"],
        "d (Å)":   r["d(Å)"],
        "2θ (°)":  r["2θ(°)"],
        "s (Å⁻¹)": r["s(Å⁻¹)"],
        "|F|":     r["|F|"],
        "|F|²":    r["|F|²"],
        "Phase (°)": r["phase°"],
        "Re(F)":   r["F_real"],
        "Im(F)":   r["F_imag"],
        "Ausgelöscht": r.get("extinct", False),
    } for r in all_results])

    if search.strip():
        mask = df_all.apply(lambda row: search.lower() in str(row).lower(), axis=1)
        df_all = df_all[mask]

    sort_col = sort_by if sort_by in df_all.columns else "|F|²"
    df_all = df_all.sort_values(sort_col, ascending=False)

    st.caption(f"{len(df_all)} Reflexe")
    st.dataframe(df_all.drop(columns=["h","k","l"]),
                 use_container_width=True, hide_index=True, height=500)

    csv = df_all.to_csv(index=False)
    st.download_button("⬇ HKL-Liste als CSV", csv, "hkl_list.csv", "text/csv")


# ────────────────────────────────────────────────────────────────────────────
# TAB 4: RECIPROCAL LATTICE (2D slice)
# ────────────────────────────────────────────────────────────────────────────
with tab_reciprocal:
    st.markdown("#### Reziprokes Gitter — 2D Schnitt")
    c1, c2 = st.columns([1,3])
    with c1:
        plane = st.selectbox("Schnittebene", ["hk0", "h0l", "0kl", "hk1", "hk2"], index=0)
        color_by = st.selectbox("Farbe", ["|F|²", "|F|", "Phase"], index=0)
        dot_scale = st.slider("Punktgröße", 10, 300, 80)

    # determine fixed index and plane
    plane_map = {
        "hk0": ("h","k", 0, 0), "h0l": ("h","l", 1, 0),
        "0kl": ("k","l", 0, 1), "hk1": ("h","k", 0, 1),
        "hk2": ("h","k", 0, 2),
    }
    ax1_lbl, ax2_lbl, fix_idx, fix_val = plane_map[plane]
    axis_labels = {"h":0,"k":1,"l":2}

    filtered_plane = []
    for r in all_results:
        hkl = [r["h"],r["k"],r["l"]]
        if hkl[fix_idx] == fix_val and not r.get("extinct", False):
            filtered_plane.append(r)

    if not filtered_plane:
        st.info(f"Keine Reflexe in der {plane}-Ebene gefunden.")
    else:
        xs  = [r[ax1_lbl] for r in filtered_plane]
        ys  = [r[ax2_lbl] for r in filtered_plane]
        val_key = "|F|²" if color_by == "|F|²" else ("|F|" if color_by == "|F|" else "phase°")
        vals = [r[val_key] for r in filtered_plane]

        fig, ax = plt.subplots(figsize=(8,7), facecolor="#07080f")
        ax.set_facecolor("#07080f")

        norm_vals = np.array(vals, dtype=float)
        if norm_vals.max() > norm_vals.min():
            norm_vals = (norm_vals - norm_vals.min()) / (norm_vals.max() - norm_vals.min())
        else:
            norm_vals = np.ones_like(norm_vals)

        sc = ax.scatter(xs, ys, c=vals, s=dot_scale * norm_vals + 10,
                        cmap="plasma", alpha=0.85,
                        edgecolors="white", linewidths=0.4)
        plt.colorbar(sc, ax=ax, label=color_by,
                     fraction=0.03, pad=0.04).ax.yaxis.label.set_color("#6688cc")

        # label strongest
        sorted_plane = sorted(filtered_plane, key=lambda r: r["|F|²"], reverse=True)
        for r in sorted_plane[:12]:
            ax.annotate(f"({r['h']}{r['k']}{r['l']})",
                        (r[ax1_lbl], r[ax2_lbl]),
                        fontsize=7, color="#c8d4f8", alpha=0.8,
                        xytext=(4,4), textcoords="offset points",
                        fontfamily="monospace")

        ax.axhline(0, color="#2a3880", lw=0.6)
        ax.axvline(0, color="#2a3880", lw=0.6)
        ax.set_xlabel(ax1_lbl, color="#6688cc", fontsize=12, fontfamily="monospace")
        ax.set_ylabel(ax2_lbl, color="#6688cc", fontsize=12, fontfamily="monospace")
        ax.tick_params(colors="#445588")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2555")
        ax.set_title(f"Reziprokes Gitter — {plane}  |  {color_by}",
                     color="#c8d4f8", fontsize=11, fontfamily="monospace")
        ax.grid(color="#1a2050", lw=0.4, ls=":")
        plt.tight_layout()

        with c2:
            st.pyplot(fig, use_container_width=True)
        plt.close()


# ────────────────────────────────────────────────────────────────────────────
# TAB 5: STRUCTURE INFO
# ────────────────────────────────────────────────────────────────────────────
with tab_info:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Gitterparameter")
        m1,m2,m3 = st.columns(3)
        m1.metric("a (Å)", f"{a_val:.5f}"); m1.metric("α (°)", f"{alpha:.3f}")
        m2.metric("b (Å)", f"{b_val:.5f}"); m2.metric("β (°)", f"{beta:.3f}")
        m3.metric("c (Å)", f"{c_val:.5f}"); m3.metric("γ (°)", f"{gamma:.3f}")
        st.metric("V (Å³)", f"{vol:.3f}")

    with c2:
        st.markdown("#### Asymmetrische Einheit")
        df_asym = pd.DataFrame(asym_atoms)[["label","element","x","y","z","occ","U_iso"]]
        df_asym = df_asym.round(5)
        st.dataframe(df_asym, use_container_width=True, hide_index=True)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Vollständige Einheitszelle")
        df_full = pd.DataFrame([{"label":a["label"],"element":a["element"],
                                  "x":round(a["x"],5),"y":round(a["y"],5),"z":round(a["z"],5)}
                                 for a in all_atoms])
        st.dataframe(df_full, use_container_width=True, hide_index=True, height=300)
        st.caption(f"{len(asym_atoms)} asymm. Atome → {len(all_atoms)} Atome (nach Symmetrie) · {len(sym_ops)} Symmetrieoperationen")

    with c4:
        st.markdown("#### Cromer-Mann Parameter (verwendete Werte)")
        elems = sorted(set(a["element"] for a in all_atoms))
        cm_rows = []
        for e in elems:
            p = CM_PARAMS.get(e.upper())
            if p:
                src = "CIF" if e.upper() in cm_from_cif else "Standard"
                cm_rows.append({"Element": e, "a1":p[0],"b1":p[1],"a2":p[2],"b2":p[3],
                                 "a3":p[4],"b3":p[5],"a4":p[6],"b4":p[7],"c":p[8],"Quelle":src})
        if cm_rows:
            st.dataframe(pd.DataFrame(cm_rows).round(4),
                         use_container_width=True, hide_index=True)