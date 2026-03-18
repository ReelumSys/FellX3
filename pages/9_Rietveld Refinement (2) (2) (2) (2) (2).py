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
import re
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Diffraction Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.3rem; }
.block-container { padding-top: 1rem; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
h1  { font-size: 1.8rem !important; }
h2  { font-size: 1.2rem !important; }
h3  { font-size: 1.05rem !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CIF PARSER  (zero external dependencies)
# ═══════════════════════════════════════════════════════════════════════════

def _cif_val(raw):
    """Strip ESD parentheses and quotes; coerce to float or str."""
    if raw in (".", "?", "", None):
        return None
    raw = re.sub(r"\(\d+\)", "", str(raw))   # remove (esd)
    raw = raw.strip("'\"")
    try:
        return float(raw)
    except ValueError:
        return raw


def _tokenise(text):
    """Yield CIF tokens, handling semicolon text blocks and inline comments."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # ── semicolon text block ──────────────────────────────────────────
        if line.startswith(";"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                block.append(lines[i])
                i += 1
            yield "\n".join(block)
            i += 1
            continue
        # strip inline comment
        line = re.sub(r"(?<!['\"])#.*", "", line)
        # tokenise respecting single/double quotes
        for tok in re.findall(r"'[^']*'|\"[^\"]*\"|[^\s]+", line):
            yield tok
        i += 1


def parse_cif(text):
    """
    Parse a CIF text and return:
      cell    : {a, b, c, alpha, beta, gamma}
      sg      : H-M space-group string
      sg_no   : space-group number (int or None)
      system  : crystal system string
      atoms   : list of {element, label, x, y, z, occ, Biso}
      formula : chemical formula string or None
    """
    result = dict(cell={}, sg="P1", sg_no=None,
                  system="Triclinic", atoms=[], formula=None)
    tokens    = list(_tokenise(text))
    n         = len(tokens)

    i = 0
    while i < n:
        tok = tokens[i]

        # skip data_ block headers
        if tok.lower().startswith("data_"):
            i += 1; continue

        # ── loop_ ─────────────────────────────────────────────────────────
        if tok.lower() == "loop_":
            i += 1
            loop_keys = []
            while i < n and tokens[i].startswith("_"):
                loop_keys.append(tokens[i].lower())
                i += 1
            lk_set = set(loop_keys)
            is_atom_loop = (
                "_atom_site_fract_x" in lk_set or
                "_atom_site_x"       in lk_set
            )
            # consume rows until we hit a keyword or new block
            while i < n:
                t = tokens[i]
                if t.startswith("_") or t.lower() in ("loop_",) or t.lower().startswith("data_"):
                    break
                row = {}
                for key in loop_keys:
                    if i >= n: break
                    row[key] = _cif_val(tokens[i])
                    i += 1
                if is_atom_loop:
                    # --- element symbol ---
                    el = (row.get("_atom_site_type_symbol") or
                          row.get("_atom_site_element_symbol") or
                          row.get("_atom_site_label") or "X")
                    if isinstance(el, str):
                        # keep only letters, capitalise first
                        el = re.sub(r"[^A-Za-z]", "", el)
                        el = el[0].upper() + el[1:].lower() if el else "X"
                    # --- label ---
                    lbl = row.get("_atom_site_label", el)
                    if not isinstance(lbl, str): lbl = str(lbl)
                    # --- coordinates ---
                    x = row.get("_atom_site_fract_x") or row.get("_atom_site_x") or 0.0
                    y = row.get("_atom_site_fract_y") or row.get("_atom_site_y") or 0.0
                    z = row.get("_atom_site_fract_z") or row.get("_atom_site_z") or 0.0
                    # --- occupancy ---
                    occ = row.get("_atom_site_occupancy", 1.0)
                    if occ is None: occ = 1.0
                    # --- Biso / Uiso ---
                    Biso = row.get("_atom_site_b_iso_or_equiv") or \
                           row.get("_atom_site_b_equiv_geom_mean")
                    if Biso is None:
                        Uiso = row.get("_atom_site_u_iso_or_equiv") or \
                               row.get("_atom_site_u_equiv_geom_mean")
                        Biso = 8 * np.pi**2 * float(Uiso) if Uiso else 0.5
                    try:
                        result["atoms"].append(dict(
                            element=str(el), label=str(lbl),
                            x=float(x),  y=float(y),  z=float(z),
                            occ=float(occ), Biso=float(Biso)))
                    except (TypeError, ValueError):
                        pass
            continue

        # ── scalar key-value pairs ─────────────────────────────────────────
        if tok.startswith("_"):
            key = tok.lower()
            i  += 1
            val = _cif_val(tokens[i]) if i < n else None
            i  += 1

            # Cell parameters
            cell_map = {
                "_cell_length_a":    "a",
                "_cell_length_b":    "b",
                "_cell_length_c":    "c",
                "_cell_angle_alpha": "alpha",
                "_cell_angle_beta":  "beta",
                "_cell_angle_gamma": "gamma",
            }
            if key in cell_map and val is not None:
                result["cell"][cell_map[key]] = float(val)

            # Space group
            if key in ("_symmetry_space_group_name_h-m",
                       "_space_group_name_h-m_alt",
                       "_symmetry_space_group_name_h-m_alt",) and isinstance(val, str):
                result["sg"] = val.strip()
            if key in ("_symmetry_int_tables_number",
                       "_space_group_it_number") and val:
                try: result["sg_no"] = int(float(val))
                except (TypeError, ValueError): pass

            # Formula
            if key == "_chemical_formula_sum" and isinstance(val, str):
                result["formula"] = val.strip()
            continue

        i += 1

    # ── infer crystal system ───────────────────────────────────────────────
    n_sg = result["sg_no"]
    if n_sg:
        if   n_sg <= 2:   result["system"] = "Triclinic"
        elif n_sg <= 15:  result["system"] = "Monoclinic"
        elif n_sg <= 74:  result["system"] = "Orthorhombic"
        elif n_sg <= 142: result["system"] = "Tetragonal"
        elif n_sg <= 194: result["system"] = "Hexagonal"
        else:             result["system"] = "Cubic"
    else:
        c  = result["cell"]
        al = c.get("alpha", 90); be = c.get("beta", 90); ga = c.get("gamma", 90)
        av = c.get("a", 1);      bv = c.get("b", 1);     cv = c.get("c", 1)
        if abs(ga - 120) < 1 and abs(al - 90) < 1 and abs(be - 90) < 1:
            result["system"] = "Hexagonal"
        elif all(abs(x - 90) < 1 for x in (al, be, ga)):
            if abs(av-bv) < 0.02 and abs(av-cv) < 0.02: result["system"] = "Cubic"
            elif abs(av-bv) < 0.02:                       result["system"] = "Tetragonal"
            else:                                          result["system"] = "Orthorhombic"
        elif abs(al-90) < 1 and abs(ga-90) < 1:           result["system"] = "Monoclinic"
        else:                                              result["system"] = "Triclinic"

    return result


# ═══════════════════════════════════════════════════════════════════════════
# CRYSTALLOGRAPHY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def d_spacing(h, k, l, a, b, c, alpha_deg, beta_deg, gamma_deg, system):
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
        V = a*b*c*np.sqrt(max(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg, 1e-20))
        if V < 1e-10: return None
        inv = (b**2*c**2*sa**2*h**2 + a**2*c**2*sb**2*k**2 + a**2*b**2*sg**2*l**2
               + 2*a*b*c**2*(ca*cb-cg)*h*k
               + 2*a**2*b*c*(cb*cg-ca)*k*l
               + 2*a*b**2*c*(ca*cg-cb)*h*l) / V**2
    return None if inv <= 1e-12 else 1.0/np.sqrt(inv)


def is_absent(h, k, l, sg):
    h, k, l = int(h), int(k), int(l)
    su = sg.upper().replace(" ", "")
    if su.startswith("I") and (h+k+l) % 2 != 0:        return True
    if su.startswith("F") and len({h%2,k%2,l%2}) > 1:  return True
    if su.startswith("C") and (h+k) % 2 != 0:           return True
    if su.startswith("A") and (k+l) % 2 != 0:           return True
    if su.startswith("B") and (h+l) % 2 != 0:           return True
    if "FD" in su or ("F" in su and "D" in su):
        if h == 0 and k == 0 and l % 4 != 0:            return True
    return False


def gen_reflections(a, b, c, alpha, beta, gamma, system, sg, wl, tt_min, tt_max):
    d_min = wl / (2*np.sin(np.radians(tt_max/2)))
    d_max = wl / (2*np.sin(np.radians(max(tt_min, 0.5)/2)))
    mh = int(2*a/d_min)+2
    mk = int(2*b/d_min)+2
    ml = int(2*c/d_min)+2
    seen = {}
    for h in range(-mh, mh+1):
        for k in range(-mk, mk+1):
            for l in range(-ml, ml+1):
                if h == k == l == 0: continue
                if is_absent(h, k, l, sg): continue
                d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma, system)
                if d is None or not (d_min <= d <= d_max): continue
                tt  = 2*np.degrees(np.arcsin(np.clip(wl/(2*d), -1, 1)))
                key = round(tt, 4)
                if key not in seen:
                    seen[key] = [h, k, l, d, tt, 1000.0]
    return sorted(seen.values(), key=lambda r: r[4])


# ── Profile functions ──────────────────────────────────────────────────────

def pseudo_voigt(x, x0, fwhm, eta):
    eta  = np.clip(eta, 0, 1); fwhm = max(fwhm, 1e-6)
    sig  = fwhm / (2*np.sqrt(2*np.log(2)))
    G    = np.exp(-0.5*((x-x0)/sig)**2)
    L    = 1.0 / (1 + ((x-x0)/(fwhm/2))**2)
    return eta*L + (1-eta)*G

def caglioti_fwhm(tt, U, V, W):
    t = np.tan(np.radians(tt/2))
    return max(np.sqrt(max(U*t**2 + V*t + W, 1e-8)), 0.005)

def chebyshev_bg(tt, coeffs):
    x = 2*(tt - tt.min())/(tt.max()-tt.min()) - 1
    T = [np.ones_like(x), x, 2*x**2-1, 4*x**3-3*x,
         8*x**4-8*x**2+1, 16*x**5-20*x**3+5*x,
         32*x**6-48*x**4+18*x**2-1, 64*x**7-112*x**5+56*x**3-7*x]
    out = np.zeros_like(x)
    for i, c in enumerate(coeffs): out += c * T[i]
    return out

def lp_factor(tt):
    th   = np.radians(tt/2)
    cos2 = np.cos(np.radians(tt))**2
    return (1+cos2) / (np.sin(th)**2 * np.cos(th) + 1e-12)

def multiplicity(h, k, l, system):
    h, k, l = abs(int(h)), abs(int(k)), abs(int(l))
    zeros = sum(x == 0 for x in (h, k, l))
    if system == "Cubic":
        return {1:8, 2:24}.get(len({h,k,l}), 48)
    elif system == "Tetragonal":
        if h == 0 and k == 0: return 2
        return (4 if h==k else 8) * (1 if l==0 else 2)
    elif system == "Hexagonal":
        return 2 if (h==0 and k==0) else (12 if l!=0 else 6)
    elif system == "Orthorhombic":
        return 2**(3-zeros)*2
    return max(2**(3-zeros), 1)


# ── Cromer-Mann scattering factors ─────────────────────────────────────────

CM = {
    "H":  ([0.4899,0.2620,0.1968,0.0499],[20.659,7.740,49.552,2.202],0.001),
    "C":  ([2.310,1.020,1.589,0.865],[20.844,10.208,0.569,51.651],0.216),
    "N":  ([12.213,3.132,2.013,1.166],[0.006,9.893,28.997,0.583],-11.529),
    "O":  ([3.049,2.287,1.546,0.867],[13.277,5.701,0.324,32.909],0.251),
    "F":  ([3.539,2.641,1.517,1.024],[10.283,4.294,0.261,26.147],0.278),
    "Na": ([4.763,3.174,1.267,1.113],[3.285,8.842,0.314,129.424],0.676),
    "Mg": ([5.420,2.174,1.227,2.307],[2.828,79.261,0.381,7.194],0.858),
    "Al": ([6.420,1.900,1.594,1.965],[3.039,0.743,31.547,85.089],1.115),
    "Si": ([6.292,3.035,1.989,1.541],[2.439,32.334,0.679,81.694],1.141),
    "P":  ([6.435,4.179,1.780,1.491],[1.907,27.157,0.526,68.164],1.115),
    "S":  ([6.905,5.203,1.438,1.586],[1.468,22.215,0.254,56.172],0.866),
    "Cl": ([11.460,7.196,6.256,1.645],[0.010,1.166,18.519,47.778],-9.557),
    "K":  ([8.219,7.439,1.052,0.866],[12.795,0.775,213.187,41.684],1.423),
    "Ca": ([8.627,7.387,1.590,1.021],[10.442,0.660,85.748,178.437],1.375),
    "Ti": ([9.760,7.359,1.699,1.902],[7.851,0.500,35.634,116.105],1.281),
    "V":  ([10.297,7.351,2.070,2.045],[6.865,0.438,26.894,102.478],1.220),
    "Cr": ([10.641,7.354,3.324,1.492],[6.104,0.392,20.263,98.740],1.183),
    "Mn": ([11.282,7.357,3.019,2.244],[5.341,0.343,17.867,83.754],1.089),
    "Fe": ([11.770,7.357,3.522,2.305],[4.761,0.307,15.354,76.881],1.037),
    "Co": ([12.284,7.341,4.003,2.349],[4.279,0.278,13.536,71.169],1.012),
    "Ni": ([12.838,7.290,4.444,2.380],[3.878,0.257,12.176,66.342],1.034),
    "Cu": ([13.338,7.168,5.616,1.674],[3.583,0.247,11.397,64.813],1.191),
    "Zn": ([14.074,7.032,5.165,2.410],[3.266,0.233,10.316,58.710],1.304),
    "Ga": ([15.235,6.701,4.359,2.962],[3.067,0.241,10.781,61.413],1.719),
    "Ge": ([16.082,6.375,3.707,3.683],[2.851,0.252,11.447,54.763],2.131),
    "Zr": ([17.876,10.948,5.418,3.657],[1.276,11.916,0.118,87.663],2.069),
    "Ba": ([20.336,19.297,10.888,5.480],[3.216,0.275,20.207,109.460],2.775),
    "La": ([20.578,19.599,11.373,3.287],[2.948,0.244,18.773,133.124],2.147),
    "Ce": ([21.167,19.770,11.851,3.330],[2.812,0.226,17.608,127.113],1.862),
    "Pr": ([22.044,19.670,12.386,2.824],[2.774,0.222,16.767,143.644],2.058),
    "Nd": ([22.685,19.685,12.774,2.851],[2.662,0.210,15.885,137.903],1.985),
    "Sm": ([24.004,19.426,13.440,2.896],[2.473,0.196,14.400,128.007],2.209),
    "Pb": ([31.062,13.064,18.442,5.970],[0.690,2.358,8.618,47.257],13.412),
    "Bi": ([33.369,12.951,16.588,6.469],[0.704,2.923,8.794,48.009],13.579),
}

def f_atom(element, s):
    # Try two-character then one-character lookup (capitalised)
    key = (element[0].upper() + element[1:].lower()) if len(element) > 1 else element.upper()
    if key not in CM:
        key = element[0].upper()
    if key not in CM:
        return 1.0
    a4, b4, c = CM[key]
    s2 = s*s
    return c + sum(ai*np.exp(-bi*s2) for ai, bi in zip(a4, b4))

def structure_factor_sq(h, k, l, atoms, wl, tt):
    s = np.sin(np.radians(tt/2)) / wl
    Fr = Fi = 0.0
    for at in atoms:
        f   = f_atom(at["element"], s)
        DW  = np.exp(-at["Biso"]*s*s)
        phi = 2*np.pi*(h*at["x"] + k*at["y"] + l*at["z"])
        Fr += at["occ"]*f*DW*np.cos(phi)
        Fi += at["occ"]*f*DW*np.sin(phi)
    return Fr*Fr + Fi*Fi


# ── Pattern calculation ────────────────────────────────────────────────────

def calc_pattern(tt_arr, refs, pr, bg_c, atoms=None, mode="lebail"):
    bg   = chebyshev_bg(tt_arr, bg_c)
    patt = np.zeros_like(tt_arr, dtype=float)
    U, V, W = pr["U"], pr["V"], pr["W"]
    eta0    = pr.get("eta0", 0.3)
    scale   = pr["scale"]
    wl      = pr["wl"]
    system  = pr.get("system", "Cubic")
    for ref in refs:
        h, k, l, d, tt_pk = ref[0], ref[1], ref[2], ref[3], ref[4]
        if not (tt_arr[0] <= tt_pk <= tt_arr[-1]): continue
        fwhm = caglioti_fwhm(tt_pk, U, V, W)
        lp   = lp_factor(tt_pk)
        mult = multiplicity(h, k, l, system)
        F2   = structure_factor_sq(h,k,l,atoms,wl,tt_pk) if (mode=="rietveld" and atoms) else ref[5]
        patt += scale * mult * lp * F2 * pseudo_voigt(tt_arr, tt_pk, fwhm, eta0)
    return patt + bg, bg

def r_factors(obs, calc, n_params=0):
    w    = 1.0 / np.maximum(obs, 1)
    Rwp  = 100*np.sqrt(np.sum(w*(obs-calc)**2) / np.sum(w*obs**2))
    Rp   = 100*np.sum(np.abs(obs-calc)) / np.sum(obs)
    chi2 = np.sum(w*(obs-calc)**2) / max(len(obs)-n_params, 1)
    return Rwp, Rp, chi2, np.sqrt(chi2)


# ═══════════════════════════════════════════════════════════════════════════
# PRESETS
# ═══════════════════════════════════════════════════════════════════════════

PRESETS = {
    "Si  (cubic Fd-3m, a=5.431 Å)":
        dict(system="Cubic", sg="Fd-3m", a=5.4309, b=5.4309, c=5.4309, al=90,be=90,ga=90,
             atoms=[{"element":"Si","label":"Si1","x":0.0, "y":0.0, "z":0.0, "occ":1.0,"Biso":0.46},
                    {"element":"Si","label":"Si2","x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.46}]),
    "LaB6 (cubic Pm-3m, a=4.157 Å)":
        dict(system="Cubic", sg="Pm-3m", a=4.1569, b=4.1569, c=4.1569, al=90,be=90,ga=90,
             atoms=[{"element":"La","label":"La1","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.20},
                    {"element":"B", "label":"B1", "x":0.5,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.50},
                    {"element":"B", "label":"B2", "x":0.0,"y":0.5,"z":0.0,"occ":1.0,"Biso":0.50},
                    {"element":"B", "label":"B3", "x":0.0,"y":0.0,"z":0.5,"occ":1.0,"Biso":0.50}]),
    "CeO2 (cubic Fm-3m, a=5.411 Å)":
        dict(system="Cubic", sg="Fm-3m", a=5.4124, b=5.4124, c=5.4124, al=90,be=90,ga=90,
             atoms=[{"element":"Ce","label":"Ce1","x":0.0, "y":0.0, "z":0.0, "occ":1.0,"Biso":0.40},
                    {"element":"O", "label":"O1", "x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.60}]),
    "Custom":
        dict(system="Cubic", sg="P1", a=4.0,b=4.0,c=4.0,al=90,be=90,ga=90,atoms=[]),
}


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def _init():
    defs = dict(
        refs=None, obs_tt=None, obs_I=None,
        lb_result=None, rv_result=None,
        atoms=[{"element":"Si","label":"Si1","x":0.0, "y":0.0, "z":0.0, "occ":1.0,"Biso":0.5},
               {"element":"Si","label":"Si2","x":0.25,"y":0.25,"z":0.25,"occ":1.0,"Biso":0.5}],
        cif_loaded=False,
        cif_a=None, cif_b=None, cif_c=None,
        cif_al=None, cif_be=None, cif_ga=None,
        cif_sg=None, cif_system=None, cif_formula=None,
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


def bragg_ticks(refs, lo, hi, y0, dy):
    xs, ys = [], []
    for r in refs[:400]:
        tp = r[4]
        if lo <= tp <= hi:
            xs += [tp, tp, None]; ys += [y0, y0+dy, None]
    return xs, ys


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Experiment Setup")
    wl = st.number_input("Wavelength λ (Å)", 0.5, 3.0, 1.54056, 0.00001, "%.5f",
                         help="CuKα1=1.54056  MoKα1=0.70930  AgKα1=0.55941")

    st.markdown("---")
    st.markdown("### 📂 Diffraction Data")
    data_mode = st.radio("Data source", ["Synthetic (preset)", "Upload XY file"],
                         label_visibility="collapsed")
    if data_mode == "Upload XY file":
        uploaded_xy = st.file_uploader("XY file (2θ  I per line)",
                                       type=["xy","dat","txt","csv"])
        if uploaded_xy:
            lines = uploaded_xy.read().decode(errors="replace").splitlines()
            pts   = []
            for ln in lines:
                ln = ln.strip()
                if not ln or ln.startswith("#"): continue
                parts = ln.split()
                if len(parts) >= 2:
                    try: pts.append((float(parts[0]), float(parts[1])))
                    except ValueError: pass
            if pts:
                arr = np.array(pts)
                st.session_state.obs_tt = arr[:,0]
                st.session_state.obs_I  = arr[:,1]
                st.success(f"Loaded {len(pts):,} data points")
    else:
        preset_key = st.selectbox("Preset material", list(PRESETS.keys()))
        preset     = PRESETS[preset_key]

    st.markdown("---")
    st.markdown("### 🔷 Unit Cell")
    if st.session_state.cif_loaded:
        st.caption("📌 Values auto-filled from CIF — edit below or clear in Rietveld tab.")

    cif = st.session_state.cif_loaded
    def _cd(field, pf, fb):
        v = st.session_state.get(f"cif_{field}")
        if cif and v is not None: return float(v)
        return preset.get(pf, fb) if data_mode=="Synthetic (preset)" else fb

    sys_opts    = ["Cubic","Tetragonal","Orthorhombic","Hexagonal","Monoclinic","Triclinic"]
    sys_default = (st.session_state.cif_system if cif and st.session_state.cif_system
                   else (preset.get("system","Cubic") if data_mode=="Synthetic (preset)" else "Cubic"))
    system      = st.selectbox("Crystal system", sys_opts, index=sys_opts.index(sys_default))

    c1, c2 = st.columns(2)
    a = c1.number_input("a (Å)", 0.5,30.0, _cd("a","a",5.43), 0.0001, "%.4f")
    if system == "Cubic":
        b = a; c_par = a; c2.markdown("**b = c = a**")
    elif system in ("Tetragonal","Hexagonal"):
        b = a
        c_par = c2.number_input("c (Å)",0.5,30.0,_cd("c","c",5.43),0.0001,"%.4f")
        if system=="Tetragonal": st.markdown("b = a")
    else:
        b     = c2.number_input("b (Å)",0.5,30.0,_cd("b","b",5.43),0.0001,"%.4f")
        c_par = c1.number_input("c (Å)",0.5,30.0,_cd("c","c",5.43),0.0001,"%.4f")

    if system in ("Monoclinic","Triclinic"):
        c3,c4,c5 = st.columns(3)
        al = c3.number_input("α°",1.0,179.0,_cd("al","al",90.0),0.01,"%.2f")
        be = c4.number_input("β°",1.0,179.0,_cd("be","be",90.0),0.01,"%.2f")
        ga = c5.number_input("γ°",1.0,179.0,_cd("ga","ga",90.0),0.01,"%.2f")
    elif system == "Hexagonal":
        al,be,ga = 90.0,90.0,120.0
    else:
        al=be=ga=90.0

    sg_default = (st.session_state.cif_sg if cif and st.session_state.cif_sg
                  else (preset.get("sg","P1") if data_mode=="Synthetic (preset)" else "P1"))
    sg = st.text_input("Space group", sg_default)

    st.markdown("---")
    st.markdown("### 📐 2θ Range & Grid")
    c1,c2   = st.columns(2)
    tt_min  = c1.number_input("Min 2θ (°)", 1.0,170.0, 10.0, 0.5)
    tt_max  = c2.number_input("Max 2θ (°)",10.0,170.0,100.0, 0.5)
    n_pts   = st.slider("Grid points", 500, 5000, 2000, 100)

    st.markdown("---")
    st.markdown("### 📊 Profile Parameters")
    U     = st.number_input("U (Caglioti)",  0.0, 5.0,  0.010, 0.001,"%.4f")
    V     = st.number_input("V (Caglioti)", -1.0, 0.0, -0.001, 0.001,"%.4f")
    W     = st.number_input("W (Caglioti)",  1e-4,5.0,  0.005, 0.001,"%.4f")
    eta0  = st.number_input("η₀ (Lorentzian frac.)",0.0,1.0,0.3,0.01)
    scale = st.number_input("Scale factor",0.001,1e9,1000.0,100.0)

    st.markdown("---")
    st.markdown("### 🌐 Background")
    n_bg = st.slider("Chebyshev terms", 2, 8, 5)

    st.markdown("---")
    if st.button("🔄 Generate Reflections & Data", type="primary", use_container_width=True):
        refs_new = gen_reflections(a,b,c_par,al,be,ga,system,sg,wl,tt_min,tt_max)
        st.session_state.refs      = refs_new
        st.session_state.lb_result = None
        st.session_state.rv_result = None

        pr0 = dict(U=U,V=V,W=W,eta0=eta0,scale=scale,wl=wl,system=system)
        bg0 = np.zeros(n_bg); bg0[0]=80.0; bg0[1]=-20.0
        tt_arr = np.linspace(tt_min,tt_max,n_pts)

        if data_mode == "Synthetic (preset)":
            atoms_pr = PRESETS[preset_key].get("atoms",[])
            if atoms_pr:
                pat,_ = calc_pattern(tt_arr,refs_new,pr0,bg0,atoms=atoms_pr,mode="rietveld")
            else:
                pat,_ = calc_pattern(tt_arr,refs_new,pr0,bg0,mode="lebail")
            noise = np.random.default_rng(42).normal(0,np.sqrt(np.abs(pat)+1)*0.04)
            st.session_state.obs_tt = tt_arr
            st.session_state.obs_I  = np.maximum(pat+noise,0)
        st.success(f"✅ {len(refs_new)} reflections generated")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════

st.title("🔬 Diffraction Analyser — Full Profile Refinement")

refs      = st.session_state.refs
obs_tt    = st.session_state.obs_tt
obs_I     = st.session_state.obs_I
have_data = obs_tt is not None and obs_I is not None and refs is not None

tab_data, tab_lb, tab_rv, tab_results = st.tabs(
    ["📈 Pattern","⚗️ Le Bail Fit","🔬 Rietveld Fit","📋 Results"])


# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — PATTERN
# ──────────────────────────────────────────────────────────────────────────
with tab_data:
    if not have_data:
        st.info("👈 Configure the sidebar and click **Generate Reflections & Data** to start.")
    else:
        ymax = obs_I.max()
        tx,ty = bragg_ticks(refs, obs_tt[0],obs_tt[-1], ymax, -ymax*0.04)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=obs_tt,y=obs_I,mode="lines",name="Observed",
                                  line=dict(color="#1f77b4",width=1.2)))
        if tx:
            fig.add_trace(go.Scatter(x=tx,y=ty,mode="lines",name="Bragg positions",
                                      line=dict(color="red",width=1),hoverinfo="skip"))
        fig.update_layout(xaxis_title="2θ (°)",yaxis_title="Intensity (counts)",
                          template="plotly_white",height=480,
                          title=f"Observed Pattern — {len(refs)} reflections | λ={wl:.5f} Å",
                          legend=dict(x=0.75,y=0.95))
        st.plotly_chart(fig,use_container_width=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Data points",f"{len(obs_tt):,}")
        c2.metric("Reflections",f"{len(refs)}")
        c3.metric("2θ range",f"{obs_tt[0]:.1f}°–{obs_tt[-1]:.1f}°")
        with st.expander("📄 Reflection list (first 60)"):
            st.dataframe(pd.DataFrame([
                {"h":int(r[0]),"k":int(r[1]),"l":int(r[2]),
                 "d (Å)":f"{r[3]:.4f}","2θ (°)":f"{r[4]:.3f}"}
                for r in refs[:60]]),use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — LE BAIL
# ──────────────────────────────────────────────────────────────────────────
with tab_lb:
    st.markdown("""
    **Le Bail method** (Armel Le Bail, 1988) — extracts integrated intensities |F_hkl|²
    iteratively without an atomic structure model.
    """)
    if not have_data:
        st.warning("Generate data first (sidebar).")
    else:
        ctrl_col, plot_col = st.columns([1,2.5])
        with ctrl_col:
            st.markdown("#### Refinement switches")
            lb_scale   = st.checkbox("Scale",           True,  key="lb_sc")
            lb_profile = st.checkbox("Profile U,V,W",   True,  key="lb_prf")
            lb_eta     = st.checkbox("Mixing η₀",       True,  key="lb_eta")
            lb_bg      = st.checkbox("Background",      True,  key="lb_bg")
            lb_iters   = st.number_input("Le Bail cycles",20,500,100,10)
            run_lb     = st.button("▶ Run Le Bail",type="primary",use_container_width=True)

        if run_lb:
            tt_arr  = obs_tt; obs = obs_I
            refs_lb = [list(r) for r in refs]
            for r in refs_lb: r[5]=float(scale)
            pr = dict(U=U,V=V,W=W,eta0=eta0,scale=scale,wl=wl,system=system)
            bg = np.zeros(n_bg); bg[0]=float(np.percentile(obs,3))
            bar = st.progress(0, text="Le Bail cycles…")

            x0k = []   # pre-declare so it exists after the loop
            for cycle in range(int(lb_iters)):
                calc_v, bgv = calc_pattern(tt_arr,refs_lb,pr,bg,mode="lebail")
                for i, ref in enumerate(refs_lb):
                    p_k  = pseudo_voigt(tt_arr,ref[4],caglioti_fwhm(ref[4],pr["U"],pr["V"],pr["W"]),pr["eta0"])
                    p_s  = p_k.sum()
                    if p_s < 1e-12: continue
                    calc_nb = np.maximum(calc_v-bgv,0)
                    obs_nb  = np.maximum(obs-bgv,0)
                    refs_lb[i][5] = max(ref[5]*np.dot(obs_nb,p_k)/(np.dot(calc_nb,p_k)+1e-6),1e-3)

                if cycle % 5 == 4:
                    x0k,lok,hik,kk = [],[],[],[]
                    if lb_scale:   x0k.append(pr["scale"]);lok.append(1e-3);hik.append(1e10);kk.append("scale")
                    if lb_profile: x0k+=[pr["U"],pr["V"],pr["W"]];lok+=[0,-2,1e-4];hik+=[20,0,20];kk+=["U","V","W"]
                    if lb_eta:     x0k.append(pr["eta0"]);lok.append(0);hik.append(1);kk.append("eta0")
                    if lb_bg:      x0k+=list(bg);lok+=[-1e6]*n_bg;hik+=[1e6]*n_bg;kk+=[f"bg{j}" for j in range(n_bg)]
                    if x0k:
                        def _res_lb(p, pr_=pr, bg_=bg, kk_=kk):
                            pr_t=pr_.copy();bg_t=bg_.copy();idx=0
                            for key in kk_:
                                if key.startswith("bg"): bg_t[int(key[2:])]=p[idx]
                                else: pr_t[key]=p[idx]
                                idx+=1
                            cv,_ = calc_pattern(tt_arr,refs_lb,pr_t,bg_t,mode="lebail")
                            return np.sqrt(1/np.maximum(obs,1))*(obs-cv)
                        try:
                            rls = least_squares(_res_lb,x0k,bounds=(lok,hik),max_nfev=30,method="trf")
                            idx=0
                            for key in kk:
                                if key.startswith("bg"): bg[int(key[2:])]=rls.x[idx]
                                else: pr[key]=rls.x[idx]
                                idx+=1
                        except Exception: pass
                bar.progress((cycle+1)/int(lb_iters),text=f"Le Bail cycle {cycle+1}/{lb_iters}")
            bar.empty()

            calc_f,bg_f = calc_pattern(tt_arr,refs_lb,pr,bg,mode="lebail")
            Rwp,Rp,chi2,GoF = r_factors(obs,calc_f,len(x0k))
            st.session_state.lb_result = dict(
                refs=refs_lb,pr=pr,bg=bg,calc=calc_f,bgv=bg_f,
                Rwp=Rwp,Rp=Rp,chi2=chi2,GoF=GoF)

        with plot_col:
            res = st.session_state.lb_result
            if res:
                tt_arr=obs_tt; obs=obs_I
                fig = make_subplots(rows=2,cols=1,row_heights=[0.72,0.28],
                                    shared_xaxes=True,vertical_spacing=0.03)
                fig.add_trace(go.Scatter(x=tt_arr,y=obs,mode="lines",name="Observed",
                                          line=dict(color="#1f77b4",width=1.2)),row=1,col=1)
                fig.add_trace(go.Scatter(x=tt_arr,y=res["calc"],mode="lines",name="Calculated",
                                          line=dict(color="#ff7f0e",width=1.8)),row=1,col=1)
                fig.add_trace(go.Scatter(x=tt_arr,y=res["bgv"],mode="lines",name="Background",
                                          line=dict(color="#9467bd",width=1,dash="dash")),row=1,col=1)
                ymax_=obs.max()
                tx,ty=bragg_ticks(res["refs"],tt_arr[0],tt_arr[-1],ymax_*1.02,-ymax_*0.04)
                if tx:
                    fig.add_trace(go.Scatter(x=tx,y=ty,mode="lines",name="Bragg",
                                              line=dict(color="#2ca02c",width=1),hoverinfo="skip"),row=1,col=1)
                fig.add_trace(go.Scatter(x=tt_arr,y=obs-res["calc"],mode="lines",name="Difference",
                                          line=dict(color="#d62728",width=1)),row=2,col=1)
                fig.add_hline(y=0,line_color="gray",line_dash="dot",row=2,col=1)
                fig.update_layout(height=520,template="plotly_white",
                                  title=f"Le Bail | Rwp={res['Rwp']:.2f}%  Rp={res['Rp']:.2f}%  χ²={res['chi2']:.3f}",
                                  xaxis2_title="2θ (°)",yaxis_title="Intensity",yaxis2_title="Δ",
                                  legend=dict(x=0.75,y=0.95),margin=dict(t=50))
                st.plotly_chart(fig,use_container_width=True)
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Rwp",f"{res['Rwp']:.2f}%")
                m2.metric("Rp", f"{res['Rp']:.2f}%")
                m3.metric("χ²", f"{res['chi2']:.3f}")
                m4.metric("GoF",f"{res['GoF']:.3f}")
            else:
                st.info("← Configure and click **Run Le Bail**")


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — RIETVELD
# ──────────────────────────────────────────────────────────────────────────
with tab_rv:
    st.markdown("""
    **Rietveld refinement** — fits the full crystal structure against the observed pattern.
    Upload a **CIF file** to import cell parameters and atom sites automatically.
    """)
    if not have_data:
        st.warning("Generate diffraction data first (sidebar).")
    else:
        ctrl_col, plot_col = st.columns([1,2.5])

        with ctrl_col:
            # ── CIF UPLOAD ────────────────────────────────────────────────
            st.markdown("#### 📂 Structural Model — CIF Upload")
            cif_file = st.file_uploader(
                "Upload a CIF file", type=["cif"], key="cif_uploader",
                help="Accepts CIF files from ICSD, COD, CCDC, Materials Project, VESTA…")

            if cif_file is not None:
                try:
                    raw_cif  = cif_file.read().decode(errors="replace")
                    parsed   = parse_cif(raw_cif)
                    cell     = parsed["cell"]

                    # store cell into session state so sidebar can pick it up
                    if cell.get("a"): st.session_state.cif_a  = cell["a"]
                    if cell.get("b"): st.session_state.cif_b  = cell.get("b", cell["a"])
                    if cell.get("c"): st.session_state.cif_c  = cell.get("c", cell["a"])
                    st.session_state.cif_al = cell.get("alpha", 90.0)
                    st.session_state.cif_be = cell.get("beta",  90.0)
                    st.session_state.cif_ga = cell.get("gamma", 90.0)
                    st.session_state.cif_sg      = parsed["sg"]
                    st.session_state.cif_system  = parsed["system"]
                    st.session_state.cif_formula = parsed.get("formula")
                    st.session_state.cif_loaded  = True

                    if parsed["atoms"]:
                        st.session_state.atoms = parsed["atoms"]

                    n_at = len(parsed["atoms"])
                    sg_str = parsed["sg"] or "?"
                    sys_str = parsed["system"]
                    form_str = f" | {parsed['formula']}" if parsed.get("formula") else ""
                    st.success(f"✅ CIF loaded: {n_at} atom sites | {sg_str} | {sys_str}{form_str}")
                    st.info("↩ Click **Generate Reflections & Data** in the sidebar to apply the new cell.")

                    # ── CIF summary table ─────────────────────────────────
                    with st.expander("📋 CIF cell parameters", expanded=True):
                        cif_rows = {
                            "a (Å)":   f"{cell.get('a','?'):.4f}" if cell.get("a") else "?",
                            "b (Å)":   f"{cell.get('b','?'):.4f}" if cell.get("b") else "=a",
                            "c (Å)":   f"{cell.get('c','?'):.4f}" if cell.get("c") else "=a",
                            "α (°)":   f"{cell.get('alpha',90):.3f}",
                            "β (°)":   f"{cell.get('beta', 90):.3f}",
                            "γ (°)":   f"{cell.get('gamma',90):.3f}",
                            "Space group": parsed["sg"],
                            "System":      parsed["system"],
                            "Formula":     parsed.get("formula","—"),
                        }
                        for lbl, val in cif_rows.items():
                            st.markdown(f"**{lbl}:** {val}")

                    # ── atom table from CIF ───────────────────────────────
                    if n_at:
                        with st.expander(f"⚛ {n_at} atom sites from CIF", expanded=False):
                            df_cif = pd.DataFrame(parsed["atoms"])[
                                ["label","element","x","y","z","occ","Biso"]].round(5)
                            st.dataframe(df_cif, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"CIF parse error: {e}")

            # clear CIF button
            if st.session_state.cif_loaded:
                if st.button("🗑 Clear CIF / reset atoms", use_container_width=True):
                    for f in ["cif_loaded","cif_a","cif_b","cif_c",
                               "cif_al","cif_be","cif_ga","cif_sg","cif_system","cif_formula"]:
                        st.session_state[f] = False if f=="cif_loaded" else None
                    st.session_state.atoms = [
                        {"element":"Si","label":"Si1","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.5}]
                    st.rerun()

            st.markdown("---")

            # ── Manual atom editor (seeded by CIF or user) ────────────────
            st.markdown("#### ⚛ Atom Sites (editable)")
            updated_atoms = []
            for i, at in enumerate(st.session_state.atoms):
                lbl_str = at.get("label", at.get("element","?"))
                with st.expander(f"Atom {i+1}: {lbl_str}", expanded=i < 2):
                    cc1,cc2 = st.columns(2)
                    el  = cc1.text_input("Element", at["element"],          key=f"el_{i}")
                    lbl = cc2.text_input("Label",   at.get("label",el),     key=f"lb_{i}")
                    c1a,c2a,c3a = st.columns(3)
                    x = c1a.number_input("x",0.0,1.0,float(at["x"]),0.0001,"%.5f",key=f"ax_{i}")
                    y = c2a.number_input("y",0.0,1.0,float(at["y"]),0.0001,"%.5f",key=f"ay_{i}")
                    z = c3a.number_input("z",0.0,1.0,float(at["z"]),0.0001,"%.5f",key=f"az_{i}")
                    o1,o2 = st.columns(2)
                    occ  = o1.number_input("Occ.", 0.0, 1.0,  float(np.clip(at["occ"],  0.0, 1.0)),  0.01, "%.3f", key=f"oc_{i}")
                    Biso = o2.number_input("Biso", 0.0, 20.0, float(np.clip(at["Biso"], 0.0, 20.0)), 0.01, "%.3f", key=f"Bs_{i}")
                    updated_atoms.append({"element":el,"label":lbl,"x":x,"y":y,"z":z,"occ":occ,"Biso":Biso})
            st.session_state.atoms = updated_atoms

            ca,cr = st.columns(2)
            if ca.button("➕ Add atom", use_container_width=True):
                idx_new = len(st.session_state.atoms)+1
                st.session_state.atoms.append(
                    {"element":"O","label":f"O{idx_new}",
                     "x":0.5,"y":0.5,"z":0.5,"occ":1.0,"Biso":1.0})
                st.rerun()
            if cr.button("➖ Remove last", use_container_width=True) and len(st.session_state.atoms)>1:
                st.session_state.atoms.pop(); st.rerun()

            st.markdown("---")

            # ── Refinement switches ───────────────────────────────────────
            st.markdown("#### Refinement switches")
            rv_scale   = st.checkbox("Scale",             True, key="rv_sc")
            rv_profile = st.checkbox("Profile U, V, W",   True, key="rv_prf")
            rv_eta     = st.checkbox("Mixing η₀",         False,key="rv_eta")
            rv_bg      = st.checkbox("Background",        True, key="rv_bg")
            rv_pos     = st.checkbox("Atomic x, y, z",    False,key="rv_pos",
                                     help="Refine fractional coordinates")
            rv_Biso    = st.checkbox("Biso",              True, key="rv_Biso")
            rv_occ     = st.checkbox("Occupancies",       False,key="rv_occ")

            run_rv = st.button("▶ Run Rietveld", type="primary", use_container_width=True)

        # ── Rietveld refinement ───────────────────────────────────────────
        if run_rv:
            atoms_work = [dict(a) for a in st.session_state.atoms]
            tt_arr = obs_tt; obs = obs_I

            if st.session_state.lb_result:
                pr = dict(st.session_state.lb_result["pr"])
                bg = st.session_state.lb_result["bg"].copy()
            else:
                pr = dict(U=U,V=V,W=W,eta0=eta0,scale=scale,wl=wl,system=system)
                bg = np.zeros(n_bg); bg[0]=float(np.percentile(obs,3))

            def pack(pr_, bg_, atl):
                p=[]
                if rv_scale:   p.append(pr_["scale"])
                if rv_profile: p+=[pr_["U"],pr_["V"],pr_["W"]]
                if rv_eta:     p.append(pr_["eta0"])
                if rv_bg:      p+=list(bg_)
                for at in atl:
                    if rv_pos:  p+=[at["x"],at["y"],at["z"]]
                    if rv_Biso: p.append(at["Biso"])
                    if rv_occ:  p.append(at["occ"])
                return np.array(p,dtype=float)

            def unpack(p,pr_,bg_,atl):
                pr_=dict(pr_);bg_=bg_.copy();atl=[dict(a) for a in atl];idx=0
                if rv_scale:   pr_["scale"]=max(p[idx],1e-6);idx+=1
                if rv_profile: pr_["U"]=max(p[idx],0);idx+=1;pr_["V"]=p[idx];idx+=1;pr_["W"]=max(p[idx],1e-4);idx+=1
                if rv_eta:     pr_["eta0"]=np.clip(p[idx],0,1);idx+=1
                if rv_bg:      bg_[:]=p[idx:idx+n_bg];idx+=n_bg
                for j in range(len(atl)):
                    if rv_pos:  atl[j]["x"]=p[idx]%1;idx+=1;atl[j]["y"]=p[idx]%1;idx+=1;atl[j]["z"]=p[idx]%1;idx+=1
                    if rv_Biso: atl[j]["Biso"]=max(p[idx],0.01);idx+=1
                    if rv_occ:  atl[j]["occ"]=np.clip(p[idx],0.01,1);idx+=1
                return pr_,bg_,atl

            def residuals(p):
                pr_t,bg_t,at_t=unpack(p,pr,bg,atoms_work)
                cal,_=calc_pattern(tt_arr,refs,pr_t,bg_t,atoms=at_t,mode="rietveld")
                return np.sqrt(1/np.maximum(obs,1))*(obs-cal)

            x0=pack(pr,bg,atoms_work)
            lo,hi=[],[]
            if rv_scale:   lo.append(1e-6);  hi.append(1e10)
            if rv_profile: lo+=[0,-2,1e-4];  hi+=[20,0,20]
            if rv_eta:     lo.append(0);      hi.append(1)
            if rv_bg:      lo+=[-1e6]*n_bg;   hi+=[1e6]*n_bg
            for _ in atoms_work:
                if rv_pos:  lo+=[0,0,0];      hi+=[1,1,1]
                if rv_Biso: lo.append(0.01);  hi.append(30)
                if rv_occ:  lo.append(0.01);  hi.append(1)

            with st.spinner("Running Rietveld refinement…"):
                try:
                    rls=least_squares(residuals,x0,bounds=(lo,hi),
                                      max_nfev=2000,ftol=1e-10,xtol=1e-10,method="trf")
                    pr_f,bg_f,at_f=unpack(rls.x,pr,bg,atoms_work)
                    conv=f"Converged in {rls.nfev} evaluations (cost={rls.cost:.4g})"
                except Exception as e:
                    st.error(f"Refinement error: {e}")
                    pr_f,bg_f,at_f=pr,bg,atoms_work
                    conv="⚠ Did not converge"

            calc_f,bgv_f=calc_pattern(tt_arr,refs,pr_f,bg_f,atoms=at_f,mode="rietveld")
            Rwp,Rp,chi2,GoF=r_factors(obs,calc_f,len(x0))
            st.session_state.rv_result=dict(
                pr=pr_f,bg=bg_f,atoms=at_f,calc=calc_f,bgv=bgv_f,
                Rwp=Rwp,Rp=Rp,chi2=chi2,GoF=GoF,conv=conv,
                formula=st.session_state.cif_formula)

        # ── Plot ─────────────────────────────────────────────────────────
        with plot_col:
            res = st.session_state.rv_result
            if res:
                tt_arr=obs_tt; obs=obs_I
                diff=obs-res["calc"]
                fig=make_subplots(rows=2,cols=1,row_heights=[0.72,0.28],
                                  shared_xaxes=True,vertical_spacing=0.03)
                fig.add_trace(go.Scatter(x=tt_arr,y=obs,mode="lines",name="Observed",
                                          line=dict(color="#1f77b4",width=1.2)),row=1,col=1)
                fig.add_trace(go.Scatter(x=tt_arr,y=res["calc"],mode="lines",name="Calculated",
                                          line=dict(color="#ff7f0e",width=1.8)),row=1,col=1)
                fig.add_trace(go.Scatter(x=tt_arr,y=res["bgv"],mode="lines",name="Background",
                                          line=dict(color="#9467bd",width=1,dash="dash")),row=1,col=1)
                ymax_=obs.max()
                tx,ty=bragg_ticks(refs,tt_arr[0],tt_arr[-1],ymax_*1.02,-ymax_*0.04)
                if tx:
                    fig.add_trace(go.Scatter(x=tx,y=ty,mode="lines",name="Bragg",
                                              line=dict(color="#2ca02c",width=1),
                                              hoverinfo="skip"),row=1,col=1)
                fig.add_trace(go.Scatter(x=tt_arr,y=diff,mode="lines",name="Difference",
                                          line=dict(color="#d62728",width=1)),row=2,col=1)
                fig.add_hline(y=0,line_color="gray",line_dash="dot",row=2,col=1)
                fstr=f" | {res['formula']}" if res.get("formula") else ""
                fig.update_layout(
                    height=520,template="plotly_white",
                    title=(f"Rietveld{fstr} | "
                           f"Rwp={res['Rwp']:.2f}%  Rp={res['Rp']:.2f}%  "
                           f"χ²={res['chi2']:.3f}  GoF={res['GoF']:.3f}"),
                    xaxis2_title="2θ (°)",yaxis_title="Intensity",yaxis2_title="Δ",
                    legend=dict(x=0.75,y=0.95),margin=dict(t=55))
                st.plotly_chart(fig,use_container_width=True)
                st.caption(res.get("conv",""))

                m1,m2,m3,m4=st.columns(4)
                m1.metric("Rwp",f"{res['Rwp']:.2f}%")
                m2.metric("Rp", f"{res['Rp']:.2f}%")
                m3.metric("χ²", f"{res['chi2']:.3f}")
                m4.metric("GoF",f"{res['GoF']:.3f}")
                pr_=res["pr"]
                p1,p2,p3,p4,p5=st.columns(5)
                p1.metric("Scale",f"{pr_['scale']:.2f}")
                p2.metric("U",    f"{pr_['U']:.5f}")
                p3.metric("V",    f"{pr_['V']:.5f}")
                p4.metric("W",    f"{pr_['W']:.5f}")
                p5.metric("η₀",   f"{pr_['eta0']:.4f}")

                st.markdown("**Refined atom sites:**")
                cols_show = [c for c in ["label","element","x","y","z","occ","Biso"] if c in res["atoms"][0]]
                st.dataframe(pd.DataFrame(res["atoms"])[cols_show].round(5),
                             use_container_width=True, hide_index=True)
            else:
                if st.session_state.cif_loaded:
                    st.info("CIF loaded ✅ — click **▶ Run Rietveld** to start refinement.")
                else:
                    st.info("← Upload a CIF file and click **▶ Run Rietveld**")


# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — RESULTS
# ──────────────────────────────────────────────────────────────────────────
with tab_results:
    st.markdown("## Results Summary")
    lb = st.session_state.lb_result
    rv = st.session_state.rv_result

    methods,Rwps,Rps,chi2s,GoFs=[],[],[],[],[]
    if lb: methods.append("Le Bail"); Rwps.append(lb["Rwp"]); Rps.append(lb["Rp"]); chi2s.append(lb["chi2"]); GoFs.append(lb["GoF"])
    if rv: methods.append("Rietveld");Rwps.append(rv["Rwp"]); Rps.append(rv["Rp"]); chi2s.append(rv["chi2"]); GoFs.append(rv["GoF"])

    if methods:
        fig_r=go.Figure()
        fig_r.add_trace(go.Bar(x=methods,y=Rwps,name="Rwp (%)",marker_color="#1f77b4"))
        fig_r.add_trace(go.Bar(x=methods,y=Rps, name="Rp (%)", marker_color="#ff7f0e"))
        fig_r.update_layout(barmode="group",template="plotly_white",
                             title="R-factor comparison",height=300,
                             yaxis_title="R (%)",legend=dict(x=0.8,y=0.95))
        st.plotly_chart(fig_r,use_container_width=True)
        st.dataframe(pd.DataFrame({
            "Method":methods,
            "Rwp (%)": [f"{v:.3f}" for v in Rwps],
            "Rp (%)":  [f"{v:.3f}" for v in Rps],
            "χ²":      [f"{v:.4f}" for v in chi2s],
            "GoF":     [f"{v:.4f}" for v in GoFs],
        }),use_container_width=True,hide_index=True)
    else:
        st.info("Run Le Bail and/or Rietveld to see results here.")

    if lb and lb.get("refs"):
        st.markdown("### Le Bail — Extracted |F²| (first 80 reflections)")
        st.dataframe(pd.DataFrame([
            {"h":int(r[0]),"k":int(r[1]),"l":int(r[2]),
             "d (Å)":round(r[3],4),"2θ (°)":round(r[4],3),"|F²|":round(r[5],2)}
            for r in lb["refs"][:80]]),use_container_width=True,height=280)

    if rv and rv.get("atoms"):
        st.markdown("### Rietveld — Refined Atomic Parameters")
        cols_show = [c for c in ["label","element","x","y","z","occ","Biso"] if c in rv["atoms"][0]]
        st.dataframe(pd.DataFrame(rv["atoms"])[cols_show].round(5),
                     use_container_width=True)

    st.markdown("### 💾 Export")
    ec1,ec2,ec3=st.columns(3)

    if ec1.button("📥 Export Pattern CSV") and have_data:
        df_exp=pd.DataFrame({"2theta_deg":obs_tt,"observed":obs_I})
        if lb: df_exp["LeBail_calc"]=lb["calc"];df_exp["LeBail_bg"]=lb["bgv"];df_exp["LeBail_diff"]=obs_I-lb["calc"]
        if rv: df_exp["Rietveld_calc"]=rv["calc"];df_exp["Rietveld_bg"]=rv["bgv"];df_exp["Rietveld_diff"]=obs_I-rv["calc"]
        ec1.download_button("⬇ Download CSV",df_exp.to_csv(index=False),"diffraction_pattern.csv","text/csv")

    if ec2.button("📥 Export Reflections CSV") and refs is not None:
        ec2.download_button("⬇ Download CSV",
            pd.DataFrame([{"h":int(r[0]),"k":int(r[1]),"l":int(r[2]),
                           "d_Ang":round(r[3],5),"2theta_deg":round(r[4],4),"F2":round(r[5],3)}
                          for r in (lb["refs"] if lb else refs)]).to_csv(index=False),
            "reflections.csv","text/csv")

    if ec3.button("📥 Export Atoms CSV") and rv and rv.get("atoms"):
        ec3.download_button("⬇ Download CSV",pd.DataFrame(rv["atoms"]).to_csv(index=False),
                            "refined_atoms.csv","text/csv")