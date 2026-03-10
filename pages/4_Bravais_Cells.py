"""
CIF → Bravais-Gitter 3D Visualisierer
======================================
Liest eine beliebige CIF-Datei ein, erkennt das Bravais-Gitter automatisch
und stellt es interaktiv in 3D dar.

Installation:
    pip install streamlit matplotlib numpy

Starten:
    streamlit run bravais_from_cif.py

    # Mit direktem Dateipfad (optional):
    streamlit run bravais_from_cif.py -- --cif ../1000028__1_.cif
"""

import sys
import re
import math
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import streamlit as st
from pathlib import Path

# ── optional CLI arg: pre-load a CIF from filesystem ─────────────────────────
PRELOAD_PATH = None
args = sys.argv[1:]
for i, arg in enumerate(args):
    if arg == "--cif" and i+1 < len(args):
        PRELOAD_PATH = Path(args[i+1])
    elif arg.endswith(".cif"):
        PRELOAD_PATH = Path(arg)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bravais aus CIF",
    page_icon="⬡",
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
    border-radius: 10px;
    padding: 12px 18px;
}
[data-testid="stMetricLabel"] { color:#4466aa !important; font-family:'Share Tech Mono',monospace; font-size:10px; }
[data-testid="stMetricValue"] { color:#00e5c8 !important; font-size:22px; }

.bravais-header {
    background: linear-gradient(135deg, rgba(20,26,70,0.8), rgba(10,15,45,0.9));
    border: 1px solid rgba(50,80,200,0.35);
    border-left: 4px solid var(--accent, #4466ff);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
}
.tag {
    display:inline-block; padding:3px 14px; border-radius:16px;
    font-family:'Share Tech Mono',monospace; font-size:11px;
    margin:3px; letter-spacing:0.08em;
}
.info-box {
    background: rgba(15,20,55,0.5);
    border: 1px solid rgba(50,80,200,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #aabcee;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CIF PARSER  (no external deps)
# ════════════════════════════════════════════════════════════════════════════

def parse_number(s):
    if s is None or str(s).strip() in ("?", "."):
        return None
    try:
        return float(re.sub(r"\(.*?\)", "", str(s).strip()))
    except:
        return None

def parse_cif(text: str) -> dict:
    """Extract the key crystallographic fields we need."""
    result = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1; continue
        # multi-line string
        if line.startswith(";"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(";"):
                i += 1
            i += 1; continue
        # key-value
        if line.startswith("_"):
            parts = _tokenize(line)
            key = parts[0].lower()
            if len(parts) >= 2:
                result[key] = parts[1]
            else:
                # value might be on next line
                i += 1
                if i < len(lines):
                    nxt = lines[i].strip()
                    if nxt and not nxt.startswith("_") and not nxt.startswith(";") and nxt.lower() != "loop_":
                        result[key] = _tokenize(nxt)[0] if _tokenize(nxt) else "?"
                continue
        i += 1
    return result

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


# ════════════════════════════════════════════════════════════════════════════
# BRAVAIS DETERMINATION
# ════════════════════════════════════════════════════════════════════════════

# Space-group number → (crystal system, centering, Bravais symbol, Bravais name)
# Full mapping of all 230 space groups
def sg_to_bravais(sg_number: int):
    n = sg_number
    if   1  <= n <= 2:   return ("Triklin",        "P", "aP", "Triklin P")
    elif 3  <= n <= 5:
        return ("Monoklin", "P" if n in (3,4) else "C", "mP" if n in (3,4) else "mC",
                "Monoklin P" if n in (3,4) else "Monoklin C")
    elif 6  <= n <= 9:
        cent = "P" if n in (6,7) else "C"
        sym  = "mP" if cent == "P" else "mC"
        return ("Monoklin", cent, sym, f"Monoklin {cent}")
    elif 10 <= n <= 15:
        cent = "P" if n in (10,11,13,14) else "C"
        sym  = "mP" if cent == "P" else "mC"
        return ("Monoklin", cent, sym, f"Monoklin {cent}")
    elif 16 <= n <= 24:  return ("Orthorhombisch", "P", "oP", "Orthorhombisch P")
    elif 25 <= n <= 46:
        if n in (38,39,40,41): return ("Orthorhombisch","A","oA","Orthorhombisch A")
        if n in (42,43):       return ("Orthorhombisch","F","oF","Orthorhombisch F")
        if n in (44,45,46):    return ("Orthorhombisch","I","oI","Orthorhombisch I")
        return ("Orthorhombisch","C","oC","Orthorhombisch C")
    elif 47 <= n <= 74:
        if n in (65,66,67,68): return ("Orthorhombisch","C","oC","Orthorhombisch C")
        if n in (69,70):       return ("Orthorhombisch","F","oF","Orthorhombisch F")
        if n in (71,72,73,74): return ("Orthorhombisch","I","oI","Orthorhombisch I")
        return ("Orthorhombisch","P","oP","Orthorhombisch P")
    elif 75 <= n <= 82:  return ("Tetragonal",     "P", "tP", "Tetragonal P")
    elif 83 <= n <= 88:  return ("Tetragonal",     "P", "tP", "Tetragonal P")
    elif 89 <= n <= 98:  return ("Tetragonal",     "P", "tP", "Tetragonal P")
    elif 99 <= n <= 110: return ("Tetragonal",     "P", "tP", "Tetragonal P")
    elif 111<= n <=122:  return ("Tetragonal",     "P", "tP", "Tetragonal P")
    elif 123<= n <=142:
        if n in (139,140,141,142): return ("Tetragonal","I","tI","Tetragonal I")
        return ("Tetragonal","P","tP","Tetragonal P")
    elif 143<= n <=146:  return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
    elif 147<= n <=148:  return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
    elif 149<= n <=155:  return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
    elif 156<= n <=161:  return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
    elif 162<= n <=167:
        if n in (160,161,166,167): return ("Rhomboedrisch","R","hR","Rhomboedrisch R")
        return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
    elif 168<= n <=176:  return ("Hexagonal",      "P", "hP", "Hexagonal P")
    elif 177<= n <=194:  return ("Hexagonal",      "P", "hP", "Hexagonal P")
    elif 195<= n <=199:  return ("Kubisch",        "P", "cP", "Kubisch P")
    elif 200<= n <=206:  return ("Kubisch",        "P", "cP", "Kubisch P")
    elif 207<= n <=214:
        if n in (207,208,212,213): return ("Kubisch","P","cP","Kubisch P")
        if n in (209,210):         return ("Kubisch","F","cF","Kubisch F (FCC)")
        if n in (211,214):         return ("Kubisch","I","cI","Kubisch I (BCC)")
        return ("Kubisch","P","cP","Kubisch P")
    elif 215<= n <=220:
        if n in (216,219): return ("Kubisch","F","cF","Kubisch F (FCC)")
        if n in (217,220): return ("Kubisch","I","cI","Kubisch I (BCC)")
        return ("Kubisch","P","cP","Kubisch P")
    elif 221<= n <=230:
        if n in (225,226,227,228): return ("Kubisch","F","cF","Kubisch F (FCC)")
        if n in (229,230):         return ("Kubisch","I","cI","Kubisch I (BCC)")
        return ("Kubisch","P","cP","Kubisch P")
    return ("Unbekannt","P","??","Unbekannt")

def bravais_from_cif_data(data: dict):
    """Determine Bravais lattice from parsed CIF data."""
    # Try space-group number first
    sg_num = None
    for key in ("_symmetry_int_tables_number","_space_group_it_number",
                "_symmetry_int_tables_number"):
        v = data.get(key)
        if v and v not in ("?","."):
            try: sg_num = int(v); break
            except: pass

    if sg_num:
        return sg_to_bravais(sg_num), sg_num

    # Fallback: derive from cell setting + centering
    setting = (data.get("_symmetry_cell_setting") or
               data.get("_space_group_crystal_system") or "").strip().lower()
    hm = (data.get("_symmetry_space_group_name_h-m") or
          data.get("_space_group_name_h-m_alt") or "").strip().strip("'\"")

    # centering from first letter of H-M symbol
    cent = "P"
    if hm:
        first = hm[0].upper()
        if first in ("P","I","F","C","A","B","R"):
            cent = first

    sys_map = {
        "triclinic":    "Triklin",
        "monoclinic":   "Monoklin",
        "orthorhombic": "Orthorhombisch",
        "tetragonal":   "Tetragonal",
        "trigonal":     "Trigonal/Rhomboedrisch",
        "hexagonal":    "Hexagonal",
        "cubic":        "Kubisch",
        "rhombohedral": "Rhomboedrisch",
    }
    system = sys_map.get(setting, "Unbekannt")

    sym_map = {
        ("Kubisch","P"): ("cP","Kubisch P"),
        ("Kubisch","I"): ("cI","Kubisch I (BCC)"),
        ("Kubisch","F"): ("cF","Kubisch F (FCC)"),
        ("Tetragonal","P"): ("tP","Tetragonal P"),
        ("Tetragonal","I"): ("tI","Tetragonal I"),
        ("Orthorhombisch","P"): ("oP","Orthorhombisch P"),
        ("Orthorhombisch","C"): ("oC","Orthorhombisch C"),
        ("Orthorhombisch","I"): ("oI","Orthorhombisch I"),
        ("Orthorhombisch","F"): ("oF","Orthorhombisch F"),
        ("Monoklin","P"): ("mP","Monoklin P"),
        ("Monoklin","C"): ("mC","Monoklin C"),
        ("Triklin","P"):  ("aP","Triklin P"),
        ("Hexagonal","P"):("hP","Hexagonal P"),
        ("Rhomboedrisch","R"):("hR","Rhomboedrisch R"),
        ("Trigonal/Rhomboedrisch","P"):("hP","Hexagonal P"),
        ("Trigonal/Rhomboedrisch","R"):("hR","Rhomboedrisch R"),
    }
    sym, name = sym_map.get((system, cent), ("??", f"{system} {cent}"))
    return (system, cent, sym, name), None


# ════════════════════════════════════════════════════════════════════════════
# LATTICE VECTOR BUILDER
# ════════════════════════════════════════════════════════════════════════════

def cell_vectors(a, b, c, alpha_deg, beta_deg, gamma_deg):
    al = math.radians(alpha_deg)
    be = math.radians(beta_deg)
    ga = math.radians(gamma_deg)
    cos_a, cos_b, cos_g = math.cos(al), math.cos(be), math.cos(ga)
    sin_g = math.sin(ga)
    v = math.sqrt(max(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2*cos_a*cos_b*cos_g, 0))
    a1 = np.array([a, 0, 0])
    a2 = np.array([b*cos_g, b*sin_g, 0])
    a3 = np.array([c*cos_b, c*(cos_a - cos_b*cos_g)/sin_g, c*v/sin_g])
    return a1, a2, a3

def centering_fracs(centering: str):
    pts = [(0,0,0)]
    if centering == "I":  pts += [(0.5,0.5,0.5)]
    elif centering == "F":pts += [(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)]
    elif centering == "C":pts += [(0.5,0.5,0)]
    elif centering == "A":pts += [(0,0.5,0.5)]
    elif centering == "B":pts += [(0.5,0,0.5)]
    elif centering == "R":pts += [(2/3,1/3,1/3),(1/3,2/3,2/3)]
    return pts

def cell_edges(a1, a2, a3):
    corners = {(i,j,k): i*a1+j*a2+k*a3 for i,j,k in itertools.product([0,1],repeat=3)}
    edges = []
    for (i,j,k) in corners:
        for di,dj,dk in [(1,0,0),(0,1,0),(0,0,1)]:
            ni,nj,nk = i+di,j+dj,k+dk
            if ni<=1 and nj<=1 and nk<=1:
                edges.append([corners[(i,j,k)], corners[(ni,nj,nk)]])
    return edges, corners


# ════════════════════════════════════════════════════════════════════════════
# COLORS
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_COLORS = {
    "Triklin":                  "#e74c3c",
    "Monoklin":                 "#e67e22",
    "Orthorhombisch":           "#27ae60",
    "Tetragonal":               "#2980b9",
    "Rhomboedrisch":            "#8e44ad",
    "Trigonal/Rhomboedrisch":   "#9b59b6",
    "Hexagonal":                "#16a085",
    "Kubisch":                  "#00e5c8",
    "Unbekannt":                "#888888",
}

BRAVAIS_INFO = {
    "cP": {"desc": "Einfach kubisch. Atome nur an den Würfelecken.", "points": 1},
    "cI": {"desc": "Raumzentrierter Würfel (BCC). Zusätzlicher Atom im Zentrum.", "points": 2},
    "cF": {"desc": "Flächenzentrierter Würfel (FCC). Dichteste Kugelpackung.", "points": 4},
    "tP": {"desc": "Tetragonale Zelle (primitiv). Quadratische Basis, gestreckte c-Achse.", "points": 1},
    "tI": {"desc": "Tetragonal raumzentriert. Zusätzlicher Punkt im Zentrum.", "points": 2},
    "oP": {"desc": "Orthorhombisch primitiv. Drei ungleiche rechtwinklige Achsen.", "points": 1},
    "oC": {"desc": "Orthorhombisch C-zentriert. Zusatzpunkte auf ab-Flächen.", "points": 2},
    "oI": {"desc": "Orthorhombisch raumzentriert.", "points": 2},
    "oF": {"desc": "Orthorhombisch flächenzentriert.", "points": 4},
    "hP": {"desc": "Hexagonal primitiv. 120°-Winkel in der Basisebene.", "points": 1},
    "hR": {"desc": "Rhomboedrisch. Würfel entlang Raumdiagonale deformiert.", "points": 1},
    "mP": {"desc": "Monoklin primitiv. Ein schiefer Winkel (β≠90°).", "points": 1},
    "mC": {"desc": "Monoklin C-zentriert.", "points": 2},
    "aP": {"desc": "Triklin. Keine Einschränkungen – niedrigste Symmetrie.", "points": 1},
}


# ════════════════════════════════════════════════════════════════════════════
# 3D VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def draw_bravais_3d(a1, a2, a3, centering, color,
                    elev=22, azim=35, supercell=1,
                    show_vectors=True, show_planes=False,
                    atom_size=120, figsize=(8,7),
                    dark=True):

    bg  = "#07080f" if dark else "#f5f4f0"
    fg  = "#c8d4f8" if dark else "#1a1a2e"
    grid_col = "#1e2555" if dark else "#d0cfc8"

    fig = plt.figure(figsize=figsize, facecolor=bg)
    ax  = fig.add_subplot(111, projection="3d", facecolor=bg)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(grid_col)
        pane.set_alpha(0.15)
    ax.grid(False)
    ax.set_axis_off()

    sc = supercell
    # draw cells
    for ti,tj,tk in itertools.product(range(sc), repeat=3):
        origin = ti*a1 + tj*a2 + tk*a3
        edges, _ = cell_edges(a1, a2, a3)
        segs = [[e[0]+origin, e[1]+origin] for e in edges]
        lc = Line3DCollection(segs, colors=color, linewidths=1.4,
                              alpha=0.5 if sc > 1 else 0.7)
        ax.add_collection3d(lc)

        # optional face planes (first cell only)
        if show_planes and ti==0 and tj==0 and tk==0:
            corners = {(i,j,k): i*a1+j*a2+k*a3
                       for i,j,k in itertools.product([0,1],repeat=3)}
            faces = [
                [corners[(0,0,0)],corners[(1,0,0)],corners[(1,1,0)],corners[(0,1,0)]],
                [corners[(0,0,0)],corners[(1,0,0)],corners[(1,0,1)],corners[(0,0,1)]],
                [corners[(0,0,0)],corners[(0,1,0)],corners[(0,1,1)],corners[(0,0,1)]],
            ]
            poly = Poly3DCollection(faces, alpha=0.06, facecolor=color, edgecolor="none")
            ax.add_collection3d(poly)

    # lattice points
    cent_pts = centering_fracs(centering)
    rng = range(sc + 1)
    shown = set()
    for i,j,k in itertools.product(rng, repeat=3):
        base = i*a1 + j*a2 + k*a3
        for (fi,fj,fk) in cent_pts:
            pt = base + fi*a1 + fj*a2 + fk*a3
            key = tuple(np.round(pt, 3))
            if key in shown: continue
            shown.add(key)
            # distinguish corner vs centering points
            is_corner = (fi == 0 and fj == 0 and fk == 0)
            s    = atom_size if is_corner else atom_size * 0.75
            ecol = "white" if dark else "#333"
            ax.scatter(*pt, s=s, c=color, edgecolors=ecol,
                       linewidths=0.8, depthshade=True, zorder=5,
                       alpha=1.0 if is_corner else 0.85)

    # lattice vectors (only for sc==1)
    if show_vectors and sc == 1:
        vcols = ["#ff4444", "#44ff88", "#4488ff"]
        vlbls = ["a", "b", "c"]
        for v, vc, vl in zip([a1, a2, a3], vcols, vlbls):
            ax.quiver(0, 0, 0, *v, color=vc, arrow_length_ratio=0.12,
                      linewidth=2.2, alpha=0.95)
            off = v * 1.15
            ax.text(*off, vl, color=vc, fontsize=13,
                    fontfamily="monospace", fontweight="bold",
                    ha="center", va="center")

    ax.view_init(elev=elev, azim=azim)

    # axis limits
    pts_all = [i*a1 + j*a2 + k*a3
               for i,j,k in itertools.product(range(sc+1), repeat=3)]
    coords = np.array(pts_all)
    pad = max(np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(a3)) * 0.25
    mn, mx = coords.min(axis=0) - pad, coords.max(axis=0) + pad
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])

    plt.tight_layout(pad=0)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⬡ BRAVAIS AUS CIF")
    st.markdown("---")

    cif_text = None
    cif_name = ""

    # ── 1. CLI argument (--cif path) ─────────────────────────────────────────
    if PRELOAD_PATH and PRELOAD_PATH.exists():
        cif_text = PRELOAD_PATH.read_text(encoding="utf-8", errors="replace")
        cif_name = PRELOAD_PATH.name

    # ── 2. Auto-search parent folders ────────────────────────────────────────
    if cif_text is None:
        script_dir = Path(__file__).resolve().parent
        search_dirs = [
            script_dir.parent,          # ../
            script_dir.parent.parent,   # ../../
            script_dir,                 # ./
        ]
        found_cifs = []
        for d in search_dirs:
            found_cifs += sorted(d.glob("*.cif"))
        found_cifs = list(dict.fromkeys(found_cifs))  # deduplicate

        if found_cifs:
            cif_names = [f.name for f in found_cifs]
            st.markdown("**CIF-Dateien gefunden**")
            selected_idx = st.selectbox(
                "Datei auswählen",
                range(len(cif_names)),
                format_func=lambda i: cif_names[i],
                index=0,
            )
            chosen = found_cifs[selected_idx]
            cif_text = chosen.read_text(encoding="utf-8", errors="replace")
            cif_name = chosen.name
            st.caption(f"📂 {chosen.parent}")

    # ── 3. Fallback: manual upload ────────────────────────────────────────────
    if cif_text is None:
        st.markdown("**Keine CIF im Parent-Ordner gefunden**")
        uploaded = st.file_uploader("CIF manuell hochladen", type=["cif"])
        if uploaded:
            cif_text = uploaded.read().decode("utf-8", errors="replace")
            cif_name = uploaded.name

    st.markdown("---")
    st.markdown("**Visualisierung**")
    elev     = st.slider("Elevation",  -90, 90, 22, key="elev")
    azim     = st.slider("Azimut",     0, 360, 35, key="azim")
    sc       = st.slider("Superzelle", 1, 3, 1, key="sc")
    show_vec = st.checkbox("Gittervektoren a, b, c", True)
    show_pln = st.checkbox("Flächen einblenden", False)
    atom_sz  = st.slider("Atom-Größe", 40, 300, 120, step=10)

    st.markdown("---")
    st.markdown("**Darstellung**")
    dark_mode = st.checkbox("Dark Mode", True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if cif_text is None:
    st.markdown("""
    <div style='text-align:center; padding:100px 40px;'>
      <div style='font-size:80px; margin-bottom:28px; filter:drop-shadow(0 0 30px #4466ff);'>⬡</div>
      <h1 style='font-family:"Share Tech Mono",monospace; font-size:28px;
                 letter-spacing:0.15em; color:#c8d4f8;'>BRAVAIS AUS CIF</h1>
      <p style='color:#334488; font-size:16px; margin-top:14px; letter-spacing:0.06em; max-width:500px; margin-left:auto; margin-right:auto;'>
        Lade eine CIF-Datei hoch oder gib einen Dateipfad an —<br>
        das Bravais-Gitter wird automatisch erkannt und in 3D visualisiert.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Parse ─────────────────────────────────────────────────────────────────────
data = parse_cif(cif_text)

a_val = parse_number(data.get("_cell_length_a"))
b_val = parse_number(data.get("_cell_length_b"))
c_val = parse_number(data.get("_cell_length_c"))
alpha = parse_number(data.get("_cell_angle_alpha")) or 90.0
beta  = parse_number(data.get("_cell_angle_beta"))  or 90.0
gamma = parse_number(data.get("_cell_angle_gamma")) or 90.0
vol   = parse_number(data.get("_cell_volume"))

if not a_val:
    st.error("Keine Gitterparameter in der CIF-Datei gefunden.")
    st.stop()

b_val = b_val or a_val
c_val = c_val or a_val

a1, a2, a3 = cell_vectors(a_val, b_val, c_val, alpha, beta, gamma)

(system, centering, symbol, bravais_name), sg_num = bravais_from_cif_data(data)

color = SYSTEM_COLORS.get(system, "#888888")
info  = BRAVAIS_INFO.get(symbol, {})

spacegroup = (data.get("_symmetry_space_group_name_h-m") or
              data.get("_space_group_name_h-m_alt") or "–").strip().strip("'\"")
crystal_name = (data.get("_chemical_name_common") or
                data.get("_chemical_name_systematic") or
                cif_name.replace(".cif","")).strip().strip("'\"")
if crystal_name in ("?","."): crystal_name = cif_name.replace(".cif","")

formula = (data.get("_chemical_formula_sum") or
           data.get("_chemical_formula_iupac") or "–").strip().strip("'\"")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='bravais-header' style='--accent:{color};'>
  <span style='font-family:"Share Tech Mono",monospace; font-size:11px;
               color:#334488; letter-spacing:0.2em;'>CIF → BRAVAIS-GITTER</span>
  <h1 style='font-size:32px; font-weight:600; color:#e8f0ff; margin:6px 0 4px;
             font-family:"DM Sans",sans-serif;'>
    {crystal_name}
  </h1>
  <div>
    <span class='tag' style='background:{color}22; border:1px solid {color}55; color:{color};'>
      {bravais_name}
    </span>
    <span class='tag' style='background:rgba(30,40,90,0.6); border:1px solid #2a3880; color:#6688cc;'>
      {symbol}
    </span>
    <span class='tag' style='background:rgba(30,40,90,0.4); border:1px solid #1e2555; color:#445588;'>
      {spacegroup}
    </span>
    <span class='tag' style='background:rgba(30,40,90,0.4); border:1px solid #1e2555; color:#445588;'>
      {formula}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_3d, col_info = st.columns([3, 2])

with col_3d:
    fig = draw_bravais_3d(
        a1, a2, a3, centering, color,
        elev=elev, azim=azim,
        supercell=sc,
        show_vectors=show_vec,
        show_planes=show_pln,
        atom_size=atom_sz,
        figsize=(8, 7),
        dark=dark_mode,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_info:
    # Cell parameters
    st.markdown("#### Gitterparameter")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("a (Å)", f"{a_val:.4f}")
        st.metric("α (°)", f"{alpha:.2f}")
    with c2:
        st.metric("b (Å)", f"{b_val:.4f}")
        st.metric("β (°)", f"{beta:.2f}")
    with c3:
        st.metric("c (Å)", f"{c_val:.4f}")
        st.metric("γ (°)", f"{gamma:.2f}")

    if vol:
        st.metric("Volumen (Å³)", f"{vol:.3f}")

    st.markdown("---")

    # Bravais info
    st.markdown("#### Bravais-Gitter")
    st.markdown(f"""
    <div class='info-box'>
      <b style='color:{color};font-family:"Share Tech Mono",monospace;
                font-size:18px;'>{symbol}</b>
      &nbsp;·&nbsp; {bravais_name}<br><br>
      {info.get('desc', '')}
    </div>
    """, unsafe_allow_html=True)

    # Centering explanation
    cent_desc = {
        "P": "**Primitiv (P):** Gitterpunkte nur an den Ecken der Einheitszelle.",
        "I": "**Raumzentriert (I):** Zusätzlicher Gitterpunkt im Zentrum der Zelle (½,½,½).",
        "F": "**Flächenzentriert (F):** Zusätzliche Punkte auf allen 6 Flächen (½,½,0) etc.",
        "C": "**C-zentriert:** Zusätzliche Punkte auf den ab-Flächen (½,½,0).",
        "R": "**Rhomboedrisch (R):** Punkte bei (⅔,⅓,⅓) und (⅓,⅔,⅔).",
        "A": "**A-zentriert:** Zusätzliche Punkte auf den bc-Flächen.",
        "B": "**B-zentriert:** Zusätzliche Punkte auf den ac-Flächen.",
    }.get(centering, "")
    if cent_desc:
        st.markdown(cent_desc)

    st.markdown("---")

    # Space group details
    st.markdown("#### Raumgruppe")
    sg_data = {
        "H-M Symbol":   spacegroup,
        "Hall Symbol":  data.get("_symmetry_space_group_name_hall","–").strip().strip("'\""),
        "Nr. (Int. Tab.)": str(sg_num) if sg_num else "–",
        "Kristallsystem": system,
        "Zentrierung":  centering,
        "Punktgruppe":  data.get("_symmetry_point_group","–"),
    }
    for k, v in sg_data.items():
        if v and v not in ("–","?","."):
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"border-bottom:1px solid #1a2050;padding:5px 0;"
                f"font-size:13px;'>"
                f"<span style='color:#445588;font-family:\"Share Tech Mono\",monospace;"
                f"font-size:11px;'>{k}</span>"
                f"<span style='color:#c8d4f8;font-family:\"Share Tech Mono\",monospace;'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

# ── Legende ──────────────────────────────────────────────────────────────────
st.markdown("---")
lc1, lc2, lc3 = st.columns(3)
with lc1:
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;font-size:13px;color:#aabcee;'>
      <div style='width:14px;height:14px;border-radius:50%;background:{color};
                  box-shadow:0 0 8px {color};'></div>
      <span>Gitterpunkt (Ecke)</span>
    </div>""", unsafe_allow_html=True)
with lc2:
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;font-size:13px;color:#aabcee;'>
      <div style='width:10px;height:10px;border-radius:50%;background:{color};opacity:0.7;
                  box-shadow:0 0 6px {color};'></div>
      <span>Zentrierungspunkt</span>
    </div>""", unsafe_allow_html=True)
with lc3:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;font-size:13px;color:#aabcee;'>
      <div style='display:flex;gap:4px;align-items:center;'>
        <span style='color:#ff4444;font-family:monospace;font-weight:bold;'>a</span>
        <span style='color:#44ff88;font-family:monospace;font-weight:bold;'>b</span>
        <span style='color:#4488ff;font-family:monospace;font-weight:bold;'>c</span>
      </div>
      <span>Gittervektoren</span>
    </div>""", unsafe_allow_html=True)