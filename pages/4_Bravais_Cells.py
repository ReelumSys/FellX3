
import streamlit as st
import pandas as pd

# Retrieve data from session state
main_df = st.session_state.get('main_df')
comp_df = st.session_state.get('comp_df')
cif_data = st.session_state.get('cif_data')

if main_df is None or comp_df is None:
    st.warning("Please upload the required XRD patterns on the Main Page first.")
    st.stop()

     1|﻿"""
     2|CIF → Bravais-Gitter 3D Visualisierer
     3|======================================
     4|Liest eine beliebige CIF-Datei ein, erkennt das Bravais-Gitter automatisch
     5|und stellt es interaktiv in 3D dar.
     6|
     7|Installation:
     8|    pip install streamlit matplotlib numpy
     9|
    10|Starten:
    11|    streamlit run bravais_from_cif.py
    12|
    13|    # Mit direktem Dateipfad (optional):
    14|    streamlit run bravais_from_cif.py -- --cif ../1000028__1_.cif
    15|"""
    16|
    17|import sys
    18|import re
    19|import math
    20|import itertools
    21|import numpy as np
    22|import matplotlib
    23|matplotlib.use("Agg")
    24|import matplotlib.pyplot as plt
    25|from mpl_toolkits.mplot3d import Axes3D          # noqa
    26|from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    27|import streamlit as st
    28|from pathlib import Path
    29|
    30|# ── optional CLI arg: pre-load a CIF from filesystem ─────────────────────────
    31|PRELOAD_PATH = None
    32|args = sys.argv[1:]
    33|for i, arg in enumerate(args):
    34|    if arg == "--cif" and i+1 < len(args):
    35|        PRELOAD_PATH = Path(args[i+1])
    36|    elif arg.endswith(".cif"):
    37|        PRELOAD_PATH = Path(arg)
    38|
    39|# ── page config ───────────────────────────────────────────────────────────────
    40|st.set_page_config(
    41|    page_title="Bravais aus CIF",
    42|    page_icon="⬡",
    43|    layout="wide",
    44|    initial_sidebar_state="expanded",
    45|)
    46|
    47|st.markdown("""
    48|<style>
    49|@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;600&display=swap');
    50|html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    51|h1,h2,h3 { font-family: 'Share Tech Mono', monospace !important; }
    52|.stApp { background: #07080f; color: #c8d4f8; }
    53|[data-testid="stSidebar"] { background: #0d0e1c !important; border-right:1px solid #1e2555; }
    54|[data-testid="stSidebar"] * { color: #c8d4f8 !important; }
    55|[data-testid="stSidebar"] label { color: #4466aa !important; font-size:12px; }
    56|
    57|[data-testid="metric-container"] {
    58|    background: rgba(20,26,70,0.6);
    59|    border: 1px solid rgba(50,80,200,0.3);
    60|    border-radius: 10px;
    61|    padding: 12px 18px;
    62|}
    63|[data-testid="stMetricLabel"] { color:#4466aa !important; font-family:'Share Tech Mono',monospace; font-size:10px; }
    64|[data-testid="stMetricValue"] { color:#00e5c8 !important; font-size:22px; }
    65|
    66|.bravais-header {
    67|    background: linear-gradient(135deg, rgba(20,26,70,0.8), rgba(10,15,45,0.9));
    68|    border: 1px solid rgba(50,80,200,0.35);
    69|    border-left: 4px solid var(--accent, #4466ff);
    70|    border-radius: 12px;
    71|    padding: 20px 28px;
    72|    margin-bottom: 20px;
    73|}
    74|.tag {
    75|    display:inline-block; padding:3px 14px; border-radius:16px;
    76|    font-family:'Share Tech Mono',monospace; font-size:11px;
    77|    margin:3px; letter-spacing:0.08em;
    78|}
    79|.info-box {
    80|    background: rgba(15,20,55,0.5);
    81|    border: 1px solid rgba(50,80,200,0.25);
    82|    border-radius: 10px;
    83|    padding: 14px 18px;
    84|    margin: 8px 0;
    85|    font-size: 14px;
    86|    line-height: 1.7;
    87|    color: #aabcee;
    88|}
    89|</style>
    90|""", unsafe_allow_html=True)
    91|
    92|
    93|# ════════════════════════════════════════════════════════════════════════════
    94|# CIF PARSER  (no external deps)
    95|# ════════════════════════════════════════════════════════════════════════════
    96|
    97|def parse_number(s):
    98|    if s is None or str(s).strip() in ("?", "."):
    99|        return None
   100|    try:
   101|        return float(re.sub(r"\(.*?\)", "", str(s).strip()))
   102|    except:
   103|        return None
   104|
   105|def parse_cif(text: str) -> dict:
   106|    """Extract the key crystallographic fields we need."""
   107|    result = {}
   108|    lines = text.splitlines()
   109|    i = 0
   110|    while i < len(lines):
   111|        line = lines[i].strip()
   112|        if not line or line.startswith("#"):
   113|            i += 1; continue
   114|        # multi-line string
   115|        if line.startswith(";"):
   116|            i += 1
   117|            while i < len(lines) and not lines[i].strip().startswith(";"):
   118|                i += 1
   119|            i += 1; continue
   120|        # key-value
   121|        if line.startswith("_"):
   122|            parts = _tokenize(line)
   123|            key = parts[0].lower()
   124|            if len(parts) >= 2:
   125|                result[key] = parts[1]
   126|            else:
   127|                # value might be on next line
   128|                i += 1
   129|                if i < len(lines):
   130|                    nxt = lines[i].strip()
   131|                    if nxt and not nxt.startswith("_") and not nxt.startswith(";") and nxt.lower() != "loop_":
   132|                        result[key] = _tokenize(nxt)[0] if _tokenize(nxt) else "?"
   133|                continue
   134|        i += 1
   135|    return result
   136|
   137|def _tokenize(line):
   138|    tokens = []
   139|    i = 0
   140|    while i < len(line):
   141|        if line[i] in ('"', "'"):
   142|            q = line[i]; j = line.find(q, i+1)
   143|            if j == -1: j = len(line)
   144|            tokens.append(line[i+1:j]); i = j+1
   145|        elif line[i] in (" ", "\t"):
   146|            i += 1
   147|        else:
   148|            j = i
   149|            while j < len(line) and line[j] not in (" ", "\t"):
   150|                j += 1
   151|            tokens.append(line[i:j]); i = j
   152|    return tokens
   153|
   154|
   155|# ════════════════════════════════════════════════════════════════════════════
   156|# BRAVAIS DETERMINATION
   157|# ════════════════════════════════════════════════════════════════════════════
   158|
   159|# Space-group number → (crystal system, centering, Bravais symbol, Bravais name)
   160|# Full mapping of all 230 space groups
   161|def sg_to_bravais(sg_number: int):
   162|    n = sg_number
   163|    if   1  <= n <= 2:   return ("Triklin",        "P", "aP", "Triklin P")
   164|    elif 3  <= n <= 5:
   165|        return ("Monoklin", "P" if n in (3,4) else "C", "mP" if n in (3,4) else "mC",
   166|                "Monoklin P" if n in (3,4) else "Monoklin C")
   167|    elif 6  <= n <= 9:
   168|        cent = "P" if n in (6,7) else "C"
   169|        sym  = "mP" if cent == "P" else "mC"
   170|        return ("Monoklin", cent, sym, f"Monoklin {cent}")
   171|    elif 10 <= n <= 15:
   172|        cent = "P" if n in (10,11,13,14) else "C"
   173|        sym  = "mP" if cent == "P" else "mC"
   174|        return ("Monoklin", cent, sym, f"Monoklin {cent}")
   175|    elif 16 <= n <= 24:  return ("Orthorhombisch", "P", "oP", "Orthorhombisch P")
   176|    elif 25 <= n <= 46:
   177|        if n in (38,39,40,41): return ("Orthorhombisch","A","oA","Orthorhombisch A")
   178|        if n in (42,43):       return ("Orthorhombisch","F","oF","Orthorhombisch F")
   179|        if n in (44,45,46):    return ("Orthorhombisch","I","oI","Orthorhombisch I")
   180|        return ("Orthorhombisch","C","oC","Orthorhombisch C")
   181|    elif 47 <= n <= 74:
   182|        if n in (65,66,67,68): return ("Orthorhombisch","C","oC","Orthorhombisch C")
   183|        if n in (69,70):       return ("Orthorhombisch","F","oF","Orthorhombisch F")
   184|        if n in (71,72,73,74): return ("Orthorhombisch","I","oI","Orthorhombisch I")
   185|        return ("Orthorhombisch","P","oP","Orthorhombisch P")
   186|    elif 75 <= n <= 82:  return ("Tetragonal",     "P", "tP", "Tetragonal P")
   187|    elif 83 <= n <= 88:  return ("Tetragonal",     "P", "tP", "Tetragonal P")
   188|    elif 89 <= n <= 98:  return ("Tetragonal",     "P", "tP", "Tetragonal P")
   189|    elif 99 <= n <= 110: return ("Tetragonal",     "P", "tP", "Tetragonal P")
   190|    elif 111<= n <=122:  return ("Tetragonal",     "P", "tP", "Tetragonal P")
   191|    elif 123<= n <=142:
   192|        if n in (139,140,141,142): return ("Tetragonal","I","tI","Tetragonal I")
   193|        return ("Tetragonal","P","tP","Tetragonal P")
   194|    elif 143<= n <=146:  return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
   195|    elif 147<= n <=148:  return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
   196|    elif 149<= n <=155:  return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
   197|    elif 156<= n <=161:  return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
   198|    elif 162<= n <=167:
   199|        if n in (160,161,166,167): return ("Rhomboedrisch","R","hR","Rhomboedrisch R")
   200|        return ("Trigonal/Rhomboedrisch","P","hP","Hexagonal P")
   201|    elif 168<= n <=176:  return ("Hexagonal",      "P", "hP", "Hexagonal P")
   202|    elif 177<= n <=194:  return ("Hexagonal",      "P", "hP", "Hexagonal P")
   203|    elif 195<= n <=199:  return ("Kubisch",        "P", "cP", "Kubisch P")
   204|    elif 200<= n <=206:  return ("Kubisch",        "P", "cP", "Kubisch P")
   205|    elif 207<= n <=214:
   206|        if n in (207,208,212,213): return ("Kubisch","P","cP","Kubisch P")
   207|        if n in (209,210):         return ("Kubisch","F","cF","Kubisch F (FCC)")
   208|        if n in (211,214):         return ("Kubisch","I","cI","Kubisch I (BCC)")
   209|        return ("Kubisch","P","cP","Kubisch P")
   210|    elif 215<= n <=220:
   211|        if n in (216,219): return ("Kubisch","F","cF","Kubisch F (FCC)")
   212|        if n in (217,220): return ("Kubisch","I","cI","Kubisch I (BCC)")
   213|        return ("Kubisch","P","cP","Kubisch P")
   214|    elif 221<= n <=230:
   215|        if n in (225,226,227,228): return ("Kubisch","F","cF","Kubisch F (FCC)")
   216|        if n in (229,230):         return ("Kubisch","I","cI","Kubisch I (BCC)")
   217|        return ("Kubisch","P","cP","Kubisch P")
   218|    return ("Unbekannt","P","??","Unbekannt")
   219|
   220|def bravais_from_cif_data(data: dict):
   221|    """Determine Bravais lattice from parsed CIF data."""
   222|    # Try space-group number first
   223|    sg_num = None
   224|    for key in ("_symmetry_int_tables_number","_space_group_it_number",
   225|                "_symmetry_int_tables_number"):
   226|        v = data.get(key)
   227|        if v and v not in ("?","."):
   228|            try: sg_num = int(v); break
   229|            except: pass
   230|
   231|    if sg_num:
   232|        return sg_to_bravais(sg_num), sg_num
   233|
   234|    # Fallback: derive from cell setting + centering
   235|    setting = (data.get("_symmetry_cell_setting") or
   236|               data.get("_space_group_crystal_system") or "").strip().lower()
   237|    hm = (data.get("_symmetry_space_group_name_h-m") or
   238|          data.get("_space_group_name_h-m_alt") or "").strip().strip("'\"")
   239|
   240|    # centering from first letter of H-M symbol
   241|    cent = "P"
   242|    if hm:
   243|        first = hm[0].upper()
   244|        if first in ("P","I","F","C","A","B","R"):
   245|            cent = first
   246|
   247|    sys_map = {
   248|        "triclinic":    "Triklin",
   249|        "monoclinic":   "Monoklin",
   250|        "orthorhombic": "Orthorhombisch",
   251|        "tetragonal":   "Tetragonal",
   252|        "trigonal":     "Trigonal/Rhomboedrisch",
   253|        "hexagonal":    "Hexagonal",
   254|        "cubic":        "Kubisch",
   255|        "rhombohedral": "Rhomboedrisch",
   256|    }
   257|    system = sys_map.get(setting, "Unbekannt")
   258|
   259|    sym_map = {
   260|        ("Kubisch","P"): ("cP","Kubisch P"),
   261|        ("Kubisch","I"): ("cI","Kubisch I (BCC)"),
   262|        ("Kubisch","F"): ("cF","Kubisch F (FCC)"),
   263|        ("Tetragonal","P"): ("tP","Tetragonal P"),
   264|        ("Tetragonal","I"): ("tI","Tetragonal I"),
   265|        ("Orthorhombisch","P"): ("oP","Orthorhombisch P"),
   266|        ("Orthorhombisch","C"): ("oC","Orthorhombisch C"),
   267|        ("Orthorhombisch","I"): ("oI","Orthorhombisch I"),
   268|        ("Orthorhombisch","F"): ("oF","Orthorhombisch F"),
   269|        ("Monoklin","P"): ("mP","Monoklin P"),
   270|        ("Monoklin","C"): ("mC","Monoklin C"),
   271|        ("Triklin","P"):  ("aP","Triklin P"),
   272|        ("Hexagonal","P"):("hP","Hexagonal P"),
   273|        ("Rhomboedrisch","R"):("hR","Rhomboedrisch R"),
   274|        ("Trigonal/Rhomboedrisch","P"):("hP","Hexagonal P"),
   275|        ("Trigonal/Rhomboedrisch","R"):("hR","Rhomboedrisch R"),
   276|    }
   277|    sym, name = sym_map.get((system, cent), ("??", f"{system} {cent}"))
   278|    return (system, cent, sym, name), None
   279|
   280|
   281|# ════════════════════════════════════════════════════════════════════════════
   282|# LATTICE VECTOR BUILDER
   283|# ════════════════════════════════════════════════════════════════════════════
   284|
   285|def cell_vectors(a, b, c, alpha_deg, beta_deg, gamma_deg):
   286|    al = math.radians(alpha_deg)
   287|    be = math.radians(beta_deg)
   288|    ga = math.radians(gamma_deg)
   289|    cos_a, cos_b, cos_g = math.cos(al), math.cos(be), math.cos(ga)
   290|    sin_g = math.sin(ga)
   291|    v = math.sqrt(max(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2*cos_a*cos_b*cos_g, 0))
   292|    a1 = np.array([a, 0, 0])
   293|    a2 = np.array([b*cos_g, b*sin_g, 0])
   294|    a3 = np.array([c*cos_b, c*(cos_a - cos_b*cos_g)/sin_g, c*v/sin_g])
   295|    return a1, a2, a3
   296|
   297|def centering_fracs(centering: str):
   298|    pts = [(0,0,0)]
   299|    if centering == "I":  pts += [(0.5,0.5,0.5)]
   300|    elif centering == "F":pts += [(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)]
   301|    elif centering == "C":pts += [(0.5,0.5,0)]
   302|    elif centering == "A":pts += [(0,0.5,0.5)]
   303|    elif centering == "B":pts += [(0.5,0,0.5)]
   304|    elif centering == "R":pts += [(2/3,1/3,1/3),(1/3,2/3,2/3)]
   305|    return pts
   306|
   307|def cell_edges(a1, a2, a3):
   308|    corners = {(i,j,k): i*a1+j*a2+k*a3 for i,j,k in itertools.product([0,1],repeat=3)}
   309|    edges = []
   310|    for (i,j,k) in corners:
   311|        for di,dj,dk in [(1,0,0),(0,1,0),(0,0,1)]:
   312|            ni,nj,nk = i+di,j+dj,k+dk
   313|            if ni<=1 and nj<=1 and nk<=1:
   314|                edges.append([corners[(i,j,k)], corners[(ni,nj,nk)]])
   315|    return edges, corners
   316|
   317|
   318|# ════════════════════════════════════════════════════════════════════════════
   319|# COLORS
   320|# ════════════════════════════════════════════════════════════════════════════
   321|
   322|SYSTEM_COLORS = {
   323|    "Triklin":                  "#e74c3c",
   324|    "Monoklin":                 "#e67e22",
   325|    "Orthorhombisch":           "#27ae60",
   326|    "Tetragonal":               "#2980b9",
   327|    "Rhomboedrisch":            "#8e44ad",
   328|    "Trigonal/Rhomboedrisch":   "#9b59b6",
   329|    "Hexagonal":                "#16a085",
   330|    "Kubisch":                  "#00e5c8",
   331|    "Unbekannt":                "#888888",
   332|}
   333|
   334|BRAVAIS_INFO = {
   335|    "cP": {"desc": "Einfach kubisch. Atome nur an den Würfelecken.", "points": 1},
   336|    "cI": {"desc": "Raumzentrierter Würfel (BCC). Zusätzlicher Atom im Zentrum.", "points": 2},
   337|    "cF": {"desc": "Flächenzentrierter Würfel (FCC). Dichteste Kugelpackung.", "points": 4},
   338|    "tP": {"desc": "Tetragonale Zelle (primitiv). Quadratische Basis, gestreckte c-Achse.", "points": 1},
   339|    "tI": {"desc": "Tetragonal raumzentriert. Zusätzlicher Punkt im Zentrum.", "points": 2},
   340|    "oP": {"desc": "Orthorhombisch primitiv. Drei ungleiche rechtwinklige Achsen.", "points": 1},
   341|    "oC": {"desc": "Orthorhombisch C-zentriert. Zusatzpunkte auf ab-Flächen.", "points": 2},
   342|    "oI": {"desc": "Orthorhombisch raumzentriert.", "points": 2},
   343|    "oF": {"desc": "Orthorhombisch flächenzentriert.", "points": 4},
   344|    "hP": {"desc": "Hexagonal primitiv. 120°-Winkel in der Basisebene.", "points": 1},
   345|    "hR": {"desc": "Rhomboedrisch. Würfel entlang Raumdiagonale deformiert.", "points": 1},
   346|    "mP": {"desc": "Monoklin primitiv. Ein schiefer Winkel (β≠90°).", "points": 1},
   347|    "mC": {"desc": "Monoklin C-zentriert.", "points": 2},
   348|    "aP": {"desc": "Triklin. Keine Einschränkungen – niedrigste Symmetrie.", "points": 1},
   349|}
   350|
   351|
   352|# ════════════════════════════════════════════════════════════════════════════
   353|# 3D VISUALISATION
   354|# ════════════════════════════════════════════════════════════════════════════
   355|
   356|def draw_bravais_3d(a1, a2, a3, centering, color,
   357|                    elev=22, azim=35, supercell=1,
   358|                    show_vectors=True, show_planes=False,
   359|                    atom_size=120, figsize=(8,7),
   360|                    dark=True):
   361|
   362|    bg  = "#07080f" if dark else "#f5f4f0"
   363|    fg  = "#c8d4f8" if dark else "#1a1a2e"
   364|    grid_col = "#1e2555" if dark else "#d0cfc8"
   365|
   366|    fig = plt.figure(figsize=figsize, facecolor=bg)
   367|    ax  = fig.add_subplot(111, projection="3d", facecolor=bg)
   368|    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
   369|        pane.fill = False
   370|        pane.set_edgecolor(grid_col)
   371|        pane.set_alpha(0.15)
   372|    ax.grid(False)
   373|    ax.set_axis_off()
   374|
   375|    sc = supercell
   376|    # draw cells
   377|    for ti,tj,tk in itertools.product(range(sc), repeat=3):
   378|        origin = ti*a1 + tj*a2 + tk*a3
   379|        edges, _ = cell_edges(a1, a2, a3)
   380|        segs = [[e[0]+origin, e[1]+origin] for e in edges]
   381|        lc = Line3DCollection(segs, colors=color, linewidths=1.4,
   382|                              alpha=0.5 if sc > 1 else 0.7)
   383|        ax.add_collection3d(lc)
   384|
   385|        # optional face planes (first cell only)
   386|        if show_planes and ti==0 and tj==0 and tk==0:
   387|            corners = {(i,j,k): i*a1+j*a2+k*a3
   388|                       for i,j,k in itertools.product([0,1],repeat=3)}
   389|            faces = [
   390|                [corners[(0,0,0)],corners[(1,0,0)],corners[(1,1,0)],corners[(0,1,0)]],
   391|                [corners[(0,0,0)],corners[(1,0,0)],corners[(1,0,1)],corners[(0,0,1)]],
   392|                [corners[(0,0,0)],corners[(0,1,0)],corners[(0,1,1)],corners[(0,0,1)]],
   393|            ]
   394|            poly = Poly3DCollection(faces, alpha=0.06, facecolor=color, edgecolor="none")
   395|            ax.add_collection3d(poly)
   396|
   397|    # lattice points
   398|    cent_pts = centering_fracs(centering)
   399|    rng = range(sc + 1)
   400|    shown = set()
   401|    for i,j,k in itertools.product(rng, repeat=3):
   402|        base = i*a1 + j*a2 + k*a3
   403|        for (fi,fj,fk) in cent_pts:
   404|            pt = base + fi*a1 + fj*a2 + fk*a3
   405|            key = tuple(np.round(pt, 3))
   406|            if key in shown: continue
   407|            shown.add(key)
   408|            # distinguish corner vs centering points
   409|            is_corner = (fi == 0 and fj == 0 and fk == 0)
   410|            s    = atom_size if is_corner else atom_size * 0.75
   411|            ecol = "white" if dark else "#333"
   412|            ax.scatter(*pt, s=s, c=color, edgecolors=ecol,
   413|                       linewidths=0.8, depthshade=True, zorder=5,
   414|                       alpha=1.0 if is_corner else 0.85)
   415|
   416|    # lattice vectors (only for sc==1)
   417|    if show_vectors and sc == 1:
   418|        vcols = ["#ff4444", "#44ff88", "#4488ff"]
   419|        vlbls = ["a", "b", "c"]
   420|        for v, vc, vl in zip([a1, a2, a3], vcols, vlbls):
   421|            ax.quiver(0, 0, 0, *v, color=vc, arrow_length_ratio=0.12,
   422|                      linewidth=2.2, alpha=0.95)
   423|            off = v * 1.15
   424|            ax.text(*off, vl, color=vc, fontsize=13,
   425|                    fontfamily="monospace", fontweight="bold",
   426|                    ha="center", va="center")
   427|
   428|    ax.view_init(elev=elev, azim=azim)
   429|
   430|    # axis limits
   431|    pts_all = [i*a1 + j*a2 + k*a3
   432|               for i,j,k in itertools.product(range(sc+1), repeat=3)]
   433|    coords = np.array(pts_all)
   434|    pad = max(np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(a3)) * 0.25
   435|    mn, mx = coords.min(axis=0) - pad, coords.max(axis=0) + pad
   436|    ax.set_xlim(mn[0], mx[0])
   437|    ax.set_ylim(mn[1], mx[1])
   438|    ax.set_zlim(mn[2], mx[2])
   439|
   440|    plt.tight_layout(pad=0)
   441|    return fig
   442|
   443|
   444|# ════════════════════════════════════════════════════════════════════════════
   445|# SIDEBAR
   446|# ════════════════════════════════════════════════════════════════════════════
   447|
   448|with st.sidebar:
   449|    st.markdown("## ⬡ BRAVAIS AUS CIF")
   450|    st.markdown("---")
   451|
   452|    cif_text = None
   453|    cif_name = ""
   454|
   455|    # ── 1. CLI argument (--cif path) ─────────────────────────────────────────
   456|    if PRELOAD_PATH and PRELOAD_PATH.exists():
   457|        cif_text = PRELOAD_PATH.read_text(encoding="utf-8", errors="replace")
   458|        cif_name = PRELOAD_PATH.name
   459|
   460|    # ── 2. Auto-search parent folders ────────────────────────────────────────
   461|    if cif_text is None:
   462|        script_dir = Path(__file__).resolve().parent
   463|        search_dirs = [
   464|            script_dir.parent,          # ../
   465|            script_dir.parent.parent,   # ../../
   466|            script_dir,                 # ./
   467|        ]
   468|        found_cifs = []
   469|        for d in search_dirs:
   470|            found_cifs += sorted(d.glob("*.cif"))
   471|        found_cifs = list(dict.fromkeys(found_cifs))  # deduplicate
   472|
   473|        if found_cifs:
   474|            cif_names = [f.name for f in found_cifs]
   475|            st.markdown("**CIF-Dateien gefunden**")
   476|            selected_idx = st.selectbox(
   477|                "Datei auswählen",
   478|                range(len(cif_names)),
   479|                format_func=lambda i: cif_names[i],
   480|                index=0,
   481|            )
   482|            chosen = found_cifs[selected_idx]
   483|            cif_text = chosen.read_text(encoding="utf-8", errors="replace")
   484|            cif_name = chosen.name
   485|            st.caption(f"📂 {chosen.parent}")
   486|
   487|    # ── 3. Fallback: manual upload ────────────────────────────────────────────
   488|    if cif_text is None:
   489|        st.markdown("**Keine CIF im Parent-Ordner gefunden**")
   490|        uploaded = st.file_uploader("CIF manuell hochladen", type=["cif"])
   491|        if uploaded:
   492|            cif_text = uploaded.read().decode("utf-8", errors="replace")
   493|            cif_name = uploaded.name
   494|
   495|    st.markdown("---")
   496|    st.markdown("**Visualisierung**")
   497|    elev     = st.slider("Elevation",  -90, 90, 22, key="elev")
   498|    azim     = st.slider("Azimut",     0, 360, 35, key="azim")
   499|    sc       = st.slider("Superzelle", 1, 3, 1, key="sc")
   500|    show_vec = st.checkbox("Gittervektoren a, b, c", True)
   501|