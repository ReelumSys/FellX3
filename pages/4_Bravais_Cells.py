"""
╔══════════════════════════════════════════════════════╗
║        BRAVAIS LATTICE VIEWER  ·  Streamlit App      ║
║  Pure-Python CIF parser · Plotly 3-D · No gemmi req  ║
╚══════════════════════════════════════════════════════╝

Install:
    pip install streamlit plotly numpy pandas

Run:
    streamlit run bravais_viewer.py
"""

import re, math, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from main import cif_file2000

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bravais Lattice Viewer",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
#  GLOBAL CSS  –  dark crystallography lab aesthetic
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

/* ── root ── */
html, body, [data-testid="stApp"] {
    background: #070b14 !important;
    color: #c8d8f0 !important;
    font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #0b1120 !important;
    border-right: 1px solid #1e3a5f44;
}
[data-testid="stSidebarContent"] { padding: 1.2rem 1rem !important; }

/* ── headings ── */
h1 { font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
     letter-spacing: 2px !important; color: #7eb8f7 !important; }
h2, h3 { font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
          color: #9dceff !important; letter-spacing: 1px !important; }

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #1e5a9f88 !important;
    border-radius: 8px !important;
    background: #0d1826 !important;
}

/* ── metric cards ── */
.crystal-card {
    background: linear-gradient(135deg, #0d1826 0%, #111e30 100%);
    border: 1px solid #1e4070;
    border-radius: 10px;
    padding: 18px 20px;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
}
.crystal-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent);
}
.crystal-card h4 { margin: 0 0 4px 0; font-family: 'Rajdhani', sans-serif;
                    font-size: 0.75rem; text-transform: uppercase;
                    letter-spacing: 2px; color: #5a8ab8; }
.crystal-card .val { font-family: 'Share Tech Mono', monospace;
                      font-size: 1.6rem; color: var(--accent); margin: 0; }
.crystal-card .sub { font-size: 0.78rem; color: #4a6a8a; margin: 4px 0 0 0; }

/* ── bravais badge ── */
.bravais-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 4px;
    background: var(--bg);
    color: var(--fg);
    border: 2px solid var(--fg);
    margin-bottom: 8px;
}

/* ── param table ── */
.param-row {
    display: flex; justify-content: space-between;
    padding: 5px 0; border-bottom: 1px solid #1a2e48;
    font-family: 'Share Tech Mono', monospace; font-size: 0.85rem;
}
.param-row span:first-child { color: #4a8abf; }
.param-row span:last-child  { color: #d0e8ff; }

/* ── section divider ── */
.sect-divider {
    border: none; border-top: 1px solid #1a3456;
    margin: 18px 0;
}

/* ── atom element chip ── */
.elem-chip {
    display: inline-block; padding: 2px 10px;
    border-radius: 30px; margin: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem; font-weight: 600;
}

/* ── plotly container ── */
.js-plotly-plot .plotly .modebar {
    background: transparent !important;
}

/* ── selectbox & sliders ── */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] > div { color: #c8d8f0 !important; }

/* ── info box ── */
.info-box {
    background: #0d1f35;
    border-left: 3px solid #2a6496;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.85rem;
    line-height: 1.6;
}

/* hide streamlit default footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  BRAVAIS LATTICE DATABASE
# ══════════════════════════════════════════════════════
BRAVAIS_DB = {
    "aP": {"system":"Triclinic",     "centering":"Primitive",     "color":"#e74c3c", "bg":"#2d0b0b"},
    "mP": {"system":"Monoclinic",    "centering":"Primitive",     "color":"#e67e22", "bg":"#2d1a0b"},
    "mS": {"system":"Monoclinic",    "centering":"C-centered",    "color":"#f0a500", "bg":"#2d210b"},
    "oP": {"system":"Orthorhombic",  "centering":"Primitive",     "color":"#2ecc71", "bg":"#0b2d1a"},
    "oS": {"system":"Orthorhombic",  "centering":"C-centered",    "color":"#27ae60", "bg":"#0b2d18"},
    "oF": {"system":"Orthorhombic",  "centering":"Face-centered", "color":"#1abc9c", "bg":"#0b2d28"},
    "oI": {"system":"Orthorhombic",  "centering":"Body-centered", "color":"#16a085", "bg":"#0b2522"},
    "tP": {"system":"Tetragonal",    "centering":"Primitive",     "color":"#3498db", "bg":"#0b1e2d"},
    "tI": {"system":"Tetragonal",    "centering":"Body-centered", "color":"#2980b9", "bg":"#0b1a2d"},
    "hR": {"system":"Rhombohedral",  "centering":"Rhombohedral",  "color":"#a569bd", "bg":"#1e0b2d"},
    "hP": {"system":"Hexagonal",     "centering":"Primitive",     "color":"#8e44ad", "bg":"#1a0b2d"},
    "cP": {"system":"Cubic",         "centering":"Primitive",     "color":"#7eb8f7", "bg":"#0b1e30"},
    "cF": {"system":"Cubic",         "centering":"Face-centered", "color":"#5dade2", "bg":"#0b1c2e"},
    "cI": {"system":"Cubic",         "centering":"Body-centered", "color":"#85c1e9", "bg":"#0b1e32"},
}

ELEMENT_COLORS = {
    "H":"#ffffff","C":"#909090","N":"#3050F8","O":"#FF0D0D","F":"#90E050",
    "Na":"#AB5CF2","Mg":"#8AFF00","Al":"#BFA6A6","Si":"#F0C8A0","P":"#FF8000",
    "S":"#FFFF30","Cl":"#1FF01F","K":"#8F40D4","Ca":"#3DFF00","Fe":"#E06633",
    "Cu":"#C88033","Zn":"#7D80B0","Br":"#A62929","Ag":"#C0C0C0","I":"#940094",
    "Ba":"#00C900","Pb":"#575961","DEFAULT":"#ff88aa",
}

ELEMENT_RADII = {
    "H":0.53,"C":0.77,"N":0.75,"O":0.73,"F":0.71,"Na":1.86,"Mg":1.60,
    "Al":1.43,"Si":1.17,"P":1.10,"S":1.04,"Cl":0.99,"K":2.27,"Ca":1.97,
    "Fe":1.26,"Cu":1.28,"Zn":1.22,"Br":1.14,"DEFAULT":1.0,
}


# ══════════════════════════════════════════════════════
#  CIF PARSER  (pure Python – no gemmi)
# ══════════════════════════════════════════════════════
def _strip_esd(val: str) -> str:
    """Remove estimated-standard-deviation parentheses: 8.897(3) → 8.897"""
    return re.sub(r'\([^)]*\)', '', val).strip()

def parse_cif(text: str) -> dict:
    """Minimal CIF parser; returns dict of key→value / key→[list] pairs."""
    # normalise line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # remove comments
    text = re.sub(r'#[^\n]*', '', text)

    data = {}

    # ── scalar values  _key   value ──────────────────
    scalar_re = re.compile(
        r"(_\S+)\s+"                       # key
        r"(?:'([^']*)'|\"([^\"]*)\"|(\S+))", # value: quoted or bare
        re.MULTILINE
    )
    for m in scalar_re.finditer(text):
        key = m.group(1).lower()
        val = m.group(2) or m.group(3) or m.group(4)
        data[key] = val

    # ── semicolon text-fields ─────────────────────────
    # Replace them before loop parsing so they don't confuse things
    text_clean = re.sub(r'\n;.*?\n;', ' ? ', text, flags=re.DOTALL)

    # ── loop_ blocks ──────────────────────────────────
    loop_blocks = re.split(r'\bloop_\b', text_clean, flags=re.IGNORECASE)
    for block in loop_blocks[1:]:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        headers = []
        value_lines = []
        mode = 'headers'
        for line in lines:
            if mode == 'headers':
                if line.startswith('_'):
                    headers.append(line.lower())
                else:
                    mode = 'values'
                    value_lines.append(line)
            else:
                if line.startswith('_') or line.lower().startswith('loop_'):
                    break
                value_lines.append(line)

        if not headers:
            continue

        # tokenise value lines
        tokens = []
        for vl in value_lines:
            # handle quoted strings
            for tok in re.findall(r"'[^']*'|\"[^\"]*\"|\S+", vl):
                tokens.append(tok.strip("'\""))

        n = len(headers)
        if n == 0 or len(tokens) < n:
            continue

        rows = [tokens[i:i+n] for i in range(0, len(tokens) - n + 1, n)]
        for i, h in enumerate(headers):
            data[h] = [row[i] for row in rows if i < len(row)]

    return data


def extract_structure(cif_data: dict) -> dict:
    """Pull out the crystal structure info we need from parsed CIF dict."""
    def flt(key, default=0.0):
        v = cif_data.get(key, str(default))
        if isinstance(v, list): v = v[0]
        try:    return float(_strip_esd(str(v)))
        except: return default

    a = flt('_cell_length_a',     8.0)
    b = flt('_cell_length_b',     a)
    c = flt('_cell_length_c',     a)
    alpha = flt('_cell_angle_alpha', 90.0)
    beta  = flt('_cell_angle_beta',  90.0)
    gamma = flt('_cell_angle_gamma', 90.0)
    volume= flt('_cell_volume',   0.0)

    sg_hm   = cif_data.get('_symmetry_space_group_name_h-m','') or \
              cif_data.get('_space_group_name_h-m_alt','')
    sg_hall = cif_data.get('_symmetry_space_group_name_hall','') or \
              cif_data.get('_space_group_name_hall','')
    sg_num  = cif_data.get('_symmetry_int_tables_number','') or \
              cif_data.get('_space_group_it_number','')
    cell_setting = cif_data.get('_symmetry_cell_setting','')

    if isinstance(sg_hm, list):   sg_hm = sg_hm[0]
    if isinstance(sg_hall, list): sg_hall = sg_hall[0]
    if isinstance(sg_num, list):  sg_num  = sg_num[0]
    if isinstance(cell_setting, list): cell_setting = cell_setting[0]

    # Atomic sites
    labels  = cif_data.get('_atom_site_label', [])
    fx_list = cif_data.get('_atom_site_fract_x', [])
    fy_list = cif_data.get('_atom_site_fract_y', [])
    fz_list = cif_data.get('_atom_site_fract_z', [])
    elem_list = cif_data.get('_atom_site_type_symbol', labels)

    atoms = []
    for i in range(len(labels)):
        try:
            fx = float(_strip_esd(fx_list[i])) if i < len(fx_list) else 0.0
            fy = float(_strip_esd(fy_list[i])) if i < len(fy_list) else 0.0
            fz = float(_strip_esd(fz_list[i])) if i < len(fz_list) else 0.0
            el = (elem_list[i] if i < len(elem_list) else labels[i])
            el = re.sub(r'[^A-Za-z]', '', el)[:2].capitalize()
            atoms.append({"label": labels[i], "element": el,
                          "fx": fx, "fy": fy, "fz": fz})
        except Exception:
            continue

    return dict(a=a,b=b,c=c,alpha=alpha,beta=beta,gamma=gamma,
                volume=volume,sg_hm=str(sg_hm).strip("'\" "),
                sg_hall=str(sg_hall).strip("'\" "),
                sg_num=str(sg_num).strip(),
                cell_setting=str(cell_setting).strip("'\" "),
                atoms=atoms)


# ══════════════════════════════════════════════════════
#  BRAVAIS CLASSIFICATION  (from space-group H-M symbol)
# ══════════════════════════════════════════════════════
SYSTEM_TO_BRAVAIS = {
    "triclinic":    {"P": "aP"},
    "monoclinic":   {"P": "mP", "C": "mS", "A": "mS", "I": "mS"},
    "orthorhombic": {"P": "oP", "C": "oS", "A": "oS", "B": "oS",
                     "F": "oF", "I": "oI"},
    "tetragonal":   {"P": "tP", "I": "tI"},
    "trigonal":     {"P": "hP", "R": "hR"},
    "rhombohedral": {"R": "hR", "P": "hP"},
    "hexagonal":    {"P": "hP"},
    "cubic":        {"P": "cP", "F": "cF", "I": "cI"},
}

def detect_bravais(sg_hm: str, cell_setting: str) -> str:
    """Determine the 2-letter Bravais symbol from the H-M space group name."""
    sg_clean = sg_hm.strip("'\" ")
    centering = sg_clean[0].upper() if sg_clean else "P"

    setting_lc = cell_setting.lower()
    if not setting_lc:
        # Infer from parameters or from H-M symbol prefix
        hm_upper = sg_clean.upper()
        for sys in SYSTEM_TO_BRAVAIS:
            if sys.upper() in hm_upper:
                setting_lc = sys
                break

    # Try every system
    for sys, cmap in SYSTEM_TO_BRAVAIS.items():
        if sys in setting_lc or setting_lc.startswith(sys[:4]):
            return cmap.get(centering, list(cmap.values())[0])

    # Last resort: guess from cell angles
    return "aP"


def classify_from_structure(s: dict) -> str:
    bravais = detect_bravais(s["sg_hm"], s["cell_setting"])
    # Sanity-check: if we got nothing, fall back by cell shape
    if bravais not in BRAVAIS_DB:
        a,b,c = s["a"],s["b"],s["c"]
        al,be,ga = s["alpha"],s["beta"],s["gamma"]
        all90 = all(abs(x-90)<0.5 for x in [al,be,ga])
        if abs(a-b)<0.01 and abs(b-c)<0.01 and all90: return "cP"
        if abs(a-b)<0.01 and all90: return "tP"
        if all90: return "oP"
        return "aP"
    return bravais


# ══════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════
def cell_matrix(a,b,c,alpha_d,beta_d,gamma_d):
    al,be,ga = math.radians(alpha_d),math.radians(beta_d),math.radians(gamma_d)
    ax = a
    bx = b*math.cos(ga);  by = b*math.sin(ga)
    cx = c*math.cos(be)
    cy = c*(math.cos(al)-math.cos(be)*math.cos(ga))/math.sin(ga)
    cz = math.sqrt(max(c**2-cx**2-cy**2, 0.0))
    return np.array([[ax,0,0],[bx,by,0],[cx,cy,cz]])

def frac_to_cart(frac, M):
    return M[0]*frac[0] + M[1]*frac[1] + M[2]*frac[2]

def cell_edges(M, origin=None):
    O = origin if origin is not None else np.zeros(3)
    a,b,c = M[0],M[1],M[2]
    return [
        (O,     O+a),   (O,     O+b),   (O,     O+c),
        (O+a,   O+a+b), (O+a,   O+a+c),
        (O+b,   O+a+b), (O+b,   O+b+c),
        (O+c,   O+a+c), (O+c,   O+b+c),
        (O+a+b, O+a+b+c),(O+a+c,O+a+b+c),(O+b+c,O+a+b+c),
    ]

def centering_fracs(symbol):
    c = symbol[1] if len(symbol)>1 else "P"
    base = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
    extra = {"I":[(0.5,0.5,0.5)],
             "F":[(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
             "S":[(0.5,0.5,0)], "C":[(0.5,0.5,0)],
             "R":[(2/3,1/3,1/3),(1/3,2/3,2/3)]}.get(c,[])
    return base + extra


# ══════════════════════════════════════════════════════
#  PLOTLY FIGURE
# ══════════════════════════════════════════════════════
def build_figure(s: dict, bravais: str, opts: dict) -> go.Figure:
    info  = BRAVAIS_DB[bravais]
    color = info["color"]
    M     = cell_matrix(s["a"],s["b"],s["c"],s["alpha"],s["beta"],s["gamma"])
    sc    = opts.get("supercell", 1)
    traces = []

    # ── unit cell edges (supercell) ───────────────────
    for i in range(sc):
        for j in range(sc):
            for k in range(sc):
                origin = i*M[0]+j*M[1]+k*M[2]
                for (p1,p2) in cell_edges(M, origin):
                    traces.append(go.Scatter3d(
                        x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]],
                        mode="lines",
                        line=dict(color=color, width=2.5),
                        showlegend=False, hoverinfo="skip",
                    ))

    # ── centering lattice points ──────────────────────
    cf = centering_fracs(bravais)
    all_pts = []
    for i in range(sc):
        for j in range(sc):
            for k in range(sc):
                off = i*M[0]+j*M[1]+k*M[2]
                for (fx,fy,fz) in cf:
                    pt = off + frac_to_cart((fx,fy,fz), M)
                    all_pts.append(pt)
    if all_pts:
        px,py,pz = zip(*[(p[0],p[1],p[2]) for p in all_pts])
        traces.append(go.Scatter3d(
            x=px,y=py,z=pz, mode="markers",
            marker=dict(size=7, color=color, opacity=0.9,
                        line=dict(color="#0a0f1a", width=1)),
            name=f"Lattice nodes ({info['centering']})",
            hovertemplate="<b>Lattice point</b><br>x=%{x:.3f} Å<br>y=%{y:.3f} Å<br>z=%{z:.3f} Å<extra></extra>",
        ))

    # ── basis vectors a b c ───────────────────────────
    O = np.zeros(3)
    for vec, lbl, col in zip(M, ["a","b","c"], ["#ff4e4e","#4ecc71","#4e9eff"]):
        # draw arrow as thick line + cone
        traces.append(go.Scatter3d(
            x=[O[0],vec[0]], y=[O[1],vec[1]], z=[O[2],vec[2]],
            mode="lines+text",
            line=dict(color=col, width=5),
            text=["", f"<b>{lbl}</b>"],
            textposition="top center",
            textfont=dict(size=14, color=col, family="Share Tech Mono"),
            name=f"Axis {lbl}",
        ))

    # ── atoms ─────────────────────────────────────────
    if opts.get("show_atoms", True) and s["atoms"]:
        elem_groups: dict[str, list] = {}
        for atom in s["atoms"]:
            for i in range(sc):
                for j in range(sc):
                    for k in range(sc):
                        off = i*M[0]+j*M[1]+k*M[2]
                        cart = off + frac_to_cart((atom["fx"],atom["fy"],atom["fz"]), M)
                        el = atom["element"]
                        elem_groups.setdefault(el, []).append(
                            (cart, atom["label"])
                        )

        for el, pts in elem_groups.items():
            ecol = ELEMENT_COLORS.get(el, ELEMENT_COLORS["DEFAULT"])
            rad  = ELEMENT_RADII.get(el, ELEMENT_RADII["DEFAULT"])
            xs,ys,zs,lbls = zip(*[(p[0],p[1],p[2],lb) for (p,lb) in pts])
            traces.append(go.Scatter3d(
                x=xs,y=ys,z=zs, mode="markers",
                marker=dict(size=rad*3.5+2, color=ecol, opacity=0.85,
                            line=dict(color="#0a0f1a", width=0.5)),
                name=el,
                hovertemplate=f"<b>%{{text}}</b> ({el})<br>"
                              f"x=%{{x:.4f}} Å<br>y=%{{y:.4f}} Å<br>z=%{{z:.4f}} Å<extra></extra>",
                text=lbls,
            ))

    # ── layout ────────────────────────────────────────
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#070b14",
        scene=dict(
            bgcolor="#070b14",
            xaxis=dict(showbackground=False, showgrid=False,
                       zeroline=False, color="#2a4a6a", title="x (Å)"),
            yaxis=dict(showbackground=False, showgrid=False,
                       zeroline=False, color="#2a4a6a", title="y (Å)"),
            zaxis=dict(showbackground=False, showgrid=False,
                       zeroline=False, color="#2a4a6a", title="z (Å)"),
            aspectmode="data",
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        height=580,
        legend=dict(
            bgcolor="rgba(7,11,20,0.85)",
            bordercolor="#1e3a5f",
            borderwidth=1,
            font=dict(family="Share Tech Mono", size=11, color="#7eb8f7"),
        ),
        uirevision="constant",
    )
    return fig


def build_all14_figure() -> go.Figure:
    """Schematic overview of all 14 Bravais lattices."""
    representative = [
        ("aP", 1.0,1.2,1.5, 70,80,85),
        ("mP", 1.0,1.2,1.5, 90,110,90),
        ("mS", 1.0,1.2,1.5, 90,110,90),
        ("oP", 1.0,1.2,1.5, 90,90,90),
        ("oS", 1.0,1.2,1.5, 90,90,90),
        ("oF", 1.0,1.2,1.5, 90,90,90),
        ("oI", 1.0,1.2,1.5, 90,90,90),
        ("tP", 1.0,1.0,1.5, 90,90,90),
        ("tI", 1.0,1.0,1.5, 90,90,90),
        ("hR", 1.0,1.0,1.0, 75,75,75),
        ("hP", 1.0,1.0,1.5, 90,90,120),
        ("cP", 1.0,1.0,1.0, 90,90,90),
        ("cF", 1.0,1.0,1.0, 90,90,90),
        ("cI", 1.0,1.0,1.0, 90,90,90),
    ]
    traces = []
    annots = []
    cols = 7
    sx, sz = 3.4, 3.4

    for idx,(sym,a,b,c,al,be,ga) in enumerate(representative):
        ri,ci = divmod(idx, cols)
        off = np.array([ci*sx, 0.0, -ri*sz])
        M   = cell_matrix(a,b,c,al,be,ga)
        info = BRAVAIS_DB[sym]
        col  = info["color"]

        for (p1,p2) in cell_edges(M, off):
            traces.append(go.Scatter3d(
                x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]],
                mode="lines", line=dict(color=col, width=2),
                showlegend=False, hoverinfo="skip",
            ))

        pts = [off + frac_to_cart(f,M) for f in centering_fracs(sym)]
        lx,ly,lz = zip(*[(p[0],p[1],p[2]) for p in pts])
        traces.append(go.Scatter3d(
            x=lx,y=ly,z=lz, mode="markers",
            marker=dict(size=5, color=col, opacity=0.9),
            name=sym,
            hovertemplate=f"<b>{sym}</b><br>{info['system']}<br>{info['centering']}<extra></extra>",
        ))

        ctr = off + 0.5*(M[0]+M[1]+M[2])
        annots.append(dict(
            x=ctr[0], y=ctr[1]+0.2, z=ctr[2]+1.3,
            text=f"<b>{sym}</b>",
            showarrow=False,
            font=dict(size=12, color=col, family="Share Tech Mono"),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#070b14",
        scene=dict(
            bgcolor="#070b14",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            annotations=annots,
            aspectmode="data",
        ),
        margin=dict(l=0,r=0,t=10,b=0),
        height=500,
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:16px">
      <span style="font-family:'Rajdhani',sans-serif;font-size:1.5rem;
                   color:#7eb8f7;letter-spacing:3px;font-weight:700">
        🔷 BRAVAIS<br>LATTICE VIEWER
      </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CIF file", type=["cif"],
                                 label_visibility="collapsed")

    st.markdown('<hr class="sect-divider">', unsafe_allow_html=True)
    st.markdown("**⚙ Display Options**")

    show_atoms  = st.checkbox("Show atomic positions", value=True)
    supercell   = st.select_slider("Supercell", options=[1,2,3], value=1)
    show_all14  = st.checkbox("Show all 14 Bravais lattices", value=False)

    st.markdown('<hr class="sect-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem;color:#3a6080;line-height:1.7">
    Parses CIF files natively.<br>
    Renders interactive 3-D lattice<br>
    visualisations with Plotly.<br><br>
    <b style="color:#4a80a0">Rotate</b> · drag<br>
    <b style="color:#4a80a0">Zoom</b> · scroll<br>
    <b style="color:#4a80a0">Pan</b> · right-drag
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  MAIN PANEL
# ══════════════════════════════════════════════════════
st.markdown("""
<h1 style="margin-bottom:0">🔷 Bravais Lattice Viewer</h1>
<p style="color:#3a6080;font-family:'Share Tech Mono',monospace;
          margin-top:2px;font-size:0.82rem;letter-spacing:1px">
  CRYSTALLOGRAPHIC STRUCTURE ANALYSIS  ·  CIF FORMAT
</p>
""", unsafe_allow_html=True)

# ── All-14 overview ────────────────────────────────────
if show_all14:
    st.markdown("### All 14 Bravais Lattices")
    st.plotly_chart(build_all14_figure(), use_container_width=True)
    st.markdown('<hr class="sect-divider">', unsafe_allow_html=True)

# ── No file yet ────────────────────────────────────────
if uploaded is None:
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="crystal-card" style="--accent:#7eb8f7">
          <h4>What is a CIF?</h4>
          <p style="font-size:0.85rem;color:#8ab0d0;line-height:1.6;margin:0">
            A <b>Crystallographic Information File</b> stores unit-cell 
            parameters, space group symmetry and atomic coordinates for 
            crystalline materials.
          </p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="crystal-card" style="--accent:#4ecc71">
          <h4>Bravais Lattices</h4>
          <p style="font-size:0.85rem;color:#8ab0d0;line-height:1.6;margin:0">
            There are <b>14 unique</b> 3-D periodic arrangements of lattice 
            points, grouped into 7 crystal systems and 4 centering types.
          </p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="crystal-card" style="--accent:#a569bd">
          <h4>Features</h4>
          <p style="font-size:0.85rem;color:#8ab0d0;line-height:1.6;margin:0">
            Interactive 3-D cell · Centering points · Atom overlay · 
            Supercell expansion · All-14 overview panel.
          </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:24px">
      👈  <b>Upload a CIF file</b> in the sidebar to begin visualisation.
      <br>Try enabling <i>"Show all 14 Bravais lattices"</i> above for a reference overview.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════
#  PARSE & DISPLAY
# ══════════════════════════════════════════════════════
raw_text = uploaded.read().decode("utf-8", errors="replace")

try:
    cif_data  = parse_cif(raw_text)
    s         = extract_structure(cif_data)
    bravais   = classify_from_structure(s)
    info      = BRAVAIS_DB[bravais]
except Exception as exc:
    st.error(f"❌ Could not parse CIF: {exc}")
    st.stop()

color = info["color"]
system = info["system"]
centering = info["centering"]

# ── Info row ───────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:24px;flex-wrap:wrap;margin:12px 0 20px 0">

  <div class="crystal-card" style="--accent:{color};flex:0 0 auto;min-width:170px">
    <h4>Bravais Lattice</h4>
    <div class="bravais-badge" style="--bg:{info['bg']};--fg:{color}">{bravais}</div>
    <div class="sub">{system} · {centering}</div>
  </div>

  <div class="crystal-card" style="--accent:#7eb8f7;flex:1;min-width:220px">
    <h4>Space Group</h4>
    <div class="val" style="font-size:1.1rem">{s['sg_hm'] or '—'}</div>
    <div class="sub">
      Hall:&nbsp;{s['sg_hall'] or '—'} &nbsp;|&nbsp; No.&nbsp;{s['sg_num'] or '—'}
      &nbsp;|&nbsp; {s['cell_setting'] or '—'}
    </div>
  </div>

  <div class="crystal-card" style="--accent:#4ecc71;flex:1;min-width:220px">
    <h4>Unit Cell Parameters</h4>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">
      <div class="param-row"><span>a</span><span>{s['a']:.5f} Å</span></div>
      <div class="param-row"><span>α</span><span>{s['alpha']:.3f}°</span></div>
      <div class="param-row"><span>b</span><span>{s['b']:.5f} Å</span></div>
      <div class="param-row"><span>β</span><span>{s['beta']:.3f}°</span></div>
      <div class="param-row"><span>c</span><span>{s['c']:.5f} Å</span></div>
      <div class="param-row"><span>γ</span><span>{s['gamma']:.3f}°</span></div>
    </div>
  </div>

  <div class="crystal-card" style="--accent:#a569bd;flex:0 0 auto;min-width:150px">
    <h4>Volume</h4>
    <div class="val">{s['volume']:.1f}</div>
    <div class="sub">Å³</div>
    <hr class="sect-divider" style="margin:10px 0">
    <h4>Atoms (asym. unit)</h4>
    <div class="val">{len(s['atoms'])}</div>
  </div>

</div>
""", unsafe_allow_html=True)

# ── Element chips ─────────────────────────────────────
if s["atoms"]:
    elems = sorted({a["element"] for a in s["atoms"]})
    chips = "".join(
        f'<span class="elem-chip" '
        f'style="background:{ELEMENT_COLORS.get(e,ELEMENT_COLORS["DEFAULT"])}22;'
        f'color:{ELEMENT_COLORS.get(e,ELEMENT_COLORS["DEFAULT"])};'
        f'border:1px solid {ELEMENT_COLORS.get(e,ELEMENT_COLORS["DEFAULT"])}66">'
        f'{e}</span>'
        for e in elems
    )
    st.markdown(f"**Elements present:** {chips}", unsafe_allow_html=True)

st.markdown('<hr class="sect-divider">', unsafe_allow_html=True)

# ── 3-D figure ────────────────────────────────────────
opts = {"show_atoms": show_atoms, "supercell": supercell}
fig  = build_figure(s, bravais, opts)
st.plotly_chart(fig, use_container_width=True)

# ── Atom site table ────────────────────────────────────
if show_atoms and s["atoms"]:
    with st.expander(f"🔬 Atom site coordinates  ({len(s['atoms'])} sites in asymmetric unit)"):
        M = cell_matrix(s["a"],s["b"],s["c"],s["alpha"],s["beta"],s["gamma"])
        rows = []
        for atom in s["atoms"]:
            cart = frac_to_cart((atom["fx"],atom["fy"],atom["fz"]), M)
            rows.append({
                "Label": atom["label"], "Element": atom["element"],
                "x (frac)": f"{atom['fx']:.5f}",
                "y (frac)": f"{atom['fy']:.5f}",
                "z (frac)": f"{atom['fz']:.5f}",
                "x (Å)": f"{cart[0]:.4f}",
                "y (Å)": f"{cart[1]:.4f}",
                "z (Å)": f"{cart[2]:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True, height=200)

# ── Quick-reference table ──────────────────────────────
with st.expander("📚 All 14 Bravais Lattices — reference"):
    ref_rows = []
    for sym, d in BRAVAIS_DB.items():
        ref_rows.append({
            "Symbol": sym, "Crystal System": d["system"],
            "Centering": d["centering"],
            "Detected ✓": "◆" if sym == bravais else ""
        })
    df = pd.DataFrame(ref_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)