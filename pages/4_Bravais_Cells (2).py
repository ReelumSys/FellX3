"""
Bravais Lattice Viewer
Reads cif_file2 content from st.session_state["cif_file2"]
(set by the upload page). Parses CIF, classifies Bravais lattice,
renders interactive 3-D visualisation.
"""

import re, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from main import cif_file2

st.set_page_config(page_title="Bravais Lattice Viewer", page_icon="🔷", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');
html,body,[data-testid="stApp"]{background:#06090f!important;color:#c0d4f0!important;font-family:'Exo 2',sans-serif!important}
[data-testid="stSidebar"]{background:#090d18!important;border-right:1px solid #1a3050}
h1,h2,h3{font-family:'Rajdhani',sans-serif!important;letter-spacing:2px!important}
h1{color:#6db3ff!important;font-weight:700!important}
h2,h3{color:#90c4ff!important;font-weight:600!important}
.stTabs [data-baseweb="tab-list"]{background:#090d18;border-bottom:1px solid #1a3555}
.stTabs [data-baseweb="tab"]{font-family:'Rajdhani',sans-serif!important;font-size:0.95rem;letter-spacing:1px;color:#3a6888;border-radius:6px 6px 0 0;padding:8px 20px}
.stTabs [aria-selected="true"]{color:#6db3ff!important;background:#0e1930!important;border-bottom:2px solid #6db3ff!important}
[data-testid="stMetricValue"]{font-family:'Share Tech Mono',monospace!important;color:#6db3ff!important}
.card{background:linear-gradient(135deg,#0a1220,#0e1930);border:1px solid #1a3555;border-radius:10px;padding:16px 18px;margin:6px 0;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;width:4px;height:100%;background:var(--ac)}
.card .lbl{font-family:'Rajdhani',sans-serif;font-size:0.7rem;text-transform:uppercase;letter-spacing:2px;color:#3a6888;margin:0 0 4px 0}
.card .val{font-family:'Share Tech Mono',monospace;font-size:1.35rem;color:var(--ac);margin:0}
.card .sub{font-size:0.76rem;color:#3a5a78;margin:4px 0 0 0}
.bv-badge{display:inline-block;padding:6px 18px;border-radius:4px;font-family:'Share Tech Mono',monospace;font-size:2rem;font-weight:700;letter-spacing:4px;border:2px solid var(--ac);color:var(--ac);background:var(--bg);margin-bottom:6px}
.pr{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #122033;font-family:'Share Tech Mono',monospace;font-size:0.82rem}
.pr span:first-child{color:#3a7aaa}
.pr span:last-child{color:#cce0ff}
.echip{display:inline-block;padding:2px 10px;border-radius:20px;margin:2px;font-family:'Share Tech Mono',monospace;font-size:0.78rem;font-weight:600}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  BRAVAIS DATABASE
# ══════════════════════════════════════════════════════════════
BRAVAIS_DB = {
    "aP":{"system":"Triclinic",    "centering":"Primitive",    "color":"#e74c3c","bg":"#2d0b0b"},
    "mP":{"system":"Monoclinic",   "centering":"Primitive",    "color":"#e67e22","bg":"#2d1a0b"},
    "mS":{"system":"Monoclinic",   "centering":"C-centered",   "color":"#f0a500","bg":"#2d210b"},
    "oP":{"system":"Orthorhombic", "centering":"Primitive",    "color":"#2ecc71","bg":"#0b2d1a"},
    "oS":{"system":"Orthorhombic", "centering":"C-centered",   "color":"#27ae60","bg":"#0b2d18"},
    "oF":{"system":"Orthorhombic", "centering":"Face-centered","color":"#1abc9c","bg":"#0b2d28"},
    "oI":{"system":"Orthorhombic", "centering":"Body-centered","color":"#16a085","bg":"#0b2522"},
    "tP":{"system":"Tetragonal",   "centering":"Primitive",    "color":"#3498db","bg":"#0b1e2d"},
    "tI":{"system":"Tetragonal",   "centering":"Body-centered","color":"#2980b9","bg":"#0b1a2d"},
    "hR":{"system":"Rhombohedral", "centering":"Rhombohedral", "color":"#a569bd","bg":"#1e0b2d"},
    "hP":{"system":"Hexagonal",    "centering":"Primitive",    "color":"#8e44ad","bg":"#1a0b2d"},
    "cP":{"system":"Cubic",        "centering":"Primitive",    "color":"#6db3ff","bg":"#0a1628"},
    "cF":{"system":"Cubic",        "centering":"Face-centered","color":"#5dade2","bg":"#0b1c2e"},
    "cI":{"system":"Cubic",        "centering":"Body-centered","color":"#85c1e9","bg":"#0b1e32"},
}

ELEMENT_COLORS = {
    "H":"#ffffff","C":"#909090","N":"#3050F8","O":"#FF0D0D","F":"#90E050",
    "Na":"#AB5CF2","Mg":"#8AFF00","Al":"#BFA6A6","Si":"#F0C8A0","P":"#FF8000",
    "S":"#FFFF30","Cl":"#1FF01F","K":"#8F40D4","Ca":"#3DFF00","Fe":"#E06633",
    "Cu":"#C88033","Zn":"#7D80B0","DEFAULT":"#ff88aa",
}
ELEMENT_RADII = {
    "H":0.53,"C":0.77,"N":0.75,"O":0.73,"Na":1.86,"Mg":1.60,"Al":1.43,
    "Si":1.17,"Cl":0.99,"K":2.27,"Ca":1.97,"Fe":1.26,"DEFAULT":1.0,
}

# ══════════════════════════════════════════════════════════════
#  CIF PARSER
# ══════════════════════════════════════════════════════════════
def _strip_esd(v):
    return re.sub(r'\([^)]*\)', '', str(v)).strip()

def parse_cif(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'#[^\n]*', '', text)
    data = {}
    scalar_re = re.compile(r"(_\S+)\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))", re.MULTILINE)
    for m in scalar_re.finditer(text):
        k = m.group(1).lower()
        v = m.group(2) or m.group(3) or m.group(4)
        data[k] = v
    text_clean = re.sub(r'\n;.*?\n;', ' ? ', text, flags=re.DOTALL)
    for block in re.split(r'\bloop_\b', text_clean, flags=re.IGNORECASE)[1:]:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        headers, value_lines, mode = [], [], 'h'
        for line in lines:
            if mode == 'h':
                if line.startswith('_'): headers.append(line.lower())
                else: mode = 'v'; value_lines.append(line)
            else:
                if line.startswith('_') or line.lower().startswith('loop_'): break
                value_lines.append(line)
        if not headers: continue
        tokens = []
        for vl in value_lines:
            for tok in re.findall(r"'[^']*'|\"[^\"]*\"|\S+", vl):
                tokens.append(tok.strip("'\""))
        n = len(headers)
        if n == 0 or len(tokens) < n: continue
        rows = [tokens[i:i+n] for i in range(0, len(tokens)-n+1, n)]
        for i, h in enumerate(headers):
            data[h] = [row[i] for row in rows if i < len(row)]
    return data

def extract_structure(cif_data):
    def flt(key, default=0.0):
        v = cif_data.get(key, str(default))
        if isinstance(v, list): v = v[0]
        try: return float(_strip_esd(str(v)))
        except: return default
    def gs(k):
        v = cif_data.get(k, '')
        if isinstance(v, list): v = v[0]
        return str(v).strip("'\" ")

    a = flt('_cell_length_a', 8.0); b = flt('_cell_length_b', a); c = flt('_cell_length_c', a)
    alpha = flt('_cell_angle_alpha', 90.0); beta = flt('_cell_angle_beta', 90.0); gamma = flt('_cell_angle_gamma', 90.0)
    volume = flt('_cell_volume', 0.0)
    sg_hm        = gs('_symmetry_space_group_name_h-m') or gs('_space_group_name_h-m_alt')
    sg_hall      = gs('_symmetry_space_group_name_hall') or gs('_space_group_name_hall')
    sg_num       = gs('_symmetry_int_tables_number') or gs('_space_group_it_number')
    cell_setting = gs('_symmetry_cell_setting')

    labels = cif_data.get('_atom_site_label', [])
    fx_l   = cif_data.get('_atom_site_fract_x', [])
    fy_l   = cif_data.get('_atom_site_fract_y', [])
    fz_l   = cif_data.get('_atom_site_fract_z', [])
    elem_l = cif_data.get('_atom_site_type_symbol', labels)
    atoms  = []
    for i in range(len(labels)):
        try:
            fx = float(_strip_esd(fx_l[i])) if i < len(fx_l) else 0.0
            fy = float(_strip_esd(fy_l[i])) if i < len(fy_l) else 0.0
            fz = float(_strip_esd(fz_l[i])) if i < len(fz_l) else 0.0
            el = (elem_l[i] if i < len(elem_l) else labels[i])
            el = re.sub(r'[^A-Za-z]', '', el)[:2].capitalize()
            atoms.append({"label": labels[i], "element": el, "fx": fx, "fy": fy, "fz": fz})
        except: continue

    return dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                volume=volume, sg_hm=sg_hm, sg_hall=sg_hall,
                sg_num=sg_num, cell_setting=cell_setting, atoms=atoms)

# ══════════════════════════════════════════════════════════════
#  BRAVAIS CLASSIFICATION
# ══════════════════════════════════════════════════════════════
SYSTEM_MAP = {
    "triclinic":   {"P":"aP"},
    "monoclinic":  {"P":"mP","C":"mS","A":"mS","I":"mS"},
    "orthorhombic":{"P":"oP","C":"oS","A":"oS","B":"oS","F":"oF","I":"oI"},
    "tetragonal":  {"P":"tP","I":"tI"},
    "trigonal":    {"P":"hP","R":"hR"},
    "rhombohedral":{"R":"hR","P":"hP"},
    "hexagonal":   {"P":"hP"},
    "cubic":       {"P":"cP","F":"cF","I":"cI"},
}

def classify_bravais(sg_hm, cell_setting):
    sg_clean  = sg_hm.strip("'\" ")
    centering = sg_clean[0].upper() if sg_clean else "P"
    sl = cell_setting.lower()
    if not sl:
        for sys in SYSTEM_MAP:
            if sys.upper() in sg_clean.upper():
                sl = sys; break
    for sys, cmap in SYSTEM_MAP.items():
        if sys in sl or sl.startswith(sys[:4]):
            return cmap.get(centering, list(cmap.values())[0])
    return "aP"

def classify_from_structure(s):
    bv = classify_bravais(s["sg_hm"], s["cell_setting"])
    if bv not in BRAVAIS_DB:
        a, b, c = s["a"], s["b"], s["c"]
        al, be, ga = s["alpha"], s["beta"], s["gamma"]
        all90 = all(abs(x-90) < 0.5 for x in [al, be, ga])
        if abs(a-b) < 0.01 and abs(b-c) < 0.01 and all90: return "cP"
        if abs(a-b) < 0.01 and all90: return "tP"
        if all90: return "oP"
        return "aP"
    return bv

# ══════════════════════════════════════════════════════════════
#  GEOMETRY
# ══════════════════════════════════════════════════════════════
def cell_matrix(a, b, c, alpha_d, beta_d, gamma_d):
    al, be, ga = math.radians(alpha_d), math.radians(beta_d), math.radians(gamma_d)
    ax = a
    bx = b*math.cos(ga);  by = b*math.sin(ga)
    cx = c*math.cos(be)
    cy = c*(math.cos(al) - math.cos(be)*math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    return np.array([[ax,0,0],[bx,by,0],[cx,cy,cz]])

def frac_to_cart(frac, M):
    return M[0]*frac[0] + M[1]*frac[1] + M[2]*frac[2]

def cell_edges(M, origin=None):
    O = origin if origin is not None else np.zeros(3)
    a, b, c = M[0], M[1], M[2]
    return [
        (O,     O+a),   (O,     O+b),   (O,     O+c),
        (O+a,   O+a+b), (O+a,   O+a+c),
        (O+b,   O+a+b), (O+b,   O+b+c),
        (O+c,   O+a+c), (O+c,   O+b+c),
        (O+a+b, O+a+b+c),(O+a+c,O+a+b+c),(O+b+c,O+a+b+c),
    ]

def centering_fracs(symbol):
    c = symbol[1] if len(symbol) > 1 else "P"
    base  = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
    extra = {"I":[(0.5,0.5,0.5)],
             "F":[(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
             "S":[(0.5,0.5,0)],"C":[(0.5,0.5,0)],
             "R":[(2/3,1/3,1/3),(1/3,2/3,2/3)]}.get(c, [])
    return base + extra

# ══════════════════════════════════════════════════════════════
#  3-D FIGURE
# ══════════════════════════════════════════════════════════════
def build_figure(s, bravais, show_atoms, supercell):
    info  = BRAVAIS_DB[bravais]
    color = info["color"]
    M     = cell_matrix(s["a"], s["b"], s["c"], s["alpha"], s["beta"], s["gamma"])
    traces = []

    # Cell edges
    for i in range(supercell):
        for j in range(supercell):
            for k in range(supercell):
                origin = i*M[0] + j*M[1] + k*M[2]
                for (p1, p2) in cell_edges(M, origin):
                    traces.append(go.Scatter3d(
                        x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]],
                        mode="lines", line=dict(color=color, width=2.5),
                        showlegend=False, hoverinfo="skip",
                    ))

    # Centering lattice points
    all_pts = []
    for i in range(supercell):
        for j in range(supercell):
            for k in range(supercell):
                off = i*M[0] + j*M[1] + k*M[2]
                for f in centering_fracs(bravais):
                    all_pts.append(off + frac_to_cart(f, M))
    if all_pts:
        px, py, pz = zip(*[(p[0],p[1],p[2]) for p in all_pts])
        traces.append(go.Scatter3d(
            x=px, y=py, z=pz, mode="markers",
            marker=dict(size=7, color=color, opacity=0.9,
                        line=dict(color="#0a0f1a", width=1)),
            name=f"Lattice nodes ({info['centering']})",
            hovertemplate="<b>Lattice point</b><br>x=%{x:.3f} Å<br>y=%{y:.3f} Å<br>z=%{z:.3f} Å<extra></extra>",
        ))

    # Basis vectors a, b, c
    O = np.zeros(3)
    for vec, lbl, col in zip(M, ["a","b","c"], ["#ff4e4e","#4ecc71","#4e9eff"]):
        traces.append(go.Scatter3d(
            x=[O[0],vec[0]], y=[O[1],vec[1]], z=[O[2],vec[2]],
            mode="lines+text", line=dict(color=col, width=5),
            text=["", f"<b>{lbl}</b>"],
            textposition="top center",
            textfont=dict(size=14, color=col, family="Share Tech Mono"),
            name=f"Axis {lbl}",
        ))

    # Atoms
    if show_atoms and s["atoms"]:
        elem_groups = {}
        for atom in s["atoms"]:
            for i in range(supercell):
                for j in range(supercell):
                    for k in range(supercell):
                        off  = i*M[0] + j*M[1] + k*M[2]
                        cart = off + frac_to_cart((atom["fx"], atom["fy"], atom["fz"]), M)
                        elem_groups.setdefault(atom["element"], []).append((cart, atom["label"]))
        for el, pts in elem_groups.items():
            ecol = ELEMENT_COLORS.get(el, ELEMENT_COLORS["DEFAULT"])
            rad  = ELEMENT_RADII.get(el, ELEMENT_RADII["DEFAULT"])
            xs, ys, zs, lbls = zip(*[(p[0],p[1],p[2],lb) for (p,lb) in pts])
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers",
                marker=dict(size=rad*3.5+2, color=ecol, opacity=0.85,
                            line=dict(color="#0a0f1a", width=0.5)),
                name=el,
                hovertemplate=f"<b>%{{text}}</b> ({el})<br>x=%{{x:.4f}} Å<br>y=%{{y:.4f}} Å<br>z=%{{z:.4f}} Å<extra></extra>",
                text=lbls,
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#06090f",
        scene=dict(
            bgcolor="#06090f",
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                       color="#2a4a6a", title="x (Å)"),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                       color="#2a4a6a", title="y (Å)"),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                       color="#2a4a6a", title="z (Å)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=580,
        legend=dict(bgcolor="rgba(6,9,15,0.85)", bordercolor="#1e3a5f",
                    borderwidth=1, font=dict(family="Share Tech Mono", size=11, color="#7eb8f7")),
        uirevision="constant",
    )
    return fig

def build_all14_figure():
    representative = [
        ("aP",1.0,1.2,1.5,70,80,85), ("mP",1.0,1.2,1.5,90,110,90),
        ("mS",1.0,1.2,1.5,90,110,90), ("oP",1.0,1.2,1.5,90,90,90),
        ("oS",1.0,1.2,1.5,90,90,90), ("oF",1.0,1.2,1.5,90,90,90),
        ("oI",1.0,1.2,1.5,90,90,90), ("tP",1.0,1.0,1.5,90,90,90),
        ("tI",1.0,1.0,1.5,90,90,90), ("hR",1.0,1.0,1.0,75,75,75),
        ("hP",1.0,1.0,1.5,90,90,120),("cP",1.0,1.0,1.0,90,90,90),
        ("cF",1.0,1.0,1.0,90,90,90), ("cI",1.0,1.0,1.0,90,90,90),
    ]
    traces, annots = [], []
    for idx, (sym,a,b,c,al,be,ga) in enumerate(representative):
        ri, ci = divmod(idx, 7)
        off = np.array([ci*3.4, 0.0, -ri*3.4])
        M   = cell_matrix(a, b, c, al, be, ga)
        col = BRAVAIS_DB[sym]["color"]
        for (p1,p2) in cell_edges(M, off):
            traces.append(go.Scatter3d(
                x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]],
                mode="lines", line=dict(color=col, width=2),
                showlegend=False, hoverinfo="skip",
            ))
        pts = [off + frac_to_cart(f, M) for f in centering_fracs(sym)]
        lx, ly, lz = zip(*[(p[0],p[1],p[2]) for p in pts])
        traces.append(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="markers",
            marker=dict(size=5, color=col, opacity=0.9),
            name=sym,
            hovertemplate=f"<b>{sym}</b> — {BRAVAIS_DB[sym]['system']}<extra></extra>",
        ))
        ctr = off + 0.5*(M[0]+M[1]+M[2])
        annots.append(dict(x=ctr[0], y=ctr[1]+0.2, z=ctr[2]+1.3,
                           text=f"<b>{sym}</b>", showarrow=False,
                           font=dict(size=12, color=col, family="Share Tech Mono")))
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#06090f",
        scene=dict(bgcolor="#06090f",
                   xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), annotations=annots, aspectmode="data"),
        margin=dict(l=0,r=0,t=10,b=0), height=500, showlegend=False,
    )
    return fig

# ══════════════════════════════════════════════════════════════
#  CIF INPUT  — set cif_file2 to the raw CIF string
# ══════════════════════════════════════════════════════════════

# ↓↓ Deliver your CIF content here ↓↓
cif_file2 = cif_file2   # replace with your variable, e.g.: cif_file2 = my_cif_string

# ─────────────────────────────────────────────────────────────
st.title("🔷 Bravais Lattice Viewer")

if not cif_file2:
    st.error("No CIF content provided. Set the `cif_file2` variable at the top of this file.")
    st.stop()

if isinstance(cif_file2, str):
    raw_cif = cif_file2
elif isinstance(cif_file2, bytes):
    raw_cif = cif_file2.decode('utf-8', errors='replace')
else:
    # UploadedFile or any file-like object
    raw_cif = cif_file2.read()
    if isinstance(raw_cif, bytes):
        raw_cif = raw_cif.decode('utf-8', errors='replace')
filename = cif_file2  # adjust if needed

cif_data = parse_cif(raw_cif)
s        = extract_structure(cif_data)
bravais  = classify_from_structure(s)
info     = BRAVAIS_DB[bravais]
color    = info["color"]

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;margin-bottom:16px'>
      <span style='font-family:Rajdhani,sans-serif;font-size:1.4rem;
                   color:#6db3ff;letter-spacing:3px;font-weight:700'>
        🔷 BRAVAIS VIEWER
      </span>
    </div>
    <div style='background:#0d1826;border:1px solid #1a3555;border-radius:8px;
                padding:10px 14px;margin-bottom:12px;
                font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#4a8abf'>
      📂 &nbsp;<b style='color:#6db3ff'>{filename}</b><br>
      {len(s["atoms"])} sites &nbsp;·&nbsp; {s["cell_setting"] or "—"}
    </div>
    """, unsafe_allow_html=True)

    st.subheader("⚙ Display Options")
    show_atoms = st.checkbox("Show atomic positions", value=True)
    supercell  = st.select_slider("Supercell", options=[1, 2, 3], value=1)

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem;color:#3a6080;line-height:1.8'>
    <b style='color:#4a80a0'>Rotate</b> · drag<br>
    <b style='color:#4a80a0'>Zoom</b> · scroll<br>
    <b style='color:#4a80a0'>Pan</b> · right-drag
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SUBTITLE
# ══════════════════════════════════════════════════════════════
st.markdown(
    f"<p style='color:#3a6080;font-family:Share Tech Mono,monospace;"
    f"font-size:0.8rem;letter-spacing:1px;margin-top:-10px'>"
    f"{filename}  ·  {s['sg_hm'] or '—'}  ·  {info['system'].upper()}</p>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
#  INFO CARDS
# ══════════════════════════════════════════════════════════════
col_a, col_b, col_c, col_d = st.columns([1.4, 1.6, 1.8, 1.2])

with col_a:
    st.markdown(f"""
    <div class="card" style="--ac:{color}">
      <p class="lbl">Bravais Lattice</p>
      <div class="bv-badge" style="--ac:{color};--bg:{info['bg']}">{bravais}</div>
      <p class="sub">{info['system']} · {info['centering']}</p>
    </div>""", unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div class="card" style="--ac:#7eb8f7">
      <p class="lbl">Space Group</p>
      <p class="val" style="font-size:1rem">{s['sg_hm'] or '—'}</p>
      <p class="sub">Hall: {s['sg_hall'] or '—'} &nbsp;|&nbsp; No. {s['sg_num'] or '—'}</p>
    </div>""", unsafe_allow_html=True)

with col_c:
    st.markdown(f"""
    <div class="card" style="--ac:#4ecc71">
      <p class="lbl">Unit Cell Parameters</p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">
        <div class="pr"><span>a</span><span>{s['a']:.5f} Å</span></div>
        <div class="pr"><span>α</span><span>{s['alpha']:.3f}°</span></div>
        <div class="pr"><span>b</span><span>{s['b']:.5f} Å</span></div>
        <div class="pr"><span>β</span><span>{s['beta']:.3f}°</span></div>
        <div class="pr"><span>c</span><span>{s['c']:.5f} Å</span></div>
        <div class="pr"><span>γ</span><span>{s['gamma']:.3f}°</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

with col_d:
    st.markdown(f"""
    <div class="card" style="--ac:#a569bd">
      <p class="lbl">Volume</p>
      <p class="val">{s['volume']:.2f}</p>
      <p class="sub">Å³</p>
      <hr style="border:none;border-top:1px solid #1a3050;margin:10px 0">
      <p class="lbl">Asymmetric unit</p>
      <p class="val">{len(s['atoms'])} sites</p>
    </div>""", unsafe_allow_html=True)

# Element chips
if s["atoms"]:
    elems = sorted({a["element"] for a in s["atoms"]})
    chips = "".join(
        f'<span class="echip" style="background:{ELEMENT_COLORS.get(e,ELEMENT_COLORS["DEFAULT"])}22;'
        f'color:{ELEMENT_COLORS.get(e,ELEMENT_COLORS["DEFAULT"])};'
        f'border:1px solid {ELEMENT_COLORS.get(e,ELEMENT_COLORS["DEFAULT"])}66">{e}</span>'
        for e in elems
    )
    st.markdown(f"<div style='margin:8px 0'><b style='color:#3a6888'>Elements:</b> {chips}</div>",
                unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🧊 3-D Lattice", "📋 Atom Sites", "📚 All 14 Bravais"])

with tab1:
    st.plotly_chart(build_figure(s, bravais, show_atoms, supercell),
                    use_container_width=True)

with tab2:
    if s["atoms"]:
        M = cell_matrix(s["a"], s["b"], s["c"], s["alpha"], s["beta"], s["gamma"])
        rows = []
        for atom in s["atoms"]:
            cart = frac_to_cart((atom["fx"], atom["fy"], atom["fz"]), M)
            rows.append({
                "Label":    atom["label"],   "Element": atom["element"],
                "x (frac)": f"{atom['fx']:.5f}",
                "y (frac)": f"{atom['fy']:.5f}",
                "z (frac)": f"{atom['fz']:.5f}",
                "x (Å)":   f"{cart[0]:.4f}",
                "y (Å)":   f"{cart[1]:.4f}",
                "z (Å)":   f"{cart[2]:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True, height=350)
    else:
        st.info("No atom sites found in CIF.")

    st.divider()
    st.subheader("All 14 Bravais Lattices — Reference")
    ref = [{"Symbol": sym, "Crystal System": d["system"],
            "Centering": d["centering"],
            "Detected ✓": "◆" if sym == bravais else ""}
           for sym, d in BRAVAIS_DB.items()]
    st.dataframe(pd.DataFrame(ref), use_container_width=True, hide_index=True)

with tab3:
    st.plotly_chart(build_all14_figure(), use_container_width=True)