"""
Bravais Lattice Viewer — Streamlit App
Visualizes crystal structures and Bravais lattices from uploaded CIF files.

Requirements:
    pip install streamlit gemmi plotly numpy
Run:
    streamlit run bravais_lattice_viewer.py
"""

import io
import math
import tempfile
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Try importing gemmi ───────────────────────────────────────────────────────
try:
    import gemmi
    GEMMI_OK = True
except ImportError:
    GEMMI_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Bravais lattice classification helpers
# ─────────────────────────────────────────────────────────────────────────────

BRAVAIS_INFO = {
    "aP":  {"system": "Triclinic",     "centering": "Primitive",      "symbol": "aP",  "color": "#e74c3c"},
    "mP":  {"system": "Monoclinic",    "centering": "Primitive",      "symbol": "mP",  "color": "#e67e22"},
    "mS":  {"system": "Monoclinic",    "centering": "C-centered",     "symbol": "mS",  "color": "#f39c12"},
    "oP":  {"system": "Orthorhombic",  "centering": "Primitive",      "symbol": "oP",  "color": "#2ecc71"},
    "oS":  {"system": "Orthorhombic",  "centering": "C-centered",     "symbol": "oS",  "color": "#27ae60"},
    "oF":  {"system": "Orthorhombic",  "centering": "Face-centered",  "symbol": "oF",  "color": "#1abc9c"},
    "oI":  {"system": "Orthorhombic",  "centering": "Body-centered",  "symbol": "oI",  "color": "#16a085"},
    "tP":  {"system": "Tetragonal",    "centering": "Primitive",      "symbol": "tP",  "color": "#3498db"},
    "tI":  {"system": "Tetragonal",    "centering": "Body-centered",  "symbol": "tI",  "color": "#2980b9"},
    "hR":  {"system": "Rhombohedral",  "centering": "Rhombohedral",   "symbol": "hR",  "color": "#9b59b6"},
    "hP":  {"system": "Hexagonal",     "centering": "Primitive",      "symbol": "hP",  "color": "#8e44ad"},
    "cP":  {"system": "Cubic",         "centering": "Primitive",      "symbol": "cP",  "color": "#34495e"},
    "cF":  {"system": "Cubic",         "centering": "Face-centered",  "symbol": "cF",  "color": "#2c3e50"},
    "cI":  {"system": "Cubic",         "centering": "Body-centered",  "symbol": "cI",  "color": "#95a5a6"},
}

def get_bravais_from_spacegroup(sg: "gemmi.SpaceGroup") -> str:
    """Return the two-letter Bravais lattice symbol from a gemmi SpaceGroup."""
    lattice_letter = sg.bravais_lattice().short_name()[:2]
    return lattice_letter


def classify_bravais(cell: "gemmi.UnitCell", spacegroup: "gemmi.SpaceGroup") -> dict:
    """Return full Bravais info dict for the structure."""
    symbol = get_bravais_from_spacegroup(spacegroup)
    info = BRAVAIS_INFO.get(symbol, {
        "system": "Unknown", "centering": "Unknown", "symbol": symbol, "color": "#bdc3c7"
    })
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Lattice geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def cell_to_vectors(cell: "gemmi.UnitCell") -> np.ndarray:
    """Convert unit cell parameters to 3x3 matrix of lattice vectors (row = vector)."""
    a, b, c = cell.a, cell.b, cell.c
    alpha = math.radians(cell.alpha)
    beta  = math.radians(cell.beta)
    gamma = math.radians(cell.gamma)

    # Standard crystallographic convention
    ax = a
    bx = b * math.cos(gamma)
    by = b * math.sin(gamma)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cz = math.sqrt(max(c**2 - cx**2 - cy**2, 0.0))

    return np.array([
        [ax,  0.0, 0.0],
        [bx,  by,  0.0],
        [cx,  cy,  cz ],
    ])


def make_unit_cell_edges(vecs: np.ndarray, origin: np.ndarray = None):
    """Return list of (start, end) pairs for the 12 unit-cell edges."""
    if origin is None:
        origin = np.zeros(3)
    a, b, c = vecs[0], vecs[1], vecs[2]
    O = origin
    corners = [
        O, O+a, O+b, O+c,
        O+a+b, O+a+c, O+b+c, O+a+b+c
    ]
    edges = [
        (O,     O+a),   (O,     O+b),   (O,     O+c),
        (O+a,   O+a+b), (O+a,   O+a+c),
        (O+b,   O+a+b), (O+b,   O+b+c),
        (O+c,   O+a+c), (O+c,   O+b+c),
        (O+a+b, O+a+b+c),(O+a+c, O+a+b+c),(O+b+c, O+a+b+c),
    ]
    return corners, edges


def lattice_points_for_bravais(symbol: str, vecs: np.ndarray) -> list[np.ndarray]:
    """Return fractional → Cartesian lattice points for the conventional cell."""
    a, b, c = vecs[0], vecs[1], vecs[2]
    frac_points = [(0, 0, 0)]  # always include origin

    centering = symbol[1] if len(symbol) > 1 else "P"
    if centering == "I":
        frac_points.append((0.5, 0.5, 0.5))
    elif centering == "F":
        frac_points += [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    elif centering == "S" or centering == "C":
        frac_points.append((0.5, 0.5, 0))
    elif centering == "R":
        frac_points += [(2/3, 1/3, 1/3), (1/3, 2/3, 2/3)]

    # Also show all 8 corners of the cell
    for fx in [0, 1]:
        for fy in [0, 1]:
            for fz in [0, 1]:
                frac_points.append((fx, fy, fz))

    cart = [f[0]*a + f[1]*b + f[2]*c for f in frac_points]
    return cart


# ─────────────────────────────────────────────────────────────────────────────
# Plotly 3-D figure builders
# ─────────────────────────────────────────────────────────────────────────────

def build_lattice_figure(vecs: np.ndarray, bravais: dict,
                          atom_positions_cart: list = None,
                          show_atoms: bool = True,
                          supercell: int = 1) -> go.Figure:
    """Build an interactive 3-D Plotly figure of the Bravais lattice."""
    color = bravais["color"]
    traces = []

    # ── Draw unit-cell edges for each supercell repeat ──
    for i in range(supercell):
        for j in range(supercell):
            for k in range(supercell):
                origin = i*vecs[0] + j*vecs[1] + k*vecs[2]
                _, edges = make_unit_cell_edges(vecs, origin)
                for (p1, p2) in edges:
                    traces.append(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode="lines",
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

    # ── Lattice-centering points ──
    lp = lattice_points_for_bravais(bravais["symbol"], vecs)
    lx, ly, lz = zip(*[(p[0], p[1], p[2]) for p in lp])
    traces.append(go.Scatter3d(
        x=lx, y=ly, z=lz,
        mode="markers",
        marker=dict(size=8, color=color, opacity=0.9,
                    line=dict(color="white", width=1)),
        name=f"Lattice points ({bravais['centering']})",
    ))

    # ── Basis vectors ──
    O = np.zeros(3)
    for vec, label, col in zip(vecs, ["a", "b", "c"], ["#e74c3c", "#27ae60", "#2980b9"]):
        traces.append(go.Scatter3d(
            x=[O[0], vec[0]], y=[O[1], vec[1]], z=[O[2], vec[2]],
            mode="lines+text",
            line=dict(color=col, width=6),
            text=["", f"<b>{label}</b>"],
            textposition="top center",
            textfont=dict(size=14, color=col),
            name=f"Vector {label}",
        ))

    # ── Atom positions ──
    if show_atoms and atom_positions_cart:
        ax_c, ay_c, az_c, atom_names = [], [], [], []
        for pos, name in atom_positions_cart:
            ax_c.append(pos[0]); ay_c.append(pos[1]); az_c.append(pos[2])
            atom_names.append(name)
        traces.append(go.Scatter3d(
            x=ax_c, y=ay_c, z=az_c,
            mode="markers",
            marker=dict(size=5, color="#f1c40f", opacity=0.75,
                        line=dict(color="#e67e22", width=1)),
            text=atom_names,
            hovertemplate="<b>%{text}</b><br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            name="Atoms",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"<b>{bravais['system']} — {bravais['centering']} ({bravais['symbol']})</b>",
            x=0.5, font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(showbackground=False, title="x (Å)"),
            yaxis=dict(showbackground=False, title="y (Å)"),
            zaxis=dict(showbackground=False, title="z (Å)"),
            aspectmode="data",
            bgcolor="#0f1117",
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="white", borderwidth=1),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
    )
    return fig


def build_all14_figure() -> go.Figure:
    """Draw all 14 Bravais lattice cells schematically in one figure."""
    from plotly.subplots import make_subplots

    # Representative cells (a, b, c, alpha, beta, gamma in degrees)
    cells_14 = [
        ("aP",  1.0, 1.2, 1.5, 70,  80,  85),
        ("mP",  1.0, 1.2, 1.5, 90,  110, 90),
        ("mS",  1.0, 1.2, 1.5, 90,  110, 90),
        ("oP",  1.0, 1.2, 1.5, 90,  90,  90),
        ("oS",  1.0, 1.2, 1.5, 90,  90,  90),
        ("oF",  1.0, 1.2, 1.5, 90,  90,  90),
        ("oI",  1.0, 1.2, 1.5, 90,  90,  90),
        ("tP",  1.0, 1.0, 1.5, 90,  90,  90),
        ("tI",  1.0, 1.0, 1.5, 90,  90,  90),
        ("hR",  1.0, 1.0, 1.0, 75,  75,  75),
        ("hP",  1.0, 1.0, 1.5, 90,  90,  120),
        ("cP",  1.0, 1.0, 1.0, 90,  90,  90),
        ("cF",  1.0, 1.0, 1.0, 90,  90,  90),
        ("cI",  1.0, 1.0, 1.0, 90,  90,  90),
    ]

    traces_all = []
    annotations = []
    # lay them out on a 3D grid (7 cols × 2 rows), offset each cell
    cols, rows = 7, 2
    spacing = 3.5

    for idx, (sym, a, b, c, al, be, ga) in enumerate(cells_14):
        row_i = idx // cols
        col_i = idx % cols
        offset = np.array([col_i * spacing, row_i * spacing * 1.6, 0.0])

        vecs = cell_to_vectors_params(a, b, c, al, be, ga)
        _, edges = make_unit_cell_edges(vecs, offset)
        info = BRAVAIS_INFO[sym]
        col = info["color"]

        for (p1, p2) in edges:
            traces_all.append(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode="lines", line=dict(color=col, width=2),
                showlegend=False, hoverinfo="skip",
            ))

        lp = lattice_points_for_bravais(sym, vecs)
        adj = [p + offset for p in lp]
        lx, ly, lz = zip(*[(p[0], p[1], p[2]) for p in adj])
        traces_all.append(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="markers",
            marker=dict(size=5, color=col, opacity=0.9),
            name=sym,
            hovertemplate=f"<b>{info['system']} {info['centering']}</b><extra>{sym}</extra>",
        ))

        # label above each cell
        center = offset + 0.5*(vecs[0] + vecs[1] + vecs[2])
        annotations.append(dict(
            x=center[0], y=center[1], z=center[2] + 1.0,
            text=f"<b>{sym}</b>",
            showarrow=False,
            font=dict(size=11, color=col),
        ))

    fig = go.Figure(data=traces_all)
    fig.update_layout(
        title=dict(text="<b>All 14 Bravais Lattices</b>", x=0.5, font=dict(size=20)),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            annotations=annotations,
            bgcolor="#0f1117",
            aspectmode="data",
        ),
        paper_bgcolor="#0f1117",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(l=0, r=0, t=60, b=0),
        height=620,
    )
    return fig


def cell_to_vectors_params(a, b, c, alpha_deg, beta_deg, gamma_deg) -> np.ndarray:
    alpha = math.radians(alpha_deg)
    beta  = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    ax = a
    bx = b * math.cos(gamma)
    by = b * math.sin(gamma)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta)*math.cos(gamma)) / math.sin(gamma)
    cz = math.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    return np.array([[ax, 0, 0],[bx, by, 0],[cx, cy, cz]])


# ─────────────────────────────────────────────────────────────────────────────
# Main Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bravais Lattice Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    h1, h2, h3 { color: #ecf0f1; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid;
    }
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Bravais Lattice Viewer")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload a CIF file",
        type=["cif"],
        help="Crystallographic Information File (.cif)",
    )

    st.markdown("---")
    st.subheader("⚙️ Display Options")
    show_atoms  = st.checkbox("Show atomic positions", value=True)
    supercell_n = st.slider("Supercell repeats", 1, 3, 1,
                             help="Repeat unit cell N×N×N times")
    show_all14  = st.checkbox("Show all 14 Bravais lattices", value=False)

    st.markdown("---")
    st.markdown("""
    **About**  
    Parses CIF files using [gemmi](https://gemmi.readthedocs.io/) and  
    renders interactive 3-D lattice visualisations with Plotly.
    """)

# ── Main panel ───────────────────────────────────────────────────────────────
st.title("🔷 Bravais Lattice Visualiser")

if not GEMMI_OK:
    st.error(
        "**gemmi** is not installed. Please run:\n"
        "```\npip install gemmi plotly streamlit\n```\nthen restart."
    )
    st.stop()

# ── All-14 panel (no upload needed) ──────────────────────────────────────────
if show_all14:
    st.subheader("All 14 Bravais Lattices")
    st.plotly_chart(build_all14_figure(), use_container_width=True)
    st.markdown("---")

# ── CIF upload panel ─────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈  Upload a CIF file in the sidebar to begin.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **What is a CIF file?**  
        A Crystallographic Information File (CIF) stores  
        structural data for crystals, including unit-cell  
        parameters, space group, and atomic coordinates.
        """)
    with col2:
        st.markdown("""
        **What is a Bravais lattice?**  
        One of the 14 distinct periodic arrangements of  
        points in 3-D space that describe all possible  
        crystal translational symmetry types.
        """)
    with col3:
        st.markdown("""
        **Features**  
        • Interactive 3-D cell view  
        • Centering-point visualisation  
        • Atomic position overlay  
        • Supercell expansion  
        • All-14 overview panel  
        """)
    st.stop()

# ── Parse CIF ────────────────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

try:
    doc = gemmi.cif.read(tmp_path)
    st.success(f"✅ Parsed **{uploaded.name}** — {len(doc)} block(s) found.")
except Exception as e:
    st.error(f"Failed to parse CIF: {e}")
    os.unlink(tmp_path)
    st.stop()

# Block selector (some CIFs have multiple structures)
block_names = [b.name for b in doc]
if len(block_names) > 1:
    chosen_block = st.selectbox("Select data block", block_names)
else:
    chosen_block = block_names[0]

block = doc[chosen_block]

# ── Extract structure ─────────────────────────────────────────────────────────
try:
    structure = gemmi.make_small_structure_from_block(block)
    cell = structure.cell
    sg   = structure.find_spacegroup()
    if sg is None:
        sg = gemmi.SpaceGroup(1)   # fallback: P1
except Exception as e:
    st.error(f"Could not read structure: {e}")
    os.unlink(tmp_path)
    st.stop()

bravais = classify_bravais(cell, sg)
vecs    = cell_to_vectors(cell)

# ── Atom positions ───────────────────────────────────────────────────────────
atom_cart = []
try:
    for site in structure.sites:
        frac = site.fract
        cart = cell.orthogonalize(frac)
        atom_cart.append((np.array([cart.x, cart.y, cart.z]), site.type_symbol or "?"))
except Exception:
    pass

os.unlink(tmp_path)

# ── Info cards ───────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns([1.6, 1.6, 1.8])

with col_a:
    st.subheader("🔷 Bravais Lattice")
    st.markdown(f"""
    <div class="metric-card" style="border-color:{bravais['color']}">
        <h3 style="color:{bravais['color']};margin:0">{bravais['symbol']}</h3>
        <p style="margin:4px 0;color:#ecf0f1;font-size:1.1em">{bravais['system']}</p>
        <p style="margin:0;color:#bdc3c7">{bravais['centering']} centering</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Space group:** {sg.xhm() if sg else 'Unknown'}")
    st.markdown(f"**Hall symbol:** {sg.hall if sg else '—'}")
    st.markdown(f"**No.:** {sg.number if sg else '—'}")

with col_b:
    st.subheader("📐 Unit Cell Parameters")
    p = {
        "a": f"{cell.a:.4f} Å",
        "b": f"{cell.b:.4f} Å",
        "c": f"{cell.c:.4f} Å",
        "α": f"{cell.alpha:.3f}°",
        "β": f"{cell.beta:.3f}°",
        "γ": f"{cell.gamma:.3f}°",
    }
    for k, v in p.items():
        st.markdown(f"**{k}** = {v}")

with col_c:
    st.subheader("📊 Crystal Metrics")
    vol = cell.volume
    st.metric("Volume", f"{vol:.2f} Å³")
    st.metric("Atoms in unit cell", len(atom_cart))
    # Estimate density bucket
    if vol < 100:
        density_hint = "Very compact"
    elif vol < 500:
        density_hint = "Compact"
    elif vol < 2000:
        density_hint = "Moderate"
    else:
        density_hint = "Large/porous"
    st.metric("Cell volume class", density_hint)

st.markdown("---")

# ── 3-D visualisation ─────────────────────────────────────────────────────────
st.subheader("🧊 3-D Lattice View")

fig = build_lattice_figure(
    vecs, bravais,
    atom_positions_cart=atom_cart if show_atoms else None,
    show_atoms=show_atoms,
    supercell=supercell_n,
)
st.plotly_chart(fig, use_container_width=True)

# ── Atom table ────────────────────────────────────────────────────────────────
if show_atoms and atom_cart:
    with st.expander(f"🔬 Atomic positions ({len(atom_cart)} atoms)"):
        import pandas as pd
        rows = [{"Element": name, "x (Å)": f"{pos[0]:.4f}",
                 "y (Å)": f"{pos[1]:.4f}", "z (Å)": f"{pos[2]:.4f}"}
                for pos, name in atom_cart]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

# ── Lattice system reference ──────────────────────────────────────────────────
with st.expander("📚 All 14 Bravais Lattices — Quick Reference"):
    import pandas as pd
    ref = []
    for sym, info in BRAVAIS_INFO.items():
        ref.append({"Symbol": sym, "Crystal System": info["system"],
                    "Centering": info["centering"]})
    df = pd.DataFrame(ref)
    # Highlight the detected one
    def highlight_row(row):
        if row["Symbol"] == bravais["symbol"]:
            return [f"background-color:{bravais['color']}33"] * len(row)
        return [""] * len(row)
    st.dataframe(df.style.apply(highlight_row, axis=1),
                 use_container_width=True, hide_index=True)