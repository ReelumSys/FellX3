"""
Bravais Lattice Cell Visualizer for Minerals
3D interactive visualization of all 14 Bravais lattice types,
with mineral-specific unit cells and atom positions.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go




im = 'images/favicon.png'
st.set_page_config(
    page_title="FellX v0.8",
    page_icon=im,
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Bravais lattice definitions (all 14)
# centering: P=primitive, I=body, F=face, C=base, R=rhombohedral
# ─────────────────────────────────────────────────────────────────────────────
BRAVAIS_LATTICES = {
    # Triclinic
    "Triclinic P (aP)": {
        "system": "Triclinic", "centering": "P",
        "a": 1.0, "b": 1.3, "c": 1.6,
        "alpha": 75, "beta": 85, "gamma": 95,
        "color": "#e74c3c", "symbol": "aP",
    },
    # Monoclinic
    "Monoclinic P (mP)": {
        "system": "Monoclinic", "centering": "P",
        "a": 1.0, "b": 1.4, "c": 1.2,
        "alpha": 90, "beta": 105, "gamma": 90,
        "color": "#e67e22", "symbol": "mP",
    },
    "Monoclinic C (mC)": {
        "system": "Monoclinic", "centering": "C",
        "a": 1.0, "b": 1.4, "c": 1.2,
        "alpha": 90, "beta": 105, "gamma": 90,
        "color": "#f39c12", "symbol": "mC",
    },
    # Orthorhombic
    "Orthorhombic P (oP)": {
        "system": "Orthorhombic", "centering": "P",
        "a": 1.0, "b": 1.3, "c": 1.6,
        "alpha": 90, "beta": 90, "gamma": 90,
        "color": "#2ecc71", "symbol": "oP",
    },
    "Orthorhombic C (oC)": {
        "system": "Orthorhombic", "centering": "C",
        "a": 1.0, "b": 1.3, "c": 1.6,
        "alpha": 90, "beta": 90, "gamma": 90,
        "color": "#27ae60", "symbol": "oC",
    },
    "Orthorhombic I (oI)": {
        "system": "Orthorhombic", "centering": "I",
        "a": 1.0, "b": 1.3, "c": 1.6,
        "alpha": 90, "beta": 90, "gamma": 90,
        "color": "#1abc9c", "symbol": "oI",
    },
    "Orthorhombic F (oF)": {
        "system": "Orthorhombic", "centering": "F",
        "a": 1.0, "b": 1.3, "c": 1.6,
        "alpha": 90, "beta": 90, "gamma": 90,
        "color": "#16a085", "symbol": "oF",
    },
    # Tetragonal
    "Tetragonal P (tP)": {
        "system": "Tetragonal", "centering": "P",
        "a": 1.0, "b": 1.0, "c": 1.5,
        "alpha": 90, "beta": 90, "gamma": 90,
        "color": "#3498db", "symbol": "tP",
    },
    "Tetragonal I (tI)": {
        "system": "Tetragonal", "centering": "I",
        "a": 1.0, "b": 1.0, "c": 1.5,
        "alpha": 90, "beta": 90, "gamma": 90,
        "color": "#2980b9", "symbol": "tI",
    },
    # Trigonal / Rhombohedral
    "Rhombohedral R (hR)": {
        "system": "Trigonal", "centering": "R",
        "a": 1.0, "b": 1.0, "c": 1.0,
        "alpha": 70, "beta": 70, "gamma": 70,
        "color": "#9b59b6", "symbol": "hR",
    },
    # Hexagonal
    "Hexagonal P (hP)": {
        "system": "Hexagonal", "centering": "P",
        "a": 1.0, "b": 1.0, "c": 1.6,
        "alpha": 90, "beta": 90, "gamma": 120,
        "color": "#8e44ad", "symbol": "hP",
    },
    # Cubic
    "Cubic P (cP)": {
        "system": "Cubic", "centering": "P",
        "a": 1.0, "b": 1.0, "c": 1.0,
        "alpha": 90, "beta": 90, "gamma": 90,
        "color": "#c0392b", "symbol": "cP",
    },
    "Cubic I (cI)": {
        "system": "Cubic", "centering": "I",
        "a": 1.0, "b": 1.0, "c": 1.0,
        "alpha": 90, "beta": 90, "gamma": 90,
        "color": "#e74c3c", "symbol": "cI",
    },
    "Cubic F (cF)": {
        "system": "Cubic", "centering": "F",
        "a": 1.0, "b": 1.0, "c": 1.0,
        "alpha": 90, "beta": 90, "gamma": 90,
        "color": "#ec407a", "symbol": "cF",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Mineral → Bravais lattice mapping + real unit cell parameters
# ─────────────────────────────────────────────────────────────────────────────
MINERALS = {
    "Quartz (SiO₂)": {
        "bravais": "Hexagonal P (hP)",
        "a": 4.9133, "b": 4.9133, "c": 5.4053,
        "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
        "space_group": "P3₂21  (No. 154)",
        "atoms": [
            {"element": "Si", "x": 0.4697, "y": 0.0000, "z": 0.0000, "color": "#4fc3f7", "r": 0.12},
            {"element": "Si", "x": 0.0000, "y": 0.4697, "z": 0.6667, "color": "#4fc3f7", "r": 0.12},
            {"element": "Si", "x": 0.5303, "y": 0.5303, "z": 0.3333, "color": "#4fc3f7", "r": 0.12},
            {"element": "O",  "x": 0.4135, "y": 0.2669, "z": 0.1188, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.2669, "y": 0.4135, "z": 0.8812, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.7331, "y": 0.1466, "z": 0.4521, "color": "#ef5350", "r": 0.09},
        ],
    },
    "Calcite (CaCO₃)": {
        "bravais": "Rhombohedral R (hR)",
        "a": 4.9896, "b": 4.9896, "c": 17.0610,
        "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
        "space_group": "R3̄c  (No. 167)",
        "atoms": [
            {"element": "Ca", "x": 0.0000, "y": 0.0000, "z": 0.0000, "color": "#ab47bc", "r": 0.15},
            {"element": "C",  "x": 0.0000, "y": 0.0000, "z": 0.2500, "color": "#78909c", "r": 0.08},
            {"element": "O",  "x": 0.2573, "y": 0.0000, "z": 0.2500, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.0000, "y": 0.2573, "z": 0.2500, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.7427, "y": 0.7427, "z": 0.2500, "color": "#ef5350", "r": 0.09},
        ],
    },
    "Forsterite (Mg₂SiO₄)": {
        "bravais": "Orthorhombic F (oF)",
        "a": 4.7540, "b": 10.1971, "c": 5.9806,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        "space_group": "Pbnm  (No. 62)",
        "atoms": [
            {"element": "Mg", "x": 0.0000, "y": 0.0000, "z": 0.0000, "color": "#66bb6a", "r": 0.13},
            {"element": "Mg", "x": 0.5000, "y": 0.5000, "z": 0.0000, "color": "#66bb6a", "r": 0.13},
            {"element": "Mg", "x": 0.0000, "y": 0.2211, "z": 0.5000, "color": "#66bb6a", "r": 0.13},
            {"element": "Mg", "x": 0.5000, "y": 0.7789, "z": 0.5000, "color": "#66bb6a", "r": 0.13},
            {"element": "Si", "x": 0.0000, "y": 0.0940, "z": 0.4232, "color": "#4fc3f7", "r": 0.12},
            {"element": "Si", "x": 0.5000, "y": 0.4060, "z": 0.4232, "color": "#4fc3f7", "r": 0.12},
            {"element": "O",  "x": 0.0000, "y": 0.0926, "z": 0.7656, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.5000, "y": 0.4074, "z": 0.7656, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.0000, "y": 0.4512, "z": 0.2199, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.5000, "y": 0.0488, "z": 0.2199, "color": "#ef5350", "r": 0.09},
        ],
    },
    "Albite (NaAlSi₃O₈)": {
        "bravais": "Triclinic P (aP)",
        "a": 8.1360, "b": 12.7870, "c": 7.1582,
        "alpha": 94.253, "beta": 116.605, "gamma": 87.756,
        "space_group": "P1̄  (No. 2)",
        "atoms": [
            {"element": "Na", "x": 0.2690, "y": 0.9890, "z": 0.1470, "color": "#ffca28", "r": 0.14},
            {"element": "Al", "x": 0.0088, "y": 0.1680, "z": 0.2082, "color": "#ff8a65", "r": 0.11},
            {"element": "Si", "x": 0.0036, "y": 0.8200, "z": 0.2390, "color": "#4fc3f7", "r": 0.12},
            {"element": "Si", "x": 0.6900, "y": 0.1120, "z": 0.3150, "color": "#4fc3f7", "r": 0.12},
            {"element": "Si", "x": 0.6813, "y": 0.8820, "z": 0.3610, "color": "#4fc3f7", "r": 0.12},
            {"element": "O",  "x": 0.0055, "y": 0.1310, "z": 0.9680, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.5934, "y": 0.9970, "z": 0.2800, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.8194, "y": 0.1085, "z": 0.1902, "color": "#ef5350", "r": 0.09},
            {"element": "O",  "x": 0.0203, "y": 0.3027, "z": 0.2700, "color": "#ef5350", "r": 0.09},
        ],
    },
    "Halite (NaCl)": {
        "bravais": "Cubic F (cF)",
        "a": 5.6402, "b": 5.6402, "c": 5.6402,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        "space_group": "Fm3̄m  (No. 225)",
        "atoms": [
            {"element": "Na", "x": 0.0, "y": 0.0, "z": 0.0, "color": "#ffca28", "r": 0.14},
            {"element": "Na", "x": 0.5, "y": 0.5, "z": 0.0, "color": "#ffca28", "r": 0.14},
            {"element": "Na", "x": 0.5, "y": 0.0, "z": 0.5, "color": "#ffca28", "r": 0.14},
            {"element": "Na", "x": 0.0, "y": 0.5, "z": 0.5, "color": "#ffca28", "r": 0.14},
            {"element": "Cl", "x": 0.5, "y": 0.0, "z": 0.0, "color": "#b0bec5", "r": 0.17},
            {"element": "Cl", "x": 0.0, "y": 0.5, "z": 0.0, "color": "#b0bec5", "r": 0.17},
            {"element": "Cl", "x": 0.0, "y": 0.0, "z": 0.5, "color": "#b0bec5", "r": 0.17},
            {"element": "Cl", "x": 0.5, "y": 0.5, "z": 0.5, "color": "#b0bec5", "r": 0.17},
        ],
    },
    "Pyrite (FeS₂)": {
        "bravais": "Cubic P (cP)",
        "a": 5.4166, "b": 5.4166, "c": 5.4166,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        "space_group": "Pa3̄  (No. 205)",
        "atoms": [
            {"element": "Fe", "x": 0.0,   "y": 0.0,   "z": 0.0,   "color": "#ffd54f", "r": 0.13},
            {"element": "Fe", "x": 0.5,   "y": 0.0,   "z": 0.5,   "color": "#ffd54f", "r": 0.13},
            {"element": "Fe", "x": 0.0,   "y": 0.5,   "z": 0.5,   "color": "#ffd54f", "r": 0.13},
            {"element": "Fe", "x": 0.5,   "y": 0.5,   "z": 0.0,   "color": "#ffd54f", "r": 0.13},
            {"element": "S",  "x": 0.385, "y": 0.385, "z": 0.385, "color": "#fff176", "r": 0.11},
            {"element": "S",  "x": 0.615, "y": 0.615, "z": 0.385, "color": "#fff176", "r": 0.11},
            {"element": "S",  "x": 0.615, "y": 0.385, "z": 0.615, "color": "#fff176", "r": 0.11},
            {"element": "S",  "x": 0.385, "y": 0.615, "z": 0.615, "color": "#fff176", "r": 0.11},
        ],
    },
}

CRYSTAL_SYSTEM_INFO = {
    "Triclinic":     {"axes": "a≠b≠c",  "angles": "α≠β≠γ≠90°", "lattices": ["P"],           "minerals": "Albite, Kyanite, Microcline"},
    "Monoclinic":    {"axes": "a≠b≠c",  "angles": "α=γ=90°, β≠90°", "lattices": ["P","C"],  "minerals": "Orthoclase, Gypsum, Augite"},
    "Orthorhombic":  {"axes": "a≠b≠c",  "angles": "α=β=γ=90°", "lattices": ["P","C","I","F"],"minerals": "Forsterite, Aragonite, Topaz"},
    "Tetragonal":    {"axes": "a=b≠c",  "angles": "α=β=γ=90°", "lattices": ["P","I"],        "minerals": "Zircon, Rutile, Vesuvianite"},
    "Trigonal":      {"axes": "a=b≠c",  "angles": "α=β=90°, γ=120°", "lattices": ["R"],      "minerals": "Calcite, Dolomite, Quartz"},
    "Hexagonal":     {"axes": "a=b≠c",  "angles": "α=β=90°, γ=120°", "lattices": ["P"],      "minerals": "Quartz, Apatite, Beryl"},
    "Cubic":         {"axes": "a=b=c",  "angles": "α=β=γ=90°", "lattices": ["P","I","F"],    "minerals": "Halite, Pyrite, Garnet, Fluorite"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def cell_vectors(a, b, c, alpha, beta, gamma):
    """Return Cartesian vectors a1, a2, a3 for the unit cell."""
    al, be, ga = np.radians(alpha), np.radians(beta), np.radians(gamma)
    # a along x
    a1 = np.array([a, 0, 0])
    # b in xy-plane
    a2 = np.array([b * np.cos(ga), b * np.sin(ga), 0])
    # c general
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c*c - cx*cx - cy*cy, 0))
    a3 = np.array([cx, cy, cz])
    return a1, a2, a3

def frac_to_cart(fx, fy, fz, a1, a2, a3):
    return fx * a1 + fy * a2 + fz * a3

def unit_cell_edges(a1, a2, a3):
    """Return list of (start, end) pairs for the 12 edges of the parallelepiped."""
    O = np.zeros(3)
    corners = {
        '000': O,
        '100': a1,
        '010': a2,
        '001': a3,
        '110': a1 + a2,
        '101': a1 + a3,
        '011': a2 + a3,
        '111': a1 + a2 + a3,
    }
    edges = [
        ('000','100'),('000','010'),('000','001'),
        ('100','110'),('100','101'),
        ('010','110'),('010','011'),
        ('001','101'),('001','011'),
        ('110','111'),('101','111'),('011','111'),
    ]
    return [(corners[s], corners[e]) for s, e in edges]

def centering_points(centering):
    """Return fractional coords of extra lattice points for the centering type."""
    pts = [(0,0,0)]
    if centering == "I":
        pts += [(0.5, 0.5, 0.5)]
    elif centering == "F":
        pts += [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    elif centering == "C":
        pts += [(0.5, 0.5, 0)]
    elif centering == "R":
        pts += [(1/3, 2/3, 1/3), (2/3, 1/3, 2/3)]
    elif centering == "A":
        pts += [(0, 0.5, 0.5)]
    elif centering == "B":
        pts += [(0.5, 0, 0.5)]
    return pts

def sphere_mesh(cx, cy, cz, r, n=12):
    """Return (x,y,z) arrays for a sphere surface (for Mesh3d)."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones(n), np.cos(v))
    return x, y, z

# ─────────────────────────────────────────────────────────────────────────────
# Plot builders
# ─────────────────────────────────────────────────────────────────────────────

def build_cell_figure(a, b, c, alpha, beta, gamma, centering, color,
                      atoms=None, show_atoms=True, show_axes=True,
                      show_centering=True, title="Unit Cell"):
    """Build a 3D Plotly figure of a Bravais unit cell."""
    a1, a2, a3 = cell_vectors(a, b, c, alpha, beta, gamma)
    edges = unit_cell_edges(a1, a2, a3)

    traces = []

    # ── Cell edges ────────────────────────────────────────────────────────────
    for s, e in edges:
        traces.append(go.Scatter3d(
            x=[s[0], e[0]], y=[s[1], e[1]], z=[s[2], e[2]],
            mode="lines",
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Corner lattice points ─────────────────────────────────────────────────
    corners_frac = [
        (0,0,0),(1,0,0),(0,1,0),(0,0,1),
        (1,1,0),(1,0,1),(0,1,1),(1,1,1)
    ]
    cx_list = [frac_to_cart(f[0],f[1],f[2],a1,a2,a3) for f in corners_frac]
    traces.append(go.Scatter3d(
        x=[p[0] for p in cx_list],
        y=[p[1] for p in cx_list],
        z=[p[2] for p in cx_list],
        mode="markers",
        marker=dict(size=7, color=color, symbol="circle",
                    line=dict(color="white", width=1)),
        name="Lattice points",
        hoverinfo="skip",
    ))

    # ── Extra centering points ────────────────────────────────────────────────
    if show_centering and centering != "P":
        extra_frac = centering_points(centering)[1:]
        ec_list = [frac_to_cart(f[0],f[1],f[2],a1,a2,a3) for f in extra_frac]
        traces.append(go.Scatter3d(
            x=[p[0] for p in ec_list],
            y=[p[1] for p in ec_list],
            z=[p[2] for p in ec_list],
            mode="markers",
            marker=dict(size=10, color="#ffffff", symbol="circle",
                        line=dict(color=color, width=3)),
            name=f"{centering}-centering points",
        ))

    # ── Axes arrows (a, b, c) ─────────────────────────────────────────────────
    if show_axes:
        axis_data = [
            (a1 * 1.15, "a", "#f44336"),
            (a2 * 1.15, "b", "#4caf50"),
            (a3 * 1.15, "c", "#2196f3"),
        ]
        for vec, label, acolor in axis_data:
            traces.append(go.Scatter3d(
                x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
                mode="lines+text",
                line=dict(color=acolor, width=5),
                text=["", f"<b>{label}</b>"],
                textfont=dict(size=14, color=acolor),
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            ))

    # ── Atom spheres ──────────────────────────────────────────────────────────
    element_legend = set()
    if show_atoms and atoms:
        for atom in atoms:
            pos = frac_to_cart(atom["x"], atom["y"], atom["z"], a1, a2, a3)
            r_sphere = atom["r"] * max(a, b, c)
            elem = atom["element"]
            acolor = atom["color"]
            show_leg = elem not in element_legend
            element_legend.add(elem)

            traces.append(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers",
                marker=dict(
                    size=r_sphere * 35,
                    color=acolor,
                    opacity=0.85,
                    line=dict(color="white", width=1),
                ),
                name=elem if show_leg else None,
                showlegend=show_leg,
                hovertemplate=f"<b>{elem}</b><br>"
                              f"x={atom['x']:.3f}, y={atom['y']:.3f}, z={atom['z']:.3f}<extra></extra>",
            ))

    # ── Layout ────────────────────────────────────────────────────────────────
    all_pts = [frac_to_cart(f[0],f[1],f[2],a1,a2,a3)
               for f in [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]]
    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]; zs = [p[2] for p in all_pts]
    pad = max(a, b, c) * 0.25
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(range=[min(xs)-pad, max(xs)+pad], showbackground=False,
                       showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[min(ys)-pad, max(ys)+pad], showbackground=False,
                       showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(range=[min(zs)-pad, max(zs)+pad], showbackground=False,
                       showgrid=False, zeroline=False, showticklabels=False),
            bgcolor="#0e1117",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.9)),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig


def build_all14_figure():
    """Build a 3×5 grid overview of all 14 Bravais lattice types."""
    from plotly.subplots import make_subplots

    names = list(BRAVAIS_LATTICES.keys())
    cols = 4
    rows = (len(names) + cols - 1) // cols  # 4 rows

    specs = [[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
    subtitles = names + [""] * (rows * cols - len(names))

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=subtitles,
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    for idx, (name, bl) in enumerate(BRAVAIS_LATTICES.items()):
        row = idx // cols + 1
        col = idx % cols + 1
        a1, a2, a3 = cell_vectors(bl["a"], bl["b"], bl["c"],
                                   bl["alpha"], bl["beta"], bl["gamma"])
        edges = unit_cell_edges(a1, a2, a3)
        color = bl["color"]

        for s, e in edges:
            fig.add_trace(go.Scatter3d(
                x=[s[0], e[0]], y=[s[1], e[1]], z=[s[2], e[2]],
                mode="lines", line=dict(color=color, width=3),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=col)

        corners = [frac_to_cart(f[0],f[1],f[2],a1,a2,a3)
                   for f in [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]]
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in corners],
            y=[p[1] for p in corners],
            z=[p[2] for p in corners],
            mode="markers",
            marker=dict(size=5, color=color),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col)

        if bl["centering"] != "P":
            extra = centering_points(bl["centering"])[1:]
            ec = [frac_to_cart(f[0],f[1],f[2],a1,a2,a3) for f in extra]
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in ec], y=[p[1] for p in ec], z=[p[2] for p in ec],
                mode="markers",
                marker=dict(size=8, color="white", line=dict(color=color, width=2)),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=col)

    scene_settings = dict(
        showbackground=False, showgrid=False,
        zeroline=False, showticklabels=False,
    )
    for i in range(1, rows * cols + 1):
        fig.update_layout(**{
            f"scene{i if i > 1 else ''}": dict(
                xaxis=scene_settings, yaxis=scene_settings, zaxis=scene_settings,
                bgcolor="#0e1117", aspectmode="cube",
                camera=dict(eye=dict(x=1.8, y=1.4, z=1.0)),
            )
        })

    fig.update_layout(
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=10),
        height=950,
        title=dict(text="All 14 Bravais Lattice Types", font=dict(size=18)),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bravais Cell Visualizer", page_icon="🔷", layout="wide")

st.title("🔷 Bravais Lattice Cell Visualizer")
st.markdown(
    "Interactive 3D visualization of the **14 Bravais lattice types** "
    "and real mineral unit cells with atomic positions."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ View Options")

    view_mode = st.radio(
        "Mode",
        ["🔬 Mineral Cell", "📐 Generic Bravais Lattice", "🗂️ All 14 Lattices"],
        index=0,
    )

    if view_mode == "🔬 Mineral Cell":
        mineral_sel = st.selectbox("Select Mineral", list(MINERALS.keys()))

    elif view_mode == "📐 Generic Bravais Lattice":
        lattice_sel = st.selectbox("Select Bravais Lattice", list(BRAVAIS_LATTICES.keys()))

    st.divider()
    show_atoms      = st.toggle("Show Atoms",             value=True)
    show_axes       = st.toggle("Show a/b/c Axes",        value=True)
    show_centering  = st.toggle("Show Centering Points",  value=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

if view_mode == "🗂️ All 14 Lattices":
    st.subheader("All 14 Bravais Lattice Types")
    with st.spinner("Rendering 14 lattices…"):
        fig = build_all14_figure()
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Crystal System Reference")
    rows = []
    for system, info in CRYSTAL_SYSTEM_INFO.items():
        rows.append({
            "Crystal System": system,
            "Axes": info["axes"],
            "Angles": info["angles"],
            "Bravais Types": " · ".join(info["lattices"]),
            "Example Minerals": info["minerals"],
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif view_mode == "📐 Generic Bravais Lattice":
    bl = BRAVAIS_LATTICES[lattice_sel]

    col_info, col_plot = st.columns([1, 2])
    with col_info:
        st.subheader(lattice_sel)
        system = bl["system"]
        info = CRYSTAL_SYSTEM_INFO.get(system, {})
        st.markdown(f"""
| Property | Value |
|---|---|
| **Crystal System** | {system} |
| **Centering** | {bl['centering']} |
| **Symbol** | {bl['symbol']} |
| **Axes** | {info.get('axes', '—')} |
| **Angles** | {info.get('angles', '—')} |
| **Example Minerals** | {info.get('minerals', '—')} |
        """)

        st.divider()
        st.markdown("**Unit Cell Parameters (normalized)**")
        c1, c2 = st.columns(2)
        with c1:
            a = st.number_input("a", value=bl["a"], step=0.05, format="%.3f", key="bl_a")
            b = st.number_input("b", value=bl["b"], step=0.05, format="%.3f", key="bl_b")
            c = st.number_input("c", value=bl["c"], step=0.05, format="%.3f", key="bl_c")
        with c2:
            alpha = st.number_input("α (°)", value=float(bl["alpha"]), step=1.0, key="bl_al")
            beta  = st.number_input("β (°)", value=float(bl["beta"]),  step=1.0, key="bl_be")
            gamma = st.number_input("γ (°)", value=float(bl["gamma"]), step=1.0, key="bl_ga")

    with col_plot:
        fig = build_cell_figure(
            a, b, c, alpha, beta, gamma,
            bl["centering"], bl["color"],
            atoms=None, show_atoms=False,
            show_axes=show_axes, show_centering=show_centering,
            title=f"{lattice_sel}  |  {system}",
        )
        st.plotly_chart(fig, use_container_width=True)

else:  # Mineral Cell
    mineral = MINERALS[mineral_sel]
    bl_name = mineral["bravais"]
    bl = BRAVAIS_LATTICES[bl_name]
    system = bl["system"]
    info = CRYSTAL_SYSTEM_INFO.get(system, {})

    col_info, col_plot = st.columns([1, 2])
    with col_info:
        st.subheader(mineral_sel)
        st.markdown(f"""
| Property | Value |
|---|---|
| **Bravais Lattice** | {bl_name} |
| **Crystal System** | {system} |
| **Space Group** | {mineral.get('space_group','—')} |
| **a** | {mineral['a']} Å |
| **b** | {mineral['b']} Å |
| **c** | {mineral['c']} Å |
| **α** | {mineral['alpha']}° |
| **β** | {mineral['beta']}° |
| **γ** | {mineral['gamma']}° |
        """)

        # Volume
        a1, a2, a3 = cell_vectors(
            mineral["a"], mineral["b"], mineral["c"],
            mineral["alpha"], mineral["beta"], mineral["gamma"]
        )
        V = abs(np.dot(a1, np.cross(a2, a3)))
        st.metric("Cell Volume", f"{V:.2f} Å³")

        st.divider()
        st.markdown("**Atom Legend**")
        elem_seen = {}
        for atom in mineral["atoms"]:
            if atom["element"] not in elem_seen:
                elem_seen[atom["element"]] = atom["color"]
        for elem, col in elem_seen.items():
            st.markdown(
                f'<span style="background:{col};padding:2px 10px;border-radius:4px;'
                f'color:#000;font-weight:bold">{elem}</span>',
                unsafe_allow_html=True
            )

    with col_plot:
        fig = build_cell_figure(
            mineral["a"], mineral["b"], mineral["c"],
            mineral["alpha"], mineral["beta"], mineral["gamma"],
            bl["centering"], bl["color"],
            atoms=mineral["atoms"] if show_atoms else None,
            show_atoms=show_atoms,
            show_axes=show_axes, show_centering=show_centering,
            title=f"{mineral_sel}  ·  {bl_name}  ·  {mineral.get('space_group','')}",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Atom table
    if show_atoms:
        st.subheader("Atomic Positions (fractional coordinates)")
        import pandas as pd
        df_atoms = pd.DataFrame([
            {"Element": a["element"], "x": a["x"], "y": a["y"], "z": a["z"]}
            for a in mineral["atoms"]
        ])
        # Convert fractional to Cartesian
        a1, a2, a3 = cell_vectors(
            mineral["a"], mineral["b"], mineral["c"],
            mineral["alpha"], mineral["beta"], mineral["gamma"]
        )
        cartesian = [frac_to_cart(row.x, row.y, row.z, a1, a2, a3) for row in df_atoms.itertuples()]
        df_atoms["X (Å)"] = [f"{p[0]:.4f}" for p in cartesian]
        df_atoms["Y (Å)"] = [f"{p[1]:.4f}" for p in cartesian]
        df_atoms["Z (Å)"] = [f"{p[2]:.4f}" for p in cartesian]
        st.dataframe(df_atoms, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Unit cell geometry computed via the metric tensor. "
    "Centering: P=primitive · I=body · F=face-centred · C=base-centred · R=rhombohedral. "
    "Atomic radii are scaled for display purposes."
)