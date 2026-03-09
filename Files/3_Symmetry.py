"""
HKL Phase Calculator for Minerals
Calculates structure factors F(hkl) and phases using crystallographic data.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Atomic scattering factor coefficients (a1,b1,a2,b2,a3,b3,a4,b4,c)
# Source: International Tables for Crystallography Vol. C
# ─────────────────────────────────────────────────────────────────────────────
SCATTERING_FACTORS = {
    "Si": ([6.2915, 3.0353, 1.9891, 0.5399, 1.1410], [2.4386, 32.3337, 0.6785, 81.6937, 0.0], 1.1407),
    "O":  ([3.0485, 2.2868, 1.0624, 0.1156, 0.0],    [13.2771, 5.7011, 0.3239, 32.9089, 0.0], 0.3006),
    "Al": ([6.4202, 1.9002, 1.5936, 1.9646, 0.0],    [3.0387, 0.7426, 31.5472, 85.0886, 0.0], 1.1151),
    "Ca": ([8.6266, 7.3873, 1.5899, 1.0211, 0.0],    [10.4421, 0.6599, 85.7484, 178.437, 0.0], 1.3751),
    "Fe": ([11.7695, 7.3573, 3.5222, 2.3045, 0.0],   [4.7611, 0.3072, 15.3535, 76.8805, 0.0], 1.0369),
    "Mg": ([5.4204, 2.1735, 1.2269, 2.3073, 0.0],    [2.8275, 79.2611, 0.3808, 7.1937, 0.0], 0.8584),
    "Na": ([6.4202, 1.9002, 1.5936, 1.9646, 0.0],    [3.0387, 0.7426, 31.5472, 85.0886, 0.0], 0.4655),
    "K":  ([8.2186, 7.4398, 1.0519, 0.8659, 0.0],    [12.7949, 0.7748, 213.187, 41.6841, 0.0], 1.4228),
    "Ti": ([9.7595, 7.3558, 1.6991, 1.9021, 0.0],    [7.8508, 0.5, 35.6338, 116.105, 0.0], 1.2807),
    "Mn": ([11.2819, 7.3573, 3.5490, 2.1645, 0.0],   [5.3409, 0.3432, 17.8674, 83.7543, 0.0], 1.0896),
    "H":  ([0.4899, 0.2620, 0.1967, 0.0490, 0.0],    [20.6593, 7.7404, 49.5519, 2.2016, 0.0], 0.0010),
    "C":  ([2.3100, 1.0200, 1.5886, 0.8650, 0.0],    [20.8439, 10.2075, 0.5687, 51.6512, 0.0], 0.2156),
}

# ─────────────────────────────────────────────────────────────────────────────
# Predefined minerals
# ─────────────────────────────────────────────────────────────────────────────
MINERALS = {
    "Quartz (SiO₂)": {
        "system": "Hexagonal",
        "a": 4.9133, "b": 4.9133, "c": 5.4053,
        "alpha": 90, "beta": 90, "gamma": 120,
        "atoms": [
            {"element": "Si", "x": 0.4697, "y": 0.0000, "z": 0.0000, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.0000, "y": 0.4697, "z": 0.6667, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.5303, "y": 0.5303, "z": 0.3333, "occ": 1.0, "Biso": 0.5},
            {"element": "O",  "x": 0.4135, "y": 0.2669, "z": 0.1188, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.2669, "y": 0.4135, "z": 0.8812, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.7331, "y": 0.1466, "z": 0.4521, "occ": 1.0, "Biso": 0.8},
        ],
    },
    "Calcite (CaCO₃)": {
        "system": "Trigonal",
        "a": 4.9896, "b": 4.9896, "c": 17.0610,
        "alpha": 90, "beta": 90, "gamma": 120,
        "atoms": [
            {"element": "Ca", "x": 0.0000, "y": 0.0000, "z": 0.0000, "occ": 1.0, "Biso": 0.6},
            {"element": "C",  "x": 0.0000, "y": 0.0000, "z": 0.2500, "occ": 1.0, "Biso": 0.5},
            {"element": "O",  "x": 0.2573, "y": 0.0000, "z": 0.2500, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.0000, "y": 0.2573, "z": 0.2500, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.7427, "y": 0.7427, "z": 0.2500, "occ": 1.0, "Biso": 1.0},
        ],
    },
    "Forsterite (Mg₂SiO₄)": {
        "system": "Orthorhombic",
        "a": 4.7540, "b": 10.1971, "c": 5.9806,
        "alpha": 90, "beta": 90, "gamma": 90,
        "atoms": [
            {"element": "Mg", "x": 0.0000, "y": 0.0000, "z": 0.0000, "occ": 1.0, "Biso": 0.5},
            {"element": "Mg", "x": 0.5000, "y": 0.5000, "z": 0.0000, "occ": 1.0, "Biso": 0.5},
            {"element": "Mg", "x": 0.0000, "y": 0.2211, "z": 0.5000, "occ": 1.0, "Biso": 0.5},
            {"element": "Mg", "x": 0.5000, "y": 0.7789, "z": 0.5000, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.0000, "y": 0.0940, "z": 0.4232, "occ": 1.0, "Biso": 0.4},
            {"element": "Si", "x": 0.5000, "y": 0.4060, "z": 0.4232, "occ": 1.0, "Biso": 0.4},
            {"element": "O",  "x": 0.0000, "y": 0.0926, "z": 0.7656, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.5000, "y": 0.4074, "z": 0.7656, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.0000, "y": 0.4512, "z": 0.2199, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.5000, "y": 0.0488, "z": 0.2199, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.2724, "y": 0.1643, "z": 0.2801, "occ": 1.0, "Biso": 0.8},
            {"element": "O",  "x": 0.7276, "y": 0.8357, "z": 0.2801, "occ": 1.0, "Biso": 0.8},
        ],
    },
    "Albite (NaAlSi₃O₈)": {
        "system": "Triclinic",
        "a": 8.1360, "b": 12.7870, "c": 7.1582,
        "alpha": 94.253, "beta": 116.605, "gamma": 87.756,
        "atoms": [
            {"element": "Na", "x": 0.2690, "y": 0.9890, "z": 0.1470, "occ": 1.0, "Biso": 1.5},
            {"element": "Al", "x": 0.0088, "y": 0.1680, "z": 0.2082, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.0036, "y": 0.8200, "z": 0.2390, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.6900, "y": 0.1120, "z": 0.3150, "occ": 1.0, "Biso": 0.5},
            {"element": "Si", "x": 0.6813, "y": 0.8820, "z": 0.3610, "occ": 1.0, "Biso": 0.5},
            {"element": "O",  "x": 0.0055, "y": 0.1310, "z": 0.9680, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.5934, "y": 0.9970, "z": 0.2800, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.8194, "y": 0.1085, "z": 0.1902, "occ": 1.0, "Biso": 1.0},
            {"element": "O",  "x": 0.0203, "y": 0.3027, "z": 0.2700, "occ": 1.0, "Biso": 1.0},
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Crystallography functions
# ─────────────────────────────────────────────────────────────────────────────

def deg2rad(deg):
    return np.radians(deg)

def compute_metric_tensor(a, b, c, alpha, beta, gamma):
    """Compute the metric tensor G for the unit cell."""
    ca, cb, cg = np.cos(deg2rad(alpha)), np.cos(deg2rad(beta)), np.cos(deg2rad(gamma))
    G = np.array([
        [a*a,    a*b*cg, a*c*cb],
        [a*b*cg, b*b,    b*c*ca],
        [a*c*cb, b*c*ca, c*c  ]
    ])
    return G

def d_spacing(h, k, l, a, b, c, alpha, beta, gamma):
    """Calculate d-spacing for (hkl) reflection."""
    G = compute_metric_tensor(a, b, c, alpha, beta, gamma)
    Ginv = np.linalg.inv(G)
    hkl = np.array([h, k, l])
    q2 = hkl @ Ginv @ hkl
    return 1.0 / np.sqrt(q2) if q2 > 0 else np.inf

def atomic_scattering_factor(element, sin_theta_over_lambda):
    """Compute f(s) using 4-Gaussian approximation."""
    if element not in SCATTERING_FACTORS:
        return 1.0
    a_coeff, b_coeff, c = SCATTERING_FACTORS[element]
    s2 = sin_theta_over_lambda**2
    f = c
    for a_i, b_i in zip(a_coeff, b_coeff):
        f += a_i * np.exp(-b_i * s2)
    return f

def debye_waller(Biso, sin_theta_over_lambda):
    """Debye-Waller temperature factor."""
    return np.exp(-Biso * sin_theta_over_lambda**2)

def structure_factor(h, k, l, atoms, a, b, c, alpha, beta, gamma):
    """
    Compute the complex structure factor F(hkl).
    Returns (|F|, phase_degrees, F_real, F_imag)
    """
    d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma)
    sin_tol = 1.0 / (2 * d) if d > 0 else 0.0

    F = 0.0 + 0.0j
    for atom in atoms:
        x, y, z = atom["x"], atom["y"], atom["z"]
        elem     = atom["element"]
        occ      = atom.get("occ", 1.0)
        Biso     = atom.get("Biso", 0.5)

        f = atomic_scattering_factor(elem, sin_tol)
        DW = debye_waller(Biso, sin_tol)
        phase = 2 * np.pi * (h*x + k*y + l*z)
        F += occ * f * DW * np.exp(1j * phase)

    amplitude = abs(F)
    phase_deg = np.degrees(np.angle(F))
    return amplitude, phase_deg, F.real, F.imag

def generate_hkl_list(hmax, kmax, lmax):
    """Generate all (h,k,l) triplets within limits."""
    hkl = []
    for h in range(-hmax, hmax + 1):
        for k in range(-kmax, kmax + 1):
            for l in range(-lmax, lmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl.append((h, k, l))
    return hkl

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="HKL Phase Calculator", page_icon="💎", layout="wide")

st.title("💎 HKL Phase Calculator for Minerals")
st.markdown(
    "Calculate X-ray structure factors **F(hkl)**, amplitudes, and phases "
    "for crystalline minerals using atomic scattering factors and Debye-Waller correction."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    mineral_choice = st.selectbox("Select Mineral", list(MINERALS.keys()))
    mineral = MINERALS[mineral_choice]

    st.subheader("Unit Cell Parameters")
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("a (Å)", value=float(mineral["a"]), format="%.4f")
        b = st.number_input("b (Å)", value=float(mineral["b"]), format="%.4f")
        c = st.number_input("c (Å)", value=float(mineral["c"]), format="%.4f")
    with col2:
        alpha = st.number_input("α (°)", value=float(mineral["alpha"]), format="%.3f")
        beta  = st.number_input("β (°)", value=float(mineral["beta"]),  format="%.3f")
        gamma = st.number_input("γ (°)", value=float(mineral["gamma"]), format="%.3f")

    st.subheader("HKL Range")
    hmax = st.slider("h_max", 1, 6, 3)
    kmax = st.slider("k_max", 1, 6, 3)
    lmax = st.slider("l_max", 1, 6, 3)

    st.subheader("Filtering")
    min_amplitude = st.slider("Min |F| to display", 0.0, 50.0, 0.0, 0.5)
    wavelength    = st.number_input("X-ray wavelength λ (Å)", value=1.5406, format="%.4f",
                                    help="Cu Kα = 1.5406 Å")

# ── Calculation ───────────────────────────────────────────────────────────────
atoms = mineral["atoms"]
hkl_list = generate_hkl_list(hmax, kmax, lmax)

results = []
for h, k, l in hkl_list:
    d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma)
    # Apply Bragg condition: 2d sinθ = λ → sinθ = λ/(2d)
    sin_theta = wavelength / (2 * d) if d > 0 else 999
    if sin_theta > 1.0:
        continue  # unphysical reflection

    two_theta = np.degrees(2 * np.arcsin(sin_theta))
    amp, phase, F_re, F_im = structure_factor(h, k, l, atoms, a, b, c, alpha, beta, gamma)

    if amp >= min_amplitude:
        results.append({
            "h": h, "k": k, "l": l,
            "d (Å)": round(d, 4),
            "2θ (°)": round(two_theta, 3),
            "|F(hkl)|": round(amp, 3),
            "Phase (°)": round(phase, 2),
            "F_real": round(F_re, 3),
            "F_imag": round(F_im, 3),
            "I (∝|F|²)": round(amp**2, 2),
        })

df = pd.DataFrame(results)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Table", "🌀 Phase Diagram", "📈 Diffraction Pattern", "🔬 Argand Diagram"]
)

with tab1:
    st.subheader(f"Structure Factors — {mineral_choice}")
    st.markdown(f"**Crystal system:** {mineral['system']} | **Reflections found:** {len(df)}")
    if not df.empty:
        st.dataframe(
            df.style.background_gradient(subset=["|F(hkl)|", "I (∝|F|²)"], cmap="plasma"),
            use_container_width=True, height=500
        )
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, "hkl_phases.csv", "text/csv")
    else:
        st.warning("No reflections satisfy the current filters.")

with tab2:
    st.subheader("Phase Distribution (Polar Plot)")
    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=df["|F(hkl)|"],
            theta=df["Phase (°)"],
            mode="markers",
            marker=dict(
                size=6,
                color=df["|F(hkl)|"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="|F(hkl)|"),
            ),
            text=[f"({r.h},{r.k},{r.l})" for r in df.itertuples()],
            hovertemplate="<b>%{text}</b><br>|F|=%{r:.2f}<br>φ=%{theta:.1f}°<extra></extra>",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(showticklabels=True)),
            title=f"Phase vs Amplitude — {mineral_choice}",
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Phase histogram
        fig2 = px.histogram(df, x="Phase (°)", nbins=36, color_discrete_sequence=["#6c63ff"],
                            title="Phase Angle Distribution")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Simulated Powder Diffraction Pattern")
    if not df.empty:
        df_sorted = df.sort_values("2θ (°)")
        fig = go.Figure()
        for _, row in df_sorted.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["2θ (°)"], row["2θ (°)"]],
                y=[0, row["I (∝|F|²)"]],
                mode="lines",
                line=dict(color="steelblue", width=2),
                showlegend=False,
                hovertemplate=f"({int(row['h'])},{int(row['k'])},{int(row['l'])})<br>"
                              f"2θ={row['2θ (°)']:.2f}°<br>I={row['I (∝|F|²)']:.1f}<extra></extra>",
            ))
        fig.update_layout(
            title=f"Powder Diffraction Pattern — {mineral_choice} (λ={wavelength} Å)",
            xaxis_title="2θ (°)",
            yaxis_title="Intensity |F(hkl)|²",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Argand Diagram (Complex Plane)")
    st.markdown("Each reflection is a vector in the complex plane. Length = |F|, angle = phase φ.")
    if not df.empty:
        # Pick top N by amplitude for readability
        n_show = st.slider("Number of reflections to show", 5, min(100, len(df)), min(30, len(df)))
        df_top = df.nlargest(n_show, "|F(hkl)|")

        fig = go.Figure()
        for _, row in df_top.iterrows():
            fig.add_trace(go.Scatter(
                x=[0, row["F_real"]], y=[0, row["F_imag"]],
                mode="lines+markers",
                marker=dict(size=[4, 8]),
                line=dict(width=1.5),
                name=f"({int(row['h'])},{int(row['k'])},{int(row['l'])})",
                hovertemplate=f"({int(row['h'])},{int(row['k'])},{int(row['l'])})<br>"
                              f"F_re={row['F_real']:.2f}, F_im={row['F_imag']:.2f}<br>"
                              f"|F|={row['|F(hkl)|']:.2f}, φ={row['Phase (°)']:.1f}°<extra></extra>",
                showlegend=False,
            ))

        fig.add_trace(go.Scatter(
            x=df_top["F_real"], y=df_top["F_imag"],
            mode="markers+text",
            marker=dict(size=8, color=df_top["|F(hkl)|"], colorscale="Plasma", showscale=True,
                        colorbar=dict(title="|F|")),
            text=[f"({int(r.h)}{int(r.k)}{int(r.l)})" for r in df_top.itertuples()],
            textposition="top center",
            textfont=dict(size=9),
            showlegend=False,
        ))

        fig.update_layout(
            title="Argand Diagram of Structure Factors",
            xaxis_title="F_real",
            yaxis_title="F_imag",
            xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="gray"),
            yaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="gray", scaleanchor="x"),
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Atomic scattering factors: 4-Gaussian fit (Int. Tables Vol. C). "
    "Temperature correction via isotropic Debye-Waller factor. "
    "Reflections filtered by Bragg condition (sinθ/λ ≤ 1)."
)