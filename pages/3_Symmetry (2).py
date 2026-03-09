import streamlit as st
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import py3Dmol
import tempfile
import pandas as pd


im = 'images/favicon.png'
st.set_page_config(
    page_title="FellX v0.8",
    page_icon=im,
    layout="wide",
)


st.set_page_config(page_title="FellX v0.8", layout="wide")

st.title("🔷 Bravais Lattice & Cell Viewer")
st.write("Upload a CIF file to calculate Bravais lattice and visualize the unit cell.")


def analyze_structure(cif_path):
    structure = Structure.from_file(cif_path)
    sga = SpacegroupAnalyzer(structure, symprec=1e-3)

    crystal_system = sga.get_crystal_system()
    lattice_type = sga.get_lattice_type()
    space_group_symbol = sga.get_space_group_symbol()
    space_group_number = sga.get_space_group_number()

    conventional = sga.get_conventional_standard_structure()
    primitive = sga.get_primitive_standard_structure()

    return {
        "structure": structure,
        "conventional": conventional,
        "primitive": primitive,
        "crystal_system": crystal_system,
        "lattice_type": lattice_type,
        "space_group_symbol": space_group_symbol,
        "space_group_number": space_group_number
    }


def display_structure(structure, height=500, width=600):
    mol = py3Dmol.view(height=height, width=width)
    mol.addModel(structure.to(fmt="cif"), "cif")
    mol.setStyle({"stick": {}})
    mol.addUnitCell()
    mol.zoomTo()
    return mol


uploaded_file = st.file_uploader("Upload CIF file", type=["cif"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        results = analyze_structure(tmp_path)

        st.success("Analysis Complete ✅")

        # --- Basic Info ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Bravais Information")
            st.write(f"**Crystal System:** {results['crystal_system']}")
            st.write(f"**Lattice Type:** {results['lattice_type']}")
            st.write(f"**Space Group:** {results['space_group_symbol']} "
                     f"(No. {results['space_group_number']})")
            st.write(f"**Bravais Lattice:** "
                     f"{results['lattice_type']} {results['crystal_system']}")

        with col2:
            st.subheader("📐 Lattice Parameters")
            lattice = results["structure"].lattice
            df = pd.DataFrame({
                "Parameter": ["a (Å)", "b (Å)", "c (Å)",
                              "α (°)", "β (°)", "γ (°)"],
                "Value": [lattice.a, lattice.b, lattice.c,
                          lattice.alpha, lattice.beta, lattice.gamma]
            })
            st.table(df)

        # --- 3D Visualization ---
        st.subheader("🧬 Conventional Cell")
        viewer = display_structure(results["conventional"])
        st.components.v1.html(viewer._make_html(), height=500)

        st.subheader("🧱 Primitive Cell")
        viewer2 = display_structure(results["primitive"])
        st.components.v1.html(viewer2._make_html(), height=500)

    except Exception as e:
        st.error(f"Error analyzing file: {e}")