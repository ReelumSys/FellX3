
import streamlit as st
import pandas as pd

# Data from Main Page
main_df = st.session_state.get('main_df')
comp_df = st.session_state.get('comp_df')
cif_data = st.session_state.get('cif_data')

if main_df is None:
    st.warning("Main XRD pattern missing. Please upload it on the Main Page.")
    st.stop()

     1|﻿import streamlit as st
     2|from pymatgen.core import Structure
     3|from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
     4|import py3Dmol
     5|import tempfile
     6|import pandas as pd
     7|from main import cif_file2
     8|
     9|
    10|im = 'images/favicon.png'
    11|st.set_page_config(
    12|    page_title="FellX v0.8",
    13|    page_icon=im,
    14|    layout="wide",
    15|)
    16|
    17|
    18|st.set_page_config(page_title="FellX v0.8", layout="wide")
    19|
    20|st.title("🔷 Bravais Lattice & Cell Viewer")
    21|st.write("Upload a CIF file to calculate Bravais lattice and visualize the unit cell.")
    22|
    23|
    24|def analyze_structure(cif_path):
    25|    structure = Structure.from_file(cif_path)
    26|    sga = SpacegroupAnalyzer(structure, symprec=1e-3)
    27|
    28|    crystal_system = sga.get_crystal_system()
    29|    lattice_type = sga.get_lattice_type()
    30|    space_group_symbol = sga.get_space_group_symbol()
    31|    space_group_number = sga.get_space_group_number()
    32|
    33|    conventional = sga.get_conventional_standard_structure()
    34|    primitive = sga.get_primitive_standard_structure()
    35|
    36|    return {
    37|        "structure": structure,
    38|        "conventional": conventional,
    39|        "primitive": primitive,
    40|        "crystal_system": crystal_system,
    41|        "lattice_type": lattice_type,
    42|        "space_group_symbol": space_group_symbol,
    43|        "space_group_number": space_group_number
    44|    }
    45|
    46|
    47|def display_structure(structure, height=500, width=600):
    48|    mol = py3Dmol.view(height=height, width=width)
    49|    mol.addModel(structure.to(fmt="cif"), "cif")
    50|    mol.setStyle({"stick": {}})
    51|    mol.addUnitCell()
    52|    mol.zoomTo()
    53|    return mol
    54|
    55|
    56|uploaded_file = cif_file2
    57|
    58|if uploaded_file:
    59|    with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
    60|        tmp.write(uploaded_file.read())
    61|        tmp_path = tmp.name
    62|
    63|    try:
    64|        results = analyze_structure(tmp_path)
    65|
    66|        st.success("Analysis Complete ✅")
    67|
    68|        # --- Basic Info ---
    69|        col1, col2 = st.columns(2)
    70|
    71|        with col1:
    72|            st.subheader("📊 Bravais Information")
    73|            st.write(f"**Crystal System:** {results['crystal_system']}")
    74|            st.write(f"**Lattice Type:** {results['lattice_type']}")
    75|            st.write(f"**Space Group:** {results['space_group_symbol']} "
    76|                     f"(No. {results['space_group_number']})")
    77|            st.write(f"**Bravais Lattice:** "
    78|                     f"{results['lattice_type']} {results['crystal_system']}")
    79|
    80|        with col2:
    81|            st.subheader("📐 Lattice Parameters")
    82|            lattice = results["structure"].lattice
    83|            df = pd.DataFrame({
    84|                "Parameter": ["a (Å)", "b (Å)", "c (Å)",
    85|                              "α (°)", "β (°)", "γ (°)"],
    86|                "Value": [lattice.a, lattice.b, lattice.c,
    87|                          lattice.alpha, lattice.beta, lattice.gamma]
    88|            })
    89|            st.table(df)
    90|
    91|        # --- 3D Visualization ---
    92|        st.subheader("🧬 Conventional Cell")
    93|        viewer = display_structure(results["conventional"])
    94|        st.components.v1.html(viewer._make_html(), height=500)
    95|
    96|        st.subheader("🧱 Primitive Cell")
    97|        viewer2 = display_structure(results["primitive"])
    98|        st.components.v1.html(viewer2._make_html(), height=500)
    99|
   100|    except Exception as e:
   101|        st.error(f"Error analyzing file: {e}")