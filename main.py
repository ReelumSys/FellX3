import streamlit as st
import pandas as pd
import base64
import numpy as np
from PIL import Image
import os
from pymatgen.core import Structure
from ase.io import read, write
from pymatgen.io.cif import CifWriter

st.set_page_config(
    page_title="Multipage App",
)

# --- Session State Initialization ---
if 'main_pattern' not in st.session_state:
    st.session_state.main_pattern = None
if 'comp_pattern' not in st.session_state:
    st.session_state.comp_pattern = None
if 'cif_data' not in st.session_state:
    st.session_state.cif_data = None
if 'main_df' not in st.session_state:
    st.session_state.main_df = None
if 'comp_df' not in st.session_state:
    st.session_state.comp_df = None

image = Image.open('./images/large.PNG')
new_img = image.resize((180, 100))
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image(new_img)

st.title("Main Page")
st.sidebar.success("Select a page above.")

st.markdown('###### Upload two .txt patterns separately and let them be calculated.')

# 1. Main XRD Pattern
uploaded_file = st.file_uploader("Upload Main XRD Pattern", type=["txt"])
if uploaded_file:
    # Store raw file and process to DataFrame
    st.session_state.main_pattern = uploaded_file
    df1 = pd.read_fwf(uploaded_file)
    st.session_state.main_df = df1
    st.success('Main pattern uploaded.')
elif st.session_state.main_df is None:
    st.warning('Please input a .txt file.')

# 2. Comparing XRD Pattern
uploaded_file2 = st.file_uploader("Upload Comparing XRD Pattern", type=["txt"])
if uploaded_file2:
    st.session_state.comp_pattern = uploaded_file2
    df2 = pd.read_fwf(uploaded_file2)
    st.session_state.comp_df = df2
    st.success('Comparing pattern uploaded.')
elif st.session_state.comp_df is None:
    st.warning('Please input a .txt file.')

# 3. CIF File
cif_file = st.file_uploader("Upload CIF file", type=["cif", "CIF"],
                           help="Any standard CIF including ICSD, COD, CCDC exports")
if cif_file:
    st.session_state.cif_data = cif_file
    # Save locally as well if required by other functions
    atoms = read(cif_file)
    write("cif_file2.cif", atoms)
    st.success('CIF file uploaded.')
elif st.session_state.cif_data is None:
    st.warning('Please input a .cif file.')

# Process data if both patterns are present
if st.session_state.main_df is not None and st.session_state.comp_df is not None:
    # Use local copies for calculations
    weather1 = st.session_state.main_df.copy()
    weather1.columns = ['2Theta', 'Int']
    
    weather2 = st.session_state.comp_df.copy()
    weather2.columns = ['2Theta', 'Int2']
    
    weather3 = weather1 - weather2
    np.savetxt('testTheta2.txt', weather3, fmt='%f')
    st.info("Calculations updated based on uploaded files.")
