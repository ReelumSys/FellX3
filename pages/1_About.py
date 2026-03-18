import streamlit as st
from openai import OpenAI


im = 'images/favicon.png'
st.set_page_config(
    page_title="FellX v0.8",
    page_icon=im,
    layout="wide",
)


st.title("About")

st.write("You have discovered FellX! It is a analytical programm for XRD data. The programm comes with visual analysis as well as curve fitting and for example grain size and scherrer and Willamson and Hall.")



st.title("XRDGPT")

