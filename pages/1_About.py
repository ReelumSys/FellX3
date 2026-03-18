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

client = OpenAI(api_key=st.secrets["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"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})