import streamlit as st
from openai import OpenAI
import streamlit as st
from openai import OpenAI
import os


im = 'images/favicon.png'
st.set_page_config(
    page_title="FellX v0.8",
    page_icon=im,
    layout="wide",
)


st.title("About")

st.write("You have discovered FellX! It is a analytical programm for XRD data. The programm comes with visual analysis as well as curve fitting and for example grain size and scherrer and Willamson and Hall.")



st.title("XRDGPT")



# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XRDGPT",
    page_icon="⚛️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* Background */
[data-testid="stAppViewContainer"] {
    background: #020a0f;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(0,255,200,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(0,150,255,0.05) 0%, transparent 50%);
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #030d12; border-right: 1px solid #0a3040; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Title */
h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.8rem !important;
    letter-spacing: 0.15em !important;
    background: linear-gradient(90deg, #00ffe0, #00aaff, #00ffe0) !important;
    background-size: 200% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: shimmer 4s linear infinite !important;
    text-align: center !important;
    margin-bottom: 0 !important;
}

@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #00ffe080;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Divider */
.xrd-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ffe050, #00aaff80, #00ffe050, transparent);
    margin: 0.5rem 0 1.5rem 0;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatMessageContent"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
    color: #c8f0ff !important;
    line-height: 1.7 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(0,170,255,0.07) !important;
    border: 1px solid rgba(0,170,255,0.2) !important;
    border-radius: 4px !important;
    padding: 0.5rem 1rem !important;
    margin-bottom: 0.75rem !important;
}

/* Assistant bubble */
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(0,255,224,0.05) !important;
    border: 1px solid rgba(0,255,224,0.15) !important;
    border-radius: 4px !important;
    padding: 0.5rem 1rem !important;
    margin-bottom: 0.75rem !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #030d12 !important;
    border: 1px solid #00ffe040 !important;
    border-radius: 4px !important;
    color: #c8f0ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #00ffe0aa !important;
    box-shadow: 0 0 12px rgba(0,255,224,0.15) !important;
}

/* Sidebar text */
.stSidebar p, .stSidebar label, .stSidebar span {
    font-family: 'Share Tech Mono', monospace !important;
    color: #7ec8e3 !important;
    font-size: 0.82rem !important;
}

.stSidebar h2, .stSidebar h3 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #00ffe0 !important;
    letter-spacing: 0.1em !important;
}

/* Select box & slider */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    background: transparent !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #020a0f; }
::-webkit-scrollbar-thumb { background: #00ffe040; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #00ffe080; }

/* Status badge */
.status-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #00ffe0;
    background: rgba(0,255,224,0.08);
    border: 1px solid rgba(0,255,224,0.25);
    border-radius: 2px;
    padding: 2px 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin: 0 auto;
    display: block;
    text-align: center;
    width: fit-content;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚛️ XRDGPT")
st.markdown('<div class="subtitle">X-Ray Diffraction Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<hr class="xrd-divider">', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
    )

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = st.slider("Max Tokens", 256, 4096, 1024, 128)

    st.markdown("---")
    st.markdown("### 🧬 System Prompt")
    system_prompt = st.text_area(
        "Customize assistant behaviour",
        value=(
            "You are XRDGPT, an expert AI assistant specialising in X-ray diffraction (XRD) analysis, "
            "crystallography, materials science, and related data processing. "
            "You help researchers understand diffraction patterns, peak fitting, phase identification, "
            "Rietveld refinement, and tools like xrdfit and lmfit. "
            "Be precise, scientific, and concise. Use markdown for equations and code."
        ),
        height=180,
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown('<span class="status-badge">● ONLINE</span>', unsafe_allow_html=True)

# ── OpenAI client ─────────────────────────────────────────────────────────────
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("⚠️ No OpenAI API key found. Add it to Streamlit secrets as `OPENAI_API_KEY`.")
    st.stop()

client = OpenAI(api_key=api_key)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask XRDGPT about diffraction, peaks, phases…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state.messages,
                ],
                stream=True,
            )

            full_response = ""
            placeholder = st.empty()
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})