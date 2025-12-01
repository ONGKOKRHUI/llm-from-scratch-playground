# app.py
import streamlit as st

# Import the modules (we will create tokenizer_viz next)
from modules import tokenizer_viz

# --- PAGE CONFIGURATION ---
# This must be the first Streamlit command in the whole app
st.set_page_config(
    page_title="LLM Playground",
    page_icon="🤖",
    layout="wide"
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 LLM Playground")
page = st.sidebar.radio(
    "Go to Project Phase:",
    [
        "1. Tokenizer Sandbox", 
        "2. Pre-Training (Coming Soon)", 
        "3. Generation (Coming Soon)",
        "4. Post-Training (Coming Soon)",
        "5. Evaluation (Coming Soon)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Project 1: Build an LLM Playground")

# --- PAGE ROUTING ---
if page == "1. Tokenizer Sandbox":
    tokenizer_viz.app()
    
elif page == "2. Pre-Training (Coming Soon)":
    st.title("🏗️ Pre-Training Lab")
    st.write("This module is under construction.")
    
# ... (Other pages would be similar)