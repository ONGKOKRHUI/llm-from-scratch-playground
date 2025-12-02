# app.py
import streamlit as st

# Import the modules (we will create tokenizer_viz next)
from modules import tokenizer_viz, training_demo

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
        "2. Pre-Training", 
        "3. Generation",
        "4. Post-Training",
        "5. Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Project 1: Build an LLM Playground")

# --- PAGE ROUTING ---
if page == "1. Tokenizer Sandbox":
    tokenizer_viz.app()
    
elif page == "2. Pre-Training":
    training_demo.app()
    
# ... (Other pages would be similar)