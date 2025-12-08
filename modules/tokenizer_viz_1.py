# modules/tokenizer_viz.py
import streamlit as st
import tiktoken
import re 
import html

# --- HELPER FUNCTIONS ---
def clean_text_data(text, remove_html=True, remove_digits=False, normalize_whitespace=True):
    if remove_html:
        text = re.sub(r'<[^>]+>', '', text)
    if remove_digits:
        text = re.sub(r'\d+', '', text)
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_colored_token_html(tokens, encoding):
    colors = [
        "#FFD700", "#ADFF2F", "#00FFFF", "#FF69B4", "#dda0dd", 
        "#87CEFA", "#FFA07A", "#98FB98"
    ]
    html_output = '<div style="font-family: monospace; line-height: 2.5; font-size: 18px;">'
    
    for i, token_id in enumerate(tokens):
        try:
            token_text = encoding.decode([token_id])
            #html.escape(token_text) replaces < → &lt;, > → &gt;, & → &amp;
            token_text = html.escape(token_text)
            display_text = token_text.replace(' ', '&nbsp;').replace('\n', '<br>')
            if not display_text: display_text = ""
            color = colors[i % len(colors)]
            
            html_output += (
                f'<span style="background-color: {color}; padding: 2px 5px; '
                f'border-radius: 4px; border: 1px solid #ccc; margin-right: 2px;" '
                f'title="Token ID: {token_id}">{display_text}</span>'
            )
        except:
            html_output += f'<span style="border:1px solid red;">?</span>'
            
    html_output += '</div>'
    return html_output

# --- MAIN MODULE ENTRY POINT ---
def app():
    st.title("🧩 Lab 1: The Tokenizer Sandbox")
    st.markdown("""
    This tool visualizes how raw text is broken down into **Tokens** (numbers) that an LLM can read.
    """)

    # Input Area
    default_text = """<p>Welcome to the LLM Playground!</p>
This is an example of how tokenization works.
Uncommon words like 'neuroscience' or 'antidisestablishmentarianism' might get split."""

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Text")
        raw_text = st.text_area("Enter text here:", value=default_text, height=300)
        # Note: We use a unique key to avoid conflicts if you add controls in other modules
        st.subheader("Tokenizer Settings")
        model_name = st.selectbox(
            "Model Encoding", 
            ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"], 
            index=0,
            key="tok_model_select" 
        )
        do_clean_html = st.checkbox("Remove HTML Tags", value=True, key="tok_clean_html")
        do_norm_space = st.checkbox("Normalize Whitespace", value=False, key="tok_norm_space")

    # Process Logic
    try:
        cleaned_text = clean_text_data(
            raw_text, 
            remove_html=do_clean_html, 
            normalize_whitespace=do_norm_space
        )
        
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(cleaned_text)
        
        token_count = len(tokens)
        word_count = len(cleaned_text.split())
        
        with col1:
            st.caption(f"Original Char Count: {len(raw_text)}")
            if raw_text != cleaned_text:
                with st.expander("See Cleaned Text"):
                    st.code(cleaned_text, language=None)

        with col2:
            st.subheader("2. Token Analysis")
            m1, m2 = st.columns(2)
            m1.metric("Token Count", token_count)
            ratio = round(token_count / word_count, 2) if word_count > 0 else 0
            m2.metric("Tokens/Word Ratio", ratio)
            
            tab_viz, tab_ids = st.tabs(["🎨 Visualizer", "🔢 Token IDs"])
            
            with tab_viz:
                html_view = get_colored_token_html(tokens, encoding)
                st.markdown(html_view, unsafe_allow_html=True)
                
            with tab_ids:
                st.code(str(tokens), language="json")

    except Exception as e:
        st.error(f"Error processing text: {e}")