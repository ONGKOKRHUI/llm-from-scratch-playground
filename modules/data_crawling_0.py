# modules/data_pipeline.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import random
import os

# --- 1. DATA LOADING & GENERATION ---
def generate_large_messy_sample(target_length=50000):
    """
    Generates a huge, realistic 'Common Crawl' style string if no file is found.
    Includes headers, messy HTML, duplicate content, and garbage text.
    """
    headers = "WARC/1.0\nWARC-Type: response\nContent-Type: application/http\n\nHTTP/1.1 200 OK\n\n"
    
    templates = [
        "<div><a href='/buy'>CLICK HERE FOR DISCOUNT PHARMACY!!!</a></div>\n",
        "<p>The quick brown fox jumps over the lazy dog. A classic sentence.</p>\n",
        "<script>var tracker_id = 'x89s7f89s7df'; console.log('tracking');</script>\n",
        "<nav>Home | About Us | Contact | Terms | Privacy</nav>\n",
        "<footer>© 2023 Copyright. All rights reserved. Do not copy.</footer>\n",
        "<p>Artificial Intelligence is transforming the world.</p>\n", # Useful line
        "<span class='ad'>$$$ MAKE MONEY FAST $$$</span>\n",
        "Copyright 2024.\n", # Short line (RefinedWeb target)
        "123-456-7890\n", # PII
        "user@example.com\n", # PII
    ]
    
    # Build a massive string
    content = [headers]
    current_len = len(headers)
    
    while current_len < target_length:
        chunk = random.choice(templates)
        # Randomly duplicate chunks to test Dolma deduplication
        if random.random() < 0.3:
            chunk = chunk * 3
            
        content.append(chunk)
        current_len += len(chunk)
        
    return "".join(content)

def load_common_crawl_data():
    """
    Tries to load 'data/common_crawl_sample.txt'. 
    If missing, generates a synthetic 50k char block.
    """
    path = "data/common_crawl_sample.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), "Loaded from file (Real Sample)"
    else:
        return generate_large_messy_sample(55000), "Generated Synthetic Sample (50k+ chars)"

# --- 2. ROBUST CRAWLER ---
def fetch_url(url):
    """Fetches a URL pretending to be a real browser."""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status() 
        return response.text, None
    except Exception as e:
        return None, f"Error: {e}"

# --- 3. CLEANING FILTERS (THE 'REFINERIES') ---

def filter_html(text):
    """
    Industry Context: 'Trafilatura' or 'BeautifulSoup'.
    Removes HTML tags to leave only text.
    """
    if not text: return ""
    soup = BeautifulSoup(text, "html.parser")
    for script in soup(["script", "style", "nav", "footer", "aside", "form"]):
        script.extract()
    text = soup.get_text(separator=' ')
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for phrase in lines if phrase.strip())
    return '\n'.join(chunks)

def filter_refinedweb(text):
    """
    Industry Context: 'RefinedWeb' (Falcon LLM).
    Logic: Strict quality heuristics. If it looks like a menu or code, delete it.
    """
    lines = text.split('\n')
    cleaned_lines = []
    dropped_count = 0
    
    for line in lines:
        # Rule 1: Remove short lines (likely menus/titles)
        if len(line.split()) < 3: 
            dropped_count += 1
            continue
        # Rule 2: Remove lines without terminal punctuation (High quality text usually ends in .)
        if not line.strip().endswith(('.', '!', '?', '"')): 
            dropped_count += 1
            continue
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines), dropped_count

def filter_dolma_dedup(text):
    """
    Industry Context: 'Dolma' (OLMo) / 'FineWeb'.
    Logic: Deduplication. In reality, they use 'MinHash' to find fuzzy duplicates.
    Here, we simulate it by removing exact line duplicates.
    """
    seen = set()
    lines = text.split('\n')
    unique_lines = []
    dedup_count = 0
    
    for line in lines:
        if line in seen:
            dedup_count += 1
        else:
            unique_lines.append(line)
            seen.add(line)
            
    return '\n'.join(unique_lines), dedup_count

def filter_pii(text):
    """
    Industry Context: 'StarCoder' / 'The Stack'.
    Logic: Privacy. Remove Emails and Phone Numbers to prevent the model from leaking user data.
    """
    # Redact Emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '<EMAIL_REDACTED>', text)
    # Redact Phones (Simple regex)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '<PHONE_REDACTED>', text)
    return text

# --- 4. MAIN UI ---
def app():
    st.title("🕷️ Lab 0: Data Pipeline")
    st.markdown("""
    **"Garbage In, Garbage Out."** Modern LLMs (Llama-3, Falcon, OLMo) aren't just trained on the internet. 
    They are trained on a *highly curated* version of it.
    """)
    
    tab_crawl, tab_clean = st.tabs(["1️⃣ Collection (The Swamp)", "2️⃣ Cleaning (The Refinery)"])

    # ==========================
    # TAB 1: COLLECTION
    # ==========================
    with tab_crawl:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Method A: Targeted Crawl")
            st.info("Used for: Specialized datasets (e.g., scraping StackOverflow for coding models).")
            url = st.text_input("Enter URL:", "https://en.wikipedia.org/wiki/Artificial_intelligence")
            if st.button("🕷️ Scrape URL"):
                with st.spinner("Fetching..."):
                    raw, err = fetch_url(url)
                    if err: st.error(err)
                    else:
                        st.session_state['raw_html'] = raw
                        st.success(f"Fetched {len(raw):,} chars.")
                        with st.expander("View Raw HTML Source"):
                            st.code(raw[:2000] + "...", language="html")

        with col2:
            st.subheader("Method B: Common Crawl")
            st.info("Used for: General Purpose LLMs (GPT-4, Llama). A snapshot of the entire web.")
            
            with st.expander("Why can't I verify 'Real' Common Crawl here?"):
                st.markdown("""
                **1. Scale:** A monthly Common Crawl dump is **Petabytes** of data. 
                **2. Format:** It uses `.warc` (Web Archive) files, which contain binary headers, not just text.
                **3. Industry Practice:** Engineers use massive clusters (AWS/Databricks) to process this, not local scripts.
                
                *Below, we simulate a raw 50k character chunk.*
                """)
            
            if st.button("📦 Load 50k Char Sample"):
                raw, source_msg = load_common_crawl_data()
                st.session_state['raw_html'] = raw
                st.success(f"{source_msg}. Size: {len(raw):,} chars")
                st.caption("Preview (First 500 chars):")
                st.code(raw[:2000], language="html")

    # ==========================
    # TAB 2: CLEANING
    # ==========================
    with tab_clean:
        if 'raw_html' not in st.session_state:
            st.warning("Please load data in Tab 1 first.")
        else:
            st.subheader("The Cleaning Pipeline")
            st.markdown("Select which industrial techniques to apply:")
            
            # --- EDUCATIONAL SETTINGS ---
            col_set, col_viz = st.columns([1.2, 2])
            
            with col_set:
                # FILTER 1: HTML
                st.markdown("#### 1. Format Cleaning")
                use_extract = st.checkbox("HTML Extraction (Trafilatura)", value=True)
                with st.expander("What is this?"):
                    st.write("Tools like `Trafilatura` or `BeautifulSoup` strip HTML tags (`<div>`, `<script>`) to leave only human text.")
                
                # FILTER 2: DEDUP
                st.markdown("#### 2. Deduplication")
                use_dedup = st.checkbox("Deduplication (Dolma/FineWeb)", value=True)
                with st.expander("What is Dolma?"):
                    st.image("https://github.com/allenai/dolma/raw/main/docs/images/dolma-logo.png", width=100)
                    st.write("""
                    **Dolma (Data for Open Language Models)** focuses on transparency.
                    **Technique:** They use **MinHash** (LSH) to find documents that are 90% similar and delete the duplicates.
                    *Why?* If an LLM sees the same sentence 50 times, it 'memorizes' it instead of learning to reason.
                    """)

                # FILTER 3: QUALITY
                st.markdown("#### 3. Quality Filtering")
                use_heuristic = st.checkbox("Heuristic Filters (RefinedWeb)", value=True)
                with st.expander("What is RefinedWeb?"):
                    st.write("""
                    **RefinedWeb (Falcon LLM)** proved that strict filtering beats more data.
                    **Technique:** 'Heuristics' (Rules of Thumb).
                    1. Delete lines without punctuation (usually ads/menus).
                    2. Delete lines with too many symbols.
                    *Result:* A dataset that looks like a book, not a website.
                    """)
                
                # FILTER 4: PRIVACY
                st.markdown("#### 4. Safety")
                use_pii = st.checkbox("PII Redaction (StarCoder)", value=True)
                with st.expander("What is PII?"):
                    st.write("Personally Identifiable Information. Models like StarCoder run regex scans to replace emails/phones with `<REDACTED>` tags so the AI doesn't leak private info.")

                run_btn = st.button("Run Pipeline", type="primary")

            # --- VISUALIZATION ---
            with col_viz:
                if run_btn:
                    text = st.session_state['raw_html']
                    stats = []
                    
                    # 1. HTML
                    if use_extract:
                        text = filter_html(text)
                        stats.append("HTML Tags Removed")
                    
                    # 2. Dedup
                    if use_dedup:
                        text, count = filter_dolma_dedup(text)
                        stats.append(f"Dolma: Removed {count} duplicates")
                        
                    # 3. RefinedWeb
                    if use_heuristic:
                        text, count = filter_refinedweb(text)
                        stats.append(f"RefinedWeb: Filtered {count} low-quality lines")
                        
                    # 4. PII
                    if use_pii:
                        text = filter_pii(text)
                        stats.append("PII Redacted")
                        
                    st.success("Pipeline Complete!")
                    for s in stats:
                        st.caption(f"✅ {s}")
                        
                    st.subheader("Final Dataset Preview")
                    st.text_area("Cleaned Text", text, height=500)
                    st.metric("Final Character Count", len(text), delta=len(text) - len(st.session_state['raw_html']))