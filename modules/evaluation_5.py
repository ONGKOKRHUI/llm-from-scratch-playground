# modules/evaluation_lab.py
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import re

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_eval_model():
    # We use GPT-2 to calculate Perplexity (how well it predicts text)
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

# --- METRIC 1: PERPLEXITY CALCULATOR ---
def calculate_perplexity(text, model, tokenizer):
    """
    Perplexity = exp(Cross Entropy Loss).
    Lower is better. A PPL of 10 means the model is as confused as if it had to pick from 10 words.
    """
    encodings = tokenizer(text, return_tensors="pt")
    
    # max_length for GPT-2 is 1024. We clamp it for the demo.
    input_ids = encodings.input_ids[:, :1024]
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
    ppl = torch.exp(loss).item()
    return ppl

# --- METRIC 2: N-GRAM OVERLAP (ROUGE-1 Proxy) ---
def simple_rouge_score(reference, candidate):
    """
    Calculates the overlap of words (unigrams) between reference and candidate.
    Recall = (Overlapping Words) / (Total Words in Reference)
    Precision = (Overlapping Words) / (Total Words in Candidate)
    F1 = Harmonic Mean
    """
    # Simple tokenization by splitting on whitespace and removing punctuation
    def tokenize(text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return set(text.split())

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    overlap = ref_tokens.intersection(cand_tokens)
    
    recall = len(overlap) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    precision = len(overlap) / len(cand_tokens) if len(cand_tokens) > 0 else 0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
        
    return recall, precision, f1

# --- MAIN APP ---
def app():
    st.title("📊 Lab 5: Evaluation & Metrics")
    st.markdown("""
    How do we know if an LLM is "smart"? We grade it.
    """)
    
    tab_ppl, tab_rouge, tab_judge = st.tabs([
        "1️⃣ Perplexity (Math)", 
        "2️⃣ Overlap (Reference)",
        "3️⃣ LLM-as-a-Judge (AI)"
    ])
    
    # Load Model
    with st.spinner("Loading Evaluator Model..."):
        tokenizer, model = load_eval_model()

    # ==========================
    # TAB 1: PERPLEXITY
    # ==========================
    with tab_ppl:
        st.subheader("The 'Confusion' Score")
        st.markdown("Perplexity measures how surprised the model is by your text. **Lower is better.**")
        
        col1, col2 = st.columns(2)
        with col1:
            text_a = st.text_area("Text A (Fluent English)", 
                                  "The quick brown fox jumps over the lazy dog.", height=100)
        with col2:
            text_b = st.text_area("Text B (Broken/Garbage)", 
                                  "Dog lazy the over jumps fox brown quick The.", height=100)
            
        if st.button("Calculate Perplexity"):
            ppl_a = calculate_perplexity(text_a, model, tokenizer)
            ppl_b = calculate_perplexity(text_b, model, tokenizer)
            
            c1, c2 = st.columns(2)
            c1.metric("PPL (Text A)", f"{ppl_a:.2f}", delta="Low Confusion (Good)", delta_color="normal")
            c2.metric("PPL (Text B)", f"{ppl_b:.2f}", delta="High Confusion (Bad)", delta_color="inverse")
            
            st.info("Notice that Text B has the exact same words, but the *order* confuses the model, causing high perplexity.")

    # ==========================
    # TAB 2: REFERENCE METRICS
    # ==========================
    with tab_rouge:
        st.subheader("Comparing to a 'Gold' Answer")
        st.markdown("Used in translation and summarization. Does the model match the human answer?")
        
        ref_text = st.text_area("Gold Standard Reference (Human)", "The Eiffel Tower is located in Paris, France.")
        cand_text = st.text_area("Model Output (Candidate)", "The Eiffel Tower stands in the city of Paris.")
        
        if st.button("Calculate ROUGE Scores"):
            rec, prec, f1 = simple_rouge_score(ref_text, cand_text)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Recall (ROUGE-1)", f"{rec:.1%}", help="How much of the Reference did the model capture?")
            m2.metric("Precision", f"{prec:.1%}", help="How much of the Model's output was actually correct?")
            m3.metric("F1 Score", f"{f1:.1%}", help="The balance between the two.")
            
            st.caption("Note: ROUGE is imperfect. 'Paris is where the Eiffel Tower is' might get a low score despite being correct.")

    # ==========================
    # TAB 3: LLM-AS-A-JUDGE
    # ==========================
    with tab_judge:
        st.subheader("The Modern Way: AI Grading AI")
        st.markdown("""
        Standard metrics (like ROUGE) are bad at nuance. The industry standard is now using a **Judge Prompt** to have a powerful model (like GPT-4) grade a smaller model's answer.
        """)
        
        st.markdown("#### The Judge Template")
        judge_prompt = """
        You are an impartial judge. Please rate the quality of the following answer on a scale of 1-10.
        
        Question: {question}
        Student Answer: {answer}
        
        Score (1-10): [SCORE]
        Reasoning: [REASONING]
        """
        st.code(judge_prompt, language="text")
        
        st.markdown("#### Try the concept")
        q = st.text_input("Question", "Explain gravity.")
        a = st.text_area("Student Answer", "Gravity is when things fall down because the earth is heavy.")
        
        st.info("""
        **Simulation:** In a real app, clicking 'Grade' would send this prompt to the OpenAI API. 
        Since this is a local playground, act as the judge yourself!
        """)
        
        rating = st.slider("Give a Score", 1, 10, 5)
        if rating > 7:
            st.success("Pass")
        else:
            st.error("Fail")