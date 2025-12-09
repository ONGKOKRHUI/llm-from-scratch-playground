import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel
import math
import re
import os
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import openai
from sentence_transformers import SentenceTransformer, util

# Load OpenAI key from .env
openai.api_key = os.getenv("CHATGPT_API_TOKEN")

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
    $$ Perplexity= e^{Cross Entropy Loss} $$
    Cross-entropy tells us: how surprised the model is by the correct next token.
    """
    #tokenize the text into pytorch tensors pt
    encodings = tokenizer(text, return_tensors="pt")
    
    # max_length for GPT-2 is 1024. We clamp it for the demo. Truncate to GPT-2 max length
    input_ids = encodings.input_ids[:, :1024]
    
    #Since we’re evaluating, not training, we don’t need gradient computation
    with torch.no_grad(): 
        """Forward pass with labels
        input_ids: inputs for the model to read (tokenized text)
        labels:  for the model to predict
        You are saying:
        “Using the text, predict the next token for every position — 
        and compare your predictions to this same text.”
        But important:
        HuggingFace automatically shifts the labels by one position inside the model.
        setting labels as input id tells the model:
        “Predict the next token for each position in this sequence.”
        Internally, the model shifts labels by 1 position:
        Input IDs:   [A, B, C, D]
        Labels:      [B, C, D, <eos>]
        So the model learns:
        From A → predict B
        From AB → predict C
        From ABC → predict D
        The model returns a loss automatically when labels are provided.
        """
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    #.item() converts a PyTorch tensor to a Python float.
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
        """
        Converts text to lowercase
        Removes punctuation using regex
        ([^\w\s] removes anything that is not a word character or whitespace)
        Splits text on whitespace into words
        Converts list of words into a set
        → This removes duplicates
        """
        text = re.sub(r'[^\w\s]', '', text.lower())
        return set(text.split())

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    overlap = ref_tokens.intersection(cand_tokens)
    #recall is the ratio of overlapping words to the total number of words in the reference
    #What % of reference is covered
    recall = len(overlap) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    #precision is the ratio of overlapping words to the total number of words in the candidate
    #What % of candidate is covered
    precision = len(overlap) / len(cand_tokens) if len(cand_tokens) > 0 else 0
    #f1 is the harmonic mean of recall and precision
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
        
    return recall, precision, f1

# --- METRIC 2: ROUGE ---
@st.cache_data
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# --- METRIC 3: BERTScore ---
@st.cache_data
def compute_bertscore(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang='en', rescale_with_baseline=True)
    return P[0].item(), R[0].item(), F1[0].item()

# --- METRIC 4: Embedding Similarity ---
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def compute_embedding_similarity(reference, candidate, model):
    ref_emb = model.encode(reference, convert_to_tensor=True)
    cand_emb = model.encode(candidate, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(ref_emb, cand_emb).item()
    return similarity

# --- LLM as a Judge ---
def grade_with_openai(question, answer):
    prompt = f"""
    You are an impartial judge. Rate the quality of the answer on a scale of 1-10.

    Question: {question}
    Student Answer: {answer}

    Score (1-10):
    Reasoning:
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    output_text = response['choices'][0]['message']['content']
    return output_text

# --- MAIN APP ---
def app():
    st.title("📊 Lab 5: Evaluation & Metrics (Advanced)")
    st.markdown("""
    Welcome to the advanced LLM evaluation lab!  
    Here, you can evaluate text using **Perplexity**, **ROUGE scores**, **Semantic similarity**, and even let a **powerful AI judge** the quality of answers.
    """)

    tab_ppl, tab_rouge, tab_semantic, tab_judge = st.tabs([
        "1️⃣ Perplexity", 
        "2️⃣ ROUGE Scores",
        "3️⃣ Semantic Similarity",
        "4️⃣ LLM-as-a-Judge"
    ])

    # Load models
    with st.spinner("Loading Evaluator Models... This may take a few seconds..."):
        tokenizer, model = load_eval_model()
        sent_model = load_sentence_model()
    st.success("Models loaded successfully!")

    # ==========================
    # TAB 1: PERPLEXITY
    # ==========================
    with tab_ppl:
        st.subheader("📈 Perplexity")
        st.markdown("""
        Perplexity measures how 'confused' a language model is when reading text.  
        - **Lower values** = text is fluent and easy to predict  
        - **Higher values** = text is unusual, jumbled, or hard to predict
        """)
        text = st.text_area("Enter Text for Perplexity Evaluation", 
                            "The quick brown fox jumps over the lazy dog.", height=150)

        if st.button("Compute Perplexity"):
            ppl = calculate_perplexity(text, model, tokenizer)
            st.metric("Perplexity Score", f"{ppl:.2f}", delta="Lower is better ✅")
            st.info("Tip: Changing word order or grammar will increase perplexity even if words are correct.")

    # ==========================
    # TAB 2: ROUGE
    # ==========================
    with tab_rouge:
        st.subheader("📝 ROUGE Scores")
        st.markdown("""
        ROUGE measures **overlap between the candidate text and a reference text**.  
        It's commonly used in summarization, translation, and text generation evaluation.
        """)
        ref = st.text_area("Reference (Gold Standard)", "The Eiffel Tower is located in Paris, France.")
        cand = st.text_area("Candidate (Model Output)", "The Eiffel Tower stands in the city of Paris.")

        if st.button("Compute ROUGE"):
            scores = compute_rouge(ref, cand)
            st.success("ROUGE evaluation completed!")
            for k, v in scores.items():
                st.metric(f"{k.upper()} F1", f"{v.fmeasure:.2%}", 
                          delta=f"P: {v.precision:.2%} | R: {v.recall:.2%}")

            st.caption("Note: ROUGE is word-overlap based, so paraphrasing can lead to lower scores even if meaning is correct.")

    # ==========================
    # TAB 3: SEMANTIC SIMILARITY
    # ==========================
    with tab_semantic:
        st.subheader("🤖 Semantic Similarity")
        st.markdown("""
        Goes beyond exact word match:
        - **BERTScore**: Measures contextual similarity between reference and candidate  
        - **Embedding Cosine Similarity**: Measures overall semantic closeness
        """)
        ref = st.text_area("Reference (Semantic)", "The Eiffel Tower is located in Paris, France.")
        cand = st.text_area("Candidate (Semantic)", "The Eiffel Tower stands in the city of Paris.")

        if st.button("Compute Semantic Similarity"):
            P, R, F1 = compute_bertscore(ref, cand)
            emb_sim = compute_embedding_similarity(ref, cand, sent_model)
            
            st.success("Semantic evaluation completed!")
            col1, col2 = st.columns(2)
            col1.metric("BERTScore F1", f"{F1:.2%}", delta=f"P:{P:.2%} R:{R:.2%}")
            col2.metric("Embedding Cosine Similarity", f"{emb_sim:.2%}", delta="Higher is better ✅")

            st.info("Semantic metrics capture meaning, not just exact words. Paraphrasing is better evaluated here.")

    # ==========================
    # TAB 4: LLM as a Judge
    # ==========================
    with tab_judge:
        st.subheader("🧑‍⚖️ LLM as a Judge")
        st.markdown("""
        Let a **powerful AI** grade answers automatically:
        - Enter a question and a student/model answer  
        - GPT-4 will provide a **score (1-10)** and **reasoning**
        """)
        question = st.text_input("Question", "Explain gravity.")
        answer = st.text_area("Student Answer", "Gravity is when things fall down because the earth is heavy.")

        st.info("Press 'Grade Answer' to let GPT-4 evaluate. Make sure your OpenAI API key is set in the .env file.")

        if st.button("Grade Answer"):
            with st.spinner("Getting GPT-4 judgment..."):
                result = grade_with_openai(question, answer)
                st.success("Judgment completed!")
                st.code(result, language="text")
                st.balloons()  # Fun visual feedback for users

            st.info("Tip: You can also use this tool to evaluate your own answers.")