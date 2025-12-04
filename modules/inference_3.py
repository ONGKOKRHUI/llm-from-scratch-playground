# modules/inference_lab.py
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import altair as alt

# --- CACHED MODEL LOADING ---
# We use @st.cache_resource so we don't redownload the 500MB model every time you click a button.
@st.cache_resource
def load_model():
    model_name = "gpt2" # Small, fast, and classic
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Ensure pad token is set (GPT-2 lacks one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    return tokenizer, model
# --- HELPER: PROBABILITY VISUALIZER ---
def plot_top_k(model, tokenizer, text_input, k=10):
    """
    Runs a single forward pass to get the probabilities of the NEXT token.
    """
    inputs = tokenizer(text_input, return_tensors="pt")
    
    # Run the model (no generation, just one forward pass)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits for the LAST token in the sequence
    next_token_logits = outputs.logits[0, -1, :]
    
    # Convert logits to probabilities (Softmax)
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    
    # Get the top K candidates
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    # Decode indices to words
    # We strip whitespace to make the chart labels cleaner
    top_k_tokens = [tokenizer.decode([idx]).strip() for idx in top_k_indices]
    
    # Create DataFrame for Charting
    df = pd.DataFrame({
        "Token": top_k_tokens,
        "Probability": top_k_probs.tolist()
    })
    
    return df

# --- MAIN PAGE LOGIC ---
def app():
    st.title("🎛️ Lab 3: Decoding Strategies")
    st.markdown("""
    This lab explores **Inference**: How does the model choose the next word?
    Unlike a calculator (2+2=4), an LLM is probabilistic. We can control its "creativity."
    The **Bar Chart** shows the top 10 words the model is considering for the *next* step.
                """)
    # 1. Load Resources
    with st.spinner("Loading GPT-2 Model... (this may take a minute first time)"):
        tokenizer, model = load_model()
        st.success("GPT-2 Model Loaded!")
    
    st.subheader("📝 Input Prompt")
    prompt_text = st.text_area("Start a sentence:", value="The secret to artificial intelligence is", height=100)

    # 2. Controls & Inputs
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("⚙️ The Knobs")
        
        # Decoding Strategy Selector
        strategy = st.radio(
            "Decoding Method",
            ["Greedy Search", "Beam Search", "Random Sampling"],
            help="Greedy = Fast/Repetitive. Beam = Smart/Coherent. Sampling = Creative."
        )
        
        st.divider()
        
        # Dynamic Sliders based on strategy
        max_length = st.slider("Max Length", 10, 100, 50)
        
        temperature = 1.0
        top_k = 50
        top_p = 0.95
        num_beams = 1
        
        if strategy == "Random Sampling":
            temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 
                help="Low = Robotic/Focused. High = Creative/Chaotic.")
            top_k = st.slider("Top-K", 1, 100, 50, 
                help="Limit choices to the top K most likely words.")
            top_p = st.slider("Top-P (Nucleus)", 0.5, 1.0, 0.95, 
                help="Limit choices to the cumulative probability (cleaner than Top-K).")
        
        elif strategy == "Beam Search":
            num_beams = st.slider("Number of Beams", 2, 10, 5, 
                help="How many future paths to explore simultaneously.")
            st.caption("Beam search finds the 'best' overall sentence, not just the next word.")

        generate_btn = st.button("✨ Generate Text", type="primary")

    # --- RIGHT COLUMN: VISUALIZATION ---
    with col_right:
        st.subheader("3. Next Token Probability")
        
        if prompt_text:
            # Calculate probabilities for the *very next word* based on current input
            df_probs = plot_top_k(model, tokenizer, prompt_text, k=10)
            
            # Create a nice Altair Bar Chart
            chart = alt.Chart(df_probs).mark_bar().encode(
                x=alt.X('Probability', axis=alt.Axis(format='%')),
                y=alt.Y('Token', sort='-x', title="Candidate Word"),
                color=alt.Color('Probability', legend=None),
                tooltip=['Token', alt.Tooltip('Probability', format='.2%')]
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
            
            st.caption("These are the words the model is deciding between right now!") 

    # 3. Generation Logic
    if generate_btn:
        inputs = tokenizer(prompt_text, return_tensors="pt")
        
        # Base arguments
        gen_kwargs = {
            "max_length": max_length + len(inputs['input_ids'][0]),
            "pad_token_id": tokenizer.eos_token_id,
            "no_repeat_ngram_size": 2 if strategy != "Greedy Search" else 0 # Prevent "the the the"
        }
        """
        For every next token, the model outputs a probability distribution:
        Example:
        Model is predicting the next word after:
        "The cat sat on the"
        It might output:
        Token	Probability
        "mat"	0.62
        "floor"	0.25
        "bed"	0.10
        Different decoding strategies use this distribution differently.
        """
        # Strategy-specific arguments
        if strategy == "Greedy Search":
            # Always pick the highest probability word at each step.
            gen_kwargs["do_sample"] = False
        
        elif strategy == "Beam Search":
            """
            num_beams = 5 → Keep the top 5 most probable sentences-in-progress.
            Beam search (5 beams) may explore:
            1.exciting 2.bright 3.uncertain 4.transformative 5.rapidly evolving
            For each, it continues generating further words and scores the entire sentence.
            Finally, it picks the best-scoring full sentence.
            """
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["early_stopping"] = True
            
        elif strategy == "Random Sampling":
            """
            This is the creative mode.
            Temperature - Controls how "flat" the probability distribution is, higher flatter
            Top-K - If top_k = 5, and the model predicts 50 possible tokens, we ignore the bottom 45.
            Top-P - Consider only the smallest set of words whose total probability >= p.
                    Let the model output a probability distribution over all tokens:
                    p₁ ≥ p₂ ≥ p₃ ≥ ... ≥ pₙ
                    Top-P chooses the smallest set of tokens such that:
                    p₁ + p₂ + ... + p_k ≥ p   (example: p = 0.9)
                    Within that set, we sample proportionally to their probabilities.
                    Intuition: Only consider tokens that together make up the top p portion of the model's confidence.
                    This dynamically adapts:
                    If distribution is spike-shaped → only 1-2 words selected
                    If flat → more words selected
            """
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p

        # Run Generation
        with st.spinner("Generating..."):
            output_tokens = model.generate(inputs['input_ids'], **gen_kwargs)
            output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # 4. Display Results
        st.divider()
        st.subheader("Result:")
        st.success(output_text)
        
        # Educational Breakdown
        with st.expander("🔍 What just happened?"):
            if strategy == "Greedy Search":
                st.write("The model always picked the single most likely next word. Note how it might get stuck in loops or be boring.")
            elif strategy == "Beam Search":
                st.write(f"The model explored {num_beams} different future paths simultaneously and picked the one with the highest total probability.")
                # We can try to visualize this concept
                st.write("") 
            elif strategy == "Random Sampling":
                st.write(f"The model rolled a dice! High temperature ({temperature}) flattened the probabilities, making rare words more likely.")