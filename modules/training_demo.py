# modules/training_demo.py
import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import time
from models.tiny_gpt import TinyLLM
from models.gpt import GPT

# --- CONFIGURATION & DATA ---
# A tiny snippet of Shakespeare for the model to learn from (Fallback data)
DEFAULT_TEXT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
"""

# --- MODEL ARCHITECTURE (The "Brain") ---
# This is a simplified "Bigram" Language Model.
# In a real LLM, this would be replaced by a Transformer Block.



# --- HELPER FUNCTIONS ---

def get_batch(data, block_size, batch_size):
    """Generates a small batch of inputs (x) and targets (y)"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def text_to_tensor(text, vocab_to_int):
    return torch.tensor([vocab_to_int[c] for c in text], dtype=torch.long)

def tensor_to_text(tensor, int_to_vocab):
    return "".join([int_to_vocab[i.item()] for i in tensor])

# --- MAIN PAGE LOGIC ---
def app():
    st.title("🏗️ Lab 2: Pre-Training Simulator")
    st.markdown("""
    Watch a baby AI model learn from scratch! 
    This simulation trains a neural network to predict the next character in a sequence.
    """)

    # 1. Setup Data
    col_data, col_params = st.columns([1, 1])
    
    with col_data:
        st.subheader("1. Training Data")
        text_input = st.text_area("Corpus (The 'Internet'):", value=DEFAULT_TEXT, height=150)
        
        # Build Vocabulary
        chars = sorted(list(set(text_input))) #unique sorted list of characters
        
        chars.append("<unk>")

        vocab_size = len(chars)
        #converting characters → token IDs
        vocab_to_int = { ch:i for i,ch in enumerate(chars) } #assigns each character a unique integer ID
        #converting generated token IDs → text
        int_to_vocab = { i:ch for i,ch in enumerate(chars) } #creates the reverse lookup:
        
        # Any character not in the vocab is mapped to <unk>
        def text_to_tensor(text, vocab_to_int):
            return torch.tensor([vocab_to_int.get(c, vocab_to_int["<unk>"]) for c in text], dtype=torch.long)

        # Encode data
        data_tensor = text_to_tensor(text_input, vocab_to_int)
        
        st.caption(f"Vocabulary Size: {vocab_size} unique characters")
        st.caption(f"Total Tokens: {len(data_tensor)}")

    with col_params:
        st.subheader("2. Hyperparameters")
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
        max_steps = st.slider("Training Steps", 50, 500, 100)
        batch_size = 32
        block_size = 8 # Context length

    # 2. Visualization Placeholders
    st.divider()
    st.subheader("3. Live Training Monitor")

    # --- Initialize session state ---
    if "model_type" not in st.session_state:
        st.session_state.model_type = None

    # Model selection buttons
    sml_model_btn = st.button("🧠 Small model", type="primary")
    big_model_btn = st.button("🤖 Big model", type="primary")

    if sml_model_btn:
        st.session_state.model_type = "small"

    if big_model_btn:
        st.session_state.model_type = "big"

    st.caption(f"Selected Model: {st.session_state.model_type}")

    # 1. User provides initial context
    user_context = st.text_input("Generation Prompt:", value="", placeholder="Enter a prompt here:") 

    start_btn = st.button("🚀 Start Training", type="primary")

    #------chart visualisation------
    col_chart, col_preview = st.columns([2, 1])
    
    with col_chart:
        chart_placeholder = st.empty()
    
    with col_preview:
        st.markdown("**Model Output Evolution**")
        text_placeholder = st.empty()

    #-------run training and generation------
    if start_btn:
        if st.session_state.model_type == "small":
            model = TinyLLM(vocab_size)
        elif st.session_state.model_type == "big":
            model = GPT(vocab_size)
        else:
            st.warning("Please select a model first!")
            return

        # Move to GPU if available (Recommended for Transformers!)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        loss_history = []
        progress_bar = st.progress(0)
        
        # Training Loop
        for step in range(max_steps):
            # Sample a batch of data
            xb, yb = get_batch(data_tensor, block_size, batch_size)
            xb, yb = xb.to(device), yb.to(device)

            # Evaluate the loss
            logits, loss = model(xb, yb)
            
            # Backpropagation (Learning)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # Track stats
            loss_history.append(loss.item())
            
            # Update UI every 50 steps
            #if step % 50 == 0 or step == max_steps - 1:
            if step % 10 == 0:
                # 1. Update Chart
                chart_data = pd.DataFrame(loss_history, columns=["Loss"])
                chart_placeholder.line_chart(chart_data)
                
                # 2. Update Progress
                progress_bar.progress((step + 1) / max_steps)
                # 3. Generate Sample Text
                # Convert user text to tensor (if not empty)
                if user_context:
                    #unsqueeze(0) makes it batch size 1, because your model expects (batch_size, seq_len).
                    context_tensor = text_to_tensor(user_context, vocab_to_int).unsqueeze(0)  # shape (1, seq_len)
                else:
                    # Context is just a zero (first char) to start generation
                    #Shape (1,1) → batch size 1, sequence length 1
                    context_tensor = torch.zeros((1, 1), dtype=torch.long)  # fallback

                generated_ids = model.generate(context_tensor, max_new_tokens=100)
                decoded_text = tensor_to_text(generated_ids[0], int_to_vocab)
                
                text_placeholder.code(decoded_text)
                
                # Slow down slightly so user can see updates
                #time.sleep(0.05)
        
        st.success(f"Training Complete! Final Loss: {loss.item():.4f}")