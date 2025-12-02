# modules/training_demo.py
import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import time

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

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, n_embd=32):
        """
        Each character is stored as a point in a 32-dimensional space.

        The model learns:

        which characters are close together

        which are far apart

        which directions represent common patterns

        That's how embeddings represent meaning.

        If your vocabulary = 50 characters, then:
        Embedding: 32 numbers
        ↓
        Linear layer: produces 50 numbers
        ↓
        Softmax: turns them into probabilities for each of the 50 tokens
        
        In LLM, the LM Head always maps:

        hidden_size → vocab_size

        For GPT-2:

        hidden_size = 768

        vocab_size ≈ 50k

        For GPT-3:

        hidden_size = 12,288

        vocab_size = 50k
        """
        super().__init__()
        # 1. Token Embeddings: Looking up the "meaning" vector for each token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # 2. A tiny "Linear" head to predict the next token logits acoss all possible tokens
        # In a real GPT, there are Self-Attention layers here!
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None): 
        """
        Symbol	Meaning	                        Example in your code
        B	    Batch size	                    32 sequences per training step
        T	    Time steps (context length)	    block_size = 8 characters
        C	    Channels / feature size	        embedding size = 32
        So a tensor shaped (B, T, C) is:
        B sequences, each of length T, and each token represented by C features.
        """
        # idx and targets are both (B,T) tensor of integers (32, 8) before embedding
        
        # Get embeddings
        logits = self.token_embedding_table(idx) # After embedding (B,T,C) (32, 8, 32)
        logits = self.lm_head(logits) # After linear layer (B,T,vocab_size) (32, 8, vocab_size)

        '''
        Why we flatten?

        Because PyTorch expects:

        input	shape	meaning
        logits	(N, C)	N predictions of C classes (32, 8, 50) -> (256, 50)
        targets	(N)	N correct class indices (32, 8) → (256)

        Our model outputs (B,T,C) and (B,T), so we flatten B*T → N.
        
        Cross entropy does:
        softmax on logits
        compares predicted probability distribution vs correct token
        computes negative log likelihood
        averages over all 256 positions
        '''
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #Flatten logits for cross-entropy
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        '''
        At every training step:

        The model predicts the next token for every position in every sequence

        We calculate how wrong it was

        This error drives backpropagation

        The embeddings and linear layer update
        '''

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Simple generation loop
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

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
        chars = sorted(list(set(text_input)))
        vocab_size = len(chars)
        vocab_to_int = { ch:i for i,ch in enumerate(chars) }
        int_to_vocab = { i:ch for i,ch in enumerate(chars) }
        
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
    
    col_chart, col_preview = st.columns([2, 1])
    
    with col_chart:
        chart_placeholder = st.empty()
    
    with col_preview:
        st.markdown("**Model Output Evolution**")
        text_placeholder = st.empty()
        
    start_btn = st.button("🚀 Start Training", type="primary")

    if start_btn:
        # Initialize Model
        model = TinyLLM(vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        loss_history = []
        progress_bar = st.progress(0)
        
        # Training Loop
        for step in range(max_steps):
            # Sample a batch of data
            xb, yb = get_batch(data_tensor, block_size, batch_size)

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
                #st.experimental_rerun()
                # 3. Generate Sample Text
                # Context is just a zero (first char) to start generation
                context = torch.zeros((1, 1), dtype=torch.long)
                generated_ids = model.generate(context, max_new_tokens=100)
                decoded_text = tensor_to_text(generated_ids[0], int_to_vocab)
                
                text_placeholder.code(decoded_text)
                
                # Slow down slightly so user can see updates
                #time.sleep(0.05)
        
        st.success(f"Training Complete! Final Loss: {loss.item():.4f}")