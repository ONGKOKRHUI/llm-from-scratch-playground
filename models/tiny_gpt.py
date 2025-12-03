import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import time

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