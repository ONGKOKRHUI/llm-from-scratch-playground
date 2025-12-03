"""
a real Decoder-only Transformer architecture 
(the same architecture used by GPT-2 and Llama).
"""

# models/gpt.py
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- COMPONENT 1: SELF-ATTENTION HEAD ---
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        # Key, Query, Value projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register the "causal mask" (lower triangular matrix) as a buffer so it's not a learned parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        
        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        
        # Masking: Don't let the model cheat by seeing future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

# --- COMPONENT 2: MULTI-HEAD ATTENTION ---
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs of all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# --- COMPONENT 3: FEED FORWARD NETWORK ---
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Grow by 4x (standard Transformer practice)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Project back down
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# --- COMPONENT 4: TRANSFORMER BLOCK ---
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections (x + ...) allow gradients to flow easily
        x = x + self.sa(self.ln1(x))   # Communication (Self-Attention)
        x = x + self.ffwd(self.ln2(x)) # Computation (Feed Forward)
        return x

# --- THE MAIN MODEL: GPT ---
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=4, block_size=128):
        super().__init__()
        # 1. Token Embeddings table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 2. Position Embeddings (so the model knows order of words)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 3. Transformer Blocks (The "Hidden Layers")
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head=n_head, block_size=block_size) 
            for _ in range(n_layer)
        ])
        
        # 4. Final Layer Norm
        self.ln_f = nn.LayerNorm(n_embd) 
        # 5. Language Model Head (project back to vocabulary)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens (because positional embeddings can't handle > block_size)
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx