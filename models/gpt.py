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
    """
    The attention head:
    - Turns each token into key, query, value vectors
    - Computes how much each token should pay attention to earlier tokens
    - Uses softmax to make those weights sum to 1
    - Combines earlier tokens' values based on the attention weights
    - Produces a new representation for each token based on its context

    This is how the model learns things like:

    “the subject of this sentence is X, so the verb should agree”
    “this word refers to something earlier”
    “this phrase starts here and ends there”
    """
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        """
        In self-attention, each input token is transformed into three vectors:
        - Key (K): Encodes “what this token represents” in terms of attention.
        - Query (Q): Encodes “what this token is looking for” in other tokens.
        - Value (V): Contains the actual content/information of the token to be mixed via attention.
        nn.Linear(n_embd, head_size) is a fully connected layer that projects the input embeddings of size n_embd 
            (embedding dimension) into the smaller head_size for this attention head.
        bias=False is standard in GPT-like implementations; the bias isn't needed for attention computations.
        
        Creates a lower triangular matrix of size (block_size, block_size)
        Purpose: This is the causal mask, used to prevent the model from “seeing the future” 
            in autoregressive generation.
        register_buffer stores tensors that are not parameters (so they won't be updated during
            training) but still move with the model between CPU/GPU.
        Standard dropout layer applied to attention weights to prevent overfitting.
        dropout=0.2 means 20% of attention connections are randomly zeroed during training.
        """
        super().__init__() #inherit the PyTorch behaviors like tracking parameters, moving to GPU, etc.
        # Key, Query, Value projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register the "causal mask" (lower triangular matrix) as a buffer so it's not a learned parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Each token gets 3 projections:
        query vector → tells what this token wants to attend to
        key vector → tells what this token contains
        value vector → the actual information passed on after attention
        These come from separate linear layers.
        """
        B, T, C = x.shape # batch size, sequence length, embedding dimension
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        
        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        """
        k.transpose(-2,-1) converts (B,T,H) → (B,H,T)
        Matrix multiply: (B,T,H) @ (B,H,T) → (B,T,T)
        Resulting matrix: wei[b, i, j] = attention score between token i and token j
        Why * C**-0.5?
        This is scaled dot-product attention.
        If you don't scale, the dot products grow with dimension 
        → softmax becomes too peaked → gradients unstable.
        Multiplying by 1 / sqrt(C) stabilizes training.
        """
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        
        # Masking: Don't let the model cheat by seeing future tokens
        """
        The causal mask makes sure:
        token i CANNOT see tokens j > i (future tokens)
        Masking with -inf makes softmax give 0 probability to those positions.
        """
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        """
        This turns each row into a probability distribution over past tokens:
        for each token i:
            softmax(wei[i]) tells how much token i should attend to token j
        """
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei) #Randomly disable some attention connections to improve generalization.
        
        # Perform the weighted aggregation of the values
        """
        this performs
        output[i] = sum_j( attention_weight[i,j] * value[j] )
        Each token becomes a weighted mixture of previous tokens' values.
        """
        v = self.value(x) # (B,T,head_size)
        #This is the output of one head.
        #Later, the MultiHeadAttention module concatenates all heads together.
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

# --- COMPONENT 2: MULTI-HEAD ATTENTION ---
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    """
    What Multi-Head Attention Does
    A single attention head can only focus on one type of relationship at a time.
    But language is complex:
    one head might track subject-verb agreement
    another might track coreference ("he" → "John")
    another might track sentence boundaries
    another might track syntax patterns
    So Transformers run multiple attention heads in parallel, each learning its own pattern.
    Multi-head attention = multiple independent self-attention heads + final linear mixing layer
    One attention head learns one pattern → too limited.
    Multiple heads → they specialize.
    Together, they allow the model to “look at” the sequence from many perspectives at once.
    """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        """
        Create multiple heads
        num_heads = e.g. 4
        Each head has dimension head_size
        Total embedding = num_heads * head_size = n_embd
        Each head computes its own attention:
        Head 1 → looks for subject relationships etc 
        """
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        #After concatenating all heads, you have a vector of size:
        #[B, T, num_heads * head_size] = [B, T, n_embd]
        #The projection mixes these head outputs together, allowing the model to:
        # combine information from different attention heads
        # learn better final representations
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs of all heads
        """
        Each head returns a tensor of shape (B, T, head_size)
        Concatenating across heads gives:
        (B, T, num_heads * head_size) = (B, T, n_embd)
        This mixes all head outputs back into a single embedding of 
        shape (B, T, n_embd)
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# --- COMPONENT 3: FEED FORWARD NETWORK ---
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    """
    This is the position-wise feed-forward network (FFN) from the Transformer architecture.
    It is applied to each token independently, after attention has mixed information between tokens.
    Where attention handles “communication”, FFN handles processing.
    FeedForward = “thinking” step.
    Self-attention mixes information between tokens
    FeedForward transforms each token's representation individually
    You can think of it like:
    Attention = reading and gathering information, “Look around the room and gather useful information.”
    FeedForward = processing and reasoning. “Think deeply about what you just gathered.”
    """
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            # This expands the embedding dimension 4× (standard Transformer practice)
            # Gives the model more capacity to learn complex transformations.
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(), #Adds non-linearity, ReLU enables modeling complex relationships.
            nn.Linear(4 * n_embd, n_embd), # Compress: Brings the dimension back to the original embedding size.
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# --- COMPONENT 4: TRANSFORMER BLOCK ---
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    """
    A GPT model is basically:
    A stack of many Transformer Blocks
    (your code uses 4 blocks, GPT-2 uses 12-48 blocks, GPT-3 uses 96 blocks)
    Each block performs:
    Communication: tokens talk to each other via attention
    Computation: each token processes information via FeedForward

    Residual + LayerNorm: stabilizes training and preserves gradients
    A Transformer Block = LN → Attention → Residual → LN → FFN → Residual
    """
    def __init__(self, n_embd, n_head, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        #If n_embd = 64 and n_head = 4, So each head output is dimension 16, and all 4 heads together = 64.
        head_size = n_embd // n_head
        #this performs self attention across all tokens across multiple heads
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        #this performs applies a per-token MLP, expands embedding dim (×4), applies nonlinearity,compresses back to n_embd
        self.ffwd = FeedForward(n_embd)
        #Layer normalization helps stabilize training by normalizing activations.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections (x + ...) allow gradients to flow easily
        """
        ⭐ 1. Pre-norm: self.ln1(x)
        Unlike the original Transformer, GPT/LLama use pre-LayerNorm (normalization before sublayers).
        Benefits:
        more stable training, especially for deep networks
        avoids exploding gradients
        ⭐ 2. Self-attention: self.sa(...)
        The MultiHeadAttention module:
        applies multiple attention heads
        mixes information across tokens
        outputs a new contextual embedding
        ⭐ 3. Residual connection: x + (...)
        Residual connections allow:
        gradients to flow easily
        the model to “add improvements” instead of fully replacing representations
        deep networks (many layers) not to degrade performance
        This idea is borrowed from ResNets.
        🗣️ Phase 1: Communication (Self-Attention)
        Each token asks:
        “Which other tokens should I pay attention to?”
        It gathers a weighted summary of other tokens.
        🧮 Phase 2: Computation (FeedForward)
        Each token then processes the gathered information:
        “Now that I have context, what do I compute next?”
        This is done independently per token.
        """
        x = x + self.sa(self.ln1(x))   # Communication (Self-Attention)
        x = x + self.ffwd(self.ln2(x)) # Computation (Feed Forward)
        return x

# --- THE MAIN MODEL: GPT ---
class GPT(nn.Module):
    """
    A full GPT model contains:
    Token embeddings → converts integers → vectors
    Positional embeddings → injects order information
    Transformer blocks (stacked) → attention + feedforward
    LayerNorm → stabilize
    LM head → convert back to logits over vocabulary 
    Next-token logits → cross-entropy loss
    """
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=4, block_size=128):
        super().__init__()
        # 1. Token Embeddings table, Input = integer token IDs, Output = learned vectors of size n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 2. Position Embeddings (so the model knows order of words)
        # final_embedding = token_embedding + position_embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 3. Transformer Blocks (The "Hidden Layers")
        # If n_layer = 4, you have 4 sequential Transformer blocks:
        # Each block:
        # mixes token information via attention
        # computes new representations via feedforward
        # uses residuals + LayerNorm
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head=n_head, block_size=block_size) 
            for _ in range(n_layer)
        ])
        
        # 4. Final Layer Norm
        self.ln_f = nn.LayerNorm(n_embd) 
        # 5. Language Model Head (project back to vocabulary)
        # This converts each final token embedding into a vector of size vocab_size.
        # Those numbers are logits, unnormalized scores for next-token probabilities.
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.block_size = block_size

    def forward(self, idx, targets=None):
        #The Forward Pass (Predicting Next Token)
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        
        x = tok_emb + pos_emb # (B,T,C)
        """
        Each block transforms the representation using:
        attention → sees context
        feedforward → computes
        residuals → preserves
        normalization → stabilizes
        """
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C) final normalization
        #This is raw next-token predictions for each position.
        logits = self.lm_head(x) # (B,T,vocab_size) 

        #This compares the predicted distribution vs. the true next token.
        #used during training to compute loss
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressive generation loop:
        It predicts one token at a time, then feeds the prediction back into 
        itself to predict the next one.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens (because positional embeddings can't handle > block_size)
            # If block size = 128 and sequence grows to 129 tokens, we drop the oldest one:
            # [last 128 tokens only]
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            """
            logits shape is: (B, T, vocab_size)
            It contains next-token predictions for every position in the sequence.
            """
            #Take only the final token’s prediction
            #We only care about the logits for the last position, 
            # because that is where the “next token” prediction is.
            logits = logits[:, -1, :] # becomes (B, C)
            #turns raw model scores into a probability distribution
            probs = F.softmax(logits, dim=-1) # (B, C) 
            #Randomly picks 1 token according to the probabilities.
            # This is why generation is non-deterministic unless you 
            # use greedy sampling.
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled token to the current sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx