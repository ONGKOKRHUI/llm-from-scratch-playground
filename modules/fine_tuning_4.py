# modules/fine_tuning_lab.py
import streamlit as st
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import os
import json
import altair as alt

# --- FILE TO SAVE HUMAN FEEDBACK ---
FEEDBACK_FILE = "data/rlhf_dataset.json"

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_models(device="cpu"):
    # 1. Base Model
    base_name = "gpt2"
    base_tokenizer = GPT2Tokenizer.from_pretrained(base_name)
    base_model = GPT2LMHeadModel.from_pretrained(base_name).to(device)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    # 2. Instruct Model
    sft_name = "google/flan-t5-small"
    sft_tokenizer = AutoTokenizer.from_pretrained(sft_name)
    sft_model = AutoModelForSeq2SeqLM.from_pretrained(sft_name).to(device)
    
    return (base_tokenizer, base_model), (sft_tokenizer, sft_model)

def save_feedback(prompt, chosen, rejected):
    os.makedirs("data", exist_ok=True)
    new_entry = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []
    data.append(new_entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=4)
    return len(data)

# --- LORA IMPLEMENTATION (Educational) ---
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        """
        in_dim: input dimension of your layer (hidden size)
        out_dim: output dimension (hidden size for the linear layer)
        rank: the “low-rank” size of the adapters. Lower rank → fewer trainable parameters.
        alpha: a scaling factor controlling how much the LoRA adapters affect the output
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 1. The Frozen Pre-trained Weight (Simulated large matrix)
        #obtains the original parameter weights 
        #requires_grad=False ensures it does not get updated during training.
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=False)
        
        # 2. The Trainable Low-Rank Matrices (A and B)
        # Matrix A: (in_dim, rank) -> projects the input down to a smaller “rank” dimension → compresses input.
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank))
        # Matrix B: (rank, out_dim) -> projects it back to the output dimension → re-expands.
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        
        # Scaling factor - adjusts the magnitude of the LoRA contribution.
        self.scaling = alpha / rank

    def forward(self, x):
        # The Original Path (Frozen) computes the original linear transformation (frozen, no learning).
        original_out = x @ self.weight
        
        # The LoRA Path (Trainable) computes the learnable delta.
        # x -> A -> B -> Scale -- Multiply by scaling to control magnitude.
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        # Combine
        return original_out + lora_out

    def get_param_count(self):
        # Calculate distinct parameters
        #frozen: number of parameters in the main weight (not trainable)
        frozen = self.weight.numel() #.numel() counts total elements in a tensor.
        #trainable: number of parameters in LoRA adapters (tiny)
        trainable = self.lora_A.numel() + self.lora_B.numel()
        return frozen, trainable

# --- MAIN APP ---
def app():
    st.title("🔧 Lab 4: Post-Training Studio")
    st.markdown("Explore how we turn raw models into helpful assistants.")
    
    tab_sft, tab_rlhf, tab_lora = st.tabs([
        "1️⃣ SFT: Base vs. Instruct", 
        "2️⃣ RLHF: Human Feedback",
        "3️⃣ LoRA: Efficient Tuning"
    ])
    
    # Detect GPU once
    if "device" not in st.session_state:
        st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"
        st.toast(f"Using device: {st.session_state.device}")

    # Load models only if needed (Optimization)
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False

    # ==========================
    # TAB 1: SFT BATTLE (Supervised Fine-Tuning VS Base Language Model)
    # ==========================
    with tab_sft:
        st.markdown("### Base Model vs. Instruction Tuned Model")
        
        if not st.session_state.models_loaded:
            if st.button("Load Models (GPT-2 & Flan-T5)"):
                with st.spinner(f"Loading models on {st.session_state.device}..."):
                    st.session_state.models = load_models(device=st.session_state.device)
                    st.session_state.models_loaded = True
                    st.rerun()
        
        # If models are loaded
        if st.session_state.models_loaded:
            (base_tok, base_model), (sft_tok, sft_model) = st.session_state.models
            
            prompt = st.text_input("Enter a prompt:", value="The capital of France is", key="sft_prompt")
            
            if st.button("⚔️ Compare Models", type="primary"):
                c1, c2 = st.columns(2)
                
                ######## Base
                # Tokenize the prompt
                inputs = base_tok(prompt, return_tensors="pt")
                # Generate
                out = base_model.generate(**inputs, max_new_tokens=40, do_sample=False)
                #Decode back to text: skip_special_tokens=True removes tokens like <pad> or <eos>.
                res_base = base_tok.decode(out[0], skip_special_tokens=True)
                c1.info(f"**GPT-2 (Base):**\n\n{res_base}")
                
                # SFT
                inputs = sft_tok(prompt, return_tensors="pt")
                #Generate output (note: do_sample defaults to deterministic for T5 here).
                out = sft_model.generate(**inputs, max_new_tokens=40)
                res_sft = sft_tok.decode(out[0], skip_special_tokens=True)
                c2.success(f"**Flan-T5 (Instruct):**\n\n{res_sft}")

    # ==========================
    # TAB 2: RLHF COLLECTOR
    # ==========================
    with tab_rlhf:
        st.markdown("### Reinforcement Learning from Human Feedback")
        if not st.session_state.models_loaded:
            st.warning("Please load models in Tab 1 first.")
        else:
            if "rlhf_prompt" not in st.session_state:
                st.session_state.rlhf_prompt = "Explain quantum physics like I'm 5."
                st.session_state.ans_a = ""
                st.session_state.ans_b = ""
                st.session_state.generated = False

            st.write(f"**Prompt:** {st.session_state.rlhf_prompt}")
            
            if not st.session_state.generated:
                if st.button("Generate Candidates"):
                    (base_tok, base_model), _ = st.session_state.models
                    inputs = base_tok(st.session_state.rlhf_prompt, return_tensors="pt")
                    
                    out_a = base_model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=1.1)
                    st.session_state.ans_a = base_tok.decode(out_a[0], skip_special_tokens=True)
                    
                    out_b = base_model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=1.1)
                    st.session_state.ans_b = base_tok.decode(out_b[0], skip_special_tokens=True)
                    
                    st.session_state.generated = True
                    st.rerun()

            if st.session_state.generated:
                c1, c2 = st.columns(2)
                with c1:
                    st.info(st.session_state.ans_a)
                    if st.button("👈 A is Better"):
                        c = save_feedback(st.session_state.rlhf_prompt, st.session_state.ans_a, st.session_state.ans_b)
                        st.toast(f"Saved! Total: {c}")
                        st.session_state.generated = False
                        st.rerun()
                with c2:
                    st.info(st.session_state.ans_b)
                    if st.button("B is Better 👉"):
                        c = save_feedback(st.session_state.rlhf_prompt, st.session_state.ans_b, st.session_state.ans_a)
                        st.toast(f"Saved! Total: {c}")
                        st.session_state.generated = False
                        st.rerun()
            
            with st.expander("View Collected Data"):
                 if os.path.exists(FEEDBACK_FILE):
                    with open(FEEDBACK_FILE) as f: st.json(json.load(f))

    # ==========================
    # TAB 3: LORA SIMULATOR
    # ==========================
    with tab_lora:
        st.markdown("""
        ### Low-Rank Adaptation (LoRA)
        Training a full model (billions of params) is expensive. LoRA freezes the main model 
        and only trains tiny "Adapter" matrices ($A$ and $B$).
        
        $$ W_{new} = W_{frozen} + (A \\times B) $$
        """)
        
        col_viz, col_math = st.columns([1, 1])
        
        with col_math:
            st.subheader("🧮 Parameter Calculator")
            # allows user to decide the model hidden size and output dimension
            d_model = st.number_input("Model Dimension (Hidden Size)", value=4096, step=128)
            d_out = st.number_input("Output Dimension", value=4096, step=128)
            # Lower r → far fewer trainable parameters.
            rank = st.slider("LoRA Rank (r)", 1, 64, 8, help="Lower rank = fewer parameters.")
            
            # Instantiate our educational layer
            demo_layer = LoRALayer(d_model, d_out, rank, alpha=16)
            frozen_params, trainable_params = demo_layer.get_param_count()
            
            total = frozen_params + trainable_params
            savings = 100 - (trainable_params / frozen_params * 100)
            
            st.metric("Frozen Parameters (Weight W)", f"{frozen_params:,}")
            st.metric("Trainable Parameters (A + B)", f"{trainable_params:,}", delta=f"{savings:.4f}% smaller")
            
        with col_viz:
            st.subheader("📊 Visualization")
            
            # Create data for the chart
            data = pd.DataFrame({
                'Layer Type': ['Full Fine-Tuning', 'LoRA Fine-Tuning'],
                'Parameters': [frozen_params, trainable_params]
            })
            
            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    x=alt.X('Layer Type:N', title='Layer Type'),
                    y=alt.Y('Parameters:Q', scale=alt.Scale(type='linear'), title='Parameters'),
                    color='Layer Type:N',
                    tooltip=['Layer Type:N', 'Parameters:Q']
                )
                .properties(
                    title="Comparison of Parameters"
                )
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            st.info("Notice the number of parameters tunable for LoRA compared to full fine-tuning is significantly lower.")
            
        st.divider()
        st.subheader("The Code Behind It")
        st.markdown("Here is how we implement the LoRA forward pass in PyTorch manually:")
        
        code_snippet = """
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8):
        # 1. Freeze the massive weight matrix
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=False)
        
        # 2. Create tiny adapters
        self.A = nn.Parameter(torch.randn(in_dim, rank)) 
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scale = 0.1

    def forward(self, x):
        # Original frozen path
        frozen_out = x @ self.weight
        
        # Trainable adapter path (Low Rank)
        adapter_out = (x @ self.A @ self.B) * self.scale
        
        return frozen_out + adapter_out
        """
        st.code(code_snippet, language="python")