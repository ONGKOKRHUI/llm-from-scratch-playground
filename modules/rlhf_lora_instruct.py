# app_rlhf_lora_flant5.py
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import json

# -----------------------------
# Configuration
# -----------------------------
FEEDBACK_FILE = "data/rlhf_dataset.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LoRA Layer for T5
# -----------------------------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=32, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        return x + (x @ self.lora_A @ self.lora_B) * self.scaling

# -----------------------------
# Save human feedback
# -----------------------------
def save_feedback(prompt, chosen, rejected):
    os.makedirs("data", exist_ok=True)
    entry = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []
    data.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -----------------------------
# Prepare dataset for LoRA
# -----------------------------
def prepare_dataset(tokenizer, max_len=64):
    if not os.path.exists(FEEDBACK_FILE):
        return None, None
    with open(FEEDBACK_FILE) as f:
        data = json.load(f)
    input_ids, labels = [], []
    for entry in data:
        text = entry["prompt"] + " " + entry["chosen"]
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        input_ids.append(enc.input_ids)
        labels.append(enc.input_ids)
    if input_ids:
        return torch.cat(input_ids), torch.cat(labels)
    return None, None

# -----------------------------
# Train LoRA
# -----------------------------
def train_lora(base_model, lora_layer, tokenizer, epochs=3, lr=1e-3, device=DEVICE):
    base_model.eval()
    optimizer = torch.optim.Adam(lora_layer.parameters(), lr=lr)
    input_ids, labels = prepare_dataset(tokenizer)
    if input_ids is None:
        st.warning("No feedback data to train on!")
        return
    input_ids, labels = input_ids.to(device), labels.to(device)
    lora_layer.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = base_model.get_encoder()(input_ids)[0]  # encoder output
        logits = lora_layer(embeddings)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()
        st.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# -----------------------------
# Generate text with LoRA
# -----------------------------
def generate_with_lora(base_model, lora_layer, tokenizer, prompt, max_new_tokens=40, device=DEVICE):
    base_model.eval()
    lora_layer.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    encoder_out = base_model.get_encoder()(inputs.input_ids)[0]
    adapted_enc = lora_layer(encoder_out)
    # Use adapted encoder output in the decoder
    outputs = base_model.generate(
        inputs_embeds=adapted_enc,
        max_new_tokens=max_new_tokens
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model

# -----------------------------
# Streamlit UI
# -----------------------------
def app():
    st.title("🔧 RLHF + LoRA on Flan-T5-Large")
    st.markdown("Collect human feedback, train LoRA adapters, and see visible improvements!")

    # ---Load Model---
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            st.session_state.tokenizer, st.session_state.base_model = load_model()
            st.session_state.lora_layer = LoRALayer(
                st.session_state.base_model.config.d_model,
                st.session_state.tokenizer.vocab_size,
                rank=32,
                alpha=32
            )
            st.session_state.model_loaded = True
            st.success("Model loaded!")

    # --- RLHF Feedback ---
    st.subheader("1️⃣ Generate Candidates and Collect Feedback")
    prompt = st.text_input("Enter a prompt:", "Explain photosynthesis in simple terms.")

    if "ans_a" not in st.session_state:
        st.session_state.ans_a, st.session_state.ans_b = "", ""
        st.session_state.generated = False

    # → returns False → if condition fails, return True otherwise
    if st.session_state.get("model_loaded", False): 
        tokenizer = st.session_state.tokenizer
        base_model = st.session_state.base_model
        lora_layer = st.session_state.lora_layer

    if not st.session_state.generated:
        if st.button("Generate Candidates"):
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out_a = base_model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=1.1)
            out_b = base_model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=1.1)
            st.session_state.ans_a = tokenizer.decode(out_a[0], skip_special_tokens=True)
            st.session_state.ans_b = tokenizer.decode(out_b[0], skip_special_tokens=True)
            st.session_state.generated = True
            st.rerun()

    if st.session_state.generated:
        c1, c2 = st.columns(2)
        with c1:
            st.info(st.session_state.ans_a)
            if st.button("👈 A is Better"):
                save_feedback(prompt, st.session_state.ans_a, st.session_state.ans_b)
                st.session_state.generated = False
                st.success("Feedback saved!")
                st.rerun()
        with c2:
            st.info(st.session_state.ans_b)
            if st.button("B is Better 👉"):
                save_feedback(prompt, st.session_state.ans_b, st.session_state.ans_a)
                st.session_state.generated = False
                st.success("Feedback saved!")
                st.rerun()

    # --- Train LoRA ---
    st.subheader("2️⃣ Train LoRA Adapter on Feedback")
    if st.button("Train LoRA"):
        train_lora(base_model, lora_layer, tokenizer, epochs=3)
        st.success("LoRA training completed!")

    # --- Generate Improved Text ---
    st.subheader("3️⃣ Generate Improved Text with LoRA")
    improved_prompt = st.text_input("Prompt for improved generation:", "Explain photosynthesis in simple terms.", key="improved")
    if st.button("Generate Improved"):
        result = generate_with_lora(base_model, lora_layer, tokenizer, improved_prompt)
        st.success(result)

    # --- View Collected Feedback ---
    with st.expander("View Collected Feedback"):
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE) as f:
                st.json(json.load(f))
