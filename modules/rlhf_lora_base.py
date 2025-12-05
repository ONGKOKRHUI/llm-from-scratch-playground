# app_rlhf_lora.py
import streamlit as st
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import json

# -----------------------------
# Configuration
# -----------------------------
FEEDBACK_FILE = "data/rlhf_dataset.json"

# -----------------------------
# LoRA Layer
# -----------------------------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        return x @ self.weight + (x @ self.lora_A @ self.lora_B) * self.scaling

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
# Prepare dataset
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
def train_lora(base_model, lora_layer, tokenizer, epochs=3, lr=1e-3, device="cpu"):
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
        outputs = base_model.transformer.wte(input_ids)
        logits = lora_layer(outputs)
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
def generate_with_lora(base_model, lora_layer, tokenizer, prompt, device="cpu"):
    base_model.eval()
    lora_layer.to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = base_model.transformer.wte(input_ids)
    logits = lora_layer(outputs)
    predicted_ids = torch.argmax(logits, dim=-1)
    text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return text

# -----------------------------
# Streamlit App
# -----------------------------
def app():
    st.title("🔧 RLHF + LoRA Studio")
    st.markdown("Collect feedback, fine-tune a small LoRA adapter, and see improved generations.")

    # -----------------------------
    # Device Selection
    # -----------------------------
    device_option = st.radio("Device", ["cpu", "cuda" if torch.cuda.is_available() else "cuda (unavailable)"])
    device = "cpu" if "unavailable" in device_option else device_option

    # -----------------------------
    # Load Base Model
    # -----------------------------
    @st.cache_resource
    def load_base_model(device):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.to(device)
        model.eval()
        return tokenizer, model

    tokenizer, base_model = load_base_model(device)

    # Initialize LoRA
    lora_layer = LoRALayer(base_model.config.n_embd, base_model.config.vocab_size)

    # -----------------------------
    # RLHF Feedback Collection
    # -----------------------------
    st.subheader("1️⃣ Generate Candidates and Collect Feedback")

    prompt = st.text_input("Enter a prompt:", "The capital of France is")
    if "ans_a" not in st.session_state:
        st.session_state.ans_a, st.session_state.ans_b = "", ""
        st.session_state.generated = False

    if not st.session_state.generated:
        if st.button("Generate Candidates"):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out_a = base_model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=1.1)
            out_b = base_model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=1.1)
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

    # -----------------------------
    # Train LoRA Adapter
    # -----------------------------
    st.subheader("2️⃣ Train LoRA Adapter on Feedback")
    if st.button("Train LoRA"):
        train_lora(base_model, lora_layer, tokenizer, epochs=2, device=device)
        st.success("LoRA training completed!")

    # -----------------------------
    # Generate Improved Text
    # -----------------------------
    st.subheader("3️⃣ Generate with LoRA")
    improved_prompt = st.text_input("Enter a prompt for improved generation:", "The capital of France is", key="improved")
    if st.button("Generate Improved"):
        result = generate_with_lora(base_model, lora_layer, tokenizer, improved_prompt, device=device)
        st.success(result)

    # -----------------------------
    # View Collected Feedback
    # -----------------------------
    with st.expander("View Collected Feedback"):
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE) as f:
                st.json(json.load(f))
