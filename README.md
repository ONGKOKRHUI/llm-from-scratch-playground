# NexusAI: Advanced LLM Architecture & Fine-Tuning Workbench

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=for-the-badge&logo=openai&logoColor=white)

## 🚀 Executive Summary

**NexusAI** is a comprehensive, interactive laboratory designed to bridge the gap between theoretical Large Language Model (LLM) concepts and practical engineering application.

Unlike standard "wrapper" projects that simply call an API, NexusAI provides a **"glass-box" approach** to AI. It allows users to visualize, intervene, and engineer every stage of the LLM lifecycle—from raw text tokenization and pre-training loop dynamics to advanced post-training techniques like Low-Rank Adaptation (LoRA) and Reinforcement Learning from Human Feedback (RLHF).

This workbench was built to demonstrate **full-stack AI engineering competence**, combining low-level PyTorch implementations of transformer components with high-level orchestration using modern libraries.

---

## 🛠️ Tech Stack & Engineering Highlights

This project leverages a hybrid tech stack to maximize both educational depth and industrial relevance:

*   **Core ML Engine:** `PyTorch` & `Transformers` (Hugging Face).
*   **Interface:** `Streamlit` for rapid, interactive dashboarding and real-time visualization.
*   **Low-Level Implementation:**
    *   **Custom Training Loops:** Raw PyTorch loops implementing backpropagation, optimizer stepping, and loss tracking.
    *   **Manual LoRA Implementation:** A from-scratch implementation of Low-Rank Adaptation layers (`W + A x B`) to demonstrate mathematical understanding of PEFT (Parameter-Efficient Fine-Tuning).
*   **High-Level Integration:**
    *   `LangChain` concepts for model abstraction.
    *   `Tiktoken` for BPE (Byte Pair Encoding) visualization.
    *   `Altair` for probability distribution charting.
*   **LLM-as-a-Judge:** Integration with **OpenAI GPT-4** and **Google Gemini** APIs to perform automated evaluation of model outputs.

---

## 🔬 Modules & Capabilities

NexusAI is structured into **7 distinct laboratories**, each targeting a critical phase of the LLM pipeline:

### 1. The Tokenization Sandbox
*   **Goal:** Demystify how raw text is converted into machine-readable tensors.
*   **Features:**
    *   Real-time BPE visualization using `tiktoken`.
    *   Color-coded token chunks to visualize vocabulary efficiency.
    *   Comparison of token counts vs. word counts to understand context window usage.

### 2. Pre-Training Simulator (The "Brain")
*   **Goal:** Visualize the training dynamics of a Transformer model.
*   **Features:**
    *   **Live Training Loop:** A functioning PyTorch training loop on a "NanoGPT" architecture.
    *   **Real-time Metrics:** Dynamic loss curves (Training vs. Validation) updated in real-time.
    *   **Evolutionary Generation:** Watch the model evolve from outputting gibberish to coherent text as it learns.

### 3. Inference & Decoding Laboratory
*   **Goal:** Explore the probabilistic nature of text generation.
*   **Features:**
    *   **Next-Token Probability:** Bar charts showing the top-K candidates for the next word.
    *   **Strategy Playground:** Interactive controls for **Temperature**, **Top-K**, **Top-P (Nucleus Sampling)**, and **Beam Search**.
    *   **Impact Analysis:** See how high temperature leads to "hallucinations" while Beam Search yields deterministic coherence.

### 4. Fine-Tuning Studio (SFT)
*   **Goal:** Demonstrate the difference between a "Base Model" and an "Instruct Model".
*   **Features:**
    *   **Side-by-Side Comparison:** Compare raw GPT-2 (base) against Flan-T5 (instruction-tuned) on the same prompt.
    *   **Parameter Visualization:** Visual breakdown of trainable vs. frozen parameters.

### 5. RLHF & LoRA (Base Model)
*   **Goal:** Implement the "ChatGPT recipe" (RLHF) on a base model.
*   **Features:**
    *   **Preference Data Collection:** A UI for humans to rank Model A vs. Model B outputs.
    *   **LoRA Training:** Fine-tune a custom LoRA adapter on the collected preference data.
    *   **Feedback Loop:** Immediate generation using the newly trained adapter to verify improvements.

### 6. RLHF & LoRA (Instruct Model)
*   **Goal:** Advanced alignment on a larger, pre-instruction-tuned model (Flan-T5 Large).
*   **Features:**
    *   Scales the RLHF workflow to a more capable model.
    *   Demonstrates how efficient fine-tuning can alter the behavior of even large models using consumer hardware.

### 7. Evaluation & Metrics Suite
*   **Goal:** Quantify model performance using academic and industrial standards.
*   **Features:**
    *   **Mathematical Metrics:** Perplexity (PPL) calculation.
    *   **Reference Metrics:** ROUGE-1, ROUGE-2, and ROUGE-L scoring against gold-standard references.
    *   **AI Evaluation:** An **"LLM-as-a-Judge"** system where GPT-4 or Gemini grades the local model's responses on a scale of 1-10 with reasoning.

---

## 💻 Installation & Setup Guide

Follow these steps to deploy NexusAI on a fresh machine.

### Prerequisites
*   **Python 3.10+**
*   **Git**
*   *(Optional)* **NVIDIA GPU** with CUDA installed (for faster training).

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/nexus-ai-workbench.git
cd nexus-ai-workbench
```

### 2. Set Up Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

**Using venv (Standard Python):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

**Using Conda:**
```bash
conda create -n nexus-ai python=3.10
conda activate nexus-ai
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
To use the "LLM-as-a-Judge" features or the OpenAI/Gemini integrations, you need to set up your API keys.

1.  Create a file named `.env` in the root directory.
2.  Add your keys as follows (leave blank if you don't have one):

```ini
# .env file
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
```

### 5. Launch the Workbench
Run the Streamlit application:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🗺️ Future Roadmap

*   **RAG (Retrieval-Augmented Generation) Module:** Implementing a Vector Database (ChromaDB) tab to demonstrate how LLMs interact with external knowledge bases.
*   **Quantization Lab:** Adding 4-bit and 8-bit quantization (via `bitsandbytes`) to demonstrate running large models on consumer hardware.
*   **Model Deployment:** A section on containerizing the model (Docker) and serving it via a REST API (FastAPI).

---

## 📝 Author

**[Your Name Here]**
*AI Engineer & Full-Stack Developer*

---

*Built with ❤️ using PyTorch & Streamlit.*
