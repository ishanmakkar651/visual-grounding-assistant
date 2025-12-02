# Visual Grounding Assistant 🔍
**Make any technical diagram or chart searchable & explorable with a single click.**

An end-to-end Vision-Language project running on a free Colab T4 GPU.

## Tech Stack
- **Model:** Qwen2-VL-7B-Instruct (4-bit Quantized)
- **Training:** QLoRA Fine-tuning
- **Backend:** FastAPI (Tunnelled via Ngrok/Colab)
- **Frontend:** Streamlit

## Status
- [x] Sprint 0: Environment Setup & Model Verification
- [ ] Sprint 1: Data Preparation (arXiv figures)
- [ ] Sprint 2: Fine-tuning
- [ ] Sprint 3: UI Deployment

## Quick Start
1. Open the notebook in notebooks/00_verify_gpu.ipynb
2. Run all cells.
