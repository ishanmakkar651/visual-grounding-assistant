# 👁️ VisionEngine Pro: Visual Grounding SaaS

**A Commercial-Grade Multimodal AI System for Technical Document Analysis.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![AI Model](https://img.shields.io/badge/Model-Qwen2--VL--7B-violet)](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **"Turn static PDF schematics into interactive, searchable databases."**

## 🎯 Overview
VisionEngine Pro is an end-to-end **Visual Grounding Assistant**. Unlike standard OCR which just reads text, this system understands **spatial context**. You can upload a complex engineering diagram, click on a specific component (e.g., a valve, a resistor, or a logic gate), and the AI will:
1.  **Identify** the component.
2.  **Explain** its function in the context of the diagram.
3.  **Draw a bounding box** to visually confirm what it is looking at.

## ✨ Key Features (SaaS Edition)
* **📄 Multi-Page PDF Support:** Automatically splits and processes multi-page engineering manuals.
* **💬 Conversational Interface:** ChatGPT-style interface with context memory. Ask follow-up questions about specific regions.
* **🖱️ Point-and-Click Analysis:** Click anywhere on the canvas to trigger a spatial query.
* **🎯 Visual Feedback:** The AI draws Green Bounding Boxes on the canvas to show exactly what it detected.
* **⚡ Zero-Latency UI:** Custom split-screen interface with Zoom/Pan controls built in pure JS/Canvas (No heavy frontend frameworks).

## 🏗️ Technical Architecture
* **Core Model:** [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) (Fine-tuned with QLoRA).
* **Quantization:** 4-bit NF4 (BitsAndBytes) for running on consumer GPUs (T4).
* **Backend:** FastAPI (Python) with Python-Multipart for handling image/PDF streams.
* **Session Management:** UUID-based session storage to handle multiple users simultaneously.
* **Frontend:** HTML5 Canvas + TailwindCSS for a responsive, dark-mode dashboard.

## 🚀 Quick Start (Google Colab)
You can run the full SaaS backend on a free Google Colab T4 GPU.

1.  **Clone the Repo**
2.  **Install Requirements** (`pip install -r requirements.txt`)
3.  **Run the Server:**
    ```bash
    uvicorn src.api:app --host 0.0.0.0 --port 8000
    ```
4.  **Access UI:** Open `http://localhost:8000`

## 📂 Project Structure
```bash
├── src/
│   ├── api.py           # SaaS Backend (Session handling, Model Inference)
│   ├── storage/         # Temp storage for user uploads (Session isolated)
│   └── static/
│       └── index.html   # The "Masterpiece" Dark Mode UI
├── requirements.txt     # Production dependencies
└── README.md
```
