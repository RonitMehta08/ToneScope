# 🧠 ToneScope — Multi-Turn Conversation Emotion Arc Analyzer

> *Where minds decode emotion at machine speed.*

**ToneScope** is a production-grade system that analyzes the emotional DNA of multi-turn conversations. It tracks sentiment shifts, detects sarcasm/passive-aggression/irony, predicts escalation risk, and visualizes the complete emotional arc — all through an interactive dashboard.

Built for **CEREBRO 2026** — PS-01: Tone Intelligence in Conversations.

---

## ✨ Key Features

| Feature | Description |
|:---|:---|
| **🎭 Sarcasm-Corrected Emotion Detection** | Detects sarcasm and flips mispredicted literal emotions to their true meaning |
| **📈 Escalation Prediction** | Rolling 0→1 risk score that predicts conflict before it happens |
| **🔍 Emotion Cause Attribution** | Identifies *which prior message* triggered each tone shift |
| **👥 Per-Speaker Emotion Tracking** | Independent emotional trajectory for each participant |
| **📊 7 Interactive Visualizations** | Emotion arcs, sentiment timelines, escalation heatmaps, cause graphs, radar charts |
| **📁 Multi-Format Chat Ingestion** | WhatsApp (.txt), Slack (.json), Discord (.json/.csv), Telegram (.json) |

---

## 🏗️ Architecture

```
Chat Export → Parser → Preprocessor → RoBERTa Encoder
                                          ↓
                              ┌──────────────────────┐
                              │   Multi-Task Heads    │
                              │  • Emotion (7-class)  │
                              │  • Sarcasm (binary)   │
                              │  • Escalation (0→1)   │
                              └──────────────────────┘
                                          ↓
                              ┌──────────────────────┐
                              │   Post-Processing     │
                              │  • Sarcasm Correction │
                              │  • Emotion Arc Build  │
                              │  • Cause Attribution  │
                              │  • Escalation Zones   │
                              └──────────────────────┘
                                          ↓
                              ┌──────────────────────┐
                              │ Streamlit Dashboard   │
                              │  (Interactive Plotly)  │
                              └──────────────────────┘
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch Dashboard (No Training Needed)

```bash
streamlit run app/streamlit_app.py
```

The dashboard auto-falls back to pre-trained HuggingFace models. Upload a chat export or try the demo conversation.

### 3. Train Custom Models (Better Accuracy)

See [training_guide.md](training_guide.md) for detailed step-by-step instructions.

```bash
# Download datasets
python scripts/download_datasets.py

# Train emotion model (~2-4 hours on RTX 4050)
python scripts/train_emotion.py

# Train sarcasm model (~30 minutes)
python scripts/train_sarcasm.py

# Evaluate
python scripts/evaluate.py --dataset all --split test
```

---

## 📊 Datasets (All Real, No Synthetic)

| Dataset | Source | Size | Purpose |
|:---|:---|:---|:---|
| **GoEmotions** | Google Research / Reddit | 58K | Base emotion classification |
| **MELD** | *Friends* dialogues | 13K | Conversational emotion |
| **DailyDialog** | Daily conversations | 13K | Multi-turn emotion |
| **iSarcasmEval** | SemEval-2022 / Author-annotated | 4.5K | Sarcasm detection |

---

## 🖥️ Tech Stack

- **Model:** RoBERTa-base (125M params) — fits in 6GB VRAM with FP16
- **Framework:** PyTorch + HuggingFace Transformers
- **Dashboard:** Streamlit + Plotly
- **Language:** Python 3.10+

---

## 📁 Project Structure

```
Cerebro/
├── app/                      # Streamlit dashboard
│   ├── streamlit_app.py      # Main application
│   └── components/           # Visualization components
├── src/                      # Core source code
│   ├── chat_parsers/         # Multi-format chat parsers
│   ├── models/               # ML model architectures
│   ├── data/                 # Dataset loading & preprocessing
│   ├── inference/            # Inference pipeline
│   └── analysis/             # Post-inference analysis modules
├── scripts/                  # Training & evaluation scripts
├── tests/                    # Test suite
├── data/sample_chats/        # Sample chat exports for demo
├── requirements.txt          # Dependencies
├── training_guide.md         # Step-by-step training commands
└── README.md                 # This file
```

---

## 🧪 Testing

```bash
pytest tests/ -v --tb=short
```

---

## 👥 Team

Built for the CEREBRO 2026 Hackathon — Department of CSE, IEM Kolkata.

---

## 📄 License

This project is developed for the CEREBRO 2026 hackathon and is intended for educational and competition purposes.
