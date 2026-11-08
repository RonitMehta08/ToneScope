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
ToneScope/
├── app/                      # Streamlit interactive dashboard
│   ├── streamlit_app.py      # Main application entry point
│   └── components/           # Modular Plotly visualization components
├── src/                      # Core source code
│   ├── chat_parsers/         # Multi-format chat parsers (WhatsApp, Slack, Discord, Telegram)
│   ├── models/               # Multi-task RoBERTa & sarcasm corrector architectures
│   ├── data/                 # Dataset loader & context preprocessor
│   ├── inference/            # End-to-end inference pipeline & fallback loader
│   └── analysis/             # Arc construction, cause attribution & escalation risk
├── scripts/                  # Training, evaluation & dataset download scripts
├── tests/                    # Comprehensive unit test suite (26 tests)
├── data/
│   ├── sample_chats/         # Pre-packaged chat exports for instant demo testing
│   ├── raw/                  # (Directory tracked with .gitkeep; raw data excluded)
│   └── processed/            # (Directory tracked with .gitkeep; processed data excluded)
├── models/                   # (Directory tracked with .gitkeep; model binaries excluded)
├── requirements.txt          # Python project dependencies
├── training_guide.md         # Detailed manual training instructions & CLI commands
└── README.md                 # Project documentation
```

---

## 📦 What's Included vs. Excluded in this Repository

To keep the repository lightweight, portable, and compliant with GitHub's 100 MB file size limit, large binary files and raw datasets are excluded via `.gitignore`. The directory skeleton is preserved via `.gitkeep`.

| Component | Status in Git | Why | How to Obtain / Run |
|:---|:---:|:---|:---|
| **Application & Pipeline Code** | ✅ **Included** | Complete Streamlit dashboard, modular visualization components, parsers, and inference logic | Ready to run out of the box |
| **Sample Chat Exports** | ✅ **Included** | Real multi-turn sample conversations (`data/sample_chats/`) for Discord, Slack, and WhatsApp | Use immediately in the dashboard or test suite |
| **Unit Test Suite** | ✅ **Included** | 26 unit tests covering parsers, arcs, cause attribution, escalation, and sarcasm correction | Run `pytest tests/ -v` |
| **Model Checkpoints** (`models/*`) | ❌ **Excluded** | Fine-tuned `.safetensors`, `optimizer.pt`, and checkpoint dumps are ~500 MB – 1 GB each (exceeding GitHub's limit) | **Auto-Fallback (Default):** The dashboard auto-downloads pre-trained HuggingFace base models (`roberta-base`) on first run — no manual download required.<br><br>**Custom Training:** Run `python scripts/train_emotion.py` and `python scripts/train_sarcasm.py` to regenerate the fine-tuned checkpoints into `models/`. |
| **Raw & Processed Datasets** (`data/raw/`, `data/processed/`) | ❌ **Excluded** | Full corpus data (GoEmotions 58K, MELD 13K, DailyDialog 13K, iSarcasmEval) would bloat repo size | Run `python scripts/download_datasets.py` to download and cache all datasets via HuggingFace Hub. |
| **Caches & Virtual Envs** (`venv/`, `.cache/`, `.pytest_cache/`) | ❌ **Excluded** | Machine-specific binaries, environments, and transient caches | Created locally via `python -m venv venv` |


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
