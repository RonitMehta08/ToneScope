# ToneScope — Training Guide

Step-by-step manual commands for training, evaluation, and running the dashboard.
Run these in your terminal from the project root (`c:\Users\ronit\Desktop\Cerebro`).

---

## Prerequisites

### Step 1: Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** For PyTorch with CUDA (RTX 4050), install the CUDA-enabled version:
> ```powershell
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 3: Install spaCy model (optional, for advanced preprocessing)

```powershell
python -m spacy download en_core_web_sm
```

---

## Step 4: Download Datasets

```powershell
python scripts/download_datasets.py
```

This downloads (and caches via HuggingFace Hub):
- GoEmotions (58K Reddit comments, 27 emotions)
- MELD (13K utterances from Friends, 7 emotions)
- DailyDialog (13K dialogues, 7 emotions)
- iSarcasmEval (author-annotated sarcasm data)
- RoBERTa-base model weights

**Estimated time:** 5–15 minutes depending on internet speed.

---

## Step 5: Train Emotion Model (HEAVY — ~2-4 hours on RTX 4050)

```powershell
python scripts/train_emotion.py
```

**Advanced options:**
```powershell
# Fewer epochs (faster but slightly lower accuracy)
python scripts/train_emotion.py --epochs 3

# Higher learning rate
python scripts/train_emotion.py --lr 3e-5

# Train on MELD only (faster, conversation-focused)
python scripts/train_emotion.py --no_goemotions --epochs 6

# Smaller batch size if you get OOM errors
python scripts/train_emotion.py --batch_size 2 --accumulation 16
```

**If you get CUDA Out of Memory:**
1. Reduce `--batch_size` to 2
2. Increase `--accumulation` to 16
3. Reduce `--max_length` to 96

**Output:** Model saved to `models/emotion_model/`

---

## Step 6: Train Sarcasm Model (MODERATE — ~30-60 minutes)

```powershell
python scripts/train_sarcasm.py
```

**Options:**
```powershell
python scripts/train_sarcasm.py --epochs 5 --lr 2e-5
```

**Output:** Model saved to `models/sarcasm_model/`

---

## Step 7: Evaluate Models

```powershell
# Evaluate on MELD test set
python scripts/evaluate.py --dataset meld --split test

# Evaluate on DailyDialog test set
python scripts/evaluate.py --dataset dailydialog --split test

# Evaluate all
python scripts/evaluate.py --dataset all --split test
```

---

## Step 8: Run Tests

```powershell
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v --tb=short
```

---

## Step 9: Launch Dashboard

```powershell
streamlit run app/streamlit_app.py
```

The dashboard will open at `http://localhost:8501`.

**To use:**
1. Upload a chat export (WhatsApp .txt, Slack .json, Discord .json/.csv, Telegram .json)
2. Or click "Load Demo Conversation" to see a showcase
3. Explore the tabs: Emotion Arc, Speaker Lanes, Escalation Map, Sarcasm/Irony, Cause Attribution

---

## Quick Start (Skip Training — Use Pre-trained Models)

If you want to run the dashboard immediately without training:

```powershell
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The pipeline auto-falls back to pre-trained HuggingFace models:
- `SamLowe/roberta-base-go_emotions` for emotion detection
- Rule-based heuristics for sarcasm detection

This works out of the box but custom-trained models will give better results.

---

## Troubleshooting

| Issue | Solution |
|:---|:---|
| CUDA OOM | Reduce batch_size to 2, increase accumulation to 16 |
| Slow training | Use `--no_goemotions` to train only on MELD/DailyDialog |
| Dataset download fails | Check internet connection, retry, or use `--no_<dataset>` flags |
| Import errors | Make sure you're in the project root and venv is activated |
| Streamlit port in use | Use `streamlit run app/streamlit_app.py --server.port 8502` |
