"""
ToneScope Configuration — Centralized constants and hyperparameters.
"""
from pathlib import Path
from dataclasses import dataclass, field


# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_CHATS_DIR = DATA_DIR / "sample_chats"
RAW_DATASETS_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
for _dir in [DATA_DIR, MODELS_DIR, SAMPLE_CHATS_DIR, RAW_DATASETS_DIR, PROCESSED_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─── Emotion Labels ──────────────────────────────────────────────────
EKMAN_EMOTIONS: list[str] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]

EMOTION_TO_ID: dict[str, int] = {e: i for i, e in enumerate(EKMAN_EMOTIONS)}
ID_TO_EMOTION: dict[int, str] = {i: e for e, i in EMOTION_TO_ID.items()}
NUM_EMOTION_LABELS: int = len(EKMAN_EMOTIONS)

# GoEmotions → Ekman mapping (27 labels → 7)
GOEMOTIONS_TO_EKMAN: dict[str, str] = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "neutral",
    "caring": "neutral",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "neutral",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "neutral",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}

# Sarcasm labels
SARCASM_LABELS: list[str] = ["not_sarcastic", "sarcastic"]
NUM_SARCASM_LABELS: int = 2

# Irony subtypes (from iSarcasmEval)
IRONY_TYPES: list[str] = [
    "sarcasm",
    "irony",
    "satire",
    "understatement",
    "overstatement",
    "rhetorical_question",
    "none",
]


# ─── Emotion Colors (for visualization) ──────────────────────────────
EMOTION_COLORS: dict[str, str] = {
    "anger": "#EF4444",
    "disgust": "#8B5CF6",
    "fear": "#6366F1",
    "joy": "#22C55E",
    "sadness": "#3B82F6",
    "surprise": "#F59E0B",
    "neutral": "#9CA3AF",
}

ESCALATION_COLORS: dict[str, str] = {
    "safe": "#22C55E",
    "caution": "#F59E0B",
    "alert": "#EF4444",
}


# ─── Model Configuration ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Hyperparameters for the multi-task emotion model."""

    base_model_name: str = "roberta-base"
    max_seq_length: int = 128
    context_window_size: int = 5  # Number of prior turns for context
    hidden_size: int = 768  # RoBERTa base hidden dim
    context_gru_hidden: int = 256
    context_gru_layers: int = 2
    dropout: float = 0.1
    num_emotion_labels: int = NUM_EMOTION_LABELS
    num_sarcasm_labels: int = NUM_SARCASM_LABELS


@dataclass
class TrainingConfig:
    """Training hyperparameters optimized for RTX 4050 (6GB VRAM)."""

    # Batch sizes (effective_batch = per_device * accumulation_steps)
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8  # effective batch = 32

    # Memory optimization
    fp16: bool = True
    gradient_checkpointing: bool = True

    # Optimizer
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"

    # Schedule
    num_train_epochs: int = 4
    max_steps: int = -1  # -1 means use num_train_epochs

    # Logging & checkpointing
    logging_steps: int = 50
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_weighted_f1"
    greater_is_better: bool = True

    # Seed
    seed: int = 42


@dataclass
class EscalationConfig:
    """Configuration for escalation detection."""

    rolling_window_size: int = 5
    sentiment_weight: float = 0.4
    sarcasm_weight: float = 0.25
    lexical_weight: float = 0.2
    response_latency_weight: float = 0.15
    safe_threshold: float = 0.3
    caution_threshold: float = 0.6


@dataclass
class CauseAttributionConfig:
    """Configuration for emotion cause attribution."""

    max_lookback_turns: int = 10
    similarity_threshold: float = 0.3
    min_emotion_shift_magnitude: float = 0.2


# ─── Singleton instances ──────────────────────────────────────────────
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
ESCALATION_CONFIG = EscalationConfig()
CAUSE_ATTRIBUTION_CONFIG = CauseAttributionConfig()
